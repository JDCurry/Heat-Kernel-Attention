"""
Compound Loss for Diffusion-Native Language Modeling

Three-term objective:
  L_total = λ₁ · L_rec + λ₂ · L_tok + λ₃ · L_spec

Where:
  L_rec:  Cosine reconstruction — recover clean embedding direction
  L_tok:  Cross-entropy readout — project to vocab, match target tokens
  L_spec: Spectral coherence — penalize high-frequency roughness on graph

The key insight: CE is not the primary ontology. It is the readout constraint
that keeps the continuous field legible to the discrete world.

Curriculum weighting phases:
  Early:  High reconstruction, low CE, no spectral
  Middle: Balance reconstruction and CE, introduce spectral
  Late:   Tune for downstream calibration + performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from ..model.corruption import build_line_graph_laplacian


class CosineReconstructionLoss(nn.Module):
    """
    Cosine reconstruction loss: 1 - cos(recovered, clean).

    Measures whether the recovered embedding points in the same direction
    as the clean embedding. Focuses on the semantically meaningful part
    of the space (direction) rather than magnitude.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, recovered: torch.Tensor, clean: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recovered: (batch, seq_len, d_model) — model output embeddings
            clean: (batch, seq_len, d_model) — original clean embeddings

        Returns:
            loss: scalar, mean over batch and sequence
        """
        # Normalize along embedding dimension
        rec_norm = F.normalize(recovered, p=2, dim=-1)
        clean_norm = F.normalize(clean, p=2, dim=-1)

        # Cosine similarity: (batch, seq_len)
        cos_sim = (rec_norm * clean_norm).sum(dim=-1)

        # Loss: 1 - cosine similarity, averaged
        loss = (1.0 - cos_sim).mean()

        return loss


class SpectralCoherenceLoss(nn.Module):
    """
    Spectral coherence loss: X̂ᵀ L X̂

    Penalizes high-frequency roughness over the sequence graph.
    Encourages the recovered state to be coherent with respect to
    sequence geometry.

    CAUTION: Start with small weight. Oversmoothing destroys the
    semantic edges the model needs for token discrimination.
    Monitor token recovery accuracy as diagnostic.
    """

    def __init__(self, max_seq_len: int = 256):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Precompute Laplacian
        L = build_line_graph_laplacian(max_seq_len, torch.device("cpu"))
        self.register_buffer("L", L)

    def forward(self, recovered: torch.Tensor) -> torch.Tensor:
        """
        Args:
            recovered: (batch, seq_len, d_model) — model output embeddings

        Returns:
            loss: scalar, mean spectral energy (normalized by batch, seq, dim)
        """
        batch, seq_len, d_model = recovered.shape

        L = self.L[:seq_len, :seq_len]  # (seq_len, seq_len)

        # Compute X̂ᵀ L X̂ per embedding dimension
        # L @ X̂: (seq_len, seq_len) @ (batch, seq_len, d_model) -> matmul
        Lx = torch.matmul(L.unsqueeze(0), recovered)  # (batch, seq_len, d_model)

        # X̂ᵀ L X̂ = sum over seq of X̂ * (L @ X̂), per dimension
        # (batch, seq_len, d_model) * (batch, seq_len, d_model) -> sum over seq
        energy = (recovered * Lx).sum(dim=1)  # (batch, d_model)

        # Mean over batch and dimensions, normalize by seq_len
        loss = energy.mean() / seq_len

        return loss


class MSEReconstructionLoss(nn.Module):
    """
    MSE reconstruction loss (for ablation: cosine vs cosine+MSE).
    """

    def forward(
        self, recovered: torch.Tensor, clean: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(recovered, clean)


class CurriculumScheduler:
    """
    Manages curriculum weighting of loss terms over training.

    Three phases:
      Early (0-30%):  λ_rec=1.0, λ_tok=0.3, λ_spec=0.0
      Middle (30-70%): λ_rec=0.7, λ_tok=0.7, λ_spec=0.05
      Late (70-100%):  λ_rec=0.5, λ_tok=1.0, λ_spec=0.1

    Transitions are linear interpolation for smooth training.
    """

    def __init__(
        self,
        total_steps: int,
        # Phase boundaries (fraction of total steps)
        early_end: float = 0.3,
        late_start: float = 0.7,
        # Early phase weights
        early_rec: float = 1.0,
        early_tok: float = 0.3,
        early_spec: float = 0.0,
        # Middle phase weights
        mid_rec: float = 0.7,
        mid_tok: float = 0.7,
        mid_spec: float = 0.05,
        # Late phase weights
        late_rec: float = 0.5,
        late_tok: float = 1.0,
        late_spec: float = 0.1,
    ):
        self.total_steps = total_steps
        self.early_end = early_end
        self.late_start = late_start

        self.phases = {
            "early": {"rec": early_rec, "tok": early_tok, "spec": early_spec},
            "middle": {"rec": mid_rec, "tok": mid_tok, "spec": mid_spec},
            "late": {"rec": late_rec, "tok": late_tok, "spec": late_spec},
        }

    def get_weights(self, step: int) -> Dict[str, float]:
        """
        Get loss weights for the current training step.

        Returns:
            dict with keys 'rec', 'tok', 'spec' and float values
        """
        progress = step / max(self.total_steps, 1)

        early = self.phases["early"]
        mid = self.phases["middle"]
        late = self.phases["late"]

        if progress < self.early_end:
            # Early phase or transitioning to middle
            alpha = progress / self.early_end
            return {
                k: early[k] + alpha * (mid[k] - early[k])
                for k in ["rec", "tok", "spec"]
            }
        elif progress < self.late_start:
            # Middle phase
            return dict(mid)
        else:
            # Transitioning to late
            alpha = (progress - self.late_start) / (1.0 - self.late_start)
            alpha = min(alpha, 1.0)
            return {
                k: mid[k] + alpha * (late[k] - mid[k])
                for k in ["rec", "tok", "spec"]
            }

    def get_phase_name(self, step: int) -> str:
        progress = step / max(self.total_steps, 1)
        if progress < self.early_end:
            return "early"
        elif progress < self.late_start:
            return "middle"
        else:
            return "late"


class CompoundLoss(nn.Module):
    """
    Compound loss for diffusion-native language modeling.

    L_total = λ_rec · L_rec + λ_tok · L_tok + λ_spec · L_spec

    Operates in three modes:
    1. 'standard': Just cross-entropy (for baselines A1-A3, B3)
    2. 'compound': Full three-term loss with curriculum (for B1-B4)
    3. 'auxiliary': CE is primary (λ_tok=1.0 always), reconstruction and
                   spectral are small fixed auxiliary losses. No curriculum.
                   This is the Option C approach — can't hurt perplexity,
                   might provide calibration regularization.
    """

    def __init__(
        self,
        mode: str = "compound",
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        use_mse: bool = False,       # Ablation: add MSE to cosine
        mse_weight: float = 0.1,     # Weight for MSE when use_mse=True
        # Auxiliary mode weights (fixed, no curriculum)
        aux_rec_weight: float = 0.05,
        aux_spec_weight: float = 0.01,
    ):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size
        self.aux_rec_weight = aux_rec_weight
        self.aux_spec_weight = aux_spec_weight

        if mode in ("compound", "auxiliary"):
            self.cosine_loss = CosineReconstructionLoss()
            self.spectral_loss = SpectralCoherenceLoss(max_seq_len)
            if use_mse:
                self.mse_loss = MSEReconstructionLoss()
            self.use_mse = use_mse
            self.mse_weight = mse_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        recovered_hidden: Optional[torch.Tensor] = None,
        clean_embeddings: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute compound loss.

        Args:
            logits: (batch, seq_len, vocab_size) — model output logits
            targets: (batch, seq_len) — target token IDs
            recovered_hidden: (batch, seq_len, d_model) — recovered embeddings
                              (only needed for compound/auxiliary mode)
            clean_embeddings: (batch, seq_len, d_model) — original clean embeddings
                              (only needed for compound/auxiliary mode)
            weights: dict with 'rec', 'tok', 'spec' weights from curriculum
                     (only used in compound mode; auxiliary uses fixed weights)

        Returns:
            total_loss: scalar
            components: dict with individual loss values for logging
        """
        # Always compute CE (needed for all modes)
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), targets.view(-1)
        )

        if self.mode == "standard":
            return ce_loss, {
                "total": ce_loss.item(),
                "ce": ce_loss.item(),
            }

        if self.mode == "auxiliary":
            # Option C: CE is primary, auxiliary terms are small constants
            components = {"ce": ce_loss.item()}
            total = ce_loss  # λ_tok = 1.0 always

            if recovered_hidden is not None and clean_embeddings is not None:
                rec_loss = self.cosine_loss(recovered_hidden, clean_embeddings)
                components["rec_cosine"] = rec_loss.item()
                total = total + self.aux_rec_weight * rec_loss

                if self.use_mse:
                    mse_loss = self.mse_loss(recovered_hidden, clean_embeddings)
                    components["rec_mse"] = mse_loss.item()
                    total = total + self.aux_rec_weight * self.mse_weight * mse_loss

            if recovered_hidden is not None and self.aux_spec_weight > 0:
                spec_loss = self.spectral_loss(recovered_hidden)
                components["spec"] = spec_loss.item()
                total = total + self.aux_spec_weight * spec_loss

            components["total"] = total.item()
            components["weights"] = {
                "rec": self.aux_rec_weight,
                "tok": 1.0,
                "spec": self.aux_spec_weight,
            }
            return total, components

        # Compound mode (original B-series)
        if weights is None:
            weights = {"rec": 0.7, "tok": 0.7, "spec": 0.05}

        components = {"ce": ce_loss.item()}

        total = weights["tok"] * ce_loss

        # Reconstruction loss
        if recovered_hidden is not None and clean_embeddings is not None:
            rec_loss = self.cosine_loss(recovered_hidden, clean_embeddings)
            components["rec_cosine"] = rec_loss.item()
            total = total + weights["rec"] * rec_loss

            if self.use_mse:
                mse_loss = self.mse_loss(recovered_hidden, clean_embeddings)
                components["rec_mse"] = mse_loss.item()
                total = total + weights["rec"] * self.mse_weight * mse_loss

        # Spectral coherence
        if recovered_hidden is not None and weights["spec"] > 0:
            spec_loss = self.spectral_loss(recovered_hidden)
            components["spec"] = spec_loss.item()
            total = total + weights["spec"] * spec_loss

        components["total"] = total.item()
        components["weights"] = weights

        return total, components
