"""
Diffusion Attention Module

Ported from existing Heat Kernel Attention codebase (Curry, 2025).
Supports three modes:
  - Standard softmax (baseline)
  - Fixed-t diffusion attention
  - Constrained learned-t diffusion attention

The key insight: softmax(QK^T / sqrt(d)) is equilibrium diffusion.
Finite-time diffusion (temperature = 2t) provides calibration benefits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256

    # Attention mode: "softmax", "diffusion_fixed", "diffusion_learned"
    mode: str = "diffusion_fixed"

    # Fixed diffusion time
    fixed_t: float = 0.28

    # Positional decay (heat kernel locality)
    # alpha=0: standard diffusion, alpha=1: full heat kernel
    alpha: float = 0.0

    # Learned t constraints (for mode="diffusion_learned")
    t_min: float = 0.05
    t_max: float = 2.0

    # Depth scaling: t_center = t_base * sqrt(L_base / n_layers)
    # Set these when constructing per-layer configs
    t_center: Optional[float] = None


def compute_effective_radius(t: float, alpha: float, epsilon: float = 1e-6) -> int:
    """Compute effective attention radius from diffusion parameters."""
    if alpha <= 0:
        return -1  # Infinite (global attention)
    raw_radius = math.sqrt(4 * t * math.log(1 / epsilon) / alpha)
    return int(math.ceil(raw_radius))


class DiffusionMultiHeadAttention(nn.Module):
    """
    Multi-head attention with diffusion time control.

    Implements: score_ij = (q_i · k_j) / (2t) - alpha * d(i,j)^2 / (4t)
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.mode = config.mode
        self.alpha = config.alpha
        self.dropout_p = config.dropout
        self.max_seq_len = config.max_seq_len

        # Projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Diffusion time setup
        if config.mode == "softmax":
            # Standard attention: temperature = sqrt(d_k), so t = sqrt(d_k)/2
            self.register_buffer("t", torch.tensor(math.sqrt(self.d_k) / 2.0))
            self._learned_t = False
        elif config.mode == "diffusion_fixed":
            self.register_buffer("t", torch.tensor(config.fixed_t))
            self._learned_t = False
        elif config.mode == "diffusion_learned":
            # Learned log-t per head, constrained to [t_min, t_max]
            t_center = config.t_center if config.t_center else config.fixed_t
            init_log_t = math.log(t_center)
            self.log_t = nn.Parameter(
                torch.full((config.n_heads,), init_log_t)
            )
            self.log_t_min = math.log(config.t_min)
            self.log_t_max = math.log(config.t_max)
            self._learned_t = True
        else:
            raise ValueError(f"Unknown attention mode: {config.mode}")

        # Precompute positional distance matrix for heat kernel locality
        if self.alpha > 0:
            pos = torch.arange(config.max_seq_len, dtype=torch.float32)
            dist_sq = (pos.unsqueeze(0) - pos.unsqueeze(1)) ** 2
            self.register_buffer("dist_sq", dist_sq)

        # For tracking
        self._current_t = None

    def get_t(self) -> torch.Tensor:
        """Get current diffusion time (scalar or per-head)."""
        if self._learned_t:
            # Clamp log_t to valid range, then exponentiate
            clamped = torch.clamp(self.log_t, self.log_t_min, self.log_t_max)
            return torch.exp(clamped)  # (n_heads,)
        else:
            return self.t

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (1, 1, seq_len, seq_len) causal mask

        Returns:
            output: (batch, seq_len, d_model)
            info: dict with attention statistics
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # q, k, v: (batch, n_heads, seq_len, d_k)

        # Get diffusion time
        t = self.get_t()
        self._current_t = t.detach()

        # Compute content scores: Q·K^T
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, heads, seq, seq)

        # Apply diffusion scaling: divide by 2t
        if self._learned_t:
            # t is (n_heads,) -> reshape to (1, n_heads, 1, 1)
            t_scale = t.view(1, self.n_heads, 1, 1)
            scores = scores / (2 * t_scale)
        else:
            scores = scores / (2 * t)

        # Apply positional decay (heat kernel locality)
        if self.alpha > 0:
            dist_sq = self.dist_sq[:seq_len, :seq_len]  # (seq, seq)
            if self._learned_t:
                penalty = -self.alpha * dist_sq.unsqueeze(0).unsqueeze(0) / (
                    4 * t_scale
                )
            else:
                penalty = -self.alpha * dist_sq / (4 * t)
                penalty = penalty.unsqueeze(0).unsqueeze(0)
            scores = scores + penalty

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax normalization
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Dropout
        if self.dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        # Apply to values
        out = torch.matmul(attn_weights, v)  # (batch, heads, seq, d_k)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.W_o(out)

        info = {
            "t": t.detach().cpu() if self._learned_t else t.item(),
            "mode": self.mode,
        }

        return out, info


class TransformerBlock(nn.Module):
    """Transformer block with diffusion attention."""

    def __init__(self, attn_config: AttentionConfig, d_ff: int):
        super().__init__()

        d_model = attn_config.d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = DiffusionMultiHeadAttention(attn_config)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(attn_config.dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        # Pre-norm attention
        attn_out, info = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))

        return x, info
