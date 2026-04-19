"""
Transformer Model for Diffusion-Native Language Modeling

Supports two operational modes:
1. Standard: Token embeddings → transformer → logits → CE loss
2. Diffusion-native: Token embeddings → corrupt → transformer → recover →
   compound loss (cosine rec + CE readout + spectral coherence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from dataclasses import dataclass, field

from .attention import AttentionConfig, TransformerBlock, compute_effective_radius


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 1024
    dropout: float = 0.1

    # Attention mode: "softmax", "diffusion_fixed", "diffusion_learned"
    attention_mode: str = "diffusion_fixed"

    # Diffusion parameters
    fixed_t: float = 0.16  # depth-scaled for 12 layers
    alpha: float = 0.0     # positional decay (0 = no locality)
    t_min: float = 0.05
    t_max: float = 2.0

    # Depth scaling law: t_base at L_base layers
    t_base: float = 0.28
    L_base: int = 4

    # Training mode: "standard" (CE only) or "diffusion_native" (compound loss)
    training_mode: str = "standard"


class DiffusionNativeTransformer(nn.Module):
    """
    Transformer with optional diffusion-native training.

    In 'standard' mode: behaves like a normal LM transformer.
    In 'diffusion_native' mode: accepts corrupted embeddings as input,
    outputs recovered embeddings for compound loss computation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Compute depth-scaled t for each layer
        t_center = config.t_base * math.sqrt(config.L_base / config.n_layers)

        # Build transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            attn_config = AttentionConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                mode=config.attention_mode,
                fixed_t=t_center,
                alpha=config.alpha,
                t_min=config.t_min,
                t_max=config.t_max,
                t_center=t_center,
            )
            self.blocks.append(TransformerBlock(attn_config, config.d_ff))

        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .unsqueeze(0)
            .unsqueeze(0),
        )

        # Initialize
        self.apply(self._init_weights)

        # Report
        t_eff = t_center
        radius = compute_effective_radius(t_eff, config.alpha)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model: {config.n_layers}L, d={config.d_model}, "
              f"{config.attention_mode}, t_center={t_eff:.4f}")
        if config.alpha > 0:
            print(f"  Heat kernel: alpha={config.alpha}, radius={radius}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Training mode: {config.training_mode}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token + position embeddings (before any corruption)."""
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        emb = self.token_emb(input_ids) + self.pos_emb(positions)
        return emb

    def forward_from_embeddings(
        self, h: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass from embeddings (for diffusion-native mode).

        Args:
            h: (batch, seq_len, d_model) — possibly corrupted embeddings

        Returns:
            dict with 'logits' and 'hidden' (pre-lm-head representation)
        """
        batch, seq_len, _ = h.shape

        h = self.dropout(h)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        t_values = []
        for block in self.blocks:
            h, info = block(h, mask=mask)
            t_values.append(info.get("t", None))

        h = self.ln_f(h)
        logits = self.lm_head(h)

        return {
            "logits": logits,
            "hidden": h,
            "t_values": t_values,
        }

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass from token IDs.

        Returns dict with 'logits', 'hidden', 'embeddings', 't_values'.
        """
        emb = self.get_embeddings(input_ids)
        output = self.forward_from_embeddings(emb)
        output["embeddings"] = emb
        return output
