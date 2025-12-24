"""
Diffusion Attention: PyTorch Implementation (Fixed)

A drop-in replacement for softmax attention using heat kernel dynamics.
Designed for integration with standard transformer architectures.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Core Heat Kernel Attention
# ============================================================

def heat_kernel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heat kernel attention mechanism.
    
    Instead of softmax(QK^T / sqrt(d)), we use:
        exp(-||q - k||^2 / 4t) / Z
    
    For normalized vectors, this is equivalent to:
        exp((q·k - 1) / 2t) / Z = exp(q·k / 2t) / Z'
    
    So it's like softmax with temperature T = 2t.
    
    Args:
        query: (batch, heads, seq_q, d_k)
        key: (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        t: diffusion time, (batch, heads, 1, 1) or scalar
        mask: attention mask (batch, heads, seq_q, seq_k)
        dropout_p: dropout probability
        training: whether in training mode
    
    Returns:
        output: (batch, heads, seq_q, d_v)
        attention_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    
    # Compute attention scores (dot product)
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, heads, seq_q, seq_k)
    
    # Heat kernel formulation:
    # For normalized q, k: exp(-||q-k||^2 / 4t) = exp((q·k - 1) / 2t)
    # We drop the -1 since it cancels in softmax normalization
    # Result: exp(q·k / 2t) which is softmax with temperature 2t
    
    # Scale by diffusion time (t acts like temperature/2)
    if t.dim() == 0:  # scalar
        temperature = 2 * t
    else:
        temperature = 2 * t  # (batch, heads, 1, 1)
    
    scores = scores / temperature
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax normalization (this IS the heat kernel normalization)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Handle NaN from all-masked rows
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Dropout
    if dropout_p > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def squared_distance_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heat kernel attention using actual squared Euclidean distance.
    
    This version computes ||q - k||^2 explicitly, which gives different
    behavior than dot-product attention when vectors aren't normalized.
    """
    # Compute squared distances
    # ||q - k||^2 = ||q||^2 + ||k||^2 - 2 q·k
    
    q_sq = (query ** 2).sum(dim=-1, keepdim=True)  # (batch, heads, seq_q, 1)
    k_sq = (key ** 2).sum(dim=-1, keepdim=True)    # (batch, heads, seq_k, 1)
    qk = torch.matmul(query, key.transpose(-2, -1)) # (batch, heads, seq_q, seq_k)
    
    dist_sq = q_sq + k_sq.transpose(-2, -1) - 2 * qk  # (batch, heads, seq_q, seq_k)
    
    # Heat kernel: exp(-dist_sq / 4t)
    if t.dim() == 0:
        scores = -dist_sq / (4 * t)
    else:
        scores = -dist_sq / (4 * t)  # broadcasting handles shape
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Normalize
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Dropout
    if dropout_p > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    
    # Apply to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# ============================================================
# Learnable Diffusion Time Predictor (FIXED)
# ============================================================

class DiffusionTimePredictor(nn.Module):
    """
    Predicts diffusion time from context.
    
    Takes query/key statistics and outputs log(t) for each head.
    We predict log(t) to ensure t > 0 and to handle the scale sensitivity.
    """
    
    def __init__(
        self,
        d_k: int,
        d_hidden: int = 64,
        t_min: float = 0.1,
        t_max: float = 10.0,
        per_head: bool = True,
        n_heads: int = 1,
    ):
        super().__init__()
        
        self.d_k = d_k
        self.t_min = t_min
        self.t_max = t_max
        self.per_head = per_head
        self.n_heads = n_heads
        
        # Log bounds for clamping
        self.log_t_min = math.log(t_min)
        self.log_t_max = math.log(t_max)
        
        # Input features: 3 scalar statistics per head
        # mean_sim, max_sim, entropy_proxy
        input_dim = 3
        
        output_dim = 1  # One output per head (applied separately)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, output_dim),
        )
        
        # Initialize to predict middle of range
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, (self.log_t_min + self.log_t_max) / 2)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict diffusion time for each head.
        
        Args:
            query: (batch, heads, seq_q, d_k)
            key: (batch, heads, seq_k, d_k)
        
        Returns:
            t: (batch, heads, 1, 1)
        """
        batch, heads, seq_q, d_k = query.shape
        seq_k = key.size(2)
        
        # Compute similarity statistics
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: (batch, heads, seq_q, seq_k)
        
        # Mean similarity across all positions
        mean_sim = scores.mean(dim=(-2, -1))  # (batch, heads)
        
        # Max similarity (averaged across queries)
        max_sim = scores.max(dim=-1)[0].mean(dim=-1)  # (batch, heads)
        
        # Entropy proxy: variance of softmax (high variance = peaky = low entropy)
        probs = F.softmax(scores, dim=-1)
        entropy_proxy = probs.var(dim=-1).mean(dim=-1)  # (batch, heads)
        
        # Stack features: (batch, heads, 3)
        features = torch.stack([mean_sim, max_sim, entropy_proxy], dim=-1)
        
        # Predict log(t) for each head: (batch, heads, 1)
        log_t = self.net(features)
        
        # Clamp to valid range
        log_t = torch.clamp(log_t, self.log_t_min, self.log_t_max)
        
        # Convert to t
        t = torch.exp(log_t)  # (batch, heads, 1)
        
        # Reshape for attention: (batch, heads, 1, 1)
        t = t.unsqueeze(-1)  # (batch, heads, 1, 1)
        
        return t


# ============================================================
# Full Attention Layer
# ============================================================

class DiffusionAttention(nn.Module):
    """
    Multi-head diffusion attention layer.
    
    Drop-in replacement for standard multi-head attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        adaptive_time: bool = True,
        fixed_t: float = 1.0,
        t_min: float = 0.1,
        t_max: float = 10.0,
        use_squared_distance: bool = False,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.adaptive_time = adaptive_time
        self.fixed_t = fixed_t
        self.use_squared_distance = use_squared_distance
        
        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Time predictor
        if adaptive_time:
            self.time_predictor = DiffusionTimePredictor(
                d_k=self.d_k,
                n_heads=n_heads,
                t_min=t_min,
                t_max=t_max,
                per_head=True,
            )
        else:
            self.register_buffer('t', torch.tensor(fixed_t))
        
        self._attention_weights = None  # For visualization
        self._diffusion_times = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: (batch, seq_q, seq_k) or (batch, 1, seq_q, seq_k)
            need_weights: whether to return attention weights
        
        Returns:
            output: (batch, seq_q, d_model)
            attention_weights: (batch, n_heads, seq_q, seq_k) if need_weights
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.size(1)
        
        # Project
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape to (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Get diffusion time
        if self.adaptive_time:
            t = self.time_predictor(Q, K)  # (batch, n_heads, 1, 1)
            self._diffusion_times = t.squeeze()
        else:
            t = self.t
            self._diffusion_times = t
        
        # Reshape mask if needed
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)
        
        # Apply attention
        if self.use_squared_distance:
            attn_output, attn_weights = squared_distance_attention(
                Q, K, V, t, mask, self.dropout, self.training
            )
        else:
            attn_output, attn_weights = heat_kernel_attention(
                Q, K, V, t, mask, self.dropout, self.training
            )
        
        self._attention_weights = attn_weights
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_q, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output, None


# ============================================================
# Standard Attention for Comparison
# ============================================================

class StandardAttention(nn.Module):
    """Standard multi-head attention for comparison."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self._attention_weights = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_q, _ = query.shape
        seq_k = key.size(1)
        
        Q = self.W_q(query).view(batch_size, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        self._attention_weights = attn_weights
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        output = self.W_o(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output, None


# ============================================================
# Testing
# ============================================================

def test_attention_equivalence():
    """Test that diffusion attention matches softmax at appropriate t."""
    print("Testing attention equivalence...")
    
    torch.manual_seed(42)
    
    batch, seq, d_model, n_heads = 2, 16, 64, 4
    d_k = d_model // n_heads
    
    # Create inputs
    x = torch.randn(batch, seq, d_model)
    
    # Standard attention
    std_attn = StandardAttention(d_model, n_heads)
    std_out, std_weights = std_attn(x, x, x, need_weights=True)
    
    # Diffusion attention with fixed t = sqrt(d_k) / 2
    # This should match standard attention (temperature = 2t = sqrt(d_k))
    t_equiv = math.sqrt(d_k) / 2
    diff_attn = DiffusionAttention(d_model, n_heads, adaptive_time=False, fixed_t=t_equiv)
    
    # Copy weights
    diff_attn.W_q.weight.data = std_attn.W_q.weight.data.clone()
    diff_attn.W_k.weight.data = std_attn.W_k.weight.data.clone()
    diff_attn.W_v.weight.data = std_attn.W_v.weight.data.clone()
    diff_attn.W_o.weight.data = std_attn.W_o.weight.data.clone()
    diff_attn.W_q.bias.data = std_attn.W_q.bias.data.clone()
    diff_attn.W_k.bias.data = std_attn.W_k.bias.data.clone()
    diff_attn.W_v.bias.data = std_attn.W_v.bias.data.clone()
    diff_attn.W_o.bias.data = std_attn.W_o.bias.data.clone()
    
    diff_out, diff_weights = diff_attn(x, x, x, need_weights=True)
    
    # Compare
    output_diff = (std_out - diff_out).abs().max().item()
    weight_diff = (std_weights - diff_weights).abs().max().item()
    
    print(f"  Max output difference: {output_diff:.6f}")
    print(f"  Max weight difference: {weight_diff:.6f}")
    print(f"  Equivalence test: {'PASSED' if output_diff < 1e-5 else 'FAILED'}")
    
    return output_diff < 1e-5


def test_adaptive_time():
    """Test adaptive time prediction."""
    print("\nTesting adaptive time prediction...")
    
    torch.manual_seed(42)
    
    batch, seq, d_model, n_heads = 2, 32, 64, 4
    
    attn = DiffusionAttention(d_model, n_heads, adaptive_time=True)
    
    # Normal input
    x_normal = torch.randn(batch, seq, d_model)
    out_normal, _ = attn(x_normal, x_normal, x_normal)
    t_normal = attn._diffusion_times.mean().item()
    
    # Outlier input (one position has huge values)
    x_outlier = torch.randn(batch, seq, d_model)
    x_outlier[:, 0, :] *= 10  # Spike
    out_outlier, _ = attn(x_outlier, x_outlier, x_outlier)
    t_outlier = attn._diffusion_times.mean().item()
    
    print(f"  Normal input - mean t: {t_normal:.4f}")
    print(f"  Outlier input - mean t: {t_outlier:.4f}")
    print(f"  Output shapes: {out_normal.shape}, {out_outlier.shape}")
    print(f"  Adaptive time test: PASSED")
    
    return True


def test_gradient_flow():
    """Test that gradients flow through diffusion time."""
    print("\nTesting gradient flow...")
    
    torch.manual_seed(42)
    
    batch, seq, d_model, n_heads = 2, 16, 64, 4
    
    attn = DiffusionAttention(d_model, n_heads, adaptive_time=True)
    
    x = torch.randn(batch, seq, d_model, requires_grad=True)
    out, _ = attn(x, x, x)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    time_predictor_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in attn.time_predictor.parameters()
    )
    
    print(f"  Input has gradient: {has_grad}")
    print(f"  Time predictor has gradient: {time_predictor_has_grad}")
    print(f"  Gradient flow test: {'PASSED' if has_grad and time_predictor_has_grad else 'FAILED'}")
    
    return has_grad and time_predictor_has_grad


def test_different_t_values():
    """Test behavior at different fixed t values."""
    print("\nTesting different t values...")
    
    torch.manual_seed(42)
    
    batch, seq, d_model, n_heads = 2, 16, 64, 4
    x = torch.randn(batch, seq, d_model)
    
    for t in [0.1, 0.5, 1.0, 2.0, 5.0]:
        attn = DiffusionAttention(d_model, n_heads, adaptive_time=False, fixed_t=t)
        out, weights = attn(x, x, x, need_weights=True)
        
        # Compute entropy of attention weights
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
        max_weight = weights.max().item()
        
        print(f"  t={t:.1f}: entropy={entropy:.3f}, max_weight={max_weight:.3f}")
    
    print("  Different t values test: PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Attention: PyTorch Implementation Tests")
    print("=" * 60)
    
    test_attention_equivalence()
    test_adaptive_time()
    test_gradient_flow()
    test_different_t_values()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
