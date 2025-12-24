"""
Analyze Attention Patterns in Trained Diffusion Attention Models

This script loads a trained model and analyzes:
1. Actual attention weight distributions
2. How much weight falls within different radii
3. Empirical sparsity achievable without quality loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import the model classes
try:
    from diffusion_attention_torch import DiffusionAttention, StandardAttention
    HAS_ATTENTION_MODULE = True
except ImportError:
    HAS_ATTENTION_MODULE = False
    print("Warning: diffusion_attention_torch not found, will reconstruct model")


def analyze_attention_pattern(
    attn_weights: torch.Tensor,
    thresholds: list = [0.9, 0.95, 0.99, 0.999]
) -> dict:
    """
    Analyze attention weight distribution vs distance.
    """
    batch, heads, seq_q, seq_k = attn_weights.shape
    device = attn_weights.device
    
    # Create distance matrix
    positions_q = torch.arange(seq_q, device=device)
    positions_k = torch.arange(seq_k, device=device)
    distance = (positions_q.unsqueeze(1) - positions_k.unsqueeze(0)).abs().float()
    
    results = {
        'mean_entropy': [],
        'weight_at_distance': [],
        'radius_for_threshold': {t: [] for t in thresholds},
        'outside_mass': {},  # NEW: mass outside different radii
    }
    
    # Flatten batch and heads
    attn_flat = attn_weights.view(-1, seq_q, seq_k)
    
    # Compute entropy
    entropy = -(attn_flat * (attn_flat + 1e-10).log()).sum(dim=-1)
    results['mean_entropy'] = entropy.mean().item()
    results['max_entropy'] = math.log(seq_k)
    
    # For each distance, compute mean attention weight
    max_distance = int(distance.max().item())
    weight_by_distance = []
    
    for d in range(max_distance + 1):
        mask = (distance == d)
        if mask.sum() > 0:
            weights_at_d = attn_flat[:, mask].mean().item()
        else:
            weights_at_d = 0.0
        weight_by_distance.append(weights_at_d)
    
    results['weight_at_distance'] = weight_by_distance
    
    # Compute outside mass for different radii (GPT's suggestion)
    # For each query position, what fraction of its attention weight falls outside radius R?
    for radius in [5, 10, 20, 50, 100]:
        if radius < seq_k:
            outside_mask = distance > radius  # (seq_q, seq_k)
            # For causal attention, only count valid positions
            causal_mask = positions_q.unsqueeze(1) >= positions_k.unsqueeze(0)  # (seq_q, seq_k)
            valid_outside = outside_mask & causal_mask  # (seq_q, seq_k)
            
            # For each (batch*head, query), compute sum of attention outside radius
            # attn_flat is (B*H, seq_q, seq_k)
            # valid_outside is (seq_q, seq_k)
            # We want: for each query q, sum attn_flat[:, q, k] where valid_outside[q, k] is True
            
            # Expand mask to broadcast with attn_flat
            valid_outside_expanded = valid_outside.unsqueeze(0)  # (1, seq_q, seq_k)
            
            # Zero out positions not outside, then sum over keys
            outside_weights = attn_flat * valid_outside_expanded.float()  # (B*H, seq_q, seq_k)
            outside_per_query = outside_weights.sum(dim=-1)  # (B*H, seq_q) - sum over keys
            
            # Average over all queries and batches
            outside_mass = outside_per_query.mean().item()
            results['outside_mass'][radius] = outside_mass
    
    # For each query, find radius needed to capture threshold of weight
    for threshold in thresholds:
        radii = []
        # Sample to speed up (don't need every single query)
        sample_size = min(attn_flat.shape[0] * seq_q, 1000)
        indices = np.random.choice(attn_flat.shape[0] * seq_q, sample_size, replace=False)
        
        for idx in indices:
            b = idx // seq_q
            q = idx % seq_q
            
            attn_row = attn_flat[b, q]
            distances_from_q = distance[q]
            
            # Sort by distance
            sorted_indices = distances_from_q.argsort()
            sorted_weights = attn_row[sorted_indices]
            cumsum = sorted_weights.cumsum(dim=0)
            
            above_threshold = (cumsum >= threshold).nonzero(as_tuple=True)[0]
            if len(above_threshold) > 0:
                radius_idx = above_threshold[0].item()
                radius = distances_from_q[sorted_indices[radius_idx]].item()
            else:
                radius = seq_k - 1
            radii.append(radius)
        
        results['radius_for_threshold'][threshold] = {
            'mean': np.mean(radii),
            'median': np.median(radii),
            'p95': np.percentile(radii, 95),
            'max': np.max(radii),
        }
    
    return results


def load_model_and_config(checkpoint_path: str, device: str = 'cuda'):
    """Load model checkpoint and config."""
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / 'config.json'
    
    # Load config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        print(f"Warning: config.json not found at {config_path}")
        config = {}
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    return checkpoint, config


def extract_attention_weights_from_checkpoint(
    checkpoint: dict,
    config: dict,
    input_ids: torch.Tensor,
    device: str = 'cuda'
) -> list:
    """
    Extract attention patterns by reconstructing forward pass.
    
    Returns list of attention weight tensors, one per layer.
    """
    # Get model params from config - handle nested 'model' key
    model_config = config.get('model', config)
    
    d_model = model_config.get('d_model', 256)
    n_heads = model_config.get('n_heads', 4)
    n_layers = model_config.get('n_layers', 4)
    vocab_size = model_config.get('vocab_size', 50257)
    seq_len = model_config.get('max_seq_len', model_config.get('seq_len', 256))
    fixed_t = model_config.get('fixed_t', 1.0)
    
    print(f"  Model params: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, fixed_t={fixed_t}")
    
    d_k = d_model // n_heads
    batch_size, input_seq_len = input_ids.shape
    
    # Get state dict
    state_dict = checkpoint if isinstance(checkpoint, dict) and 'token_emb.weight' in checkpoint else checkpoint.get('model_state_dict', checkpoint)
    
    # Debug: print available keys for FFN
    ffn_keys = [k for k in state_dict.keys() if 'ffn' in k or 'mlp' in k or 'fc' in k]
    if ffn_keys:
        print(f"\nAvailable FFN/MLP keys (sample): {ffn_keys[:5]}")
    
    # Extract embeddings
    token_emb_weight = state_dict['token_emb.weight'].to(device)
    pos_emb_weight = state_dict['pos_emb.weight'].to(device)
    
    # Embed input
    positions = torch.arange(input_seq_len, device=device).unsqueeze(0)
    hidden = F.embedding(input_ids, token_emb_weight) + F.embedding(positions, pos_emb_weight)
    
    # Create causal mask
    causal_mask = torch.tril(torch.ones(input_seq_len, input_seq_len, device=device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    all_attention_weights = []
    
    # Process each layer
    for layer_idx in range(n_layers):
        prefix = f'blocks.{layer_idx}'
        
        # Layer norm 1
        ln1_weight = state_dict[f'{prefix}.ln1.weight'].to(device)
        ln1_bias = state_dict[f'{prefix}.ln1.bias'].to(device)
        normed = F.layer_norm(hidden, (d_model,), ln1_weight, ln1_bias)
        
        # Attention projections
        W_q = state_dict[f'{prefix}.attn.W_q.weight'].to(device)
        W_k = state_dict[f'{prefix}.attn.W_k.weight'].to(device)
        W_v = state_dict[f'{prefix}.attn.W_v.weight'].to(device)
        W_o = state_dict[f'{prefix}.attn.W_o.weight'].to(device)
        
        # Compute Q, K, V
        q = F.linear(normed, W_q).view(batch_size, input_seq_len, n_heads, d_k).transpose(1, 2)
        k = F.linear(normed, W_k).view(batch_size, input_seq_len, n_heads, d_k).transpose(1, 2)
        v = F.linear(normed, W_v).view(batch_size, input_seq_len, n_heads, d_k).transpose(1, 2)
        
        # Compute attention scores with diffusion scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / (2 * fixed_t)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        all_attention_weights.append({
            'layer': layer_idx,
            'weights': attn_weights.detach(),
            't': fixed_t,
        })
        
        # Continue forward pass for next layer
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, input_seq_len, d_model)
        attn_out = F.linear(attn_out, W_o)
        
        # Residual
        hidden = hidden + attn_out
        
        # FFN (layer norm 2 + MLP)
        ln2_weight = state_dict[f'{prefix}.ln2.weight'].to(device)
        ln2_bias = state_dict[f'{prefix}.ln2.bias'].to(device)
        normed2 = F.layer_norm(hidden, (d_model,), ln2_weight, ln2_bias)
        
        fc1_weight = state_dict[f'{prefix}.ffn.0.weight'].to(device)
        fc1_bias = state_dict[f'{prefix}.ffn.0.bias'].to(device)
        fc2_weight = state_dict[f'{prefix}.ffn.3.weight'].to(device)
        fc2_bias = state_dict[f'{prefix}.ffn.3.bias'].to(device)
        
        ffn_out = F.linear(normed2, fc1_weight, fc1_bias)
        ffn_out = F.gelu(ffn_out)
        ffn_out = F.linear(ffn_out, fc2_weight, fc2_bias)
        
        hidden = hidden + ffn_out
    
    return all_attention_weights


def test_with_random_data():
    """Test attention analysis with random data (baseline)."""
    print("=" * 70)
    print("BASELINE: Random Attention Patterns")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch, heads, seq_len, d_k = 4, 8, 256, 64
    t = 0.28
    
    q = torch.randn(batch, heads, seq_len, d_k, device=device)
    k = torch.randn(batch, heads, seq_len, d_k, device=device)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (2 * t)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    
    results = analyze_attention_pattern(attn_weights)
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Diffusion time t: {t}")
    print(f"  Heads: {heads}")
    
    print(f"\nAttention Statistics:")
    print(f"  Mean entropy: {results['mean_entropy']:.2f} / {results['max_entropy']:.2f} ({results['mean_entropy']/results['max_entropy']*100:.1f}% of uniform)")
    
    print(f"\nMass outside radius (lower = more local):")
    for radius, mass in sorted(results['outside_mass'].items()):
        print(f"  Radius {radius:3d}: {mass*100:.2f}% outside")
    
    print(f"\nRadius needed to capture X% of attention weight:")
    for threshold in [0.9, 0.95, 0.99, 0.999]:
        r = results['radius_for_threshold'][threshold]
        print(f"  {threshold*100:.1f}%: mean={r['mean']:.1f}, median={r['median']:.1f}, p95={r['p95']:.1f}")
    
    print(f"\nAttention weight vs distance (first 20 positions):")
    weights = results['weight_at_distance'][:20]
    for d, w in enumerate(weights):
        bar = '*' * int(w * 500)
        print(f"  d={d:2d}: {w:.4f} {bar}")
    
    return results


def analyze_trained_model(checkpoint_path: str, device: str = 'cuda'):
    """Analyze attention patterns in a trained model."""
    print("\n" + "=" * 70)
    print(f"TRAINED MODEL: {checkpoint_path}")
    print("=" * 70)
    
    # Load model
    checkpoint, config = load_model_and_config(checkpoint_path, device)
    
    print(f"\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create some input data (random token IDs)
    vocab_size = config.get('vocab_size', 50257)
    seq_len = min(config.get('seq_len', 256), 256)  # Cap at 256 for speed
    batch_size = 4
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Extract attention weights
    print(f"\nExtracting attention patterns...")
    attention_data = extract_attention_weights_from_checkpoint(
        checkpoint, config, input_ids, device
    )
    
    # Analyze each layer
    print(f"\n" + "-" * 70)
    print("LAYER-BY-LAYER ANALYSIS")
    print("-" * 70)
    
    for layer_data in attention_data:
        layer_idx = layer_data['layer']
        attn_weights = layer_data['weights']
        t = layer_data['t']
        
        results = analyze_attention_pattern(attn_weights)
        
        print(f"\n--- Layer {layer_idx} (t={t}) ---")
        print(f"  Entropy: {results['mean_entropy']:.2f} / {results['max_entropy']:.2f} ({results['mean_entropy']/results['max_entropy']*100:.1f}% of uniform)")
        
        print(f"  Mass outside radius:")
        for radius, mass in sorted(results['outside_mass'].items()):
            print(f"    R={radius:3d}: {mass*100:.2f}%")
        
        print(f"  Radius for thresholds:")
        for threshold in [0.9, 0.95, 0.99]:
            r = results['radius_for_threshold'][threshold]
            print(f"    {threshold*100:.0f}%: mean={r['mean']:.1f}, p95={r['p95']:.1f}")
    
    # Summary comparison
    print(f"\n" + "=" * 70)
    print("SUMMARY: Trained vs Random")
    print("=" * 70)
    
    # Get first and last layer for comparison
    first_layer = analyze_attention_pattern(attention_data[0]['weights'])
    last_layer = analyze_attention_pattern(attention_data[-1]['weights'])
    
    print(f"\nRadius needed for 95% attention mass:")
    print(f"  Random baseline:  mean={72.7:.1f} (from earlier test)")
    print(f"  Layer 0 (first):  mean={first_layer['radius_for_threshold'][0.95]['mean']:.1f}")
    print(f"  Layer {len(attention_data)-1} (last):   mean={last_layer['radius_for_threshold'][0.95]['mean']:.1f}")
    
    print(f"\nMass outside radius=20:")
    print(f"  Random baseline:  ~{35:.1f}% (estimate)")
    print(f"  Layer 0:          {first_layer['outside_mass'].get(20, 0)*100:.1f}%")
    print(f"  Layer {len(attention_data)-1}:          {last_layer['outside_mass'].get(20, 0)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--random-only", action="store_true",
                       help="Test with random data only")
    args = parser.parse_args()
    
    # Always run random baseline first
    random_results = test_with_random_data()
    
    # Analyze trained model if provided
    if args.checkpoint and not args.random_only:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        analyze_trained_model(args.checkpoint, device)
    elif not args.random_only:
        # Try default path
        default_path = "12layer_t016/best_model.pt"
        if Path(default_path).exists():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            analyze_trained_model(default_path, device)
        else:
            print(f"\nNo checkpoint found at {default_path}")
            print("Use --checkpoint <path> to specify model location")


if __name__ == "__main__":
    main()
