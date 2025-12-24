"""
Analyze Attention Patterns in Trained Diffusion Attention Models

This script loads a trained model and analyzes:
1. Actual attention weight distributions
2. How much weight falls within different radii
3. Empirical sparsity achievable without quality loss

"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import argparse
from pathlib import Path


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def analyze_attention_pattern(
    attn_weights: torch.Tensor,
    thresholds: list = [0.9, 0.95, 0.99, 0.999]
) -> dict:
    """
    Analyze attention weight distribution vs distance.
    
    Args:
        attn_weights: (batch, heads, seq_q, seq_k) attention weights
        thresholds: cumulative weight thresholds to measure radius for
    
    Returns:
        Dictionary of statistics
    """
    batch, heads, seq_q, seq_k = attn_weights.shape
    device = attn_weights.device
    
    # Create distance matrix
    positions_q = torch.arange(seq_q, device=device)
    positions_k = torch.arange(seq_k, device=device)
    distance = (positions_q.unsqueeze(1) - positions_k.unsqueeze(0)).abs().float()
    
    # For each query position, compute cumulative weight vs distance
    results = {
        'mean_entropy': [],
        'weight_at_distance': [],
        'radius_for_threshold': {t: [] for t in thresholds},
    }
    
    # Flatten batch and heads
    attn_flat = attn_weights.view(-1, seq_q, seq_k)  # (B*H, seq_q, seq_k)
    
    # Compute entropy
    entropy = -(attn_flat * (attn_flat + 1e-10).log()).sum(dim=-1)  # (B*H, seq_q)
    results['mean_entropy'] = entropy.mean().item()
    results['max_entropy'] = math.log(seq_k)  # Uniform distribution entropy
    
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
    
    # For each query, find radius needed to capture threshold of weight
    for threshold in thresholds:
        radii = []
        for b in range(attn_flat.shape[0]):
            for q in range(seq_q):
                # Sort keys by distance
                attn_row = attn_flat[b, q]  # (seq_k,)
                distances_from_q = distance[q]  # (seq_k,)
                
                # Sort by distance
                sorted_indices = distances_from_q.argsort()
                sorted_weights = attn_row[sorted_indices]
                cumsum = sorted_weights.cumsum(dim=0)
                
                # Find first index where cumsum >= threshold
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


def analyze_model_attention(model, dataloader, device, max_batches=10):
    """
    Run model on data and analyze attention patterns.
    """
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            x = x.to(device)
            
            # Forward pass - need to capture attention weights
            # This requires modifying the model to return attention weights
            # For now, we'll compute them manually
            
            batch_size, seq_len = x.shape
            
            # Get embeddings
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            hidden = model.token_emb(x) + model.pos_emb(positions)
            hidden = model.dropout(hidden)
            
            # Process each layer
            mask = model.causal_mask[:, :, :seq_len, :seq_len]
            
            for layer_idx, block in enumerate(model.blocks):
                # Get Q, K, V
                batch, seq, d_model = hidden.shape
                n_heads = block.attn.n_heads
                d_k = d_model // n_heads
                
                q = block.attn.W_q(hidden).view(batch, seq, n_heads, d_k).transpose(1, 2)
                k = block.attn.W_k(hidden).view(batch, seq, n_heads, d_k).transpose(1, 2)
                v = block.attn.W_v(hidden).view(batch, seq, n_heads, d_k).transpose(1, 2)
                
                # Compute attention scores
                if hasattr(block.attn, 'fixed_t'):
                    t = block.attn.fixed_t
                else:
                    t = 1.0  # Default
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / (2 * t)
                scores = scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = F.softmax(scores, dim=-1)
                
                # Analyze this layer's attention
                results = analyze_attention_pattern(attn_weights)
                results['layer'] = layer_idx
                results['batch'] = batch_idx
                results['t'] = t
                all_results.append(results)
                
                # Continue forward pass
                hidden = block(hidden, mask=mask)
    
    return all_results


def print_analysis(results_list):
    """Pretty print analysis results."""
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    
    # Aggregate by layer
    by_layer = {}
    for r in results_list:
        layer = r['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(r)
    
    for layer in sorted(by_layer.keys()):
        layer_results = by_layer[layer]
        
        print(f"\n--- Layer {layer} ---")
        print(f"  Diffusion time t: {layer_results[0]['t']}")
        
        # Average entropy
        mean_entropy = np.mean([r['mean_entropy'] for r in layer_results])
        max_entropy = layer_results[0]['max_entropy']
        print(f"  Mean entropy: {mean_entropy:.2f} / {max_entropy:.2f} ({mean_entropy/max_entropy*100:.1f}% of uniform)")
        
        # Radius analysis
        print(f"  Radius needed to capture X% of attention weight:")
        for threshold in [0.9, 0.95, 0.99]:
            radii = [r['radius_for_threshold'][threshold] for r in layer_results]
            mean_radius = np.mean([x['mean'] for x in radii])
            p95_radius = np.mean([x['p95'] for x in radii])
            print(f"    {threshold*100:.0f}%: mean={mean_radius:.1f}, p95={p95_radius:.1f}")


def test_with_random_data():
    """Test attention analysis with random data (baseline)."""
    print("=" * 70)
    print("BASELINE: Random Attention Patterns")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch, heads, seq_len, d_k = 4, 8, 256, 64
    t = 0.28
    
    # Random Q, K
    q = torch.randn(batch, heads, seq_len, d_k, device=device)
    k = torch.randn(batch, heads, seq_len, d_k, device=device)
    
    # Compute attention with causal mask
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
    
    print(f"\nRadius needed to capture X% of attention weight:")
    for threshold in [0.9, 0.95, 0.99, 0.999]:
        r = results['radius_for_threshold'][threshold]
        print(f"  {threshold*100:.1f}%: mean={r['mean']:.1f}, median={r['median']:.1f}, p95={r['p95']:.1f}, max={r['max']:.0f}")
    
    # Weight decay with distance
    print(f"\nAttention weight vs distance (first 20 positions):")
    weights = results['weight_at_distance'][:20]
    for d, w in enumerate(weights):
        bar = '*' * int(w * 500)
        print(f"  d={d:2d}: {w:.4f} {bar}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze attention patterns")
    parser.add_argument("--checkpoint", type=str, default="12layer_t016/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--random", action="store_true",
                       help="Test with random data only")
    args = parser.parse_args()
    
    # Always run random baseline first
    random_results = test_with_random_data()
    
    if args.checkpoint:
        print("\n" + "=" * 70)
        print(f"TRAINED MODEL: {args.checkpoint}")
        print("=" * 70)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # TODO: Load and analyze trained model
        print("(Model analysis not yet implemented)")


if __name__ == "__main__":
    main()
