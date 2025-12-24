"""
Analyze Attention Radius Over Training

This script loads checkpoints from heat kernel training and measures:
1. Actual attention radius (how far attention reaches in practice)
2. How radius stabilizes over training
3. Comparison across different α values

This provides visual evidence that sparsity is learned, not imposed.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def compute_attention_radius_stats(
    attn_weights: torch.Tensor,
    thresholds: List[float] = [0.9, 0.95, 0.99]
) -> Dict[str, float]:
    """
    Compute radius statistics for attention weights.
    
    Args:
        attn_weights: (batch, heads, seq_q, seq_k) attention weights
        thresholds: cumulative mass thresholds
    
    Returns:
        Dictionary with radius stats for each threshold
    """
    batch, heads, seq_q, seq_k = attn_weights.shape
    device = attn_weights.device
    
    # Create distance matrix
    positions = torch.arange(seq_k, device=device).float()
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    
    # Flatten batch and heads
    attn_flat = attn_weights.view(-1, seq_q, seq_k)
    
    results = {}
    
    for threshold in thresholds:
        radii = []
        
        # Sample queries for speed
        n_samples = min(500, attn_flat.shape[0] * seq_q)
        
        for _ in range(n_samples):
            b = np.random.randint(attn_flat.shape[0])
            q = np.random.randint(seq_q)
            
            attn_row = attn_flat[b, q]
            d = distance[q]
            
            # Sort by distance
            sorted_idx = d.argsort()
            cumsum = attn_row[sorted_idx].cumsum(dim=0)
            
            # Find radius for threshold
            above = (cumsum >= threshold).nonzero(as_tuple=True)[0]
            if len(above) > 0:
                r = d[sorted_idx[above[0]]].item()
            else:
                r = seq_k - 1
            radii.append(r)
        
        results[f'radius_{int(threshold*100)}'] = {
            'mean': np.mean(radii),
            'median': np.median(radii),
            'std': np.std(radii),
            'p95': np.percentile(radii, 95),
        }
    
    # Also compute "outside mass" for fixed radii
    for fixed_r in [5, 10, 20]:
        outside_mask = distance > fixed_r
        causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
        valid_outside = outside_mask & causal_mask
        
        outside_weights = attn_flat * valid_outside.unsqueeze(0).float()
        outside_mass = outside_weights.sum(dim=-1).mean().item()
        results[f'mass_outside_{fixed_r}'] = outside_mass
    
    return results


def load_and_analyze_checkpoint(
    checkpoint_path: Path,
    config: dict,
    device: str = 'cuda',
    n_batches: int = 5,
) -> Dict:
    """Load checkpoint and analyze attention patterns."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Get model config
    model_config = config.get('model', config)
    d_model = model_config.get('d_model', 256)
    n_heads = model_config.get('n_heads', 4)
    n_layers = model_config.get('n_layers', 4)
    vocab_size = model_config.get('vocab_size', 50257)
    seq_len = model_config.get('max_seq_len', 256)
    t = model_config.get('t', 0.28)
    alpha = model_config.get('alpha', 0.0)
    
    d_k = d_model // n_heads
    
    # Generate random input
    batch_size = 4
    
    all_layer_results = []
    
    for batch_idx in range(n_batches):
        torch.manual_seed(42 + batch_idx)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Get embeddings
        token_emb = state_dict['token_emb.weight'].to(device)
        pos_emb = state_dict['pos_emb.weight'].to(device)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden = F.embedding(input_ids, token_emb) + F.embedding(positions, pos_emb)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Process each layer
        for layer_idx in range(n_layers):
            prefix = f'blocks.{layer_idx}'
            
            # Layer norm
            ln1_weight = state_dict[f'{prefix}.ln1.weight'].to(device)
            ln1_bias = state_dict[f'{prefix}.ln1.bias'].to(device)
            normed = F.layer_norm(hidden, (d_model,), ln1_weight, ln1_bias)
            
            # Projections
            W_q = state_dict[f'{prefix}.attn.W_q.weight'].to(device)
            W_k = state_dict[f'{prefix}.attn.W_k.weight'].to(device)
            W_v = state_dict[f'{prefix}.attn.W_v.weight'].to(device)
            
            q = F.linear(normed, W_q).view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
            k = F.linear(normed, W_k).view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
            
            # Compute attention scores
            content_scores = torch.matmul(q, k.transpose(-2, -1)) / (2 * t)
            
            # Add positional penalty if alpha > 0
            if alpha > 0:
                pos = torch.arange(seq_len, device=device).float()
                distance = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
                position_penalty = -alpha * (distance ** 2) / (4 * t)
                content_scores = content_scores + position_penalty.unsqueeze(0).unsqueeze(0)
            
            scores = content_scores.masked_fill(causal_mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            # Analyze
            layer_results = compute_attention_radius_stats(attn_weights)
            layer_results['layer'] = layer_idx
            layer_results['batch'] = batch_idx
            all_layer_results.append(layer_results)
            
            # Continue forward pass (simplified - just for hidden state)
            W_o = state_dict[f'{prefix}.attn.W_o.weight'].to(device)
            v = F.linear(normed, state_dict[f'{prefix}.attn.W_v.weight'].to(device))
            v = v.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            attn_out = F.linear(attn_out, W_o)
            hidden = hidden + attn_out
            
            # FFN
            ln2_weight = state_dict[f'{prefix}.ln2.weight'].to(device)
            ln2_bias = state_dict[f'{prefix}.ln2.bias'].to(device)
            normed2 = F.layer_norm(hidden, (d_model,), ln2_weight, ln2_bias)
            
            try:
                fc1_w = state_dict[f'{prefix}.ffn.0.weight'].to(device)
                fc1_b = state_dict[f'{prefix}.ffn.0.bias'].to(device)
                fc2_w = state_dict[f'{prefix}.ffn.3.weight'].to(device)
                fc2_b = state_dict[f'{prefix}.ffn.3.bias'].to(device)
            except KeyError:
                fc1_w = state_dict[f'{prefix}.ffn.fc1.weight'].to(device)
                fc1_b = state_dict[f'{prefix}.ffn.fc1.bias'].to(device)
                fc2_w = state_dict[f'{prefix}.ffn.fc2.weight'].to(device)
                fc2_b = state_dict[f'{prefix}.ffn.fc2.bias'].to(device)
            
            ffn_out = F.gelu(F.linear(normed2, fc1_w, fc1_b))
            ffn_out = F.linear(ffn_out, fc2_w, fc2_b)
            hidden = hidden + ffn_out
    
    # Aggregate results by layer
    aggregated = {}
    for layer_idx in range(n_layers):
        layer_data = [r for r in all_layer_results if r['layer'] == layer_idx]
        aggregated[layer_idx] = {
            'radius_95_mean': np.mean([r['radius_95']['mean'] for r in layer_data]),
            'radius_95_std': np.mean([r['radius_95']['std'] for r in layer_data]),
            'mass_outside_10': np.mean([r['mass_outside_10'] for r in layer_data]),
            'mass_outside_20': np.mean([r['mass_outside_20'] for r in layer_data]),
        }
    
    return aggregated


def analyze_experiment(exp_dir: Path, device: str = 'cuda') -> Dict:
    """Analyze all checkpoints in an experiment directory."""
    
    # Load config
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        print(f"No config found at {config_path}")
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    model_config = config.get('model', config)
    alpha = model_config.get('alpha', 0.0)
    t = model_config.get('t', 0.28)
    
    print(f"\nAnalyzing {exp_dir.name}: α={alpha}, t={t}")
    
    # Find checkpoints
    checkpoints = sorted(exp_dir.glob('checkpoint_*.pt'))
    best_model = exp_dir / 'best_model.pt'
    
    if best_model.exists():
        checkpoints.append(best_model)
    
    results = {
        'alpha': alpha,
        't': t,
        'effective_radius_theoretical': config.get('effective_radius', 'inf'),
        'checkpoints': {}
    }
    
    for ckpt in checkpoints:
        step = ckpt.stem.split('_')[-1] if 'checkpoint' in ckpt.stem else 'best'
        print(f"  Analyzing {ckpt.name}...")
        
        try:
            analysis = load_and_analyze_checkpoint(ckpt, config, device)
            results['checkpoints'][step] = analysis
        except Exception as e:
            print(f"    Error: {e}")
    
    return results


def plot_radius_comparison(all_results: Dict[str, Dict], output_path: Path = None):
    """Plot radius comparison across experiments."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Radius for 95% mass by experiment
    ax1 = axes[0]
    
    for exp_name, results in all_results.items():
        alpha = results['alpha']
        
        # Get best model results
        if 'best' in results['checkpoints']:
            data = results['checkpoints']['best']
            layers = sorted(data.keys())
            radii = [data[l]['radius_95_mean'] for l in layers]
            
            label = f"α={alpha}"
            ax1.plot(layers, radii, 'o-', label=label, markersize=8)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Radius for 95% Attention Mass', fontsize=12)
    ax1.set_title('Attention Radius by Layer', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mass outside radius=10
    ax2 = axes[1]
    
    for exp_name, results in all_results.items():
        alpha = results['alpha']
        
        if 'best' in results['checkpoints']:
            data = results['checkpoints']['best']
            layers = sorted(data.keys())
            mass = [data[l]['mass_outside_10'] * 100 for l in layers]
            
            label = f"α={alpha}"
            ax2.plot(layers, mass, 'o-', label=label, markersize=8)
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Attention Mass Outside R=10 (%)', fontsize=12)
    ax2.set_title('Long-Range Attention by Layer', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze attention radius")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory containing experiment logs")
    parser.add_argument("--experiments", type=str, nargs='+', 
                       default=['baseline_a00', 'hk_a025_t028', 'hk_a050_t028', 'hk_a100_t050'],
                       help="Experiment names to analyze")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="radius_comparison.png",
                       help="Output figure path")
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    all_results = {}
    
    for exp_name in args.experiments:
        exp_dir = log_dir / exp_name
        if exp_dir.exists():
            results = analyze_experiment(exp_dir, args.device)
            if results:
                all_results[exp_name] = results
        else:
            print(f"Experiment not found: {exp_dir}")
    
    if all_results:
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY: Attention Radius by Experiment")
        print("=" * 60)
        
        for exp_name, results in all_results.items():
            print(f"\n{exp_name} (α={results['alpha']}, t={results['t']}):")
            print(f"  Theoretical radius: {results['effective_radius_theoretical']}")
            
            if 'best' in results['checkpoints']:
                data = results['checkpoints']['best']
                for layer in sorted(data.keys()):
                    r = data[layer]['radius_95_mean']
                    m = data[layer]['mass_outside_10'] * 100
                    print(f"  Layer {layer}: radius_95={r:.1f}, mass_outside_10={m:.1f}%")
        
        # Plot
        output_path = Path(args.output)
        plot_radius_comparison(all_results, output_path)
    
    else:
        print("No experiments found to analyze")


if __name__ == "__main__":
    main()
