"""
Depth × Radius Composition Law Experiments

Tests the hypothesis that effective context scales as L × r, where:
- L = number of layers
- r = per-layer attention radius

This would show that local attention works via "iterated local transport" -
each layer propagates information r tokens, so L layers propagate L×r tokens.

Experiments:
- Fix α=1.0 (strong locality, r≈2-3 tokens)
- Vary depth: L ∈ {4, 8, 12, 16}
- Use depth-scaled t = 0.28 / sqrt(L/4)

Expected effective context:
- 4 layers:  4 × 3 = 12 tokens
- 8 layers:  8 × 3 = 24 tokens  
- 12 layers: 12 × 3 = 36 tokens
- 16 layers: 16 × 3 = 48 tokens

Usage:
    # Run all depth experiments
    python depth_radius_experiment.py --run_all
    
    # Run specific depth
    python depth_radius_experiment.py --n_layers 8
    
    # Analyze results
    python depth_radius_experiment.py --analyze

"""

import os
import sys
import math
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from tqdm import tqdm

# Import heat kernel attention
from heat_kernel_attention import (
    HeatKernelConfig, 
    HeatKernelAttention, 
    HeatKernelTransformerBlock,
    compute_effective_radius
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    t: float = 0.28
    alpha: float = 1.0  # Fixed at full locality
    learnable_t: bool = False
    learnable_alpha: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    eval_interval: int = 500
    save_interval: int = 2000
    max_tokens: Optional[int] = 500000
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "./logs"
    exp_name: str = "depth_exp"


# ============================================================
# Model (same as before)
# ============================================================

class HeatKernelTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        hk_config = HeatKernelConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            t=config.t,
            alpha=config.alpha,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        
        self.blocks = nn.ModuleList([
            HeatKernelTransformerBlock(hk_config, config.d_ff)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .unsqueeze(0).unsqueeze(0)
        )
        
        self.apply(self._init_weights)
        self.effective_radius = compute_effective_radius(config.t, config.alpha)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)
        
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for block in self.blocks:
            h, _ = block(h, mask=mask)
        
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return {"logits": logits}


# ============================================================
# Data Loading
# ============================================================

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y


def load_wikitext(split="train", max_tokens=None):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    all_tokens = []
    for item in dataset:
        if item["text"].strip():
            tokens = tokenizer.encode(item["text"])
            all_tokens.extend(tokens)
            if max_tokens and len(all_tokens) >= max_tokens:
                break
    
    if max_tokens:
        all_tokens = all_tokens[:max_tokens]
    
    return torch.tensor(all_tokens, dtype=torch.long)


# ============================================================
# Metrics
# ============================================================

def compute_ece(confidences, accuracies, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_confidences = []
    all_accuracies = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            output = model(x)
            logits = output["logits"]
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)
            correct = (predictions == y).float()
            
            all_confidences.append(confidences.cpu())
            all_accuracies.append(correct.cpu())
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    confidences = torch.cat(all_confidences).numpy().flatten()
    accuracies = torch.cat(all_accuracies).numpy().flatten()
    ece = compute_ece(confidences, accuracies)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "ece": ece,
        "mean_confidence": float(confidences.mean()),
    }


# ============================================================
# Training
# ============================================================

def train_single_depth(n_layers: int, log_dir: str = "./logs", device: str = "cuda"):
    """Train a model with specified depth."""
    
    # Depth-scaled t
    t = 0.28 / math.sqrt(n_layers / 4)
    alpha = 1.0  # Fixed full locality
    
    exp_name = f"depth_L{n_layers}_a100"
    
    print(f"\n{'='*60}")
    print(f"Depth × Radius Experiment: {n_layers} Layers")
    print(f"{'='*60}")
    print(f"  t = 0.28 / sqrt({n_layers}/4) = {t:.4f}")
    print(f"  α = {alpha}")
    print(f"  Effective radius: {compute_effective_radius(t, alpha)} tokens")
    print(f"  Expected effective context: ~{n_layers * 3} tokens")
    print(f"{'='*60}\n")
    
    device = torch.device(device)
    torch.manual_seed(42)
    
    # Create log directory
    log_path = Path(log_dir) / exp_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Model config
    model_config = ModelConfig(
        n_layers=n_layers,
        t=t,
        alpha=alpha,
    )
    
    # Save config
    config_dict = {
        "model": asdict(model_config),
        "effective_radius": compute_effective_radius(t, alpha),
        "expected_context": n_layers * 3,
    }
    with open(log_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load data
    print("Loading data...")
    train_tokens = load_wikitext("train", 500000)
    val_tokens = load_wikitext("validation")
    
    train_dataset = TextDataset(train_tokens, model_config.max_seq_len)
    val_dataset = TextDataset(val_tokens, model_config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    print("Creating model...")
    model = HeatKernelTransformer(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, total_steps)
    
    # Training
    metrics_log = {"val_loss": [], "val_perplexity": [], "val_ece": []}
    best_val_loss = float('inf')
    global_step = 0
    
    print("Starting training...")
    model.train()
    pbar = tqdm(train_loader, desc=f"L={n_layers}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        output = model(x)
        loss = F.cross_entropy(
            output["logits"].view(-1, model_config.vocab_size),
            y.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        global_step += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Evaluate
        if global_step % 500 == 0:
            model.eval()
            val_metrics = evaluate(model, val_loader, device)
            model.train()
            
            metrics_log["val_loss"].append((global_step, val_metrics["loss"]))
            metrics_log["val_perplexity"].append((global_step, val_metrics["perplexity"]))
            metrics_log["val_ece"].append((global_step, val_metrics["ece"]))
            
            print(f"\nStep {global_step}: loss={val_metrics['loss']:.4f}, "
                  f"ppl={val_metrics['perplexity']:.1f}, ECE={val_metrics['ece']:.4f}")
            
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": asdict(model_config),
                    "val_loss": val_metrics["loss"],
                }, log_path / "best_model.pt")
        
        # Save checkpoint
        if global_step % 2000 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(model_config),
                "step": global_step,
            }, log_path / f"checkpoint_{global_step}.pt")
    
    # Save metrics
    with open(log_path / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: L={n_layers}, Best Loss={best_val_loss:.4f}")
    print(f"Saved to: {log_path}")
    print(f"{'='*60}\n")
    
    return best_val_loss


def analyze_depth_results(log_dir: str = "./logs"):
    """Analyze results across all depth experiments."""
    
    print("\n" + "="*70)
    print("DEPTH × RADIUS COMPOSITION LAW ANALYSIS")
    print("="*70)
    
    log_path = Path(log_dir)
    results = []
    
    # Find all depth experiments
    for exp_dir in sorted(log_path.glob("depth_L*_a100")):
        config_file = exp_dir / "config.json"
        metrics_file = exp_dir / "metrics.json"
        
        if not config_file.exists() or not metrics_file.exists():
            continue
        
        with open(config_file) as f:
            config = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        model_config = config.get("model", config)
        n_layers = model_config.get("n_layers", 4)
        t = model_config.get("t", 0.28)
        alpha = model_config.get("alpha", 1.0)
        effective_radius = config.get("effective_radius", compute_effective_radius(t, alpha))
        
        # Find best metrics
        val_losses = metrics.get("val_loss", [])
        val_eces = metrics.get("val_ece", [])
        
        if val_losses:
            best_idx = np.argmin([v[1] for v in val_losses])
            best_loss = val_losses[best_idx][1]
            best_ppl = math.exp(best_loss)
            early_ece = val_eces[0][1] if val_eces else None
        else:
            continue
        
        results.append({
            "n_layers": n_layers,
            "t": t,
            "effective_radius": effective_radius,
            "expected_context": n_layers * effective_radius,
            "best_loss": best_loss,
            "best_ppl": best_ppl,
            "early_ece": early_ece,
        })
    
    if not results:
        print("No depth experiments found!")
        return
    
    # Sort by depth
    results.sort(key=lambda x: x["n_layers"])
    
    # Print table
    print(f"\n{'Layers':>8} {'t':>8} {'Radius':>8} {'Context':>10} {'Best Loss':>12} {'Best PPL':>10} {'Early ECE':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['n_layers']:>8} {r['t']:>8.4f} {r['effective_radius']:>8} "
              f"{r['expected_context']:>10.0f} {r['best_loss']:>12.4f} "
              f"{r['best_ppl']:>10.1f} {r['early_ece']:>12.4f}")
    
    # Test composition law
    print("\n" + "-"*70)
    print("COMPOSITION LAW TEST: Does perplexity improve with L × r?")
    print("-"*70)
    
    contexts = [r["expected_context"] for r in results]
    ppls = [r["best_ppl"] for r in results]
    
    if len(results) >= 2:
        # Simple correlation
        correlation = np.corrcoef(contexts, ppls)[0, 1]
        print(f"Correlation (context vs perplexity): {correlation:.3f}")
        
        if correlation < -0.5:
            print("✓ Strong negative correlation: more context → lower perplexity")
            print("  This supports the composition law hypothesis!")
        elif correlation < 0:
            print("~ Weak negative correlation: trend in expected direction")
        else:
            print("✗ No clear relationship found")
    
    # Generate plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        layers = [r["n_layers"] for r in results]
        
        # Plot 1: Perplexity vs Depth
        axes[0].plot(layers, ppls, 'bo-', markersize=10, linewidth=2)
        axes[0].set_xlabel("Number of Layers (L)")
        axes[0].set_ylabel("Best Validation Perplexity")
        axes[0].set_title("Perplexity vs Depth (α=1.0, local attention)")
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Perplexity vs Effective Context
        axes[1].plot(contexts, ppls, 'ro-', markersize=10, linewidth=2)
        axes[1].set_xlabel("Effective Context (L × r)")
        axes[1].set_ylabel("Best Validation Perplexity")
        axes[1].set_title("Composition Law: Context vs Perplexity")
        axes[1].grid(True, alpha=0.3)
        
        # Add layer labels
        for l, c, p in zip(layers, contexts, ppls):
            axes[1].annotate(f"L={l}", (c, p), textcoords="offset points", 
                           xytext=(5, 5), fontsize=9)
        
        plt.tight_layout()
        
        output_path = log_path / "depth_composition_law.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure: {output_path}")
        plt.close()
        
    except ImportError:
        print("(matplotlib not available for plotting)")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Depth × Radius Composition Law Experiments")
    parser.add_argument("--n_layers", type=int, default=None,
                       help="Train single depth (4, 8, 12, or 16)")
    parser.add_argument("--run_all", action="store_true",
                       help="Run all depth experiments")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_depth_results(args.log_dir)
    elif args.run_all:
        print("Running all depth experiments...")
        for n_layers in [4, 8, 12, 16]:
            train_single_depth(n_layers, args.log_dir, args.device)
        analyze_depth_results(args.log_dir)
    elif args.n_layers:
        train_single_depth(args.n_layers, args.log_dir, args.device)
    else:
        print("Usage:")
        print("  python depth_radius_experiment.py --n_layers 8")
        print("  python depth_radius_experiment.py --run_all")
        print("  python depth_radius_experiment.py --analyze")


if __name__ == "__main__":
    main()
