"""
Train Heat Kernel Attention Models

Quick integration of heat kernel attention (Path B) into training pipeline.
Tests whether enforced locality preserves or improves calibration.

Usage:
    # Standard diffusion (baseline)
    python train_heat_kernel.py --alpha 0.0 --t 0.28 --exp_name baseline_t028
    
    # Heat kernel with moderate locality
    python train_heat_kernel.py --alpha 0.5 --t 0.28 --exp_name hk_a05_t028
    
    # Heat kernel with strong locality  
    python train_heat_kernel.py --alpha 1.0 --t 0.28 --exp_name hk_a10_t028
    
    # Heat kernel with larger radius
    python train_heat_kernel.py --alpha 1.0 --t 0.5 --exp_name hk_a10_t050

"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

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
    
    # Heat kernel parameters
    t: float = 0.28
    alpha: float = 1.0  # 0 = standard diffusion, 1 = full heat kernel
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
    max_tokens: Optional[int] = None
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "./logs"
    exp_name: str = "heat_kernel"


# ============================================================
# Model
# ============================================================

class HeatKernelTransformer(nn.Module):
    """Transformer with Heat Kernel Attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Create heat kernel config for attention
        hk_config = HeatKernelConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            t=config.t,
            alpha=config.alpha,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            learnable_t=config.learnable_t,
            learnable_alpha=config.learnable_alpha,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HeatKernelTransformerBlock(hk_config, config.d_ff)
            for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .unsqueeze(0).unsqueeze(0)
        )
        
        # Initialize
        self.apply(self._init_weights)
        
        # Report effective radius
        self.effective_radius = compute_effective_radius(config.t, config.alpha)
        print(f"Heat Kernel Attention: t={config.t}, alpha={config.alpha}")
        print(f"Effective radius: {self.effective_radius} tokens")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)
        
        # Mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Transformer blocks
        for block in self.blocks:
            h, _ = block(h, mask=mask)
        
        # Output
        h = self.ln_f(h)
        logits = self.lm_head(h)
        
        return {"logits": logits}


# ============================================================
# Data Loading (from original training script)
# ============================================================

class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y


def load_wikitext(split: str = "train", max_tokens: Optional[int] = None):
    """Load WikiText-2 dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Tokenize
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

def compute_ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()
    
    confidences = confidences.cpu().numpy().flatten()
    accuracies = accuracies.cpu().numpy().flatten()
    
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


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    # Loss and perplexity
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    perplexity = torch.exp(loss).item()
    
    # ECE
    ece = compute_ece(logits, targets)
    
    # Confidence stats
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity,
        "ece": ece,
        "mean_confidence": confidences.mean().item(),
        "mean_accuracy": accuracies.mean().item(),
    }


# ============================================================
# Training
# ============================================================

def train(model_config: ModelConfig, train_config: TrainConfig):
    """Main training loop."""
    
    # Setup
    device = torch.device(train_config.device)
    torch.manual_seed(train_config.seed)
    
    # Create log directory
    log_dir = Path(train_config.log_dir) / train_config.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        "model": asdict(model_config),
        "train": asdict(train_config),
        "effective_radius": compute_effective_radius(model_config.t, model_config.alpha),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load data
    print("Loading data...")
    train_tokens = load_wikitext("train", train_config.max_tokens)
    val_tokens = load_wikitext("validation")
    
    train_dataset = TextDataset(train_tokens, model_config.max_seq_len)
    val_dataset = TextDataset(val_tokens, model_config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size)
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create model
    print("Creating model...")
    model = HeatKernelTransformer(model_config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    # LR scheduler
    total_steps = len(train_loader) * train_config.epochs
    scheduler = CosineAnnealingLR(optimizer, total_steps)
    
    # Training loop
    metrics_log = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "val_ece": [],
        "val_confidence": [],
        "val_accuracy": [],
    }
    
    global_step = 0
    best_val_loss = float('inf')
    
    print("Starting training...")
    
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward
            output = model(x)
            loss = F.cross_entropy(
                output["logits"].view(-1, model_config.vocab_size),
                y.view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Evaluate
            if global_step % train_config.eval_interval == 0:
                model.eval()
                val_metrics = evaluate(model, val_loader, device)
                model.train()
                
                # Log
                metrics_log["train_loss"].append((global_step, epoch_loss / (batch_idx + 1)))
                metrics_log["val_loss"].append((global_step, val_metrics["loss"]))
                metrics_log["val_perplexity"].append((global_step, val_metrics["perplexity"]))
                metrics_log["val_ece"].append((global_step, val_metrics["ece"]))
                metrics_log["val_confidence"].append((global_step, val_metrics["mean_confidence"]))
                metrics_log["val_accuracy"].append((global_step, val_metrics["mean_accuracy"]))
                
                print(f"\nStep {global_step}: "
                      f"val_loss={val_metrics['loss']:.4f}, "
                      f"ppl={val_metrics['perplexity']:.1f}, "
                      f"ECE={val_metrics['ece']:.4f}")
                
                # Save best
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": asdict(model_config),
                        "step": global_step,
                        "val_loss": val_metrics["loss"],
                    }, log_dir / "best_model.pt")
            
            # Save checkpoint
            if global_step % train_config.save_interval == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(model_config),
                    "step": global_step,
                }, log_dir / f"checkpoint_{global_step}.pt")
    
    # Save final metrics
    with open(log_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"\nTraining complete! Logs saved to {log_dir}")
    return metrics_log


def evaluate(model, dataloader, device) -> Dict[str, float]:
    """Evaluate model on dataloader."""
    model.eval()
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            all_logits.append(output["logits"])
            all_targets.append(y)
    
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    return compute_metrics(logits, targets)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Heat Kernel Attention")
    
    # Model args
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Heat kernel args
    parser.add_argument("--t", type=float, default=0.28, help="Diffusion time")
    parser.add_argument("--alpha", type=float, default=1.0, 
                       help="Positional decay strength (0=standard, 1=full heat kernel)")
    parser.add_argument("--learnable_t", action="store_true")
    parser.add_argument("--learnable_alpha", action="store_true")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_tokens", type=int, default=500000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Auto-generate experiment name
    if args.exp_name is None:
        args.exp_name = f"hk_a{args.alpha:.1f}_t{args.t:.2f}"
    
    # Create configs
    model_config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        t=args.t,
        alpha=args.alpha,
        learnable_t=args.learnable_t,
        learnable_alpha=args.learnable_alpha,
    )
    
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
    )
    
    # Print effective radius
    radius = compute_effective_radius(args.t, args.alpha)
    print(f"\n{'='*60}")
    print(f"Heat Kernel Attention Training")
    print(f"{'='*60}")
    print(f"t={args.t}, alpha={args.alpha}")
    print(f"Effective radius: {radius} tokens")
    print(f"Experiment: {args.exp_name}")
    print(f"{'='*60}\n")
    
    # Train
    train(model_config, train_config)


if __name__ == "__main__":
    main()
