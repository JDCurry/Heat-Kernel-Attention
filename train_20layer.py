"""
20-Layer Heat Kernel Attention Experiment

One more data point for the composition law.
Expected: L=20, r≈3, effective context ≈ 60 tokens

If perplexity continues to drop, we have even stronger evidence.

Usage:
    python train_20layer.py

"""

import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from tqdm import tqdm

from heat_kernel_attention import (
    HeatKernelConfig, 
    HeatKernelTransformerBlock,
    compute_effective_radius
)


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 20  # Deep!
    d_ff: int = 1024
    dropout: float = 0.1
    t: float = 0.125  # 0.28 / sqrt(20/4) = 0.125
    alpha: float = 1.0


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
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        b, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)
        
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for block in self.blocks:
            h, _ = block(h, mask=mask)
        
        h = self.ln_f(h)
        return {"logits": self.lm_head(h)}


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        return self.tokens[idx:idx + self.seq_len], self.tokens[idx + 1:idx + self.seq_len + 1]


def load_wikitext(split="train", max_tokens=None):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    tokens = []
    for item in dataset:
        if item["text"].strip():
            tokens.extend(tokenizer.encode(item["text"]))
            if max_tokens and len(tokens) >= max_tokens:
                break
    
    return torch.tensor(tokens[:max_tokens] if max_tokens else tokens, dtype=torch.long)


def compute_ece(confidences, accuracies, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.mean() > 0:
            ece += np.abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)


def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    confs, accs = [], []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            confs.append(conf.cpu())
            accs.append((pred == y).float().cpu())
    
    confs = torch.cat(confs).numpy().flatten()
    accs = torch.cat(accs).numpy().flatten()
    
    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "ece": compute_ece(confs, accs),
    }


def main():
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # Config
    n_layers = 20
    t = 0.28 / math.sqrt(n_layers / 4)  # Depth-scaled
    alpha = 1.0
    
    config = ModelConfig(n_layers=n_layers, t=t, alpha=alpha)
    radius = compute_effective_radius(t, alpha)
    
    print(f"\n{'='*60}")
    print(f"20-LAYER HEAT KERNEL EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Layers: {n_layers}")
    print(f"  t = 0.28 / sqrt({n_layers}/4) = {t:.4f}")
    print(f"  α = {alpha}")
    print(f"  Effective radius: {radius} tokens")
    print(f"  Expected context: ~{n_layers * radius} tokens")
    print(f"{'='*60}\n")
    
    # Data
    print("Loading data...")
    train_tokens = load_wikitext("train", 500000)
    val_tokens = load_wikitext("validation")
    
    train_loader = DataLoader(TextDataset(train_tokens, 256), batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(val_tokens, 256), batch_size=32)
    
    # Model
    print("Creating model...")
    model = HeatKernelTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Training
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader))
    
    log_dir = Path("./logs/depth_L20_a100")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "model": asdict(config),
            "effective_radius": radius,
            "expected_context": n_layers * radius,
        }, f, indent=2)
    
    metrics = {"val_loss": [], "val_perplexity": [], "val_ece": []}
    best_loss = float('inf')
    step = 0
    
    print("Training...")
    model.train()
    pbar = tqdm(train_loader, desc="L=20")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)["logits"]
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        step += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if step % 500 == 0:
            model.eval()
            val = evaluate(model, val_loader, device)
            model.train()
            
            metrics["val_loss"].append((step, val["loss"]))
            metrics["val_perplexity"].append((step, val["perplexity"]))
            metrics["val_ece"].append((step, val["ece"]))
            
            print(f"\nStep {step}: loss={val['loss']:.4f}, ppl={val['perplexity']:.1f}, ECE={val['ece']:.4f}")
            
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "val_loss": val["loss"],
                }, log_dir / "best_model.pt")
                print("  New best!")
        
        if step % 2000 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step,
            }, log_dir / f"checkpoint_{step}.pt")
    
    # Save metrics
    with open(log_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: L=20, Best Loss={best_loss:.4f}, PPL={math.exp(best_loss):.1f}")
    print(f"{'='*60}")
    
    # Quick composition law update
    print("\nUpdated Composition Law:")
    results = [
        (4, 16, 518.0),
        (8, 32, 483.9),
        (12, 36, 499.2),
        (16, 48, 420.8),
        (20, 20 * radius, math.exp(best_loss)),
    ]
    
    print(f"{'Layers':>8} {'Context':>10} {'PPL':>10}")
    print("-" * 30)
    for L, ctx, ppl in results:
        print(f"{L:>8} {ctx:>10.0f} {ppl:>10.1f}")
    
    contexts = [r[1] for r in results]
    ppls = [r[2] for r in results]
    corr = np.corrcoef(contexts, ppls)[0, 1]
    print(f"\nCorrelation: {corr:.3f}")


if __name__ == "__main__":
    main()
