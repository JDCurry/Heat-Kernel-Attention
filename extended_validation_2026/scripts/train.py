"""
Training Script for Diffusion-Native Language Modeling

Handles all experimental runs:
  A1: Standard softmax + CE (baseline)
  A2: Diffusion (fixed t) + CE
  A3: Diffusion (learned t, unconstrained) + CE
  B1: Diffusion (constrained t) + Compound loss (main experiment)
  B2: Diffusion (constrained t) + Compound loss, no spectral
  B3: Diffusion (constrained t) + CE only
  B4: Diffusion (constrained t) + Compound loss + MSE

Usage:
  # Quick smoke test
  python scripts/train.py --run A1 --dataset wikitext-2 --epochs 1

  # Full WikiText-103 run (with mixed precision)
  python scripts/train.py --run A1 --dataset wikitext-103 --epochs 3

  # Budgeted run (50K steps, much faster)
  python scripts/train.py --run A1 --dataset wikitext-103 --max_steps 50000
"""

import os
import sys
import math
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer import ModelConfig, DiffusionNativeTransformer
from src.model.corruption import GeometricCorruption
from src.loss.compound_loss import CompoundLoss, CurriculumScheduler
from src.data.dataset import create_dataloaders


# ============================================================
# Run Configurations
# ============================================================

def get_run_config(run_name: str, n_layers: int = 12) -> dict:
    t_base = 0.28
    L_base = 4
    t_center = t_base * math.sqrt(L_base / n_layers)

    configs = {
        "A1": {"attention_mode": "softmax", "training_mode": "standard",
               "alpha": 0.0, "description": "Standard softmax + CE"},
        "A2": {"attention_mode": "diffusion_fixed", "training_mode": "standard",
               "fixed_t": t_center, "alpha": 0.0,
               "description": f"Diffusion (fixed t={t_center:.4f}) + CE"},
        "A3": {"attention_mode": "diffusion_learned", "training_mode": "standard",
               "fixed_t": t_center, "t_min": 0.01, "t_max": 5.0, "alpha": 0.0,
               "description": "Diffusion (unconstrained learned t) + CE"},
        "B1": {"attention_mode": "diffusion_learned", "training_mode": "diffusion_native",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "loss_mode": "compound", "use_mse": False,
               "description": "Compound loss (cosine+CE+spec), constrained t"},
        "B2": {"attention_mode": "diffusion_learned", "training_mode": "diffusion_native",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "loss_mode": "compound", "use_mse": False, "no_spectral": True,
               "description": "Compound loss (no spectral), constrained t"},
        "B3": {"attention_mode": "diffusion_learned", "training_mode": "standard",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "description": "CE only, constrained t"},
        "B4": {"attention_mode": "diffusion_learned", "training_mode": "diffusion_native",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "loss_mode": "compound", "use_mse": True,
               "description": "Compound loss + MSE, constrained t"},
        # Phase C: Auxiliary loss (Option C — CE primary, small aux terms)
        "C1": {"attention_mode": "diffusion_learned", "training_mode": "diffusion_native",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "loss_mode": "auxiliary",
               "aux_rec": 0.05, "aux_spec": 0.01,
               "description": "CE + aux(rec=0.05, spec=0.01), constrained t"},
        "C2": {"attention_mode": "diffusion_learned", "training_mode": "diffusion_native",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "loss_mode": "auxiliary",
               "aux_rec": 0.1, "aux_spec": 0.02,
               "description": "CE + aux(rec=0.1, spec=0.02), constrained t"},
        "C3": {"attention_mode": "diffusion_learned", "training_mode": "diffusion_native",
               "fixed_t": t_center, "t_min": t_center * 0.5, "t_max": t_center * 1.5,
               "alpha": 0.0, "loss_mode": "auxiliary",
               "aux_rec": 0.05, "aux_spec": 0.0,
               "description": "CE + aux(rec=0.05, no spectral), constrained t"},
    }

    if run_name not in configs:
        raise ValueError(f"Unknown run: {run_name}")
    return configs[run_name]


# ============================================================
# Training Loop
# ============================================================

def train(
    run_name: str,
    dataset: str = "wikitext-103",
    n_layers: int = 12,
    d_model: int = 256,
    n_heads: int = 4,
    d_ff: int = 1024,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 3e-4,
    max_tokens: Optional[int] = None,
    max_steps: Optional[int] = None,
    seq_len: int = 256,
    eval_interval: int = 500,
    save_interval: int = 5000,
    log_dir: str = "./results",
    seed: int = 42,
    device_str: str = "cuda",
    corruption_t_min: float = 0.05,
    corruption_t_max: float = 0.5,
    amp: bool = True,
    grad_accum: int = 1,
    num_workers: int = 4,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    run_config = get_run_config(run_name, n_layers)
    print(f"\n{'='*70}")
    print(f"RUN {run_name}: {run_config['description']}")
    print(f"{'='*70}")

    model_config = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        attention_mode=run_config["attention_mode"],
        fixed_t=run_config.get("fixed_t", 0.16),
        alpha=run_config.get("alpha", 0.0),
        t_min=run_config.get("t_min", 0.05),
        t_max=run_config.get("t_max", 2.0),
        training_mode=run_config.get("training_mode", "standard"),
        max_seq_len=seq_len,
    )

    exp_dir = Path(log_dir) / f"{run_name}_{dataset}_{n_layers}L"
    exp_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "run_name": run_name,
        "run_config": run_config,
        "model_config": asdict(model_config),
        "dataset": dataset,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "seq_len": seq_len,
        "seed": seed,
        "amp": amp,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    data = create_dataloaders(
        variant=dataset, seq_len=seq_len, batch_size=batch_size,
        max_train_tokens=max_tokens, num_workers=num_workers,
    )
    train_loader = data["train"]
    val_loader = data["val"]

    print(f"\nDataset stats:")
    print(f"  Train tokens: {data['train_tokens']:,}")
    print(f"  Val tokens: {data['val_tokens']:,}")
    print(f"  Steps per epoch: {len(train_loader):,}")

    model = DiffusionNativeTransformer(model_config).to(device)

    corruption = None
    if model_config.training_mode == "diffusion_native":
        corruption = GeometricCorruption(max_seq_len=seq_len).to(device)

    loss_mode = run_config.get("loss_mode", "standard")
    use_mse = run_config.get("use_mse", False)
    aux_rec = run_config.get("aux_rec", 0.05)
    aux_spec = run_config.get("aux_spec", 0.01)
    loss_fn = CompoundLoss(
        mode=loss_mode, vocab_size=model_config.vocab_size,
        max_seq_len=seq_len, use_mse=use_mse,
        aux_rec_weight=aux_rec, aux_spec_weight=aux_spec,
    ).to(device)

    steps_per_epoch = len(train_loader)
    if max_steps is not None:
        total_steps = min(max_steps, steps_per_epoch * epochs)
    else:
        total_steps = steps_per_epoch * epochs

    curriculum = None
    if loss_mode == "compound":
        no_spectral = run_config.get("no_spectral", False)
        curriculum = CurriculumScheduler(
            total_steps=total_steps,
            early_spec=0.0,
            mid_spec=0.0 if no_spectral else 0.05,
            late_spec=0.0 if no_spectral else 0.1,
        )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, total_steps)

    # AMP setup
    use_amp = amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"Mixed precision: {amp_dtype}")

    metrics_log = {"train": [], "val": [], "t_trajectories": []}
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\nTotal steps: {total_steps:,}")
    print(f"Effective batch: {batch_size * grad_accum}")
    print(f"Training on: {device}")
    print(f"Starting training...\n")

    optimizer.zero_grad(set_to_none=True)
    should_stop = False

    for epoch in range(epochs):
        if should_stop:
            break
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                if model_config.training_mode == "diffusion_native" and corruption is not None:
                    clean_emb = model.get_embeddings(x)
                    corrupted_emb, corr_t = corruption(
                        clean_emb, t_min=corruption_t_min, t_max=corruption_t_max
                    )
                    output = model.forward_from_embeddings(corrupted_emb)

                    # Compound mode uses curriculum weights; auxiliary mode uses its own fixed weights
                    weights = curriculum.get_weights(global_step) if curriculum else None
                    loss, loss_components = loss_fn(
                        logits=output["logits"], targets=y,
                        recovered_hidden=output["hidden"],
                        clean_embeddings=clean_emb.detach(),
                        weights=weights,  # None for auxiliary mode (uses internal weights)
                    )
                else:
                    output = model(x)
                    loss, loss_components = loss_fn(logits=output["logits"], targets=y)

                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * grad_accum
                epoch_steps += 1
                global_step += 1

                pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

                # Track t values
                if global_step % 100 == 0:
                    t_vals = []
                    for block in model.blocks:
                        t = block.attn.get_t()
                        if isinstance(t, torch.Tensor) and t.dim() > 0:
                            t_vals.append(t.detach().cpu().tolist())
                        else:
                            t_val = t.item() if isinstance(t, torch.Tensor) else t
                            t_vals.append(t_val)
                    metrics_log["t_trajectories"].append(
                        {"step": global_step, "t_values": t_vals}
                    )

                # Evaluate
                if global_step % eval_interval == 0:
                    eval_batches = 20 if model_config.training_mode == "diffusion_native" else 50
                    val_metrics = evaluate(model, val_loader, device, use_amp, amp_dtype,
                                           max_batches=eval_batches)
                    metrics_log["val"].append({"step": global_step, **val_metrics})
                    metrics_log["train"].append(
                        {"step": global_step, "loss": epoch_loss / max(epoch_steps, 1)}
                    )

                    elapsed = time.time() - start_time
                    tokens_seen = global_step * batch_size * seq_len * grad_accum
                    steps_remaining = total_steps - global_step
                    eta_sec = (elapsed / global_step) * steps_remaining if global_step > 0 else 0

                    print(
                        f"\n  Step {global_step:,}/{total_steps:,}: "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"ppl={val_metrics['perplexity']:.1f}, "
                        f"ECE={val_metrics['ece']:.4f}, "
                        f"brier={val_metrics['brier']:.4f} "
                        f"[elapsed={elapsed/60:.1f}min, ETA={eta_sec/60:.1f}min, "
                        f"{tokens_seen/1e6:.1f}M tokens]"
                    )

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "config": asdict(model_config),
                            "step": global_step,
                            "val_metrics": val_metrics,
                        }, exp_dir / "best_model.pt")

                # Save checkpoint
                if global_step % save_interval == 0:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": global_step,
                        "config": asdict(model_config),
                    }, exp_dir / f"checkpoint_{global_step}.pt")

                # Save metrics periodically
                if global_step % 1000 == 0:
                    with open(exp_dir / "metrics.json", "w") as f:
                        json.dump(metrics_log, f, indent=2)

                if max_steps and global_step >= max_steps:
                    print(f"\nReached max_steps={max_steps}, stopping.")
                    should_stop = True
                    break

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE: {run_name}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    try:
        print(f"  Best perplexity: {math.exp(best_val_loss):.1f}")
    except OverflowError:
        print(f"  Best perplexity: inf")
    print(f"  Total steps: {global_step:,}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Logs: {exp_dir}")
    print(f"{'='*70}")


def evaluate(model, val_loader, device, use_amp: bool, amp_dtype,
             max_batches: int = 50) -> Dict[str, float]:
    """Streaming evaluation — no OOM from giant logits tensors."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_confidence = 0.0
    total_correct = 0.0
    total_brier = 0.0

    n_bins = 15
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.float64)
    bin_conf_sums = np.zeros(n_bins, dtype=np.float64)
    bin_correct_sums = np.zeros(n_bins, dtype=np.float64)

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                output = model(x)
                logits = output["logits"]

            logits_flat = logits.view(-1, logits.size(-1)).float().cpu()
            targets_flat = y.view(-1).cpu()

            batch_loss = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
            total_loss += batch_loss.item()

            probs = F.softmax(logits_flat, dim=-1)
            confidences, predictions = probs.max(dim=-1)
            correct = (predictions == targets_flat).float()

            total_tokens += targets_flat.numel()
            total_confidence += confidences.sum().item()
            total_correct += correct.sum().item()
            total_brier += ((confidences - correct) ** 2).sum().item()

            confidences_np = confidences.detach().cpu().numpy()
            correct_np = correct.detach().cpu().numpy()
            bin_indices = np.digitize(confidences_np, bin_boundaries, right=True) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            for bin_idx in range(n_bins):
                in_bin = bin_indices == bin_idx
                if not np.any(in_bin):
                    continue
                bin_counts[bin_idx] += in_bin.sum()
                bin_conf_sums[bin_idx] += confidences_np[in_bin].sum()
                bin_correct_sums[bin_idx] += correct_np[in_bin].sum()

    if total_tokens == 0:
        return {k: float("nan") for k in
                ["loss", "perplexity", "ece", "brier",
                 "mean_confidence", "mean_accuracy", "overconfidence"]}

    avg_loss = total_loss / total_tokens
    try:
        perplexity = math.exp(min(avg_loss, 20))
    except OverflowError:
        perplexity = float("inf")
    mean_confidence = total_confidence / total_tokens
    mean_accuracy = total_correct / total_tokens
    brier = total_brier / total_tokens

    ece = 0.0
    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] > 0:
            prop_in_bin = bin_counts[bin_idx] / total_tokens
            avg_conf = bin_conf_sums[bin_idx] / bin_counts[bin_idx]
            avg_acc = bin_correct_sums[bin_idx] / bin_counts[bin_idx]
            ece += abs(avg_acc - avg_conf) * prop_in_bin

    model.train()
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "ece": ece,
        "brier": brier,
        "mean_confidence": mean_confidence,
        "mean_accuracy": mean_accuracy,
        "overconfidence": mean_confidence - mean_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion-Native LM")
    parser.add_argument("--run", type=str, required=True,
                        choices=["A1", "A2", "A3", "B1", "B2", "B3", "B4",
                                 "C1", "C2", "C3"])
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        choices=["wikitext-2", "wikitext-103", "slimpajama"])
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Limit training tokens (None=all for WikiText, "
                             "recommended 100000000 for SlimPajama)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit total steps (recommended for budgeted runs)")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--corruption_t_min", type=float, default=0.05)
    parser.add_argument("--corruption_t_max", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision (default ON)")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    train(
        run_name=args.run, dataset=args.dataset, n_layers=args.n_layers,
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        max_tokens=args.max_tokens, max_steps=args.max_steps,
        seq_len=args.seq_len, eval_interval=args.eval_interval,
        save_interval=args.save_interval, log_dir=args.log_dir,
        seed=args.seed, device_str=args.device,
        corruption_t_min=args.corruption_t_min,
        corruption_t_max=args.corruption_t_max,
        amp=args.amp, grad_accum=args.grad_accum,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
