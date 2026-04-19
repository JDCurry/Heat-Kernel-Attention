"""
Evaluation Metrics for Diffusion-Native LM

Primary: Perplexity, ECE, Brier Score
Secondary: Learned t trajectories, overconfidence, cosine recovery accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def compute_ece(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error.

    Args:
        logits: (N, vocab_size) or (batch, seq, vocab_size)
        targets: (N,) or (batch, seq)
        n_bins: Number of calibration bins

    Returns:
        ece: scalar
    """
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    probs = F.softmax(logits_flat, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets_flat).float()

    confidences = confidences.detach().cpu().numpy()
    accuracies = accuracies.detach().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def compute_brier_score(
    logits: torch.Tensor, targets: torch.Tensor
) -> float:
    """
    Brier Score: mean squared error of confidence vs accuracy.
    Proper scoring rule combining calibration + discrimination.
    """
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    probs = F.softmax(logits_flat, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets_flat).float()

    brier = ((confidences - accuracies) ** 2).mean().item()
    return brier


def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Returns:
        dict with loss, perplexity, ece, brier, confidence, accuracy,
        overconfidence
    """
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Loss and perplexity
    loss = F.cross_entropy(logits_flat, targets_flat)
    perplexity = torch.exp(loss).item()

    # ECE and Brier
    ece = compute_ece(logits, targets)
    brier = compute_brier_score(logits, targets)

    # Confidence stats
    probs = F.softmax(logits_flat, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets_flat).float()

    mean_conf = confidences.mean().item()
    mean_acc = accuracies.mean().item()

    return {
        "loss": loss.item(),
        "perplexity": perplexity,
        "ece": ece,
        "brier": brier,
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
        "overconfidence": mean_conf - mean_acc,
    }


def compute_recovery_metrics(
    recovered: torch.Tensor,
    clean: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute embedding recovery quality metrics.
    Used as diagnostics for the compound loss.

    Args:
        recovered: (batch, seq_len, d_model)
        clean: (batch, seq_len, d_model)

    Returns:
        dict with cosine similarity, MSE, and norm ratio
    """
    # Cosine similarity
    rec_norm = F.normalize(recovered, p=2, dim=-1)
    clean_norm = F.normalize(clean, p=2, dim=-1)
    cos_sim = (rec_norm * clean_norm).sum(dim=-1).mean().item()

    # MSE
    mse = F.mse_loss(recovered, clean).item()

    # Norm ratio (recovered / clean)
    rec_norms = recovered.norm(dim=-1).mean().item()
    clean_norms = clean.norm(dim=-1).mean().item()
    norm_ratio = rec_norms / max(clean_norms, 1e-8)

    return {
        "cosine_similarity": cos_sim,
        "mse": mse,
        "norm_ratio": norm_ratio,
    }
