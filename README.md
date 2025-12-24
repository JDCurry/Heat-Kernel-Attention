# Heat Kernel Attention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18041199.svg)](https://doi.org/10.5281/zenodo.18041199)


**Provable Sparsity via Diffusion Dynamics**

This repository contains the implementation and experimental code for the paper:

> **Heat Kernel Attention: Provable Sparsity via Diffusion Dynamics**  
> Joshua D. Curry (2025)

Building on [Diffusion Attention](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5953096) (Paper 1), which established the equivalence between softmax attention and equilibrium drift-diffusion.

## Key Results

| Finding | Result |
|---------|--------|
| **Composition Law** | Effective context scales as L × r with correlation r = -0.898 |
| **Performance** | 16-layer local attention outperforms 12-layer global by 18% |
| **Efficiency** | Per-layer radius reduced from ~100 tokens to ~3 tokens (97% reduction) |
| **Speedup** | O(n²) → O(n·r) complexity, up to 683× for long sequences |

## Repository Structure

```
Heat-Kernel-Attention/
├── README.md
├── LICENSE
├── requirements.txt
│
├── diffusion_attention_torch.py      # Core diffusion attention module
├── heat_kernel_attention.py          # Sparse/locality extension with radius bounds
│
├── train_heat_kernel.py              # Main training script
├── train_heat_kernel_12layer.py      # 12-layer experiments
├── train_20layer.py                  # 20-layer depth scaling
├── depth_radius_experiment.py        # Composition law validation
├── analyze_attention_patterns.py     # Attention pattern analysis
├── analyze_attention_patterns_v3.py  # Extended analysis
├── analyze_radius_over_training.py   # Radius tracking during training
├── sparsity_analysis.py              # Sparsity statistics
│
├── logs/                             # Training logs (metrics.json, config.json)
│   ├── depth_4L_alpha1.0/
│   ├── depth_8L_alpha1.0/
│   ├── depth_12L_alpha1.0/
│   ├── depth_16L_alpha1.0/
│   └── depth_20L_alpha1.0/
│
└── manuscript/
    ├── figures/
    └── paper2_heat_kernel_attention.tex
```

## Installation

```bash
git clone https://github.com/JDCurry/Heat-Kernel-Attention.git
cd Heat-Kernel-Attention
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (16GB+ VRAM recommended)

## Quick Start

### Training a Heat Kernel Attention Model

```bash
# 12-layer model with full locality (α=1.0)
python train_heat_kernel.py --n_layers 12 --alpha 1.0 --fixed_t 0.162 --epochs 1 --max_tokens 500000

# 16-layer model (best performance)
python train_heat_kernel.py --n_layers 16 --alpha 1.0 --fixed_t 0.140 --epochs 1 --max_tokens 500000

# 20-layer model
python train_heat_kernel.py --n_layers 20 --alpha 1.0 --fixed_t 0.125 --epochs 1 --max_tokens 500000

# Standard baseline (no locality)
python train_heat_kernel.py --model standard --n_layers 12 --epochs 1 --max_tokens 500000
```

### Depth Scaling Law

The diffusion time scales with depth as: `t = 0.28 / sqrt(L/4)`

| Layers | t     | Effective Radius | Effective Context |
|--------|-------|------------------|-------------------|
| 4      | 0.280 | ~4 tokens        | 16 tokens         |
| 8      | 0.198 | ~4 tokens        | 32 tokens         |
| 12     | 0.162 | ~3 tokens        | 36 tokens         |
| 16     | 0.140 | ~3 tokens        | 48 tokens         |
| 20     | 0.125 | ~3 tokens        | 60 tokens         |

### Analyzing Attention Patterns

```bash
python analyze_attention_patterns.py --checkpoint logs/depth_12L_alpha1.0/best_model.pt --output_dir analysis/
```

## Method

Heat kernel attention adds a positional decay term to standard attention:

```
score_ij = (q_i · k_j) / 2t - α · d(i,j)² / 4t
```

Where:
- `d(i,j) = |i - j|` is positional distance
- `α ∈ [0, 1]` controls locality strength
  - `α = 0`: Standard diffusion attention (global)
  - `α = 1`: Full heat kernel locality

### Effective Radius Guarantee

For parameters (ε, t, α), attention outside radius r is bounded by ε:

```
r = sqrt(4t · ln(1/ε) / α)
```

For ε = 10⁻⁶, t = 0.16, α = 1.0: **r ≈ 3 tokens**

This is a **mathematical guarantee**, not a heuristic.

## Results

### Composition Law Validation

| Layers | Radius | Effective Context | Best PPL | 
|--------|--------|-------------------|----------|
| 4      | 4      | 16                | 518.0    |
| 8      | 4      | 32                | 483.9    |
| 12     | 3      | 36                | 499.2    |
| **16** | **3**  | **48**            | **420.8**|
| 20     | 3      | 60                | 430.7    |

**Correlation (Context vs Perplexity): r = -0.898**

### Local vs Global Attention

| Model | Layers | Radius | PPL | vs Baseline |
|-------|--------|--------|-----|-------------|
| Global (α=0) | 12 | ~100 | 497 | — |
| **Local (α=1)** | **16** | **3** | **420.8** | **-18%** |

## Computational Environment

All experiments were performed on:
- **OS**: Ubuntu 22.04.5 LTS
- **CPU**: Dual Intel Xeon E5-2699 v3 (36 cores, 72 threads @ 2.30 GHz)
- **RAM**: 128 GB DDR4
- **GPU**: NVIDIA RTX A4000 (16GB VRAM)

## Core Modules

### `diffusion_attention_torch.py`
Base diffusion attention implementation:
- `heat_kernel_attention()` - Core attention function
- `DiffusionAttention` - Multi-head attention layer
- `DiffusionTimePredictor` - Adaptive time prediction
- `StandardAttention` - Baseline for comparison

### `heat_kernel_attention.py`
Sparse/locality extensions:
- `compute_radius()` - Theoretical radius from diffusion time
- `diffusion_attention_sparse_torch()` - Sparse implementation
- `SparseDiffusionAttention` - Full sparse attention layer
- Triton kernel stubs for GPU acceleration
