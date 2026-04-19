"""
Forward Corruption Process: Geometric Corruption via Heat Kernel

The forward process corrupts clean embeddings by applying the heat kernel
operator exp(-tL) on a positional graph Laplacian. This is NOT random noise —
it is structured semantic homogenization that blurs high-frequency distinctions
between nearby tokens along the sequence graph.

The model must learn to recover sharp token distinctions from this geometric
blur, which is precisely the skill that produces calibrated outputs.

Key design choices (v1):
  - Pure Laplacian smoothing (no added Gaussian noise)
  - Fixed positional Laplacian (line graph over sequence positions)
  - Corruption level t sampled per-batch during training
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def build_line_graph_laplacian(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build the graph Laplacian for a line graph (path graph) of length seq_len.

    The line graph has edges between consecutive positions:
      A[i, i+1] = A[i+1, i] = 1

    Laplacian L = D - A where D is the degree matrix.

    For a line graph:
      L[i,i] = 2  (interior nodes, degree 2)
      L[0,0] = L[n-1,n-1] = 1  (endpoints, degree 1)
      L[i,i+1] = L[i+1,i] = -1

    Args:
        seq_len: Length of the sequence
        device: Torch device

    Returns:
        L: (seq_len, seq_len) Laplacian matrix
    """
    L = torch.zeros(seq_len, seq_len, device=device)

    # Diagonal: degree of each node
    L[0, 0] = 1.0
    L[seq_len - 1, seq_len - 1] = 1.0
    for i in range(1, seq_len - 1):
        L[i, i] = 2.0

    # Off-diagonal: -1 for adjacent nodes
    for i in range(seq_len - 1):
        L[i, i + 1] = -1.0
        L[i + 1, i] = -1.0

    return L


def compute_heat_kernel(
    L: torch.Tensor, t: float, method: str = "eigen"
) -> torch.Tensor:
    """
    Compute the heat kernel exp(-tL) for a given Laplacian.

    For small sequences, eigendecomposition is exact and fast.
    For larger sequences, we could use matrix exponential approximations,
    but for v1 on WikiText-103 with seq_len=256, eigen is fine.

    Args:
        L: (seq_len, seq_len) graph Laplacian
        t: Diffusion time (corruption level)
        method: "eigen" for eigendecomposition, "expm" for matrix exponential

    Returns:
        K: (seq_len, seq_len) heat kernel matrix exp(-tL)
    """
    if method == "eigen":
        # Eigendecomposition: L = V @ diag(λ) @ V^T
        # exp(-tL) = V @ diag(exp(-t*λ)) @ V^T
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # exp(-t * eigenvalues)
        exp_eigenvalues = torch.exp(-t * eigenvalues)

        # Reconstruct: V @ diag(exp(-t*λ)) @ V^T
        K = eigenvectors @ torch.diag(exp_eigenvalues) @ eigenvectors.T

        return K
    elif method == "expm":
        # Matrix exponential via torch
        return torch.matrix_exp(-t * L)
    else:
        raise ValueError(f"Unknown method: {method}")


class GeometricCorruption(nn.Module):
    """
    Applies geometric corruption to token embeddings via the heat kernel.

    X_t = exp(-tL) · X

    where L is a fixed positional Laplacian and t controls corruption level.

    At t=0: no corruption (identity)
    At small t: local smoothing (nearby tokens blend slightly)
    At large t: global homogenization (all tokens converge to mean)

    The model learns to invert this smoothing — recovering sharp semantic
    distinctions from geometric blur.
    """

    def __init__(self, max_seq_len: int = 256, cache_kernels: bool = True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cache_kernels = cache_kernels
        self._kernel_cache = {}

        # Precompute Laplacian eigendecomposition for efficiency
        # (computed once, reused for all t values)
        L = build_line_graph_laplacian(max_seq_len, torch.device("cpu"))
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)

    def get_heat_kernel(self, seq_len: int, t: float) -> torch.Tensor:
        """
        Get the heat kernel matrix for a given sequence length and time.

        Uses cached eigendecomposition for efficiency.
        """
        cache_key = (seq_len, round(t, 6))

        if self.cache_kernels and cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # Use precomputed eigendecomposition, truncated to seq_len
        eigenvalues = self.eigenvalues[:seq_len]
        eigenvectors = self.eigenvectors[:seq_len, :seq_len]

        # Note: for a subgraph of a line graph, the eigendecomposition of
        # the full graph truncated is NOT exactly the eigendecomposition of
        # the subgraph. For exact results we'd need to recompute.
        # But for seq_len == max_seq_len (the common case), this is exact.
        # For shorter sequences, recompute:
        if seq_len < self.max_seq_len:
            L_sub = build_line_graph_laplacian(seq_len, eigenvalues.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sub)

        exp_eigenvalues = torch.exp(-t * eigenvalues)
        K = eigenvectors @ torch.diag(exp_eigenvalues) @ eigenvectors.T

        if self.cache_kernels:
            self._kernel_cache[cache_key] = K

        return K

    def corrupt(
        self,
        embeddings: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Apply geometric corruption to embeddings.

        Args:
            embeddings: (batch, seq_len, d_model) clean embeddings
            t: corruption level (diffusion time)

        Returns:
            corrupted: (batch, seq_len, d_model) corrupted embeddings
        """
        batch, seq_len, d_model = embeddings.shape

        # Get heat kernel for this seq_len and t
        K = self.get_heat_kernel(seq_len, t)  # (seq_len, seq_len)

        # Apply: X_t = K @ X
        # K is (seq_len, seq_len), embeddings is (batch, seq_len, d_model)
        corrupted = torch.matmul(K.unsqueeze(0), embeddings)

        return corrupted

    def sample_corruption_time(
        self,
        t_min: float = 0.01,
        t_max: float = 1.0,
        schedule: str = "uniform",
    ) -> float:
        """
        Sample a corruption time for training.

        Args:
            t_min: Minimum corruption time
            t_max: Maximum corruption time
            schedule: "uniform" or "log_uniform"

        Returns:
            t: sampled corruption time
        """
        if schedule == "uniform":
            return t_min + (t_max - t_min) * torch.rand(1).item()
        elif schedule == "log_uniform":
            log_t = math.log(t_min) + (math.log(t_max) - math.log(t_min)) * torch.rand(1).item()
            return math.exp(log_t)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def forward(
        self,
        embeddings: torch.Tensor,
        t: Optional[float] = None,
        t_min: float = 0.01,
        t_max: float = 1.0,
    ) -> Tuple[torch.Tensor, float]:
        """
        Corrupt embeddings with sampled or specified t.

        Returns:
            corrupted: corrupted embeddings
            t: the corruption time used
        """
        if t is None:
            t = self.sample_corruption_time(t_min, t_max)

        corrupted = self.corrupt(embeddings, t)
        return corrupted, t
