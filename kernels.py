"""
Kernel functions module - implements various kernel functions, especially RBF kernel functions
"""
import torch
import math
import logging
from typing import Tuple, Optional


class RBFKernel:
    """Radial Basis Function (RBF) kernel class"""

    def __init__(self, bandwidth: float = -1):
        """
        Initialize RBF kernel

        Args:
        - bandwidth: Kernel bandwidth parameter, auto-determined if negative
        """
        self.bandwidth = bandwidth
        self._cached_sigma = None

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RBF kernel matrix and its derivatives between X and Y

        Args:
        - X: Input data matrix
        - Y: Input data matrix

        Returns:
        - RBF kernel matrix derivative
        - RBF kernel matrix
        """
        return self._compute_rbf_optimized(X, Y)

    def _compute_rbf_optimized(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized RBF kernel matrix computation

        Args:
        - X: Input data matrix [n, d]
        - Y: Input data matrix [m, d]

        Returns:
        - dx_K_XY: RBF kernel matrix derivative [n, d]
        - K_XY: RBF kernel matrix [n, m]
        """
        # Use more efficient distance computation
        X_norm = (X ** 2).sum(1).view(-1, 1)
        Y_norm = (Y ** 2).sum(1).view(1, -1)
        XY = torch.matmul(X, Y.t())
        dnorm2 = X_norm + Y_norm - 2 * XY

        # Determine sigma value
        sigma = self._get_sigma(dnorm2, X.shape[0])

        # Compute RBF kernel matrix
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = torch.exp(-gamma * dnorm2)

        # More efficient gradient computation
        dx_K_XY = -gamma * K_XY.unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0))
        dx_K_XY = dx_K_XY.sum(dim=1)

        return dx_K_XY, K_XY

    def _get_sigma(self, dnorm2: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Get or compute sigma value

        Args:
        - dnorm2: Squared distance matrix
        - n_samples: Number of samples

        Returns:
        - sigma value
        """
        if self.bandwidth >= 0:
            return torch.tensor(self.bandwidth, device=dnorm2.device)

        # Use cache to avoid recomputation
        if self._cached_sigma is not None:
            return self._cached_sigma

        # Auto-determine sigma value
        with torch.no_grad():
            median = torch.quantile(dnorm2.detach(), q=0.5) / (2 * math.log(n_samples + 1))
            sigma = torch.sqrt(median)
            self._cached_sigma = sigma

        return sigma

    def clear_cache(self):
        """Clear cached sigma value"""
        self._cached_sigma = None


class LegacyRBFKernel:
    """Original RBF kernel implementation (for compatibility)"""

    def __init__(self, bandwidth: float = -1):
        self.bandwidth = bandwidth

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Original RBF kernel computation method

        Args:
        - X: Input data matrix
        - Y: Input data matrix

        Returns:
        - RBF kernel matrix derivative
        - RBF kernel matrix
        """
        # Compute XX, XY and YY matrices
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        # Compute squared distances
        dnorm2 = -2 * XY + torch.diag(XX).unsqueeze(1) + torch.diag(YY).unsqueeze(0)

        # Auto-determine sigma value
        if self.bandwidth < 0:
            median = torch.quantile(dnorm2.detach(), q=0.5) / (2 * math.log(X.shape[0] + 1))
            sigma = torch.sqrt(median)
        else:
            sigma = self.bandwidth

        # Compute RBF kernel matrix
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = torch.exp(-gamma * dnorm2)

        # Compute RBF kernel matrix derivative
        dx_K_XY = -torch.matmul(K_XY, X)
        sum_K_XY = torch.sum(K_XY, dim=1)
        for i in range(X.shape[1]):
            dx_K_XY[:, i] = dx_K_XY[:, i] + torch.multiply(X[:, i], sum_K_XY)
        dx_K_XY = dx_K_XY / (1.0e-8 + sigma ** 2)

        return dx_K_XY, K_XY


def create_rbf_kernel(bandwidth: float = -1, use_optimized: bool = True) -> RBFKernel:
    """
    Factory function: create RBF kernel

    Args:
    - bandwidth: Kernel bandwidth
    - use_optimized: Whether to use optimized version

    Returns:
    - RBF kernel instance
    """
    if use_optimized:
        return RBFKernel(bandwidth)
    else:
        return LegacyRBFKernel(bandwidth)


def xRBF(sigma: float = -1) -> callable:
    """
    Original xRBF function wrapper (for compatibility)

    Args:
    - sigma: RBF bandwidth parameter

    Returns:
    - Function that computes RBF
    """
    kernel = LegacyRBFKernel(sigma)
    return kernel