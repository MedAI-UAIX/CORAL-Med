"""
SVGD sampling module - implements Stein Variational Gradient Descent sampling
"""
import torch
import logging
import random
from typing import List, Optional, Dict, Any, Callable
from .config import Constants
from .kernels import RBFKernel

class SVGDSampler:
    """SVGD (Stein Variational Gradient Descent) sampler"""

    def __init__(self,
                 kernel: RBFKernel,
                 entropy_reg: float = 10.0,
                 categorical_indices: List[int] = None,
                 feature_names: List[str] = None):
        """
        Initialize SVGD sampler

        Args:
        - kernel: RBF kernel function instance
        - entropy_reg: Entropy regularization coefficient
        - categorical_indices: List of categorical variable indices
        - feature_names: List of feature names
        """
        self.kernel = kernel
        self.entropy_reg = entropy_reg
        self.categorical_indices = categorical_indices or []
        self.feature_names = feature_names or []

        # Difficulty factor cache
        self._difficulty_cache = {}
        self._setup_difficulty_factors()

    def _setup_difficulty_factors(self):
        """Setup feature difficulty factors"""
        if self.feature_names:
            for idx, name in enumerate(self.feature_names):
                if name in Constants.DIFFICULTY_FACTORS:
                    self._difficulty_cache[idx] = Constants.DIFFICULTY_FACTORS[name]

    def sample(self,
               data: torch.Tensor,
               score_func: Callable,
               grad_optim: torch.optim.Optimizer,
               mask_matrix: torch.Tensor,
               iter_steps: int = 500,
               l1_ratio: float = 0.5,
               alpha: float = 1.0,
               problem_feature_indices: List[int] = None) -> torch.Tensor:
        """
        Execute SVGD sampling

        Args:
        - data: Initial sample data
        - score_func: Scoring function
        - grad_optim: Gradient optimizer
        - mask_matrix: Mask matrix
        - iter_steps: Number of iterations
        - l1_ratio: L1 regularization ratio
        - alpha: Elastic net regularization strength
        - problem_feature_indices: Problem feature indices list

        Returns:
        - Updated sample data
        """
        data_number = data.shape[0]
        batch_size = min(128, data_number)

        # Pre-computation
        special_alpha = alpha * 5.0
        consecutive_valid_iters = 0

        for iter_idx in range(iter_steps):
            # Compute scores
            eval_score = self._compute_scores_batched(data, score_func, data_number, batch_size)

            # Compute kernel function
            with torch.no_grad():
                eval_grad_k, eval_value_k = self.kernel(data, data)

            # Compute SVGD gradient
            grad_tensor = self._compute_svgd_gradient(
                eval_score, eval_grad_k, eval_value_k, data_number, data, l1_ratio, alpha
            )

            # Apply special feature enhancement
            if problem_feature_indices:
                grad_tensor = self._apply_problem_feature_enhancement(
                    grad_tensor, data, problem_feature_indices, special_alpha, alpha,
                    l1_ratio, iter_idx, iter_steps
                )

            # Apply categorical variable processing
            if self.categorical_indices:
                grad_tensor = self._apply_categorical_processing(grad_tensor, data)

            # Check numerical stability
            has_nan_inf = torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any()
            if has_nan_inf:
                consecutive_valid_iters = 0
                grad_tensor = self._handle_numerical_instability(grad_tensor, iter_idx)
            else:
                consecutive_valid_iters += 1

            # Apply gradient update
            self._apply_gradient_update(data, grad_tensor, mask_matrix, grad_optim)

            # Periodic checking and correction
            if iter_idx % 50 == 0 and problem_feature_indices:
                self._periodic_correction(data, problem_feature_indices)

            # Early stopping check
            if self._should_early_stop(consecutive_valid_iters, iter_idx, iter_steps):
                logging.info(f"Early stopping at iteration {iter_idx} due to stability")
                break

        return data

    def _compute_scores_batched(self,
                               data: torch.Tensor,
                               score_func: Callable,
                               data_number: int,
                               batch_size: int) -> torch.Tensor:
        """Compute scores in batches"""
        with torch.no_grad():
            if data_number > batch_size:
                all_eval_scores = []
                num_batches = (data_number + batch_size - 1) // batch_size

                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min((b + 1) * batch_size, data_number)
                    batch_data = data[start_idx:end_idx]
                    batch_scores = score_func(batch_data)
                    all_eval_scores.append(batch_scores)

                return torch.cat(all_eval_scores, dim=0)
            else:
                return score_func(data)

    def _compute_svgd_gradient(self,
                              eval_score: torch.Tensor,
                              eval_grad_k: torch.Tensor,
                              eval_value_k: torch.Tensor,
                              data_number: int,
                              data: torch.Tensor,
                              l1_ratio: float,
                              alpha: float) -> torch.Tensor:
        """Compute SVGD gradient"""
        # Basic SVGD gradient
        grad_tensor = -1.0 * (torch.matmul(eval_value_k, eval_score) / data_number)
        grad_tensor = grad_tensor - self.entropy_reg * eval_grad_k / data_number

        # Add elastic net regularization
        l1_term = alpha * l1_ratio * torch.sign(data)
        l2_term = alpha * (1 - l1_ratio) * 2 * data
        grad_tensor += l1_term + l2_term

        return grad_tensor

    def _apply_problem_feature_enhancement(self,
                                         grad_tensor: torch.Tensor,
                                         data: torch.Tensor,
                                         problem_feature_indices: List[int],
                                         special_alpha: float,
                                         alpha: float,
                                         l1_ratio: float,
                                         iter_idx: int,
                                         iter_steps: int) -> torch.Tensor:
        """Apply problem feature enhancement"""
        for idx in problem_feature_indices:
            # Use stronger L2 regularization for problem features
            special_l2_reg = (special_alpha - alpha) * (1 - l1_ratio) * 2 * data[:, idx]
            grad_tensor[:, idx] += special_l2_reg

            # Add correction force in later iterations
            if iter_idx > iter_steps * 0.8:
                current_mean = torch.mean(data[:, idx])
                correction = 0.01 * (current_mean - data[:, idx])
                grad_tensor[:, idx] += correction

        return grad_tensor

    def _apply_categorical_processing(self,
                                    grad_tensor: torch.Tensor,
                                    data: torch.Tensor) -> torch.Tensor:
        """Apply categorical variable processing"""
        for idx in self.categorical_indices:
            cat_vals = data[:, idx]
            nearest_int = torch.round(cat_vals)
            distance = torch.abs(nearest_int - cat_vals)
            direction = torch.sign(nearest_int - cat_vals)

            # Get difficulty factor
            difficulty = self._difficulty_cache.get(idx, 1.0)

            # Apply adjustment
            adjustment = direction * torch.clamp(
                distance * Constants.CAT_ADJUSTMENT_FACTOR * difficulty,
                min=0.05, max=0.7
            )
            grad_tensor[:, idx] = grad_tensor[:, idx] + adjustment

        return grad_tensor

    def _handle_numerical_instability(self,
                                    grad_tensor: torch.Tensor,
                                    iter_idx: int) -> torch.Tensor:
        """Handle numerical instability"""
        if iter_idx % 50 == 0:
            logging.info(f"Gradients have NaN or Inf at iteration {iter_idx}")

        # Clean NaN/Inf values
        return torch.where(
            torch.isnan(grad_tensor) | torch.isinf(grad_tensor),
            torch.zeros_like(grad_tensor),
            grad_tensor
        )

    def _apply_gradient_update(self,
                             data: torch.Tensor,
                             grad_tensor: torch.Tensor,
                             mask_matrix: torch.Tensor,
                             grad_optim: torch.optim.Optimizer):
        """Apply gradient update"""
        grad_optim.zero_grad()
        data.grad = torch.masked_fill(
            input=grad_tensor.contiguous(),
            mask=mask_matrix,
            value=0.0
        )
        grad_optim.step()

    def _periodic_correction(self,
                           data: torch.Tensor,
                           problem_feature_indices: List[int]):
        """Periodically check and correct outliers in special columns"""
        with torch.no_grad():
            for idx in problem_feature_indices:
                q1 = torch.quantile(data[:, idx], Constants.CLAMP_QUANTILES[0])
                q99 = torch.quantile(data[:, idx], Constants.CLAMP_QUANTILES[1])
                data[:, idx] = torch.clamp(data[:, idx], q1, q99)

    def _should_early_stop(self,
                          consecutive_valid_iters: int,
                          iter_idx: int,
                          iter_steps: int) -> bool:
        """Determine if early stopping should be applied"""
        return (consecutive_valid_iters >= 50 and iter_idx > iter_steps * 0.6)


class SimpleSVGDSampler:
    """Simplified SVGD sampler for basic use cases"""

    def __init__(self, kernel: RBFKernel, entropy_reg: float = 10.0):
        self.kernel = kernel
        self.entropy_reg = entropy_reg

    def sample(self,
               data: torch.Tensor,
               score_func: Callable,
               grad_optim: torch.optim.Optimizer,
               mask_matrix: torch.Tensor,
               iter_steps: int = 500) -> torch.Tensor:
        """
        Simplified SVGD sampling

        Args:
        - data: Initial sample data
        - score_func: Scoring function
        - grad_optim: Gradient optimizer
        - mask_matrix: Mask matrix
        - iter_steps: Number of iterations

        Returns:
        - Updated sample data
        """
        data_number = data.shape[0]

        for iter_idx in range(iter_steps):
            with torch.no_grad():
                eval_score = score_func(data)
                eval_grad_k, eval_value_k = self.kernel(data, data)

            # Basic SVGD gradient computation
            grad_tensor = -1.0 * (torch.matmul(eval_value_k, eval_score) / data_number)
            grad_tensor = grad_tensor - self.entropy_reg * eval_grad_k / data_number

            # Apply gradient update
            grad_optim.zero_grad()
            data.grad = torch.masked_fill(
                input=grad_tensor.contiguous(), 
                mask=mask_matrix, 
                value=0.0
            )
            grad_optim.step()
        
        return data