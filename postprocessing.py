"""
Post-processing module - handles post-processing of imputed data, including correlation analysis and value correction
"""
import torch
import numpy as np
import pandas as pd
import logging
import random
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple, Optional
from .config import Constants

class CorrelationAnalyzer:
    """Correlation analyzer"""

    def __init__(self,
                 categorical_indices: List[int] = None,
                 correlation_threshold: float = 0.1,
                 continuous_correlation_threshold: float = 0.2):
        """
        Initialize correlation analyzer

        Args:
        - categorical_indices: Categorical variable indices
        - correlation_threshold: Categorical correlation threshold
        - continuous_correlation_threshold: Continuous correlation threshold
        """
        self.categorical_indices = categorical_indices or []
        self.correlation_threshold = correlation_threshold
        self.continuous_correlation_threshold = continuous_correlation_threshold

    def compute_feature_correlations(self,
                                   X: torch.Tensor,
                                   mask: torch.Tensor,
                                   min_samples: int = 50) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute feature correlations

        Args:
        - X: Data tensor
        - mask: Missing value mask
        - min_samples: Minimum sample size requirement

        Returns:
        - Feature correlations dictionary
        """
        n, d = X.shape
        feature_correlations = {}

        if n <= min_samples:
            return feature_correlations

        with torch.no_grad():
            for j in self.categorical_indices:
                non_missing_j = ~mask[:, j]
                if non_missing_j.sum() > 10:
                    correlations = []

                    for k in range(d):
                        if j != k:
                            non_missing_both = non_missing_j & ~mask[:, k]
                            if non_missing_both.sum() > 10:
                                try:
                                    corr = self._compute_pairwise_correlation(
                                        X, j, k, non_missing_both
                                    )

                                    if corr > self.correlation_threshold:
                                        correlations.append((k, corr))

                                except Exception as e:
                                    if random.random() < 0.1:  # Only log 10% of errors
                                        logging.debug(f"Error computing correlation between features {j} and {k}: {e}")

                    # Sort by correlation and keep top N
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    top_n = 5
                    feature_correlations[j] = correlations[:top_n] if len(correlations) >= top_n else correlations

        return feature_correlations

    def compute_continuous_correlations(self,
                                      X: torch.Tensor,
                                      mask: torch.Tensor,
                                      problem_feature_indices: List[int]) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute continuous variable correlations

        Args:
        - X: Data tensor
        - mask: Missing value mask
        - problem_feature_indices: Problem feature indices

        Returns:
        - Continuous correlations dictionary
        """
        continuous_correlations = {}

        for j in problem_feature_indices:
            non_missing_j = ~mask[:, j]
            if non_missing_j.sum() > 10:
                correlations = []

                for k in range(X.shape[1]):
                    if j != k and k not in self.categorical_indices:
                        non_missing_both = non_missing_j & ~mask[:, k]
                        if non_missing_both.sum() > 10:
                            try:
                                x_data = X[non_missing_both, j].cpu().numpy()
                                y_data = X[non_missing_both, k].cpu().numpy()

                                corr = np.corrcoef(x_data, y_data)[0, 1]

                                if abs(corr) > self.continuous_correlation_threshold:
                                    correlations.append((k, abs(corr)))

                            except Exception as e:
                                if random.random() < 0.1:
                                    logging.debug(f"Error computing Pearson correlation between features {j} and {k}: {e}")

                correlations.sort(key=lambda x: x[1], reverse=True)
                top_n = 5
                continuous_correlations[j] = correlations[:top_n] if len(correlations) >= top_n else correlations

        return continuous_correlations

    def _compute_pairwise_correlation(self,
                                    X: torch.Tensor,
                                    j: int,
                                    k: int,
                                    non_missing_both: torch.Tensor) -> float:
        """
        Compute pairwise feature correlation

        Args:
        - X: Data tensor
        - j: First feature index
        - k: Second feature index
        - non_missing_both: Mask for both features non-missing

        Returns:
        - Correlation coefficient
        """
        if k in self.categorical_indices:
            # Categorical-Categorical: use Cramer's V
            return self._compute_cramers_v(X, j, k, non_missing_both)
        else:
            # Categorical-Continuous: use correlation ratio
            return self._compute_correlation_ratio(X, j, k, non_missing_both)

    def _compute_cramers_v(self,
                          X: torch.Tensor,
                          j: int,
                          k: int,
                          non_missing_both: torch.Tensor) -> float:
        """Compute Cramer's V correlation coefficient"""
        x_data = X[non_missing_both, j].cpu().numpy()
        y_data = X[non_missing_both, k].cpu().numpy()

        contingency = pd.crosstab(
            pd.Series(x_data.round(), name='x'),
            pd.Series(y_data.round(), name='y')
        )

        chi2 = chi2_contingency(contingency.values)[0]
        n_samples = contingency.sum().sum()
        phi2 = chi2 / n_samples

        r, c = contingency.shape
        return np.sqrt(phi2 / min(r - 1, c - 1)) if min(r - 1, c - 1) > 0 else 0

    def _compute_correlation_ratio(self,
                                  X: torch.Tensor,
                                  j: int,
                                  k: int,
                                  non_missing_both: torch.Tensor) -> float:
        """Compute correlation ratio"""
        x_cat = X[non_missing_both, j].cpu().numpy().round()
        y_num = X[non_missing_both, k].cpu().numpy()

        categories = np.unique(x_cat)

        if len(categories) > 1:
            cat_means = np.array([y_num[x_cat == cat].mean() for cat in categories])
            cat_counts = np.array([np.sum(x_cat == cat) for cat in categories])

            overall_mean = y_num.mean()
            between_var = np.sum(cat_counts * (cat_means - overall_mean) ** 2)
            total_var = np.sum((y_num - overall_mean) ** 2)

            return np.sqrt(between_var / total_var) if total_var > 0 else 0

        return 0


class PostProcessor:
    """Post-processor"""

    def __init__(self,
                 categorical_indices: List[int] = None,
                 category_values: Dict[int, torch.Tensor] = None,
                 category_probs: Dict[int, torch.Tensor] = None,
                 bandwidth: float = 10.0):
        """
        Initialize post-processor

        Args:
        - categorical_indices: Categorical variable indices
        - category_values: Valid values for categorical variables
        - category_probs: Probability distributions for categorical variables
        - bandwidth: Kernel function bandwidth
        """
        self.categorical_indices = categorical_indices or []
        self.category_values = category_values or {}
        self.category_probs = category_probs or {}
        self.bandwidth = bandwidth

    def process_categorical_variables(self,
                                    X_filled: torch.Tensor,
                                    mask: torch.Tensor,
                                    feature_correlations: Dict[int, List[Tuple[int, float]]]) -> torch.Tensor:
        """
        Process categorical variable post-processing

        Args:
        - X_filled: Filled data
        - mask: Original missing value mask
        - feature_correlations: Feature correlations

        Returns:
        - Processed data
        """
        with torch.no_grad():
            for idx in self.categorical_indices:
                missing_idx = mask[:, idx]
                if missing_idx.sum() > 0 and idx in self.category_values:
                    self._process_single_categorical(
                        X_filled, idx, missing_idx, feature_correlations
                    )

        return X_filled

    def _process_single_categorical(self,
                                  X_filled: torch.Tensor,
                                  idx: int,
                                  missing_idx: torch.Tensor,
                                  feature_correlations: Dict[int, List[Tuple[int, float]]]):
        """Process single categorical variable"""
        values = self.category_values[idx].to(X_filled.device)
        probs = self.category_probs[idx].to(X_filled.device)

        batch_size = 128
        missing_indices = torch.where(missing_idx)[0]
        num_batches = (len(missing_indices) + batch_size - 1) // batch_size

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(missing_indices))
            batch_indices = missing_indices[start_idx:end_idx]

            self._process_categorical_batch(
                X_filled, idx, batch_indices, values, probs, feature_correlations
            )

    def _process_categorical_batch(self,
                                 X_filled: torch.Tensor,
                                 idx: int,
                                 batch_indices: torch.Tensor,
                                 values: torch.Tensor,
                                 probs: torch.Tensor,
                                 feature_correlations: Dict[int, List[Tuple[int, float]]]):
        """Process categorical variable batch"""
        current_vals = X_filled[batch_indices, idx]

        # Compute base scores
        distances = torch.abs(values.unsqueeze(0) - current_vals.unsqueeze(1))
        similarity = torch.exp(-distances / self.bandwidth)
        scores = similarity * probs.unsqueeze(0)

        # Apply correlation enhancement
        if idx in feature_correlations:
            scores = self._apply_correlation_enhancement(
                X_filled, idx, batch_indices, scores, values, feature_correlations[idx]
            )

        # Select best values
        best_idx = torch.argmax(scores, dim=1)
        X_filled.data[batch_indices, idx] = values[best_idx]

    def _apply_correlation_enhancement(self,
                                     X_filled: torch.Tensor,
                                     idx: int,
                                     batch_indices: torch.Tensor,
                                     scores: torch.Tensor,
                                     values: torch.Tensor,
                                     related_features: List[Tuple[int, float]]) -> torch.Tensor:
        """Apply correlation enhancement"""
        mask = torch.isnan(X_filled)  # Recompute mask

        for related_idx, corr_strength in related_features:
            for b_idx, orig_idx in enumerate(batch_indices):
                if not mask[orig_idx, related_idx]:
                    related_val = X_filled[orig_idx, related_idx].item()

                    if related_idx in self.categorical_indices:
                        scores = self._enhance_categorical_categorical(
                            X_filled, scores, b_idx, idx, related_idx, related_val,
                            values, corr_strength, mask
                        )
                    else:
                        scores = self._enhance_categorical_continuous(
                            X_filled, scores, b_idx, idx, related_idx, related_val,
                            values, corr_strength, mask
                        )

        return scores

    def _enhance_categorical_categorical(self,
                                       X_filled: torch.Tensor,
                                       scores: torch.Tensor,
                                       b_idx: int,
                                       idx: int,
                                       related_idx: int,
                                       related_val: float,
                                       values: torch.Tensor,
                                       corr_strength: float,
                                       mask: torch.Tensor) -> torch.Tensor:
        """Categorical-categorical correlation enhancement"""
        if related_idx in self.category_values:
            rel_values = self.category_values[related_idx].to(X_filled.device)
            rel_val_rounded = rel_values[torch.abs(rel_values - related_val).argmin()]

            non_missing_both = (~mask[:, idx]) & (~mask[:, related_idx])
            rel_match = torch.abs(X_filled[:, related_idx] - rel_val_rounded) < 0.5
            idx_match = non_missing_both & rel_match

            if idx_match.sum() > 0:
                for v_idx, val in enumerate(values):
                    conditional_count = torch.sum(
                        torch.abs(X_filled[idx_match, idx] - val) < 0.5
                    ).item()
                    conditional_prob = conditional_count / idx_match.sum().item()
                    boost_factor = 1.0 + corr_strength * conditional_prob * 2.0
                    scores[b_idx, v_idx] *= boost_factor

        return scores

    def _enhance_categorical_continuous(self,
                                      X_filled: torch.Tensor,
                                      scores: torch.Tensor,
                                      b_idx: int,
                                      idx: int,
                                      related_idx: int,
                                      related_val: float,
                                      values: torch.Tensor,
                                      corr_strength: float,
                                      mask: torch.Tensor) -> torch.Tensor:
        """Categorical-continuous correlation enhancement"""
        non_missing_both = (~mask[:, idx]) & (~mask[:, related_idx])
        if non_missing_both.sum() > 0:
            rel_sim = torch.exp(
                -torch.abs(X_filled[non_missing_both, related_idx] - related_val) / self.bandwidth
            )
            rel_sim = rel_sim / (rel_sim.sum() + 1e-10)

            for v_idx, val in enumerate(values):
                class_sim = torch.exp(-torch.abs(X_filled[non_missing_both, idx] - val) / 0.5)
                weighted_prob = torch.sum(rel_sim * class_sim).item()
                boost_factor = 1.0 + corr_strength * weighted_prob * 2.0
                scores[b_idx, v_idx] *= boost_factor

        return scores

    def process_continuous_variables(self,
                                   X_filled: torch.Tensor,
                                   mask: torch.Tensor,
                                   continuous_correlations: Dict[int, List[Tuple[int, float]]],
                                   problem_feature_indices: List[int]) -> torch.Tensor:
        """
        Process continuous variable post-processing

        Args:
        - X_filled: Filled data
        - mask: Original missing value mask
        - continuous_correlations: Continuous correlations
        - problem_feature_indices: Problem feature indices

        Returns:
        - Processed data
        """
        if not problem_feature_indices or not continuous_correlations:
            return X_filled

        for idx in problem_feature_indices:
            missing_idx = mask[:, idx]
            if missing_idx.sum() > 0 and idx in continuous_correlations:
                logging.info(f"Post-processing feature {idx} using correlated features")
                self._process_continuous_feature(
                    X_filled, idx, missing_idx, continuous_correlations[idx], mask
                )

        return X_filled

    def _process_continuous_feature(self,
                                  X_filled: torch.Tensor,
                                  idx: int,
                                  missing_idx: torch.Tensor,
                                  related_features: List[Tuple[int, float]],
                                  mask: torch.Tensor):
        """Process single continuous feature"""
        batch_size = 128
        missing_indices = torch.where(missing_idx)[0]
        num_batches = (len(missing_indices) + batch_size - 1) // batch_size

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(missing_indices))
            batch_indices = missing_indices[start_idx:end_idx]

            for b_idx, orig_idx in enumerate(batch_indices):
                current_val = X_filled[orig_idx, idx].item()
                adjustments = []
                total_weight = 0.0

                for related_idx, corr_strength in related_features:
                    if not mask[orig_idx, related_idx]:
                        adjustment = self._compute_continuous_adjustment(
                            X_filled, idx, related_idx, orig_idx, corr_strength, mask
                        )
                        if adjustment is not None:
                            adjustments.append(adjustment)
                            total_weight += corr_strength

                # Apply adjustments
                if adjustments and total_weight > 0:
                    weighted_adjustment = sum(adjustments) / len(adjustments)
                    adjustment_strength = Constants.CORRECTION_STRENGTH
                    new_val = (1 - adjustment_strength) * current_val + adjustment_strength * weighted_adjustment
                    X_filled.data[orig_idx, idx] = new_val

    def _compute_continuous_adjustment(self,
                                     X_filled: torch.Tensor,
                                     idx: int,
                                     related_idx: int,
                                     orig_idx: int,
                                     corr_strength: float,
                                     mask: torch.Tensor) -> Optional[float]:
        """Compute continuous variable adjustment value"""
        related_val = X_filled[orig_idx, related_idx].item()
        non_missing_both = (~mask[:, idx]) & (~mask[:, related_idx])

        if non_missing_both.sum() > 0:
            similarities = torch.exp(
                -((X_filled[non_missing_both, related_idx] - related_val) ** 2) /
                (2 * self.bandwidth ** 2)
            )
            weights = similarities / (similarities.sum() + 1e-10)
            weighted_target = torch.sum(weights * X_filled[non_missing_both, idx])
            return weighted_target.item()

        return None

    def apply_range_constraints(self,
                              X_filled: torch.Tensor,
                              mask: torch.Tensor,
                              problem_feature_indices: List[int]) -> torch.Tensor:
        """
        Apply range constraints

        Args:
        - X_filled: Filled data
        - mask: Original missing value mask
        - problem_feature_indices: Problem feature indices

        Returns:
        - Constrained data
        """
        for idx in problem_feature_indices:
            missing_mask = mask[:, idx]
            if missing_mask.sum() > 0:
                non_missing = ~missing_mask
                if non_missing.sum() > 0:
                    q01 = torch.quantile(X_filled[non_missing, idx], Constants.CLAMP_QUANTILES[0])
                    q99 = torch.quantile(X_filled[non_missing, idx], Constants.CLAMP_QUANTILES[1])
                    X_filled.data[missing_mask, idx] = torch.clamp(
                        X_filled.data[missing_mask, idx], q01, q99
                    )
                    logging.info(
                        f"Constrained feature {idx} imputed values to quantile range: [{q01.item():.4f}, {q99.item():.4f}]"
                    )
        
        return X_filled