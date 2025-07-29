"""
Core imputer module - main imputer class implementation
"""
import torch
import numpy as np
import logging
import time
import gc
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any

from .config import ImputerConfig, Constants, optimize_torch_settings
from .kernels import create_rbf_kernel
from .svgd_sampler import SVGDSampler
from .training import ScoreNetworkTrainer
from .postprocessing import CorrelationAnalyzer, PostProcessor

# Import score-related modules
try:
    import sys
    sys.path.append('model')
    from score import score_tuple
    from score.score_trainer import Trainer
except ImportError as e:
    logging.warning(f"Unable to import score modules: {e}")
    score_tuple = None

class ModifiedNeuralGradFlowImputer:
    """Modified Neural Gradient Flow Imputer"""

    def __init__(self,
                 categorical_indices: List[int] = None,
                 entropy_reg: float = 10.0,
                 eps: float = 0.01,
                 lr: float = 1e-1,
                 opt: torch.optim.Optimizer = torch.optim.Adam,
                 niter: int = 50,
                 bandwidth: float = 10.0,
                 mlp_hidden: List[int] = None,
                 score_net_epoch: int = 2000,
                 score_net_lr: float = 1e-3,
                 score_loss_type: str = "dsm",
                 device: torch.device = None,
                 batchsize: int = 128,
                 n_pairs: int = 1,
                 noise: float = 0.1,
                 scaling: float = 0.9):
        """
        Initialize imputer

        Args:
        - categorical_indices: List of categorical variable indices
        - entropy_reg: Entropy regularization coefficient
        - eps: Small numerical stability constant
        - lr: Learning rate
        - opt: Optimizer class
        - niter: Number of iterations
        - bandwidth: Kernel function bandwidth
        - mlp_hidden: MLP hidden layer structure
        - score_net_epoch: Score network training epochs
        - score_net_lr: Score network learning rate
        - score_loss_type: Score loss type
        - device: Computing device
        - batchsize: Batch size
        - n_pairs: Number of pairs
        - noise: Noise level
        - scaling: Scaling factor
        """
        # Basic parameters
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.scaling = scaling
        self.device = device or torch.device("mps" if torch.mps.is_available() else "cpu")

        # Model parameters
        self.mlp_hidden = mlp_hidden or [256, 256]
        self.score_net_epoch = score_net_epoch
        self.score_loss_type = score_loss_type
        self.score_net_lr = score_net_lr
        self.entropy_reg = entropy_reg
        self.bandwidth = bandwidth

        # Categorical variable related
        self.categorical_indices = categorical_indices or []
        self.feature_names = None

        # Create components
        self.kernel = create_rbf_kernel(self.bandwidth)
        self.sampler = None
        self.trainer = None
        self.correlation_analyzer = None
        self.postprocessor = None

        # Cache
        self._cached_sigma = None
        self._correlation_precision = torch.float32

        logging.info(f"Imputer initialized, device: {self.device}")

    def _initialize_components(self, input_dim: int):
        """Initialize components"""
        # Initialize SVGD sampler
        self.sampler = SVGDSampler(
            kernel=self.kernel,
            entropy_reg=self.entropy_reg,
            categorical_indices=self.categorical_indices,
            feature_names=self.feature_names
        )

        # Initialize score network trainer
        if score_tuple is not None:
            self.trainer = ScoreNetworkTrainer(
                input_dim=input_dim,
                mlp_hidden=self.mlp_hidden,
                loss_type=self.score_loss_type,
                learning_rate=self.score_net_lr,
                device=self.device
            )

        # Initialize correlation analyzer
        self.correlation_analyzer = CorrelationAnalyzer(
            categorical_indices=self.categorical_indices,
            correlation_threshold=Constants.CORRELATION_THRESHOLD,
            continuous_correlation_threshold=Constants.CONTINUOUS_CORRELATION_THRESHOLD
        )

    def _prepare_data(self, X: np.ndarray) -> torch.Tensor:
        """
        Prepare input data

        Args:
        - X: Input data array

        Returns:
        - Prepared data tensor
        """
        # Convert to PyTorch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.clone().to(self.device)
        n, d = X.shape

        logging.info(f"Data shape: {n} rows x {d} columns, device: {self.device}")

        return X

    def _initialize_missing_values(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize missing values

        Args:
        - X: Input data tensor

        Returns:
        - Filled data tensor
        - Missing value mask
        - Non-missing value means
        """
        n, d = X.shape
        mask = torch.isnan(X).to(self.device)
        missing_count = mask.sum().item()
        missing_percentage = 100 * missing_count / (n * d)

        logging.info(f"Missing values count: {missing_count}, percentage: {missing_percentage:.2f}%")

        # Initialize fill values
        X_filled = X.detach().clone()
        non_missing_mean = torch.zeros(d, device=self.device)

        # Store categorical variable information
        category_values = {}
        category_probs = {}

        # Calculate initial fill values
        for j in range(d):
            non_missing = ~mask[:, j]
            if non_missing.sum() > 0:
                if j in self.categorical_indices:
                    # Categorical variables use mode
                    with torch.no_grad():
                        values, counts = torch.unique(X[non_missing, j], return_counts=True)
                        non_missing_mean[j] = values[counts.argmax()]

                        # Save categorical information
                        category_values[j] = values.cpu()
                        category_probs[j] = counts.float() / counts.sum()
                else:
                    # Continuous variables use mean
                    non_missing_mean[j] = X[non_missing, j].mean()
            else:
                non_missing_mean[j] = 0.0

        # Fill missing values with mean/mode and noise
        with torch.no_grad():
            for i in range(n):
                for j in range(d):
                    if mask[i, j]:
                        if j in self.categorical_indices:
                            # Categorical variables
                            X_filled[i, j] = non_missing_mean[j] + self.noise * torch.randn(1, device=self.device)
                            X_filled[i, j] = torch.round(X_filled[i, j])
                        else:
                            # Continuous variables
                            X_filled[i, j] = non_missing_mean[j] + self.noise * torch.randn(1, device=self.device)

        # Set post-processor
        self.postprocessor = PostProcessor(
            categorical_indices=self.categorical_indices,
            category_values=category_values,
            category_probs=category_probs,
            bandwidth=self.bandwidth
        )

        return X_filled, mask, non_missing_mean

    def _identify_problem_features(self, d: int) -> List[int]:
        """
        Identify problem feature indices

        Args:
        - d: Feature dimension

        Returns:
        - Problem feature indices list
        """
        problem_feature_indices = []

        if hasattr(self, 'feature_names') and self.feature_names is not None:
            for j in range(d):
                if j < len(self.feature_names):
                    col_name = self.feature_names[j]
                    if col_name in ['DBIL', 'duration']:
                        problem_feature_indices.append(j)
                        logging.info(f"Identified special column {col_name} at index {j}")

        return problem_feature_indices

    def _compute_correlations(self, X: torch.Tensor, mask: torch.Tensor, n: int) -> tuple[Dict, Dict]:
        """
        Compute feature correlations

        Args:
        - X: Data tensor
        - mask: Missing value mask
        - n: Number of samples

        Returns:
        - Feature correlations dictionary
        - Continuous correlations dictionary
        """
        feature_correlations = {}
        continuous_correlations = {}

        if n > 50:  # Only compute correlations when data is sufficient
            phase_start = time.time()

            # Compute categorical variable correlations
            feature_correlations = self.correlation_analyzer.compute_feature_correlations(X, mask)

            # Compute continuous correlations for problem features
            problem_feature_indices = self._identify_problem_features(X.shape[1])
            if problem_feature_indices:
                continuous_correlations = self.correlation_analyzer.compute_continuous_correlations(
                    X, mask, problem_feature_indices
                )

            corr_time = time.time() - phase_start
            logging.info(f"Correlation computation time: {corr_time:.2f} seconds")

        return feature_correlations, continuous_correlations

    def _training_loop(self, X_filled: torch.Tensor, optimizer: torch.optim.Optimizer,
                      mask: torch.Tensor, problem_feature_indices: List[int],
                      report_interval: int = 10, verbose: bool = True) -> torch.Tensor:
        """
        Main training loop

        Args:
        - X_filled: Filled data
        - optimizer: Optimizer
        - mask: Missing value mask
        - problem_feature_indices: Problem feature indices
        - report_interval: Reporting interval
        - verbose: Whether to output detailed information

        Returns:
        - Trained data
        """
        n = X_filled.shape[0]
        grad_mask = ~mask.bool()

        # Adjust batch size
        if self.batchsize > n // 2:
            e = int(np.log2(max(8, n // 2)))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batch size adjusted to {self.batchsize}")

        if verbose:
            logging.info(f"Starting imputation training, batch size = {self.batchsize}, gradient steps = 500, iterations = {self.niter}")

        # Training loop
        for i in range(self.niter):
            # Train score network
            if self.trainer is not None:
                train_dataset = score_tuple.MyDataset(data=X_filled)
                train_dataloader = DataLoader(
                    dataset=train_dataset,
                    shuffle=True,
                    batch_size=self.batchsize
                )
                self.trainer.train(train_dataloader, self.score_net_epoch, i, verbose and i == 0)

                # Use SVGD for data filling
                X_filled = self.sampler.sample(
                    data=X_filled,
                    score_func=self.trainer.get_score_function(),
                    grad_optim=optimizer,
                    mask_matrix=grad_mask,
                    iter_steps=500,
                    problem_feature_indices=problem_feature_indices
                )

            # Basic categorical variable post-processing
            if self.categorical_indices:
                with torch.no_grad():
                    for idx in self.categorical_indices:
                        missing_idx = mask[:, idx]
                        if missing_idx.sum() > 0:
                            X_filled.data[missing_idx, idx] = torch.round(X_filled.data[missing_idx, idx])

            # Report progress
            if verbose and ((i + 1) % report_interval == 0 or i == 0 or i == self.niter - 1):
                logging.info(f'Iteration {i + 1}/{self.niter}')

            # Periodic cache cleanup
            if i % 5 == 4:
                self.clear_cache()

        return X_filled

    def _enhanced_postprocessing(self, X_filled: torch.Tensor, mask: torch.Tensor,
                               feature_correlations: Dict, continuous_correlations: Dict,
                               problem_feature_indices: List[int]) -> torch.Tensor:
        """
        Enhanced post-processing

        Args:
        - X_filled: Filled data
        - mask: Original missing value mask
        - feature_correlations: Feature correlations
        - continuous_correlations: Continuous correlations
        - problem_feature_indices: Problem feature indices

        Returns:
        - Post-processed data
        """
        with torch.no_grad():
            # Process categorical variables
            if self.categorical_indices:
                X_filled = self.postprocessor.process_categorical_variables(
                    X_filled, mask, feature_correlations
                )

            # Process continuous variables
            if problem_feature_indices and continuous_correlations:
                X_filled = self.postprocessor.process_continuous_variables(
                    X_filled, mask, continuous_correlations, problem_feature_indices
                )

            # Apply range constraints
            if problem_feature_indices:
                X_filled = self.postprocessor.apply_range_constraints(
                    X_filled, mask, problem_feature_indices
                )

        return X_filled

    def clear_cache(self):
        """Clear cache and unnecessary storage attributes"""
        if hasattr(self, '_cached_sigma'):
            del self._cached_sigma
            self._cached_sigma = None

        if hasattr(self.kernel, 'clear_cache'):
            self.kernel.clear_cache()

        # Force garbage collection
        gc.collect()

        # Clear cache based on device type
        try:
            if torch.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def fit_transform(self, X: np.ndarray,
                     categorical_indices: List[int] = None,
                     verbose: bool = True,
                     report_interval: int = 10,
                     X_true: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Fit and transform input data, filling missing values

        Args:
        - X: Input data with missing values
        - categorical_indices: List of categorical variable indices
        - verbose: Whether to output log information
        - report_interval: Reporting interval
        - X_true: True data for evaluation

        Returns:
        - Data with missing values filled
        """
        # Optimize PyTorch settings
        optimize_torch_settings()

        # Performance statistics
        start_time = time.time()
        timings = {
            'preprocessing': 0,
            'model_training': 0,
            'score_training': 0,
            'imputation': 0,
            'postprocessing': 0
        }

        # Preprocessing phase
        phase_start = time.time()

        # Update categorical variable indices
        if categorical_indices is not None:
            self.categorical_indices = categorical_indices

        # Prepare data
        X = self._prepare_data(X)
        n, d = X.shape

        # Initialize components
        self._initialize_components(d)

        # Initialize missing values
        X_filled, mask, _ = self._initialize_missing_values(X)

        # Identify problem features
        problem_feature_indices = self._identify_problem_features(d)

        timings['preprocessing'] = time.time() - phase_start
        if verbose:
            logging.info(f"Preprocessing time: {timings['preprocessing']:.2f} seconds")

        # Compute feature correlations
        feature_correlations, continuous_correlations = self._compute_correlations(X, mask, n)

        # Enable gradient computation
        X_filled = X_filled.contiguous()
        X_filled.requires_grad_()

        # Set optimizer
        optimizer = self.opt([X_filled], lr=self.lr)

        # Main training loop
        phase_start = time.time()
        X_filled = self._training_loop(
            X_filled, optimizer, mask, problem_feature_indices, report_interval, verbose
        )
        timings['imputation'] = time.time() - phase_start

        # Enhanced post-processing
        phase_start = time.time()
        X_filled = self._enhanced_postprocessing(
            X_filled, mask, feature_correlations, continuous_correlations, problem_feature_indices
        )
        timings['postprocessing'] = time.time() - phase_start

        # Clean up memory
        self.clear_cache()

        # Report total time
        total_time = time.time() - start_time
        if verbose:
            logging.info(f"Imputation completed, total runtime: {total_time:.2f} seconds")
            logging.info(f"Time breakdown: preprocessing: {timings['preprocessing']:.2f}s, "
                         f"imputation: {timings['imputation']:.2f}s, "
                         f"postprocessing: {timings['postprocessing']:.2f}s")

        return X_filled.detach().cpu()


# Backward compatibility alias
NeuralGradFlowImputer = ModifiedNeuralGradFlowImputer