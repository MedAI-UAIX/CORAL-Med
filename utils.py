"""
Utility functions module - provides various helper functions and tools
"""
import os
import torch
import numpy as np
import pandas as pd
import logging
import random
import json
import datetime
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

def enable_reproducible_results(seed: int = 42):
    """
    Set random seed to ensure reproducible results

    Args:
    - seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.mps.is_available():
        # MPS specific settings
        torch.mps.manual_seed(seed)

    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    logging.info(f"Random seed set to: {seed}")

def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging

    Args:
    - log_dir: Log directory
    - log_level: Log level

    Returns:
    - Configured logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'imputation.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized, log directory: {log_dir}")

    return logger

def copy_code_folder(source_dir: str, target_dir: str, exclude_patterns: List[str] = None):
    """
    Backup code folder

    Args:
    - source_dir: Source directory
    - target_dir: Target directory
    - exclude_patterns: List of patterns to exclude
    """
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__', '*.pyc', '*.pyo', '.git', '.DS_Store',
            'results', '*.log', 'checkpoints', 'wandb'
        ]

    code_backup_dir = os.path.join(target_dir, 'code_backup')

    try:
        if os.path.exists(code_backup_dir):
            shutil.rmtree(code_backup_dir)

        def should_exclude(path: str) -> bool:
            for pattern in exclude_patterns:
                if pattern in path:
                    return True
            return False

        # Recursively copy, excluding specific files
        shutil.copytree(
            source_dir,
            code_backup_dir,
            ignore=lambda dir, files: [f for f in files if should_exclude(os.path.join(dir, f))]
        )

        logging.info(f"Code backed up to: {code_backup_dir}")

    except Exception as e:
        logging.warning(f"Code backup failed: {e}")

def save_parameters(params: Dict[str, Any], filepath: str):
    """
    Save parameters to JSON file

    Args:
    - params: Parameters dictionary
    - filepath: File path
    """
    # Handle objects that cannot be JSON serialized
    def json_serializable(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, default=json_serializable, ensure_ascii=False, indent=4)
        logging.info(f"Parameters saved to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save parameters: {e}")

def load_parameters(filepath: str) -> Dict[str, Any]:
    """
    Load parameters from JSON file

    Args:
    - filepath: File path

    Returns:
    - Parameters dictionary
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            params = json.load(f)
        logging.info(f"Parameters loaded from {filepath}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        return {}

def validate_data(data: Union[np.ndarray, pd.DataFrame],
                 name: str = "data") -> Union[np.ndarray, pd.DataFrame]:
    """
    Validate input data

    Args:
    - data: Input data
    - name: Data name

    Returns:
    - Validated data
    """
    if data is None:
        raise ValueError(f"{name} cannot be None")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError(f"{name} cannot be empty")

        # Check data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logging.warning(f"{name} has no numeric columns")

        logging.info(f"{name} validation passed: {data.shape}, numeric columns: {len(numeric_cols)}")

    elif isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError(f"{name} cannot be empty")

        if data.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")

        logging.info(f"{name} validation passed: {data.shape}")

    else:
        raise TypeError(f"{name} must be pandas.DataFrame or numpy.ndarray")

    return data

def check_missing_pattern(data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Check missing value pattern

    Args:
    - data: Input data

    Returns:
    - Missing value statistics
    """
    if isinstance(data, pd.DataFrame):
        missing_count = data.isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100

        stats = {
            'total_missing': missing_count.sum(),
            'missing_percentage_overall': (missing_count.sum() / data.size) * 100,
            'columns_with_missing': missing_count[missing_count > 0].to_dict(),
            'missing_percentage_by_column': missing_percentage[missing_percentage > 0].to_dict()
        }

    elif isinstance(data, np.ndarray):
        missing_mask = np.isnan(data)
        missing_count_by_col = np.sum(missing_mask, axis=0)
        missing_percentage_by_col = (missing_count_by_col / data.shape[0]) * 100

        stats = {
            'total_missing': np.sum(missing_mask),
            'missing_percentage_overall': (np.sum(missing_mask) / data.size) * 100,
            'missing_count_by_column': missing_count_by_col.tolist(),
            'missing_percentage_by_column': missing_percentage_by_col.tolist()
        }

    else:
        raise TypeError("Data must be pandas.DataFrame or numpy.ndarray")

    return stats

def format_time(seconds: float) -> str:
    """
    Format time display

    Args:
    - seconds: Number of seconds

    Returns:
    - Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} minutes {secs:.2f} seconds"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)} hours {int(minutes)} minutes {secs:.2f} seconds"

def create_timestamp_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Create timestamped directory

    Args:
    - base_dir: Base directory
    - prefix: Prefix

    Returns:
    - Created directory path
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    return run_dir

def get_memory_usage() -> Dict[str, float]:
    """
    Get memory usage

    Returns:
    - Memory usage statistics
    """
    import psutil

    # System memory
    system_memory = psutil.virtual_memory()

    # Current process memory
    process = psutil.Process()
    process_memory = process.memory_info()

    stats = {
        'system_total_gb': system_memory.total / (1024**3),
        'system_available_gb': system_memory.available / (1024**3),
        'system_used_percent': system_memory.percent,
        'process_rss_gb': process_memory.rss / (1024**3),
        'process_vms_gb': process_memory.vms / (1024**3)
    }

    # GPU memory (if available)
    if torch.cuda.is_available():
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)

    if torch.mps.is_available():
        try:
            stats['mps_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
        except Exception:
            pass

    return stats

def log_memory_usage(stage: str = ""):
    """
    Log memory usage

    Args:
    - stage: Stage name
    """
    try:
        memory_stats = get_memory_usage()
        log_msg = f"Memory usage - {stage}: "
        log_msg += f"System: {memory_stats['system_used_percent']:.1f}%, "
        log_msg += f"Process: {memory_stats['process_rss_gb']:.2f}GB"

        if 'gpu_allocated_gb' in memory_stats:
            log_msg += f", GPU: {memory_stats['gpu_allocated_gb']:.2f}GB"

        logging.info(log_msg)

    except Exception as e:
        logging.debug(f"Failed to get memory usage: {e}")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division to avoid division by zero

    Args:
    - numerator: Numerator
    - denominator: Denominator
    - default: Default value

    Returns:
    - Division result or default value
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    return numerator / denominator

def clip_extreme_values(data: Union[np.ndarray, torch.Tensor],
                       percentile_range: Tuple[float, float] = (1, 99)) -> Union[np.ndarray, torch.Tensor]:
    """
    Clip extreme values

    Args:
    - data: Input data
    - percentile_range: Percentile range

    Returns:
    - Clipped data
    """
    if isinstance(data, torch.Tensor):
        q_low = torch.quantile(data, percentile_range[0] / 100)
        q_high = torch.quantile(data, percentile_range[1] / 100)
        return torch.clamp(data, q_low, q_high)
    else:
        q_low = np.percentile(data, percentile_range[0])
        q_high = np.percentile(data, percentile_range[1])
        return np.clip(data, q_low, q_high)

def detect_outliers(data: Union[np.ndarray, pd.Series],
                   method: str = "iqr",
                   factor: float = 1.5) -> Union[np.ndarray, pd.Series]:
    """
    Detect outliers

    Args:
    - data: Input data
    - method: Detection method ("iqr", "zscore")
    - factor: Factor

    Returns:
    - Outlier mask
    """
    if method == "iqr":
        if isinstance(data, pd.Series):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
        else:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)

        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        return (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        if isinstance(data, pd.Series):
            z_scores = np.abs((data - data.mean()) / data.std())
        else:
            z_scores = np.abs((data - np.mean(data)) / np.std(data))

        return z_scores > factor

    else:
        raise ValueError(f"Unsupported method: {method}")

def validate_categorical_indices(categorical_indices: List[int],
                                n_features: int) -> List[int]:
    """
    Validate categorical variable indices

    Args:
    - categorical_indices: List of categorical variable indices
    - n_features: Total number of features

    Returns:
    - Validated indices list
    """
    if categorical_indices is None:
        return []

    valid_indices = []
    for idx in categorical_indices:
        if 0 <= idx < n_features:
            valid_indices.append(idx)
        else:
            logging.warning(f"Invalid categorical variable index: {idx} (total features: {n_features})")

    return valid_indices

class ProgressReporter:
    """Progress reporter"""

    def __init__(self, total_steps: int, report_interval: int = 10):
        self.total_steps = total_steps
        self.report_interval = report_interval
        self.start_time = None
        self.current_step = 0

    def start(self):
        """Start timing"""
        self.start_time = datetime.datetime.now()
        logging.info(f"Starting execution, total steps: {self.total_steps}")

    def step(self, step_name: str = ""):
        """Update progress"""
        self.current_step += 1

        if self.current_step % self.report_interval == 0 or self.current_step == self.total_steps:
            elapsed = datetime.datetime.now() - self.start_time
            progress = (self.current_step / self.total_steps) * 100

            if self.current_step > 0:
                avg_time_per_step = elapsed.total_seconds() / self.current_step
                remaining_steps = self.total_steps - self.current_step
                eta = datetime.timedelta(seconds=avg_time_per_step * remaining_steps)

                log_msg = f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%)"
                if step_name:
                    log_msg += f" - {step_name}"
                log_msg += f" - Elapsed: {format_time(elapsed.total_seconds())}"
                if remaining_steps > 0:
                    log_msg += f" - ETA: {format_time(eta.total_seconds())}"

                logging.info(log_msg)

    def finish(self):
        """Finish"""
        if self.start_time:
            elapsed = datetime.datetime.now() - self.start_time
            logging.info(f"Execution completed, total time: {format_time(elapsed.total_seconds())}")


class ConfigValidator:
    """Configuration validator"""

    @staticmethod
    def validate_imputer_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate imputer configuration

        Args:
        - config: Configuration dictionary

        Returns:
        - Validated configuration
        """
        validated = config.copy()

        # Validate numerical parameters
        numeric_params = {
            'lr': (1e-6, 1.0, 1e-1),
            'bandwidth': (0.1, 100.0, 10.0),
            'entropy_reg': (0.1, 100.0, 10.0),
            'noise': (0.0, 1.0, 0.1),
            'eps': (1e-8, 1e-1, 0.01)
        }

        for param, (min_val, max_val, default) in numeric_params.items():
            if param in validated:
                val = validated[param]
                if not isinstance(val, (int, float)) or val < min_val or val > max_val:
                    logging.warning(f"Parameter {param} value {val} out of range [{min_val}, {max_val}], using default {default}")
                    validated[param] = default

        # Validate integer parameters
        integer_params = {
            'niter': (1, 1000, 50),
            'batchsize': (8, 1024, 128),
            'score_net_epoch': (100, 10000, 2000)
        }

        for param, (min_val, max_val, default) in integer_params.items():
            if param in validated:
                val = validated[param]
                if not isinstance(val, int) or val < min_val or val > max_val:
                    logging.warning(f"Parameter {param} value {val} out of range [{min_val}, {max_val}], using default {default}")
                    validated[param] = default
        
        return validated