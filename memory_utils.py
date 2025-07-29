"""
Memory management utilities module - optimize memory usage and manage cache
"""
import os

import torch
import gc
import logging
import psutil
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

class MemoryManager:
    """Memory manager"""

    def __init__(self, device: torch.device = None):
        """
        Initialize memory manager

        Args:
        - device: Computing device
        """
        self.device = device or torch.device("cpu")
        self.cache = {}
        self.memory_threshold = 0.8  # Memory usage threshold
        self._last_cleanup_time = 0

        # Detect device type
        self.device_type = self._detect_device_type()

        logging.info(f"Memory manager initialized, device: {self.device}, type: {self.device_type}")

    def _detect_device_type(self) -> str:
        """Detect device type"""
        if self.device.type == "cuda":
            return "cuda"
        elif self.device.type == "mps":
            return "mps"
        else:
            return "cpu"

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        info = {}

        # System memory
        try:
            system_memory = psutil.virtual_memory()
            info['system'] = {
                'total_gb': system_memory.total / (1024**3),
                'available_gb': system_memory.available / (1024**3),
                'used_percent': system_memory.percent,
                'free_gb': system_memory.free / (1024**3)
            }
        except Exception as e:
            logging.debug(f"Failed to get system memory info: {e}")

        # Process memory
        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            info['process'] = {
                'rss_gb': process_memory.rss / (1024**3),
                'vms_gb': process_memory.vms / (1024**3)
            }
        except Exception as e:
            logging.debug(f"Failed to get process memory info: {e}")

        # GPU memory
        if self.device_type == "cuda" and torch.cuda.is_available():
            try:
                info['cuda'] = {
                    'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
                    'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024**3),
                    'max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / (1024**3)
                }
            except Exception as e:
                logging.debug(f"Failed to get CUDA memory info: {e}")

        elif self.device_type == "mps" and torch.mps.is_available():
            try:
                info['mps'] = {
                    'allocated_gb': torch.mps.current_allocated_memory() / (1024**3)
                }
            except Exception as e:
                logging.debug(f"Failed to get MPS memory info: {e}")

        return info

    def log_memory_status(self, stage: str = ""):
        """Log memory status"""
        try:
            info = self.get_memory_info()

            log_parts = []
            if stage:
                log_parts.append(f"[{stage}]")

            if 'system' in info:
                log_parts.append(f"System: {info['system']['used_percent']:.1f}%")

            if 'process' in info:
                log_parts.append(f"Process: {info['process']['rss_gb']:.2f}GB")

            if 'cuda' in info:
                log_parts.append(f"CUDA: {info['cuda']['allocated_gb']:.2f}GB")

            if 'mps' in info:
                log_parts.append(f"MPS: {info['mps']['allocated_gb']:.2f}GB")

            logging.info(f"Memory status - {' '.join(log_parts)}")

        except Exception as e:
            logging.debug(f"Failed to log memory status: {e}")

    def check_memory_pressure(self) -> bool:
        """Check memory pressure"""
        try:
            info = self.get_memory_info()

            # Check system memory
            if 'system' in info:
                if info['system']['used_percent'] > self.memory_threshold * 100:
                    return True

            # Check GPU memory
            if self.device_type == "cuda" and 'cuda' in info:
                try:
                    total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                    if info['cuda']['allocated_gb'] / total_memory > self.memory_threshold:
                        return True
                except Exception:
                    pass

            return False

        except Exception as e:
            logging.debug(f"Failed to check memory pressure: {e}")
            return False

    def clear_cache(self, force: bool = False):
        """Clear cache"""
        try:
            # Clear Python object cache
            self.cache.clear()

            # Force garbage collection
            gc.collect()

            # Clear device cache
            if self.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if force:
                    torch.cuda.synchronize()

            elif self.device_type == "mps" and torch.mps.is_available():
                torch.mps.empty_cache()
                if force:
                    torch.mps.synchronize()

            logging.debug("Cache cleared")

        except Exception as e:
            logging.debug(f"Failed to clear cache: {e}")

    def optimize_for_large_data(self, data_size_gb: float):
        """Optimize memory settings for large data"""
        if data_size_gb > 1.0:  # Data larger than 1GB
            logging.info(f"Detected large dataset ({data_size_gb:.2f}GB), optimizing memory settings")

            # Lower memory threshold
            self.memory_threshold = 0.7

            # Set PyTorch memory strategy
            if self.device_type == "cuda":
                # Enable memory fragmentation
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

            # More frequent garbage collection
            gc.set_threshold(700, 10, 10)

            logging.info("Large data memory optimization settings completed")

    def get_recommended_batch_size(self, base_batch_size: int, data_shape: tuple) -> int:
        """Get recommended batch size"""
        try:
            n_samples, n_features = data_shape

            # Estimate memory usage per sample (in MB)
            sample_memory_mb = n_features * 4 / (1024**2)  # float32

            # Get available memory
            info = self.get_memory_info()
            available_memory_gb = 4.0  # default value

            if 'system' in info:
                available_memory_gb = min(
                    info['system']['available_gb'],
                    info['system']['total_gb'] * (1 - self.memory_threshold)
                )

            # Calculate recommended batch size
            available_memory_mb = available_memory_gb * 1024
            max_batch_size = int(available_memory_mb / (sample_memory_mb * 4))  # Leave some margin

            recommended_batch_size = min(base_batch_size, max_batch_size, n_samples)

            if recommended_batch_size != base_batch_size:
                logging.info(f"Recommended batch size: {recommended_batch_size} (original: {base_batch_size})")

            return recommended_batch_size

        except Exception as e:
            logging.debug(f"Failed to calculate recommended batch size: {e}")
            return base_batch_size

    @contextmanager
    def memory_context(self, stage_name: str = ""):
        """Memory context manager"""
        self.log_memory_status(f"{stage_name} - start")

        try:
            yield self
        finally:
            # Clear cache
            if self.check_memory_pressure():
                logging.info(f"Detected memory pressure, clearing cache")
                self.clear_cache(force=True)

            self.log_memory_status(f"{stage_name} - finished")


class DataOptimizer:
    """Data optimizer"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.memory_manager = MemoryManager(device)

    def optimize_tensor_dtype(self, tensor: torch.Tensor,
                             precision: str = "auto") -> torch.Tensor:
        """
        Optimize tensor data type

        Args:
        - tensor: Input tensor
        - precision: Precision setting ("auto", "half", "full")

        Returns:
        - Optimized tensor
        """
        if precision == "auto":
            # Auto select precision based on data size
            if tensor.numel() > 1000000:  # More than 1 million elements
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32
        elif precision == "half":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        # Check device compatibility
        if target_dtype == torch.float16:
            if self.device.type == "cpu":
                # CPU doesn't support float16 computation, use float32
                target_dtype = torch.float32
            elif self.device.type == "mps":
                # MPS has limited float16 support, use cautiously
                if tensor.numel() < 100000:  # Use float32 for small datasets
                    target_dtype = torch.float32

        if tensor.dtype != target_dtype:
            logging.debug(f"Optimizing tensor precision: {tensor.dtype} -> {target_dtype}")
            return tensor.to(dtype=target_dtype)

        return tensor

    def create_memory_efficient_tensor(self, shape: tuple,
                                     dtype: torch.dtype = torch.float32,
                                     requires_grad: bool = False) -> torch.Tensor:
        """
        Create memory efficient tensor

        Args:
        - shape: Tensor shape
        - dtype: Data type
        - requires_grad: Whether gradients are needed

        Returns:
        - Created tensor
        """
        # Check memory pressure
        if self.memory_manager.check_memory_pressure():
            self.memory_manager.clear_cache()

        # Optimize data type
        optimized_dtype = self.optimize_tensor_dtype(
            torch.empty(1, dtype=dtype), "auto"
        ).dtype

        try:
            tensor = torch.zeros(shape, dtype=optimized_dtype, device=self.device)
            if requires_grad:
                tensor.requires_grad_(True)

            return tensor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning(f"Out of memory, trying to clear cache and retry")
                self.memory_manager.clear_cache(force=True)

                # Try using smaller data type
                if optimized_dtype == torch.float32:
                    optimized_dtype = torch.float16

                tensor = torch.zeros(shape, dtype=optimized_dtype, device=self.device)
                if requires_grad:
                    tensor.requires_grad_(True)

                return tensor
            else:
                raise

    def batch_process_large_tensor(self, tensor: torch.Tensor,
                                  process_func: callable,
                                  batch_size: int = None) -> torch.Tensor:
        """
        Batch process large tensor

        Args:
        - tensor: Input tensor
        - process_func: Processing function
        - batch_size: Batch size

        Returns:
        - Processed tensor
        """
        if batch_size is None:
            batch_size = self.memory_manager.get_recommended_batch_size(
                128, tensor.shape
            )

        n_samples = tensor.shape[0]
        results = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = tensor[i:end_idx]

            # Process batch
            with torch.no_grad():
                batch_result = process_func(batch)
                results.append(batch_result)

            # Periodic cache cleanup
            if i % (batch_size * 10) == 0:
                self.memory_manager.clear_cache()

        return torch.cat(results, dim=0)


class CacheManager:
    """Cache manager"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size

    def get(self, key: str) -> Any:
        """Get cache item"""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set cache item"""
        if key in self.cache:
            # Update existing item
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


# Global memory manager instance
_global_memory_manager = None

def get_memory_manager(device: torch.device = None) -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager

    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(device)

    return _global_memory_manager

def clear_all_cache():
    """Clear all cache"""
    global _global_memory_manager

    if _global_memory_manager is not None:
        _global_memory_manager.clear_cache(force=True)

    # Additional cleanup
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.mps.is_available():
        torch.mps.empty_cache()

@contextmanager
def memory_efficient_context(device: torch.device = None, stage_name: str = ""):
    """Memory efficient context manager"""
    memory_manager = get_memory_manager(device)
    
    with memory_manager.memory_context(stage_name):
        try:
            yield memory_manager
        finally:
            pass