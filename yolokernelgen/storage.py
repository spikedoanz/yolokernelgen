"""Improved kernel storage and retrieval with type safety and caching."""

import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache

from .types import KernelData, Config
from .exceptions import StorageError
from .naming import generate_filename, parse_filename
from .logging_config import get_storage_logger

logger = get_storage_logger()


class KernelCache:
    """In-memory LRU cache for loaded KernelData objects."""

    def __init__(self, maxsize: int = 128):
        self._cache = {}
        self._access_order = []
        self._maxsize = maxsize

    def get(self, filepath: Path) -> Optional[KernelData]:
        """Get cached kernel data."""
        key = str(filepath)
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Cache hit for {filepath.name}")
            return self._cache[key]
        return None

    def put(self, filepath: Path, kernel_data: KernelData) -> None:
        """Cache kernel data."""
        key = str(filepath)

        # Remove if already exists
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            logger.debug(f"Evicted {lru_key} from cache")

        self._cache[key] = kernel_data
        self._access_order.append(key)
        logger.debug(f"Cached {filepath.name}")

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("Cache cleared")


# Global cache instance
_kernel_cache = KernelCache()


def save_kernel(kernel_data: Union[Dict[str, Any], KernelData], cache_dir: str = ".cache/yolokernelgen") -> Path:
    """Save kernel data to filesystem with atomic writes."""
    try:
        # Convert to KernelData if needed
        if isinstance(kernel_data, dict):
            # Handle legacy format
            if kernel_data.get("version") != 1:
                kernel_data = KernelData.migrate_from_dict(kernel_data)
            else:
                kernel_data = KernelData.from_dict(kernel_data)

        # Ensure cache directory exists
        cache_path = Path(cache_dir) / "generated"
        cache_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        status_prefix = "c" if kernel_data.validation.all_passed else "r"
        filename = generate_filename(
            operation=kernel_data.operation,
            input_shape=kernel_data.metadata.input_shapes[0],
            output_shape=kernel_data.metadata.output_shapes[0],
            torch_source=kernel_data.torch_source,
            params=kernel_data.metadata.hyperparameters,
            status=status_prefix
        )

        filepath = cache_path / filename

        # Atomic write using temp file + rename
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=cache_path,
            prefix=f'tmp_{filename}_',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as tmp_file:
            json.dump(kernel_data.to_dict(), tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)

        # Atomic rename
        tmp_path.rename(filepath)
        logger.info(f"Saved kernel to {filepath}")

        # Cache the kernel data
        _kernel_cache.put(filepath, kernel_data)

        return filepath

    except Exception as e:
        logger.error(f"Failed to save kernel: {e}")
        raise StorageError(f"Failed to save kernel: {e}")


def load_kernel(filepath: Path) -> KernelData:
    """Load kernel data from filesystem with caching."""
    try:
        # Check cache first
        cached = _kernel_cache.get(filepath)
        if cached is not None:
            return cached

        if not filepath.exists():
            raise StorageError(f"Kernel file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Migrate if necessary and convert to KernelData
        kernel_data = KernelData.migrate_from_dict(raw_data)

        # Cache the loaded data
        _kernel_cache.put(filepath, kernel_data)

        logger.debug(f"Loaded kernel from {filepath}")
        return kernel_data

    except Exception as e:
        logger.error(f"Failed to load kernel from {filepath}: {e}")
        raise StorageError(f"Failed to load kernel: {e}")


def find_kernel(
    operation: str,
    input_shape: List[int],
    output_shape: List[int],
    torch_source: str,
    params: Optional[Dict[str, Any]] = None,
    cache_dir: str = ".cache/yolokernelgen",
    prefer_correct: bool = True,
    include_rejected: bool = False
) -> Optional[Path]:
    """Find existing kernel matching specifications.

    Args:
        include_rejected: If False (default), only return correct kernels when prefer_correct=True
    """
    try:
        cache_path = Path(cache_dir) / "generated"

        if not cache_path.exists():
            return None

        # Generate both possible filenames (correct and rejected)
        filename_correct = generate_filename(
            operation, input_shape, output_shape, torch_source, params, "c"
        )
        filename_rejected = generate_filename(
            operation, input_shape, output_shape, torch_source, params, "r"
        )

        # Check for files
        path_correct = cache_path / filename_correct
        path_rejected = cache_path / filename_rejected

        if prefer_correct and path_correct.exists():
            logger.debug(f"Found correct kernel: {path_correct}")
            return path_correct
        elif include_rejected and path_rejected.exists():
            logger.debug(f"Found rejected kernel: {path_rejected}")
            return path_rejected
        elif path_correct.exists():
            logger.debug(f"Found correct kernel (fallback): {path_correct}")
            return path_correct

        logger.debug(f"No kernel found for operation {operation}")
        return None

    except Exception as e:
        logger.error(f"Failed to find kernel: {e}")
        raise StorageError(f"Failed to find kernel: {e}")


def list_kernels(
    cache_dir: str = ".cache/yolokernelgen",
    status_filter: Optional[str] = None,
    only_correct: bool = False
) -> List[Dict[str, Any]]:
    """List all kernels with optional filtering.

    Args:
        only_correct: If True, only return kernels with status='correct'
    """
    try:
        cache_path = Path(cache_dir) / "generated"

        if not cache_path.exists():
            return []

        kernels = []
        for filepath in cache_path.glob("*.json"):
            try:
                info = parse_filename(filepath.name)
                info["filepath"] = str(filepath)

                # Apply filters
                if only_correct and info["status"] != "c":
                    continue

                if status_filter is not None and info["status"] != status_filter:
                    continue

                kernels.append(info)
            except ValueError:
                # Skip files that don't match naming convention
                logger.debug(f"Skipping file with invalid name: {filepath.name}")
                continue

        logger.debug(f"Listed {len(kernels)} kernels")
        return kernels

    except Exception as e:
        logger.error(f"Failed to list kernels: {e}")
        raise StorageError(f"Failed to list kernels: {e}")


def delete_kernel(filepath: Path) -> bool:
    """Delete a kernel file and remove from cache."""
    try:
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted kernel: {filepath}")

            # Remove from cache if present
            cache_key = str(filepath)
            if cache_key in _kernel_cache._cache:
                _kernel_cache._cache.pop(cache_key)
                _kernel_cache._access_order.remove(cache_key)

            return True
        return False

    except Exception as e:
        logger.error(f"Failed to delete kernel {filepath}: {e}")
        raise StorageError(f"Failed to delete kernel: {e}")


def save_test_suite(
    test_suite: List[Dict[str, Any]],
    operation: str,
    cache_dir: str = ".cache/yolokernelgen"
) -> Path:
    """Save test suite for later reuse."""
    try:
        suite_path = Path(cache_dir) / "test_suites" / operation
        suite_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"suite_{timestamp}.json"
        filepath = suite_path / filename

        # Atomic write
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=suite_path,
            prefix=f'tmp_{filename}_',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as tmp_file:
            json.dump(test_suite, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)

        tmp_path.rename(filepath)
        logger.info(f"Saved test suite to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to save test suite: {e}")
        raise StorageError(f"Failed to save test suite: {e}")


def clear_cache() -> None:
    """Clear the in-memory kernel cache."""
    _kernel_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "size": len(_kernel_cache._cache),
        "max_size": _kernel_cache._maxsize,
        "hit_rate": "N/A",  # Would need to track hits/misses
        "cached_files": list(_kernel_cache._cache.keys())
    }