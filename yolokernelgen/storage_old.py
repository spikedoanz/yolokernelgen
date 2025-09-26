"""Kernel storage and retrieval functions."""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from .naming import generate_filename, parse_filename


def save_kernel(kernel_data: Dict[str, Any], cache_dir: str = ".cache/yolokernelgen") -> Path:
    """Save kernel data to filesystem."""
    # Ensure cache directory exists
    cache_path = Path(cache_dir) / "generated"
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    status_prefix = "c" if kernel_data["validation"]["all_passed"] else "r"
    filename = generate_filename(
        operation=kernel_data["operation"],
        input_shape=kernel_data["metadata"]["input_shapes"][0],
        output_shape=kernel_data["metadata"]["output_shapes"][0],
        torch_source=kernel_data["torch_source"],
        params=kernel_data["metadata"].get("hyperparameters", {}),
        status=status_prefix
    )

    # Add UUID and timestamp if not present
    if "uuid" not in kernel_data:
        kernel_data["uuid"] = str(uuid.uuid4())
    if "timestamp" not in kernel_data:
        kernel_data["timestamp"] = datetime.now().isoformat()

    # Update status field
    kernel_data["status"] = "correct" if status_prefix == "c" else "rejected"

    # Save to file
    filepath = cache_path / filename
    with open(filepath, 'w') as f:
        json.dump(kernel_data, f, indent=2)

    return filepath


def load_kernel(filepath: Path) -> Dict[str, Any]:
    """Load kernel data from filesystem."""
    if not filepath.exists():
        raise FileNotFoundError(f"Kernel file not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def find_kernel(
    operation: str,
    input_shape: List[int],
    output_shape: List[int],
    torch_source: str,
    params: Optional[Dict[str, Any]] = None,
    cache_dir: str = ".cache/yolokernelgen",
    prefer_correct: bool = True
) -> Optional[Path]:
    """Find existing kernel matching specifications."""
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
        return path_correct
    elif path_rejected.exists():
        return path_rejected
    elif path_correct.exists():
        return path_correct

    return None


def list_kernels(
    cache_dir: str = ".cache/yolokernelgen",
    status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List all kernels with optional status filtering."""
    cache_path = Path(cache_dir) / "generated"

    if not cache_path.exists():
        return []

    kernels = []
    for filepath in cache_path.glob("*.json"):
        try:
            info = parse_filename(filepath.name)
            info["filepath"] = str(filepath)

            if status_filter is None or info["status"] == status_filter:
                kernels.append(info)
        except ValueError:
            # Skip files that don't match naming convention
            continue

    return kernels


def delete_kernel(filepath: Path) -> bool:
    """Delete a kernel file."""
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def save_test_suite(
    test_suite: List[Dict[str, Any]],
    operation: str,
    cache_dir: str = ".cache/yolokernelgen"
) -> Path:
    """Save test suite for later reuse."""
    suite_path = Path(cache_dir) / "test_suites" / operation
    suite_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = suite_path / f"suite_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump(test_suite, f, indent=2)

    return filepath