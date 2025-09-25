"""Kernel validation and test suite generation."""

import numpy as np
import torch
from typing import List, Dict, Any, Callable, Optional


def generate_test_inputs(
    shape: List[int],
    test_type: str,
    seed: int,
    dtype: str = "float32"
) -> np.ndarray:
    """Generate test input tensor based on test type."""
    np.random.seed(seed)

    if test_type == "random_uniform":
        data = np.random.uniform(-1, 1, shape).astype(dtype)
    elif test_type == "random_normal":
        data = np.random.normal(0, 1, shape).astype(dtype)
    elif test_type == "sparse":
        data = np.random.uniform(-1, 1, shape).astype(dtype)
        mask = np.random.random(shape) > 0.9  # 90% zeros
        data = data * mask
    elif test_type == "zeros":
        data = np.zeros(shape, dtype=dtype)
    elif test_type == "ones":
        data = np.ones(shape, dtype=dtype)
    elif test_type == "single_hot":
        data = np.zeros(shape, dtype=dtype)
        # Set one random element to 1
        indices = tuple(np.random.randint(0, s) for s in shape)
        data[indices] = 1.0
    elif test_type == "alternating":
        data = np.ones(shape, dtype=dtype)
        # Create alternating pattern
        data.flat[::2] = -1.0
    elif test_type == "extreme":
        # Values near dtype limits
        if dtype == "float32":
            data = np.random.choice([1e-38, 1e38, -1e38, -1e-38], shape).astype(dtype)
        else:
            data = np.random.uniform(-1, 1, shape).astype(dtype)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return data


def create_test_suite(
    input_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    num_random: int = 5,
    num_edge: int = 5,
    seeds: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Create comprehensive test suite for validation."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    test_cases = []

    # Random test cases
    random_types = ["random_uniform", "random_uniform", "random_normal", "random_normal", "sparse"]
    for i in range(min(num_random, len(random_types))):
        test_case = {
            "type": random_types[i],
            "seed": seeds[i % len(seeds)],
            "inputs": []
        }

        # Generate inputs for each input shape
        for shape in input_shapes:
            test_case["inputs"].append(
                generate_test_inputs(shape, random_types[i], seeds[i % len(seeds)])
            )

        # Generate parameters if provided
        if param_shapes:
            test_case["params"] = {}
            for name, shape in param_shapes.items():
                # Parameters typically use uniform distribution
                np.random.seed(seeds[i % len(seeds)] + hash(name) % 1000)
                test_case["params"][name] = np.random.uniform(-0.5, 0.5, shape).astype("float32")

        test_cases.append(test_case)

    # Edge test cases
    edge_types = ["zeros", "ones", "single_hot", "alternating", "extreme"]
    for i in range(min(num_edge, len(edge_types))):
        test_case = {
            "type": edge_types[i],
            "seed": seeds[i % len(seeds)] + 1000,  # Different seeds for edge cases
            "inputs": []
        }

        # Generate inputs for each input shape
        for shape in input_shapes:
            test_case["inputs"].append(
                generate_test_inputs(shape, edge_types[i], test_case["seed"])
            )

        # Generate parameters if provided
        if param_shapes:
            test_case["params"] = {}
            for name, shape in param_shapes.items():
                np.random.seed(test_case["seed"] + hash(name) % 1000)
                test_case["params"][name] = np.random.uniform(-0.5, 0.5, shape).astype("float32")

        test_cases.append(test_case)

    return test_cases


def run_torch_reference(
    torch_fn: Callable,
    inputs: List[np.ndarray],
    params: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """Run PyTorch reference implementation."""
    # Convert numpy arrays to torch tensors
    torch_inputs = [torch.from_numpy(inp.copy()) for inp in inputs]

    if params:
        torch_params = {k: torch.from_numpy(v.copy()) for k, v in params.items()}
        result = torch_fn(*torch_inputs, **torch_params)
    else:
        result = torch_fn(*torch_inputs)

    # Convert back to numpy
    if isinstance(result, torch.Tensor):
        return result.detach().cpu().numpy()
    else:
        # Handle tuple outputs
        return tuple(t.detach().cpu().numpy() for t in result)


def compare_outputs(
    reference: np.ndarray,
    generated: np.ndarray,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """Compare reference and generated outputs."""
    if reference.shape != generated.shape:
        return {
            "passed": False,
            "error": f"Shape mismatch: {reference.shape} vs {generated.shape}",
            "max_diff": float('inf'),
            "mean_diff": float('inf')
        }

    diff = np.abs(reference - generated)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    return {
        "passed": max_diff <= tolerance,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance
    }


def validate_kernel(
    kernel_source: str,
    torch_fn: Callable,
    test_suite: List[Dict[str, Any]],
    webgpu_executor: Callable,
    tolerance: float = 1e-5,
    dtype: str = "float32"
) -> Dict[str, Any]:
    """Validate WebGPU kernel against PyTorch reference."""
    validation_results = []
    all_passed = True

    for test_case in test_suite:
        # Run PyTorch reference
        reference_output = run_torch_reference(
            torch_fn,
            test_case["inputs"],
            test_case.get("params")
        )

        # Run WebGPU kernel
        try:
            # Combine inputs and params for WebGPU execution
            all_inputs = test_case["inputs"].copy()
            if "params" in test_case:
                all_inputs.extend(test_case["params"].values())

            generated_output = webgpu_executor(kernel_source, all_inputs)

            # Compare outputs
            comparison = compare_outputs(reference_output, generated_output, tolerance)
            comparison["type"] = test_case["type"]
            comparison["seed"] = test_case["seed"]

            validation_results.append(comparison)
            if not comparison["passed"]:
                all_passed = False

        except Exception as e:
            validation_results.append({
                "type": test_case["type"],
                "seed": test_case["seed"],
                "passed": False,
                "error": str(e),
                "max_diff": float('inf'),
                "mean_diff": float('inf')
            })
            all_passed = False

    return {
        "tolerance": tolerance,
        "dtype": dtype,
        "test_cases": validation_results,
        "all_passed": all_passed,
        "num_passed": sum(1 for r in validation_results if r["passed"]),
        "num_total": len(validation_results)
    }