"""Kernel validation using type-safe data structures."""

import numpy as np
import torch
from typing import List, Dict, Any, Callable, Optional
from .types import ValidationResult, TestCase
from .exceptions import ValidationError


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
    elif test_type == "large_values":
        data = np.random.uniform(-1e6, 1e6, shape).astype(dtype)
    elif test_type == "small_values":
        data = np.random.uniform(-1e-6, 1e-6, shape).astype(dtype)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return data


def create_test_suite(
    input_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    num_random: int = 5,
    num_edge: int = 5,
    seeds: Optional[List[int]] = None
) -> List[TestCase]:
    """Create comprehensive test suite for validation."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    test_cases = []

    # Random test cases
    random_types = ["random_uniform", "random_uniform", "random_normal", "random_normal", "sparse"]
    for i in range(min(num_random, len(random_types))):
        test_inputs = []

        # Generate inputs for each input shape
        for shape in input_shapes:
            test_inputs.append(
                generate_test_inputs(shape, random_types[i], seeds[i % len(seeds)]).tolist()
            )

        # Generate parameters if provided
        if param_shapes:
            for name, shape in param_shapes.items():
                # Parameters typically use uniform distribution
                np.random.seed(seeds[i % len(seeds)] + hash(name) % 1000)
                param_data = np.random.uniform(-0.5, 0.5, shape).astype("float32")
                test_inputs.append(param_data.tolist())

        test_case = TestCase(
            inputs=test_inputs,
            expected_output=[],  # Will be filled by validation
            test_type=random_types[i],
            seed=seeds[i % len(seeds)],
            tolerance=1e-5,
            passed=False  # Will be set during validation
        )
        test_cases.append(test_case)

    # Edge test cases
    edge_types = ["zeros", "ones", "single_hot", "alternating", "large_values", "small_values"]
    for i in range(min(num_edge, len(edge_types))):
        seed = seeds[i % len(seeds)] + 1000  # Different seeds for edge cases
        test_inputs = []

        # Generate inputs for each input shape
        for shape in input_shapes:
            test_inputs.append(
                generate_test_inputs(shape, edge_types[i], seed).tolist()
            )

        # Generate parameters if provided
        if param_shapes:
            for name, shape in param_shapes.items():
                np.random.seed(seed + hash(name) % 1000)
                param_data = np.random.uniform(-0.5, 0.5, shape).astype("float32")
                test_inputs.append(param_data.tolist())

        test_case = TestCase(
            inputs=test_inputs,
            expected_output=[],  # Will be filled by validation
            test_type=edge_types[i],
            seed=seed,
            tolerance=1e-5,
            passed=False  # Will be set during validation
        )
        test_cases.append(test_case)

    return test_cases


def run_torch_reference(
    torch_fn: Callable,
    inputs: List[List[float]],
    params: Optional[Dict[str, List[float]]] = None
) -> np.ndarray:
    """Run PyTorch reference implementation."""
    # Convert nested lists back to numpy arrays then to torch tensors
    numpy_inputs = [np.array(inp, dtype=np.float32) for inp in inputs]
    torch_inputs = [torch.from_numpy(inp) for inp in numpy_inputs]

    if params:
        numpy_params = {k: np.array(v, dtype=np.float32) for k, v in params.items()}
        torch_params = {k: torch.from_numpy(v) for k, v in numpy_params.items()}
        result = torch_fn(*torch_inputs, **torch_params)
    else:
        result = torch_fn(*torch_inputs)

    # Convert back to numpy
    if isinstance(result, torch.Tensor):
        return result.detach().cpu().numpy()
    else:
        # Handle tuple outputs
        return tuple(t.detach().cpu().numpy() for t in result)


def analyze_failure_pattern(
    reference: np.ndarray,
    generated: np.ndarray,
    test_type: str
) -> Dict[str, Any]:
    """Analyze failure patterns to provide specific feedback."""
    analysis = {"patterns": []}

    # Check for overflow/underflow
    if np.any(np.isinf(generated)) and not np.any(np.isinf(reference)):
        analysis["patterns"].append("overflow_detected")
        analysis["overflow_locations"] = int(np.sum(np.isinf(generated)))

    if np.any(np.isnan(generated)) and not np.any(np.isnan(reference)):
        analysis["patterns"].append("nan_detected")
        analysis["nan_locations"] = int(np.sum(np.isnan(generated)))

    # Check for boundary/edge issues
    if len(reference.shape) >= 2:  # Only for 2D+ tensors
        # Check if errors are concentrated at boundaries
        center_slice = tuple(slice(1, -1) for _ in range(len(reference.shape)))
        try:
            center_diff = np.abs(reference[center_slice] - generated[center_slice])
            edge_diff = np.abs(reference - generated)

            center_max = float(np.max(center_diff)) if center_diff.size > 0 else 0.0
            edge_max = float(np.max(edge_diff))

            if edge_max > center_max * 2 and center_max < 1e-5:
                analysis["patterns"].append("boundary_error")
                analysis["center_max_diff"] = center_max
                analysis["edge_max_diff"] = edge_max
        except (IndexError, ValueError):
            pass  # Skip boundary analysis for very small tensors

    # Check for systematic scaling issues
    if not np.any(np.isinf(generated)) and not np.any(np.isnan(generated)):
        ref_magnitude = float(np.mean(np.abs(reference)))
        gen_magnitude = float(np.mean(np.abs(generated)))

        if ref_magnitude > 0 and gen_magnitude > 0:
            scale_ratio = gen_magnitude / ref_magnitude
            if scale_ratio > 2.0 or scale_ratio < 0.5:
                analysis["patterns"].append("scaling_error")
                analysis["scale_ratio"] = scale_ratio

    # Check for indexing errors (completely wrong values)
    if not np.any(np.isinf(generated)) and not np.any(np.isnan(generated)):
        correlation = float(np.corrcoef(reference.flat, generated.flat)[0, 1])
        if np.isnan(correlation) or correlation < 0.1:
            analysis["patterns"].append("indexing_error")
            analysis["correlation"] = correlation

    # Test-specific analysis
    if test_type == "zeros" and not np.allclose(generated, 0, atol=1e-6):
        analysis["patterns"].append("zero_handling_error")
        analysis["non_zero_count"] = int(np.sum(np.abs(generated) > 1e-6))

    return analysis


def compare_outputs(
    reference: np.ndarray,
    generated: np.ndarray,
    tolerance: float = 1e-5,
    test_type: str = "unknown"
) -> Dict[str, Any]:
    """Compare reference and generated outputs with detailed failure analysis."""
    if reference.shape != generated.shape:
        return {
            "passed": False,
            "error": f"Shape mismatch: {reference.shape} vs {generated.shape}",
            "error_type": "shape_mismatch",
            "max_diff": float('inf'),
            "mean_diff": float('inf'),
            "failure_analysis": {"patterns": ["shape_mismatch"]}
        }

    diff = np.abs(reference - generated)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    passed = max_diff <= tolerance
    result = {
        "passed": passed,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance,
        "test_type": test_type
    }

    # Add detailed failure analysis if test failed
    if not passed:
        result["failure_analysis"] = analyze_failure_pattern(reference, generated, test_type)

        # Determine primary error type
        patterns = result["failure_analysis"]["patterns"]
        if "overflow_detected" in patterns:
            result["error_type"] = "overflow"
        elif "nan_detected" in patterns:
            result["error_type"] = "nan"
        elif "boundary_error" in patterns:
            result["error_type"] = "boundary"
        elif "scaling_error" in patterns:
            result["error_type"] = "scaling"
        elif "indexing_error" in patterns:
            result["error_type"] = "indexing"
        elif "zero_handling_error" in patterns:
            result["error_type"] = "zero_handling"
        else:
            result["error_type"] = "numerical"

    return result


async def validate_kernel(
    kernel_source: str,
    torch_fn: Callable,
    test_suite: List[TestCase],
    webgpu_executor: Callable,
    tolerance: float = 1e-5,
    dtype: str = "float32"
) -> ValidationResult:
    """Validate WebGPU kernel against PyTorch reference."""
    try:
        validated_test_cases = []
        all_passed = True

        for test_case in test_suite:
            # Run PyTorch reference
            reference_output = run_torch_reference(torch_fn, test_case.inputs)

            # Convert reference to list for storage
            reference_list = reference_output.tolist()

            # Run WebGPU kernel
            try:
                # Convert test case inputs back to numpy for WebGPU execution
                numpy_inputs = [np.array(inp, dtype=np.float32) for inp in test_case.inputs]
                generated_output = webgpu_executor(kernel_source, numpy_inputs)

                # Compare outputs
                comparison = compare_outputs(reference_output, generated_output, tolerance, test_case.test_type)

                # Update test case with results
                validated_case = TestCase(
                    inputs=test_case.inputs,
                    expected_output=reference_list,
                    test_type=test_case.test_type,
                    seed=test_case.seed,
                    tolerance=tolerance,
                    passed=comparison["passed"],
                    error_message=comparison.get("error"),
                    actual_output=generated_output.tolist() if comparison["passed"] else None
                )

                validated_test_cases.append(validated_case)
                if not comparison["passed"]:
                    all_passed = False

            except Exception as e:
                # Handle execution errors
                validated_case = TestCase(
                    inputs=test_case.inputs,
                    expected_output=reference_list,
                    test_type=test_case.test_type,
                    seed=test_case.seed,
                    tolerance=tolerance,
                    passed=False,
                    error_message=str(e),
                    actual_output=None
                )
                validated_test_cases.append(validated_case)
                all_passed = False

        # Generate failure summary for feedback
        failure_summary = generate_failure_summary(validated_test_cases) if not all_passed else None

        return ValidationResult(
            tolerance=tolerance,
            dtype=dtype,
            test_cases=validated_test_cases,
            all_passed=all_passed,
            num_passed=sum(1 for tc in validated_test_cases if tc.passed),
            num_total=len(validated_test_cases),
            failure_summary=failure_summary
        )

    except Exception as e:
        raise ValidationError(f"Validation failed: {e}")


def generate_failure_summary(test_cases: List[TestCase]) -> Dict[str, Any]:
    """Generate a summary of validation failures for feedback to LLM."""
    failed_tests = [tc for tc in test_cases if not tc.passed]

    if not failed_tests:
        return None

    summary = {
        "total_failures": len(failed_tests),
        "failed_test_types": [tc.test_type for tc in failed_tests],
        "error_patterns": {},
        "common_issues": [],
        "specific_feedback": []
    }

    # Analyze error patterns from error messages
    error_types = {}
    for test_case in failed_tests:
        if test_case.error_message:
            # Simple error classification
            error_msg = test_case.error_message.lower()
            if "overflow" in error_msg or "inf" in error_msg:
                error_type = "overflow"
            elif "nan" in error_msg:
                error_type = "nan"
            elif "boundary" in error_msg or "index" in error_msg:
                error_type = "indexing"
            elif "shape" in error_msg:
                error_type = "shape_mismatch"
            else:
                error_type = "numerical"
        else:
            error_type = "unknown"

        error_types[error_type] = error_types.get(error_type, 0) + 1

        # Add specific feedback based on error type
        if error_type == "overflow":
            summary["specific_feedback"].append(
                f"Test '{test_case.test_type}' failed due to overflow - check array indexing bounds"
            )
        elif error_type == "indexing":
            summary["specific_feedback"].append(
                f"Test '{test_case.test_type}' failed - likely tensor indexing error"
            )

    summary["error_patterns"] = error_types

    # Generate common issues
    if "indexing" in error_types:
        summary["common_issues"].append("Array indexing overflow")
    if "overflow" in error_types:
        summary["common_issues"].append("Numerical overflow")
    if "shape_mismatch" in error_types:
        summary["common_issues"].append("Shape mismatch errors")

    # Add performance note if mostly passed
    if len(failed_tests) <= 2:
        summary["performance_note"] = "Nearly passing - focus on edge cases"

    return summary