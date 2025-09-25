"""Test kernel generation WITH full validation against PyTorch ground truth."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Use the real webgpu executor (with dawn-python)
from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config


def test_validated_relu():
    """Test ReLU with actual PyTorch validation."""

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return False

    print("\n=== Validated ReLU Generation ===")

    torch_source = """def relu_validated(x):
    return torch.nn.functional.relu(x)"""

    input_shapes = [[2, 3, 4, 4]]  # Small for testing
    output_shapes = [[2, 3, 4, 4]]

    # THIS IS THE KEY: Provide torch_fn for validation
    def torch_fn(x):
        return F.relu(x)

    config = default_config()
    config["max_samples"] = 3  # Try a few times if needed

    try:
        print("Generating ReLU with full validation...")

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="relu",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn,  # ← This enables validation!
            config=config,
            force_regenerate=True
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Kernel status: {kernel_data['status']}")
        print(f"✓ Validation passed: {kernel_data['validation']['all_passed']}")
        print(f"✓ Tests passed: {kernel_data['validation']['num_passed']}/{kernel_data['validation']['num_total']}")

        # Show validation details
        for i, test_case in enumerate(kernel_data['validation']['test_cases'][:3]):
            status = "✓" if test_case['passed'] else "✗"
            print(f"  Test {i+1} ({test_case['type']}): {status} max_diff={test_case.get('max_diff', 'N/A'):.2e}")

        return kernel_data['status'] == 'correct'

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validated_add():
    """Test element-wise add with validation."""

    print("\n=== Validated Add Generation ===")

    torch_source = """def add_validated(a, b):
    return a + b"""

    input_shapes = [[2, 4, 8, 8], [2, 4, 8, 8]]  # Two inputs
    output_shapes = [[2, 4, 8, 8]]

    def torch_fn(a, b):
        return a + b

    config = default_config()
    config["max_samples"] = 2

    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="add",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn,  # Enable validation
            config=config,
            force_regenerate=True
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Status: {kernel_data['status']}")
        print(f"✓ Validation: {kernel_data['validation']['all_passed']}")

        return kernel_data['status'] == 'correct'

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_validated_conv2d():
    """Test Conv2D with validation - this is the real challenge."""

    print("\n=== Validated Conv2D Generation ===")

    torch_source = """def conv2d_validated(x, weight, bias):
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)"""

    # Very small shapes for easier debugging
    input_shapes = [[1, 2, 4, 4]]  # Tiny input
    output_shapes = [[1, 3, 2, 2]]  # 4-3+1=2, so 2x2 output
    param_shapes = {
        "weight": [3, 2, 3, 3],  # [out_ch, in_ch, k_h, k_w]
        "bias": [3]
    }

    def torch_fn(x, weight, bias):
        return F.conv2d(x, weight, bias, stride=1, padding=0)

    config = default_config()
    config["max_samples"] = 3

    try:
        print("This will test if our Conv2D kernels actually work...")

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            torch_fn=torch_fn,  # The real test!
            config=config,
            force_regenerate=True
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Status: {kernel_data['status']}")
        print(f"✓ Validation: {kernel_data['validation']['all_passed']}")

        if not kernel_data['validation']['all_passed']:
            print("Failed test cases:")
            for test_case in kernel_data['validation']['test_cases']:
                if not test_case['passed']:
                    print(f"  {test_case['type']}: max_diff={test_case.get('max_diff', 'N/A'):.2e}")

        return kernel_data['status'] == 'correct'

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Kernel Generation WITH Validation")
    print("=" * 50)

    # Test simple operations first
    relu_success = test_validated_relu()
    add_success = test_validated_add()

    # Then the complex one
    conv_success = test_validated_conv2d()

    successful = [relu_success, add_success, conv_success].count(True)

    print(f"\n=== Final Results ===")
    print(f"ReLU validation: {'✓' if relu_success else '✗'}")
    print(f"Add validation: {'✓' if add_success else '✗'}")
    print(f"Conv2D validation: {'✓' if conv_success else '✗'}")

    if successful == 3:
        print("\n🎉 All validations passed! Kernels are mathematically correct.")
    else:
        print(f"\n⚠️ {3-successful} validation(s) failed. This reveals kernel correctness issues.")

    print(f"\nThis is why torch_fn validation is crucial - it catches bugs!")