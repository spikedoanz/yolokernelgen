"""Example usage of the kernel generation framework."""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from yolokernelgen import (
    default_config,
    generate_kernel,
    list_kernels,
    load_kernel,
    execute_kernel
)
from yolokernelgen.config import save_config
from yolokernelgen.webgpu_executor import execute_simple_kernel


def example_relu():
    """Example: Generate and use a ReLU kernel."""
    print("\n=== ReLU Kernel Generation ===")

    # Define the PyTorch operation
    torch_source = """def relu(x):
    return torch.nn.functional.relu(x)"""

    # Define shapes
    input_shapes = [[8, 16, 32, 32]]  # NCHW format
    output_shapes = [[8, 16, 32, 32]]

    # Create torch function for validation
    def torch_fn(x):
        return F.relu(x)

    # Generate kernel
    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="relu",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn
        )
        print(f"Generated kernel: {kernel_path}")

        # Load and use the kernel
        kernel_data = load_kernel(kernel_path)
        kernel_source = kernel_data.llm_response.extracted_kernel

        # Test with real data
        test_input = np.random.randn(8, 16, 32, 32).astype(np.float32)
        result = execute_kernel(kernel_source, [test_input], output_shapes[0])

        # Verify against PyTorch
        torch_result = torch_fn(torch.from_numpy(test_input)).numpy()
        matches = np.allclose(result, torch_result, rtol=1e-5)
        print(f"Results match PyTorch: {matches}")

    except Exception as e:
        print(f"Failed to generate ReLU kernel: {e}")


def example_conv2d():
    """Example: Generate and use a Conv2d kernel."""
    print("\n=== Conv2d Kernel Generation ===")

    torch_source = """def conv2d(x, weight, bias=None):
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)"""

    # Define shapes
    input_shapes = [[1, 3, 64, 64]]  # NCHW
    output_shapes = [[1, 16, 64, 64]]  # With padding=1, same spatial size
    param_shapes = {
        "weight": [16, 3, 3, 3],  # out_channels, in_channels, kernel_h, kernel_w
        "bias": [16]
    }
    hyperparameters = {
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": [1, 1],
        "groups": 1
    }

    # Create torch function for validation
    def torch_fn(x, weight, bias):
        return F.conv2d(x, weight, bias, stride=1, padding=1)

    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=torch_fn
        )
        print(f"Generated kernel: {kernel_path}")

    except Exception as e:
        print(f"Failed to generate Conv2d kernel: {e}")


def example_simple_identity():
    """Example: Simple identity kernel (known to work)."""
    print("\n=== Simple Identity Kernel ===")

    # Manual kernel that we know works
    shape = [256, 128, 32]  # Smaller test shape
    total_elements = np.prod(shape)

    kernel_source = f"""
        @group(0) @binding(0)
        var<storage,read> input_data: array<f32>;

        @group(0) @binding(1)
        var<storage,read_write> output_data: array<f32>;

        @compute
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x + global_id.y * 65535u;
            let max_elements = {total_elements}u;

            if (index >= max_elements) {{
                return;
            }}

            output_data[index] = input_data[index];
        }}
    """

    # Test the kernel
    test_input = np.random.randn(*shape).astype(np.float32)
    input_bytes = test_input.tobytes()

    result_bytes = execute_simple_kernel(kernel_source, input_bytes, int(total_elements))
    result = np.frombuffer(result_bytes, dtype=np.float32).reshape(shape)

    matches = np.allclose(test_input, result)
    print(f"Identity kernel works: {matches}")
    if matches:
        print(f"Successfully processed {total_elements:,} elements")


def example_list_kernels():
    """Example: List all generated kernels."""
    print("\n=== List Generated Kernels ===")

    # List all kernels
    all_kernels = list_kernels()
    print(f"Total kernels: {len(all_kernels)}")

    # List only correct kernels
    correct_kernels = list_kernels(status_filter="correct")
    print(f"Correct kernels: {len(correct_kernels)}")

    # List only rejected kernels
    rejected_kernels = list_kernels(status_filter="rejected")
    print(f"Rejected kernels: {len(rejected_kernels)}")

    # Show details of first few kernels
    for kernel_info in all_kernels[:3]:
        print(f"  - {kernel_info['operation']}: "
              f"i{kernel_info['input_shape']} → o{kernel_info['output_shape']} "
              f"[{kernel_info['status']}]")


def example_custom_config():
    """Example: Using custom configuration."""
    print("\n=== Custom Configuration ===")

    # Create custom config
    custom_config = default_config()
    custom_config["max_samples"] = 3
    custom_config["llm"]["temperature"] = 0.5
    custom_config["validation"]["num_random_tests"] = 3
    custom_config["validation"]["num_edge_tests"] = 2

    # Save config for reuse
    config_path = Path("custom_config.json")
    save_config(custom_config, config_path)
    print(f"Saved custom config to {config_path}")

    # Use custom config for generation
    torch_source = """def add(a, b):
    return a + b"""

    input_shapes = [[8, 16, 32, 32], [8, 16, 32, 32]]
    output_shapes = [[8, 16, 32, 32]]

    def torch_fn(a, b):
        return a + b

    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="add",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn,
            config=custom_config
        )
        print(f"Generated kernel with custom config: {kernel_path}")

    except Exception as e:
        print(f"Failed with custom config: {e}")


def main():
    """Run all examples."""
    print("YoloKernelGen - Functional Style MVP")
    print("=" * 50)

    # Test basic functionality first
    example_simple_identity()

    # Try to generate kernels (will use mock LLM for now)
    example_relu()
    example_conv2d()

    # Show management functions
    example_list_kernels()
    example_custom_config()


if __name__ == "__main__":
    main()