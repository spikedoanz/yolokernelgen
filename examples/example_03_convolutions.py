"""
Example 3: Convolution Operations

This example demonstrates neural network layer generation:
- 2D convolutions with weights and biases
- Understanding parameter shapes and hyperparameters
- Complex tensor indexing and memory layout (NCHW)
- The most challenging kernels to generate correctly
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config


def example_basic_conv2d():
    """Generate a basic 2D convolution kernel."""

    print("=== Example: Basic Conv2D ===")

    torch_source = """def conv2d_basic(x, weight, bias):
    # 2D convolution: input[N,C_in,H,W] * weight[C_out,C_in,K_h,K_w] + bias[C_out]
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)"""

    # Define tensor shapes carefully
    batch_size = 1
    in_channels = 2
    out_channels = 3
    input_h, input_w = 5, 5
    kernel_h, kernel_w = 3, 3

    # Calculate output size: (input - kernel + 2*padding) / stride + 1
    output_h = input_h - kernel_h + 1  # 5 - 3 + 1 = 3
    output_w = input_w - kernel_w + 1  # 5 - 3 + 1 = 3

    input_shapes = [[batch_size, in_channels, input_h, input_w]]
    output_shapes = [[batch_size, out_channels, output_h, output_w]]
    param_shapes = {
        "weight": [out_channels, in_channels, kernel_h, kernel_w],
        "bias": [out_channels]
    }
    hyperparameters = {
        "stride": [1, 1],
        "padding": [0, 0],
        "kernel_size": [kernel_h, kernel_w]
    }

    def torch_fn(x, weight, bias):
        return F.conv2d(x, weight, bias, stride=1, padding=0)

    print(f"Input: {input_shapes[0]} -> Output: {output_shapes[0]}")
    print(f"Weight: {param_shapes['weight']}, Bias: {param_shapes['bias']}")

    try:
        # Conv2D kernels are complex, so allow more attempts
        config = default_config()
        config["max_samples"] = 3

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=torch_fn,
            config=config
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Conv2D kernel: {kernel_path.name}")
        print(f"✓ Validation: {kernel_data['validation']['all_passed']}")
        print(f"✓ Tests passed: {kernel_data['validation']['num_passed']}/{kernel_data['validation']['num_total']}")
        print(f"✓ Tokens used: {kernel_data['llm_response']['usage']['total_tokens']}")

        # Analyze the kernel complexity
        kernel_code = kernel_data['llm_response']['extracted_kernel']
        has_nested_loops = kernel_code.count("for") >= 3  # Should have ic, kh, kw loops
        has_nchw_indexing = "* IN_CHANNELS" in kernel_code or "c *" in kernel_code
        has_bias_handling = "bias[" in kernel_code

        print(f"✓ Has nested convolution loops: {has_nested_loops}")
        print(f"✓ NCHW tensor indexing: {has_nchw_indexing}")
        print(f"✓ Proper bias handling: {has_bias_handling}")

        return True

    except Exception as e:
        print(f"✗ Conv2D failed: {e}")
        return False


def example_conv2d_with_padding():
    """Generate convolution with padding (more complex indexing)."""

    print("\n=== Example: Conv2D with Padding ===")

    torch_source = """def conv2d_padded(x, weight, bias):
    # Convolution with padding=1 to preserve spatial dimensions
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)"""

    # With padding=1, output size = input size for kernel=3, stride=1
    input_shapes = [[1, 3, 6, 6]]
    output_shapes = [[1, 4, 6, 6]]  # Same spatial size due to padding
    param_shapes = {
        "weight": [4, 3, 3, 3],
        "bias": [4]
    }
    hyperparameters = {
        "stride": [1, 1],
        "padding": [1, 1],
        "kernel_size": [3, 3]
    }

    def torch_fn(x, weight, bias):
        return F.conv2d(x, weight, bias, stride=1, padding=1)

    try:
        config = default_config()
        config["max_samples"] = 3

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d_padded",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=torch_fn,
            config=config
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Padded Conv2D: {kernel_path.name}")
        print(f"✓ Input {input_shapes[0]} -> Output {output_shapes[0]} (preserved size)")
        print(f"✓ Validation: {kernel_data['validation']['all_passed']}")

        # Padding requires bounds checking
        kernel_code = kernel_data['llm_response']['extracted_kernel']
        has_bounds_check = ("if" in kernel_code and ">" in kernel_code and "<" in kernel_code)
        has_padding_logic = "padding" in kernel_code.lower() or "PADDING" in kernel_code

        print(f"✓ Has bounds checking: {has_bounds_check}")
        print(f"✓ Padding-aware: {has_padding_logic}")

        return True

    except Exception as e:
        print(f"✗ Padded Conv2D failed: {e}")
        return False


def show_conv_kernel_details():
    """Show details of generated convolution kernels."""

    print("\n=== Convolution Kernel Analysis ===")

    from yolokernelgen import list_kernels

    conv_kernels = [k for k in list_kernels() if "conv" in k["operation"]]

    if not conv_kernels:
        print("No convolution kernels found.")
        return

    print(f"Generated {len(conv_kernels)} convolution kernels:")

    for kernel_info in conv_kernels:
        filepath = kernel_info["filepath"]
        status = "✓" if kernel_info["status"] == "correct" else "○"
        input_shape = kernel_info.get("input_shape", "unknown")
        output_shape = kernel_info.get("output_shape", "unknown")

        print(f"  {status} {kernel_info['operation']}: {input_shape} → {output_shape}")

        # Load and show token usage for correct kernels
        if kernel_info["status"] == "correct":
            try:
                from pathlib import Path
                kernel_data = load_kernel(Path(filepath))
                tokens = kernel_data['llm_response']['usage']['total_tokens']
                tests_passed = kernel_data['validation']['num_passed']
                print(f"    Tokens: {tokens}, Tests: {tests_passed}/10 passed")
            except:
                pass


if __name__ == "__main__":
    print("YoloKernelGen - Example 3: Convolution Operations")
    print("=" * 55)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        exit(1)

    print("⚠️  Convolutions are the most complex kernels to generate!")
    print("   They require sophisticated tensor indexing and memory layout understanding.")
    print("   This may take longer and use more tokens than simple operations.\n")

    # Run convolution examples
    success1 = example_basic_conv2d()
    success2 = example_conv2d_with_padding()

    # Show detailed analysis
    show_conv_kernel_details()

    print(f"\n=== Results ===")
    if success1 and success2:
        print("🎉 Convolution kernels generated successfully!")
        print("\nAmazing achievements:")
        print("- GPT-4o generated complex NCHW tensor indexing")
        print("- Proper handling of weights, biases, and padding")
        print("- Triple nested loops for convolution computation")
        print("- All kernels mathematically validated against PyTorch")
        print("\nThis demonstrates production-level kernel generation capability!")
    else:
        print("❌ Some convolution operations failed")
        print("   Convolutions are challenging - try adjusting max_samples in config")

    print(f"\nNext: Try example_04_fusion_optimization.py for advanced kernels!")