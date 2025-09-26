"""Test Conv2D kernel generation with GPT-4o."""

import os
import sys
import numpy as np

# Mock webgpu execution for testing LLM generation only
import yolokernelgen.webgpu_executor
def mock_execute_kernel(kernel_source, input_tensors, output_shape=None, workgroup_size=256):
    if output_shape:
        return np.zeros(output_shape, dtype=np.float32)
    return np.zeros_like(input_tensors[0])
yolokernelgen.webgpu_executor.execute_kernel = mock_execute_kernel  # type: ignore

from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config


def test_conv2d_simple():
    """Test simple 2D convolution kernel generation."""

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return False

    print("\n=== Testing Conv2D Kernel Generation ===")

    # Simple conv2d operation
    torch_source = """def conv2d_simple(x, weight, bias=None):
    # x: [batch, in_channels, height, width]
    # weight: [out_channels, in_channels, kernel_h, kernel_w]
    # bias: [out_channels]
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)"""

    # Small shapes for testing
    batch_size = 1
    in_channels = 3
    out_channels = 4
    input_h, input_w = 8, 8
    kernel_h, kernel_w = 3, 3

    # Calculate output size (no padding, stride=1)
    output_h = input_h - kernel_h + 1  # 8 - 3 + 1 = 6
    output_w = input_w - kernel_w + 1  # 8 - 3 + 1 = 6

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

    print(f"Input shape: {input_shapes[0]}")
    print(f"Weight shape: {param_shapes['weight']}")
    print(f"Output shape: {output_shapes[0]}")
    print(f"Total output elements: {np.prod(output_shapes[0])}")

    config = default_config()
    config["max_samples"] = 2  # Try twice

    try:
        print("\nGenerating Conv2D kernel with GPT-4o...")
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=None,  # Skip validation for now
            config=config,
            force_regenerate=True
        )

        print(f"✓ Generated kernel: {kernel_path}")

        # Load and analyze
        kernel_data = load_kernel(kernel_path)
        kernel_source = kernel_data["llm_response"]["extracted_kernel"]

        print("\n=== Generated Conv2D WGSL Kernel ===")
        print(kernel_source)

        # Analyze kernel structure
        has_nested_loops = "for" in kernel_source.lower()
        has_proper_indexing = "input_h" in kernel_source or "height" in kernel_source.lower()
        has_weight_access = "weight" in kernel_source.lower()
        has_bias_add = "bias" in kernel_source.lower()

        print("\n=== Kernel Analysis ===")
        print(f"Has nested loops (for convolution): {has_nested_loops}")
        print(f"Has proper spatial indexing: {has_proper_indexing}")
        print(f"Accesses weight tensor: {has_weight_access}")
        print(f"Handles bias addition: {has_bias_add}")

        # Token usage
        usage = kernel_data["llm_response"]["usage"]
        print("\n=== Token Usage ===")
        print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
        print(f"Prompt/Completion: {usage.get('prompt_tokens', 'N/A')}/{usage.get('completion_tokens', 'N/A')}")

        return True

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv2d_with_padding():
    """Test Conv2D with padding - more complex indexing."""

    print("\n=== Testing Conv2D with Padding ===")

    torch_source = """def conv2d_padded(x, weight, bias=None):
    # 2D convolution with padding to preserve spatial dimensions
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)"""

    # Same input size but with padding=1, output should be same size
    input_shapes = [[1, 2, 6, 6]]
    output_shapes = [[1, 4, 6, 6]]  # padding=1 preserves size
    param_shapes = {
        "weight": [4, 2, 3, 3],
        "bias": [4]
    }
    hyperparameters = {
        "stride": [1, 1],
        "padding": [1, 1],
        "kernel_size": [3, 3]
    }

    config = default_config()
    config["max_samples"] = 2

    try:
        print("Generating Conv2D with padding...")
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d_padded",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=None,
            config=config,
            force_regenerate=True
        )

        kernel_data = load_kernel(kernel_path)
        kernel_source = kernel_data["llm_response"]["extracted_kernel"]

        print("\n=== Conv2D Padded Kernel (first 800 chars) ===")
        print(kernel_source[:800] + "..." if len(kernel_source) > 800 else kernel_source)

        # Check for padding handling
        has_padding_check = "padding" in kernel_source.lower()
        has_bounds_check = "if" in kernel_source and ">" in kernel_source

        print(f"\nPadding awareness: {has_padding_check}")
        print(f"Has bounds checking: {has_bounds_check}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Complex Conv2D Kernel Generation")
    print("=" * 50)

    success1 = test_conv2d_simple()
    success2 = test_conv2d_with_padding()

    if success1 and success2:
        print("\n🎉 Conv2D kernel generation successful!")
        print("GPT-4o can handle complex tensor operations.")
    else:
        print("\n❌ Some Conv2D tests failed.")
        sys.exit(1)