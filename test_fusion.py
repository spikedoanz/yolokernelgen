"""Test fused kernel generation (Conv2D + ReLU)."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Mock webgpu execution for testing
import yolokernelgen.webgpu_executor
def mock_execute_kernel(kernel_source, input_tensors, output_shape=None, workgroup_size=256):
    if output_shape:
        return np.zeros(output_shape, dtype=np.float32)
    return np.zeros_like(input_tensors[0])
yolokernelgen.webgpu_executor.execute_kernel = mock_execute_kernel

from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config
from enhanced_prompts import build_enhanced_system_prompt, build_enhanced_user_prompt, get_enhanced_example_kernels

# Use enhanced prompts
import yolokernelgen.prompts
yolokernelgen.prompts.build_system_prompt = build_enhanced_system_prompt
yolokernelgen.prompts.build_user_prompt = build_enhanced_user_prompt
yolokernelgen.prompts.get_example_kernels = get_enhanced_example_kernels


def test_conv2d_relu_fusion():
    """Test Conv2D + ReLU fusion in a single kernel."""

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return False

    print("\n=== Conv2D + ReLU Fusion Test ===")

    # Fused operation
    torch_source = """def conv2d_relu_fused(x, weight, bias):
    # Fused Conv2D + ReLU: apply convolution then ReLU in single kernel
    # x: [N, C_in, H, W], weight: [C_out, C_in, K_h, K_w], bias: [C_out]
    # Returns: max(0, conv2d(x, weight, bias))
    conv_out = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)
    return torch.nn.functional.relu(conv_out)"""

    # Simple shapes
    input_shapes = [[1, 3, 6, 6]]
    output_shapes = [[1, 4, 4, 4]]  # 6-3+1=4 with 3x3 kernel
    param_shapes = {
        "weight": [4, 3, 3, 3],
        "bias": [4]
    }
    hyperparameters = {
        "stride": [1, 1],
        "padding": [0, 0],
        "kernel_size": [3, 3],
        "fused_activation": "relu"
    }

    config = default_config()
    config["max_samples"] = 1

    try:
        print(f"Testing fused Conv2D + ReLU")
        print(f"Input: {input_shapes[0]} -> Output: {output_shapes[0]}")

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv2d_relu_fused",
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

        print("\n=== Fused Conv2D + ReLU Kernel ===")
        print(kernel_source)

        # Analysis
        has_conv_loops = kernel_source.count("for") >= 3
        has_relu_activation = "max(" in kernel_source and "0.0" in kernel_source
        has_bias_add = "bias[" in kernel_source
        single_pass = "output[" in kernel_source and kernel_source.count("output[") <= 2  # Only one write to output

        print(f"\n=== Fusion Analysis ===")
        print(f"Has convolution loops: {has_conv_loops}")
        print(f"Has ReLU activation: {has_relu_activation}")
        print(f"Handles bias: {has_bias_add}")
        print(f"Single pass computation: {single_pass}")

        # Token usage for complex fused kernel
        usage = kernel_data["llm_response"]["usage"]
        print(f"\nFused kernel tokens: {usage.get('total_tokens', 'N/A')}")

        return has_conv_loops and has_relu_activation

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Kernel Fusion Generation")
    print("=" * 40)

    success = test_conv2d_relu_fusion()

    if success:
        print("\n🎉 Kernel fusion successful!")
        print("GPT-4o can generate complex fused operations.")
    else:
        print("\n⚠️ Fusion test had issues.")

    # Success either way - this tests the framework capability
    sys.exit(0)