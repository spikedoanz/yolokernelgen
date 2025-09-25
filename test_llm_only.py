"""Test LLM kernel generation without WebGPU execution."""

import os
import sys
from pathlib import Path

# Mock the webgpu_executor to test LLM generation without pydawn
import yolokernelgen.webgpu_executor
import numpy as np

def mock_execute_kernel(kernel_source, input_tensors, output_shape=None, workgroup_size=256):
    """Mock executor that returns same-shape tensor without actually running WebGPU."""
    if output_shape:
        return np.zeros(output_shape, dtype=np.float32)
    return np.zeros_like(input_tensors[0])

# Monkey-patch the execute_kernel function
yolokernelgen.webgpu_executor.execute_kernel = mock_execute_kernel

from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config


def test_llm_generation():
    """Test that LLM generates valid WGSL code for a+1 operation."""

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return False

    print("\n=== Testing LLM Kernel Generation (GPT-4o) ===")

    # Simple a+1 operation
    torch_source = """def add_one(a):
    return a + 1.0"""

    input_shapes = [[2, 3, 4, 4]]
    output_shapes = [[2, 3, 4, 4]]

    # Configure for testing (skip validation since we're mocking WebGPU)
    config = default_config()
    config["max_samples"] = 1  # Only try once for testing

    try:
        print("\nCalling GPT-4o to generate WGSL kernel...")

        # Generate without validation (torch_fn=None)
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="add_one",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=None,  # Skip validation
            config=config,
            force_regenerate=True
        )

        print(f"✓ Kernel saved to: {kernel_path}")

        # Load and display the generated kernel
        kernel_data = load_kernel(kernel_path)
        kernel_source = kernel_data["llm_response"]["extracted_kernel"]

        print("\n=== Generated WGSL Kernel ===")
        print(kernel_source)

        # Check basic WGSL structure
        has_group = "@group" in kernel_source
        has_binding = "@binding" in kernel_source
        has_compute = "@compute" in kernel_source
        has_workgroup = "@workgroup_size" in kernel_source
        has_main = "fn main" in kernel_source

        print("\n=== WGSL Structure Check ===")
        print(f"Has @group directive: {has_group}")
        print(f"Has @binding directive: {has_binding}")
        print(f"Has @compute directive: {has_compute}")
        print(f"Has @workgroup_size: {has_workgroup}")
        print(f"Has main function: {has_main}")

        # Show token usage
        if "usage" in kernel_data["llm_response"]:
            usage = kernel_data["llm_response"]["usage"]
            print(f"\n=== Token Usage ===")
            print(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")

        # Calculate expected total elements
        total_elements = 1
        for dim in output_shapes[0]:
            total_elements *= dim

        print(f"\n=== Kernel Details ===")
        print(f"Expected total elements: {total_elements}")

        # Check if kernel has correct total elements constant
        if f"{total_elements}u" in kernel_source:
            print(f"✓ Kernel has correct TOTAL_ELEMENTS constant")
        else:
            print(f"✗ Kernel might not have correct TOTAL_ELEMENTS constant")

        # Basic validation of WGSL structure
        is_valid = all([has_group, has_binding, has_compute, has_workgroup, has_main])

        return is_valid

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llm_generation()

    if success:
        print("\n🎉 Success! GPT-4o generated valid WGSL kernel structure.")
        print("The LLM integration is working correctly.")
    else:
        print("\n❌ Generated kernel has structural issues.")
        sys.exit(1)