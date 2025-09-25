"""Test script for a+1 kernel generation with OpenAI API."""

import os
import sys
import numpy as np
import torch
from pathlib import Path

from yolokernelgen import generate_kernel, load_kernel, execute_kernel


def test_add_one():
    """Test generating and using an a+1 kernel."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return False

    print("\n=== Testing a+1 Kernel Generation with GPT-4o ===")

    # Define the PyTorch operation
    torch_source = """def add_one(a):
    return a + 1.0"""

    # Define shapes - small tensor for testing
    input_shapes = [[2, 3, 4, 4]]  # Small 4D tensor
    output_shapes = [[2, 3, 4, 4]]  # Same shape output

    # Create torch function for validation
    def torch_fn(a):
        return a + 1.0

    print(f"Input shape: {input_shapes[0]}")
    print(f"Output shape: {output_shapes[0]}")
    print(f"Total elements: {np.prod(input_shapes[0])}")

    try:
        print("\nGenerating kernel with GPT-4o...")
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="add_one",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn,
            force_regenerate=True  # Force regeneration for testing
        )

        print(f"\n✓ Generated kernel: {kernel_path}")

        # Load the generated kernel
        kernel_data = load_kernel(kernel_path)
        kernel_source = kernel_data["llm_response"]["extracted_kernel"]

        print("\n=== Generated WGSL Kernel ===")
        print(kernel_source[:500] + "..." if len(kernel_source) > 500 else kernel_source)

        # Test with real data
        print("\n=== Testing with Real Data ===")
        test_input = np.random.randn(*input_shapes[0]).astype(np.float32)
        print(f"Sample input values: {test_input.flat[:5]}")

        # Execute kernel
        result = execute_kernel(kernel_source, [test_input], output_shapes[0])

        # Verify against PyTorch
        torch_result = torch_fn(torch.from_numpy(test_input)).numpy()
        print(f"Sample output values: {result.flat[:5]}")
        print(f"Expected values: {torch_result.flat[:5]}")

        # Check accuracy
        max_diff = np.max(np.abs(result - torch_result))
        mean_diff = np.mean(np.abs(result - torch_result))
        matches = np.allclose(result, torch_result, rtol=1e-5)

        print(f"\n=== Validation Results ===")
        print(f"Results match PyTorch: {matches}")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")

        # Show validation details
        if "validation" in kernel_data:
            val = kernel_data["validation"]
            print(f"\nValidation suite results:")
            print(f"  Passed: {val['num_passed']}/{val['num_total']} tests")
            if val['test_cases']:
                for i, test in enumerate(val['test_cases'][:3]):  # Show first 3
                    print(f"  Test {i+1} ({test['type']}): "
                          f"{'✓' if test['passed'] else '✗'} "
                          f"max_diff={test.get('max_diff', 'N/A'):.2e}")

        # Show token usage
        if "llm_response" in kernel_data and "usage" in kernel_data["llm_response"]:
            usage = kernel_data["llm_response"]["usage"]
            print(f"\nToken usage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")

        return matches

    except Exception as e:
        print(f"\n✗ Failed to generate kernel: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_add_one()

    if success:
        print("\n🎉 Success! The a+1 kernel was generated and validated correctly.")
        print("The LLM integration is working properly with GPT-4o.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)