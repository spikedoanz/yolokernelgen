"""
Example 1: Simple Element-wise Operations

This example demonstrates the basics of YoloKernelGen:
- Generating kernels for simple operations (add, relu)
- Understanding the basic API
- Seeing how validation works
"""

import os
import torch.nn.functional as F

from yolokernelgen import generate_kernel, load_kernel


def example_add_one():
    """Generate a kernel that adds 1 to each element."""

    print("=== Example: Add One ===")

    # Define the PyTorch operation
    torch_source = """def add_one(x):
    return x + 1.0"""

    # Define tensor shapes
    input_shapes = [[4, 8]]  # Simple 2D tensor
    output_shapes = [[4, 8]]  # Same shape output

    # PyTorch function for validation
    def torch_fn(x):
        return x + 1.0

    try:
        # Generate the kernel
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="add_one",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn  # This validates correctness!
        )

        # Load and examine the result
        kernel_data = load_kernel(kernel_path)

        print(f"✓ Kernel generated: {kernel_path.name}")
        print(f"✓ Status: {kernel_data.status}")
        print(f"✓ Tests passed: {kernel_data.validation.num_passed}/{kernel_data.validation.num_total}")
        print(f"✓ Tokens used: {kernel_data.llm_response.usage.get('total_tokens', 0)}")

        # Show the generated WGSL kernel (first 300 chars)
        kernel_code = kernel_data.llm_response.extracted_kernel
        print("\nGenerated WGSL kernel (preview):")
        print("-" * 40)
        print(kernel_code[:300] + "...")
        print("-" * 40)

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def example_relu():
    """Generate a ReLU activation kernel."""

    print("\n=== Example: ReLU Activation ===")

    torch_source = """def relu(x):
    return torch.nn.functional.relu(x)"""

    input_shapes = [[2, 3, 4, 4]]  # 4D tensor (NCHW format)
    output_shapes = [[2, 3, 4, 4]]

    def torch_fn(x):
        return F.relu(x)

    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="relu",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ ReLU kernel: {kernel_path.name}")
        print(f"✓ Validation: {kernel_data.validation.all_passed}")

        # Show that ReLU correctly clamps negative values
        test_results = kernel_data.validation.test_cases
        zeros_test = next((t for t in test_results if t.test_type == 'zeros'), None)
        if zeros_test:
            print(f"✓ Zeros test passed: {zeros_test.passed} (max_diff: {getattr(zeros_test, 'max_diff', 0.0):.2e})")

        return True

    except Exception as e:
        print(f"✗ ReLU failed: {e}")
        return False


if __name__ == "__main__":
    print("YoloKernelGen - Example 1: Simple Operations")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        exit(1)

    # Run examples
    success1 = example_add_one()
    success2 = example_relu()

    print("\n=== Results ===")
    if success1 and success2:
        print("🎉 Both examples completed successfully!")
        print("\nKey takeaways:")
        print("- Generated kernels are validated against PyTorch")
        print("- WGSL code is automatically extracted and cached")
        print("- Simple operations work perfectly with minimal tokens")
    else:
        print("❌ Some examples failed - check your setup")

    print("\nNext: Try example_02_tensor_operations.py for matrix operations!")