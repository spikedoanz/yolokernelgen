"""
Example 2: Tensor Operations

This example demonstrates more complex operations:
- Matrix multiplication with proper indexing
- Element-wise operations between tensors
- Understanding parameter shapes and multi-tensor inputs
"""

import os
import torch

from yolokernelgen import generate_kernel, load_kernel, list_kernels


def example_matrix_multiplication():
    """Generate a matrix multiplication kernel."""

    print("=== Example: Matrix Multiplication ===")

    torch_source = """def matmul(a, b):
    # Matrix multiplication: A[M,K] @ B[K,N] -> C[M,N]
    return torch.matmul(a, b)"""

    # Two input tensors for matrix multiplication
    input_shapes = [[8, 4], [4, 6]]  # A: 8x4, B: 4x6
    output_shapes = [[8, 6]]         # Result: 8x6

    def torch_fn(a, b):
        return torch.matmul(a, b)

    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="matmul",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ MatMul kernel: {kernel_path.name}")
        print(f"✓ Input shapes: {input_shapes[0]} @ {input_shapes[1]} = {output_shapes[0]}")
        print(f"✓ Validation: {kernel_data['validation']['all_passed']}")
        print(f"✓ Tokens: {kernel_data['llm_response']['usage']['total_tokens']}")

        # Examine the generated kernel for key features
        kernel_code = kernel_data['llm_response']['extracted_kernel']
        has_dot_product = "sum" in kernel_code.lower() and "+=" in kernel_code
        has_nested_indexing = "row" in kernel_code.lower() and "col" in kernel_code.lower()

        print(f"✓ Has dot product computation: {has_dot_product}")
        print(f"✓ Has proper row/column indexing: {has_nested_indexing}")

        return True

    except Exception as e:
        print(f"✗ MatMul failed: {e}")
        return False


def example_element_wise_add():
    """Generate element-wise tensor addition."""

    print("\n=== Example: Element-wise Addition ===")

    torch_source = """def tensor_add(a, b):
    # Element-wise addition of two tensors
    return a + b"""

    input_shapes = [[2, 3, 4], [2, 3, 4]]  # Two tensors of same shape
    output_shapes = [[2, 3, 4]]

    def torch_fn(a, b):
        return a + b

    try:
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="tensor_add",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn
        )

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Addition kernel: {kernel_path.name}")
        print(f"✓ Handles two input tensors: {len(input_shapes)} inputs")
        print(f"✓ Validation: {kernel_data['validation']['all_passed']}")

        return True

    except Exception as e:
        print(f"✗ Addition failed: {e}")
        return False


def show_generated_kernels():
    """Show what kernels have been generated so far."""

    print("\n=== Generated Kernels Summary ===")

    # List all correct (validated) kernels
    correct_kernels = list_kernels(status_filter="correct")

    if not correct_kernels:
        print("No validated kernels found yet.")
        return

    print(f"Found {len(correct_kernels)} validated kernels:")

    operations = {}
    for kernel in correct_kernels:
        op = kernel["operation"]
        operations[op] = operations.get(op, 0) + 1

    for op, count in operations.items():
        print(f"  {op}: {count} kernel(s)")

    # Show cache directory info
    print("\nKernels cached in: .cache/yolokernelgen/generated/")
    print("  - Filenames encode operation, shapes, and parameters")
    print("  - 'c_' prefix means validated/correct")
    print("  - 'r_' prefix means rejected/failed validation")


if __name__ == "__main__":
    print("YoloKernelGen - Example 2: Tensor Operations")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        exit(1)

    # Run tensor operation examples
    success1 = example_matrix_multiplication()
    success2 = example_element_wise_add()

    # Show what we've built so far
    show_generated_kernels()

    print("\n=== Results ===")
    if success1 and success2:
        print("🎉 Tensor operations completed successfully!")
        print("\nKey insights:")
        print("- Multi-tensor inputs work seamlessly")
        print("- Complex operations like MatMul generate sophisticated kernels")
        print("- Each kernel is mathematically validated against PyTorch")
    else:
        print("❌ Some tensor operations failed")

    print("\nNext: Try example_03_convolutions.py for neural network layers!")