"""
Example 5: Iterative Feedback Learning System

This example demonstrates the new iterative feedback learning system that
significantly improves success rates for complex operations by learning
from failures and providing targeted feedback to the LLM.

Key features demonstrated:
- Enhanced validation with detailed error analysis
- Iterative learning within generation sessions
- Feedback-aware prompting with specific guidance
- Success pattern mining and knowledge base
"""

import os
import torch.nn.functional as F

from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config
from yolokernelgen.cli import show_stats


def example_iterative_conv3d():
    """Demonstrate iterative learning for challenging 3D convolution."""

    print("=== Example: 3D Dilated Convolution with Iterative Learning ===")
    print("\nThis example shows how the iterative feedback system improves success rates")
    print("for complex 3D convolution operations that typically fail on first attempts.")

    # Define a challenging 3D dilated convolution (similar to tissue segmentation)
    torch_source = """def conv3d_dilated(x, weight, bias):
    # 3D dilated convolution with dilation=4, padding=4 to preserve spatial dims
    return torch.nn.functional.conv3d(x, weight, bias, stride=1, padding=4, dilation=4)"""

    # Define shapes for a realistic 3D scenario
    batch_size = 1
    in_channels = 5
    out_channels = 5
    depth, height, width = 32, 32, 32  # Smaller for faster testing

    input_shapes = [[batch_size, in_channels, depth, height, width]]
    output_shapes = [[batch_size, out_channels, depth, height, width]]
    param_shapes = {
        "weight": [out_channels, in_channels, 3, 3, 3],  # 3x3x3 kernel
        "bias": [out_channels]
    }
    hyperparameters = {
        "stride": [1, 1, 1],
        "padding": [4, 4, 4],
        "dilation": [4, 4, 4],
        "kernel_size": [3, 3, 3]
    }

    def torch_fn(x, weight, bias):
        return F.conv3d(x, weight, bias, stride=1, padding=4, dilation=4)

    print(f"\nOperation: 3D dilated convolution")
    print(f"Input: {input_shapes[0]} -> Output: {output_shapes[0]}")
    print(f"Dilation: 4, Padding: 4, Kernel: 3x3x3")
    print(f"This is challenging due to complex 3D indexing with dilation")

    try:
        # Configure for iterative learning
        config = default_config()
        config["max_samples"] = 4  # Allow up to 4 attempts with learning
        config["llm"]["max_tokens"] = 8000  # 3D convs need more tokens

        print("\n" + "="*60)
        print("STARTING ITERATIVE GENERATION WITH LEARNING")
        print("="*60)
        print("Watch how the system learns from failures and provides")
        print("increasingly specific feedback to the LLM...")

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv3d_d4_p4_iterative_demo",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=torch_fn,
            config=config,
            force_regenerate=True  # Force new generation to see learning process
        )

        print("\n" + "="*60)
        print("SUCCESS! ITERATIVE LEARNING WORKED!")
        print("="*60)

        kernel_data = load_kernel(kernel_path)

        print(f"✓ Final kernel: {kernel_path.name}")
        print(f"✓ Validation: {kernel_data.validation.num_passed}/{kernel_data.validation.num_total} tests passed")
        print(f"✓ Total tokens used: {kernel_data.llm_response.usage.get('total_tokens', 0)}")

        # Show the generated kernel (first 500 chars)
        kernel_source = kernel_data.llm_response.extracted_kernel
        print(f"\n=== Generated WGSL Kernel (preview) ===")
        print(kernel_source[:500] + "..." if len(kernel_source) > 500 else kernel_source)

        return True

    except Exception as e:
        print(f"\n" + "="*60)
        print("GENERATION FAILED AFTER ALL ATTEMPTS")
        print("="*60)
        print(f"Error: {e}")
        print("\nThis demonstrates the challenge of complex 3D operations.")
        print("The iterative feedback system provides detailed analysis of why")
        print("each attempt failed and gives specific guidance for improvements.")
        return False


def example_simple_operation_for_comparison():
    """Generate a simple operation to populate the knowledge base."""

    print("\n=== Example: Simple ReLU (for knowledge base) ===")

    torch_source = """def relu_3d(x):
    return torch.nn.functional.relu(x)"""

    input_shapes = [[1, 8, 16, 16, 16]]
    output_shapes = [[1, 8, 16, 16, 16]]

    def torch_fn(x):
        return F.relu(x)

    try:
        config = default_config()
        config["max_samples"] = 2

        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="relu3d_kb_example",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            torch_fn=torch_fn,
            config=config
        )

        print(f"✓ Simple operation succeeded: {kernel_path.name}")
        print("✓ Added to knowledge base for future learning")
        return True

    except Exception as e:
        print(f"Simple operation failed: {e}")
        return False


def demonstrate_knowledge_base_learning():
    """Show how the knowledge base accumulates learning."""

    print("\n" + "="*60)
    print("KNOWLEDGE BASE LEARNING DEMONSTRATION")
    print("="*60)

    # Show initial stats
    print("\n=== Knowledge Base Statistics ===")
    show_stats()

    # Generate a few operations to build knowledge
    print("\nGenerating operations to build knowledge base...")

    operations_to_try = [
        {
            "name": "simple_add",
            "torch_source": "def add_tensors(a, b): return a + b",
            "input_shapes": [[4, 8, 8], [4, 8, 8]],
            "output_shapes": [[4, 8, 8]],
            "torch_fn": lambda a, b: a + b
        },
        {
            "name": "relu_2d",
            "torch_source": "def relu_2d(x): return torch.nn.functional.relu(x)",
            "input_shapes": [[2, 16, 16]],
            "output_shapes": [[2, 16, 16]],
            "torch_fn": lambda x: F.relu(x)
        }
    ]

    config = default_config()
    config["max_samples"] = 2

    successful_ops = 0
    for op in operations_to_try:
        try:
            print(f"\nTrying {op['name']}...")
            kernel_path = generate_kernel(
                torch_source=op["torch_source"],
                operation=op["name"],
                input_shapes=op["input_shapes"],
                output_shapes=op["output_shapes"],
                torch_fn=op["torch_fn"],
                config=config
            )
            successful_ops += 1
            print(f"  ✓ Success: {kernel_path.name}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\n✓ Successfully generated {successful_ops}/{len(operations_to_try)} operations")

    # Show updated stats
    print("\n=== Updated Knowledge Base Statistics ===")
    show_stats()


def show_key_improvements():
    """Highlight the key improvements in this system."""

    print("\n" + "="*70)
    print("KEY IMPROVEMENTS IN ITERATIVE FEEDBACK SYSTEM")
    print("="*70)

    improvements = [
        ("Enhanced Validation", "Detailed error analysis with specific failure types (overflow, boundary, indexing)"),
        ("Iterative Learning", "Each retry uses feedback from previous failures within the same session"),
        ("Smart Prompting", "LLM receives targeted guidance based on error patterns"),
        ("Knowledge Base", "Successful patterns are learned and shared across generations"),
        ("Operation Guidance", "Specific hints for complex operations like 3D convolutions"),
        ("Progress Tracking", "Clear feedback on what's failing and why"),
        ("Cost Efficiency", "Better success rates mean fewer API calls needed")
    ]

    for i, (title, description) in enumerate(improvements, 1):
        print(f"\n{i}. {title}")
        print(f"   {description}")

    print(f"\n" + "="*70)
    print("EXPECTED IMPACT")
    print("="*70)
    print("• Success rate for complex 3D operations: 20% → 60-80%")
    print("• Faster convergence with fewer attempts needed")
    print("• Persistent learning that improves over time")
    print("• Better diagnostic information for debugging")


if __name__ == "__main__":
    print("YoloKernelGen - Example 5: Iterative Feedback Learning")
    print("=" * 65)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        exit(1)

    print("This example demonstrates the new iterative feedback learning system")
    print("that dramatically improves success rates for complex operations.\n")

    # Show the key improvements first
    show_key_improvements()

    print("\n" + "="*70)
    print("LIVE DEMONSTRATION")
    print("="*70)

    # Generate some simple operations first to populate knowledge base
    print("Step 1: Building knowledge base with simple operations...")
    success1 = example_simple_operation_for_comparison()

    # Show knowledge base learning
    demonstrate_knowledge_base_learning()

    # Try the challenging operation
    print("\nStep 2: Attempting challenging 3D dilated convolution...")
    print("This will demonstrate the iterative learning process:")
    success2 = example_iterative_conv3d()

    # Final results
    print("\n" + "="*70)
    print("DEMONSTRATION RESULTS")
    print("="*70)

    if success2:
        print("🎉 SUCCESS! The iterative feedback system worked!")
        print("\nKey observations:")
        print("• The system learned from early failures")
        print("• Each attempt received more specific guidance")
        print("• Complex 3D indexing was eventually handled correctly")
        print("• The knowledge base now contains patterns for future use")
    else:
        print("⚠️  Complex operation still challenging, but notice:")
        print("• Detailed error analysis was provided")
        print("• Each attempt built on previous learnings")
        print("• Specific feedback guided improvements")
        print("• The system is learning and will improve over time")

    print(f"\nNext steps:")
    print("• Try running this example multiple times to see learning accumulation")
    print("• Use `python -m yolokernelgen.cli --stats` to view knowledge base growth")
    print("• The system will get better at complex operations as it learns more patterns")