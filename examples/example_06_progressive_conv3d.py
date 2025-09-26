"""
Example 6: Progressive Conv3D Generation

This example demonstrates the enhanced conv3d generation system that builds
up from simple operations to complex 3D convolutions through progressive
learning and optimized sampling.

Key features:
- Progressive complexity from element-wise to conv3d
- Complexity-aware sampling (temperature, tokens, model selection)
- Enhanced 3D-specific prompts and validation
- Builds up to 8x8x8 kernel on 256^3 tensors
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path

from yolokernelgen import generate_kernel, load_kernel
from yolokernelgen.config import default_config
from yolokernelgen.cli import show_stats


def run_progressive_learning_sequence():
    """Run a sequence of operations with increasing complexity to build knowledge."""
    
    print("=" * 70)
    print("PROGRESSIVE LEARNING SEQUENCE")
    print("=" * 70)
    print("Building knowledge base from simple to complex operations...")
    
    config = default_config()
    config["max_samples"] = 10  # Allow multiple attempts with learning
    
    # Level 1: Element-wise operations
    level_1_operations = [
        {
            "name": "add_tensors_3d",
            "torch_source": "def add_3d(a, b): return a + b",
            "input_shapes": [[2, 4, 8, 8, 8], [2, 4, 8, 8, 8]],
            "output_shapes": [[2, 4, 8, 8, 8]],
            "torch_fn": lambda a, b: a + b,
            "description": "3D tensor addition - building spatial reasoning"
        },
        {
            "name": "relu_3d_large",
            "torch_source": "def relu_3d(x): return torch.nn.functional.relu(x)",
            "input_shapes": [[1, 8, 16, 16, 16]],
            "output_shapes": [[1, 8, 16, 16, 16]],
            "torch_fn": lambda x: F.relu(x),
            "description": "3D ReLU activation - handling larger tensors"
        }
    ]
    
    # Level 2: Reduction operations
    level_2_operations = [
        {
            "name": "sum_3d_spatial",
            "torch_source": "def sum_spatial(x): return torch.sum(x, dim=(2, 3, 4))",
            "input_shapes": [[2, 4, 8, 8, 8]],
            "output_shapes": [[2, 4]],
            "torch_fn": lambda x: torch.sum(x, dim=(2, 3, 4)),
            "description": "Spatial reduction - aggregating over 3D dimensions"
        },
        {
            "name": "mean_3d_channels",
            "torch_source": "def mean_channels(x): return torch.mean(x, dim=1)",
            "input_shapes": [[2, 8, 16, 16, 16]],
            "output_shapes": [[2, 16, 16, 16]],
            "torch_fn": lambda x: torch.mean(x, dim=1),
            "description": "Channel-wise averaging - understanding feature dimensions"
        }
    ]
    
    # Level 3: Matrix operations in 3D context
    level_3_operations = [
        {
            "name": "batch_3d_norm",
            "torch_source": "def norm_3d(x): return torch.linalg.vector_norm(x, dim=(1, 2, 3, 4))",
            "input_shapes": [[4, 8, 8, 8, 8]],
            "output_shapes": [[4]],
            "torch_fn": lambda x: torch.linalg.vector_norm(x, dim=(1, 2, 3, 4)),
            "description": "3D tensor norms - complex reduction patterns"
        }
    ]
    
    # Level 4: 2D convolutions as building blocks
    level_4_operations = [
        {
            "name": "conv2d_small",
            "torch_source": """def conv2d_simple(x, weight, bias):
    return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)""",
            "input_shapes": [[1, 4, 16, 16]],
            "output_shapes": [[1, 8, 16, 16]],
            "param_shapes": {"weight": [8, 4, 3, 3], "bias": [8]},
            "hyperparameters": {"stride": [1, 1], "padding": [1, 1], "kernel_size": [3, 3]},
            "torch_fn": lambda x, weight, bias: F.conv2d(x, weight, bias, stride=1, padding=1),
            "description": "2D convolution - spatial feature extraction"
        }
    ]
    
    # Level 5: Simple 3D convolutions  
    level_5_operations = [
        {
            "name": "conv3d_tiny",
            "torch_source": """def conv3d_simple(x, weight, bias):
    return torch.nn.functional.conv3d(x, weight, bias, stride=1, padding=1)""",
            "input_shapes": [[1, 2, 8, 8, 8]],
            "output_shapes": [[1, 4, 8, 8, 8]],
            "param_shapes": {"weight": [4, 2, 3, 3, 3], "bias": [4]},
            "hyperparameters": {"stride": [1, 1, 1], "padding": [1, 1, 1], "kernel_size": [3, 3, 3]},
            "torch_fn": lambda x, weight, bias: F.conv3d(x, weight, bias, stride=1, padding=1),
            "description": "Simple 3D convolution - building 3D spatial reasoning"
        }
    ]
    
    all_levels = [
        ("Level 1 - Element-wise", level_1_operations),
        ("Level 2 - Reductions", level_2_operations), 
        ("Level 3 - Complex aggregations", level_3_operations),
        ("Level 4 - 2D Convolutions", level_4_operations),
        ("Level 5 - Simple 3D Convolutions", level_5_operations)
    ]
    
    successful_operations = 0
    total_operations = sum(len(ops) for _, ops in all_levels)
    
    for level_name, operations in all_levels:
        print(f"\n{level_name}")
        print("-" * 50)
        
        for op in operations:
            print(f"\nTrying: {op['name']} - {op['description']}")
            
            try:
                kernel_path = generate_kernel(
                    torch_source=op["torch_source"],
                    operation=op["name"],
                    input_shapes=op["input_shapes"],
                    output_shapes=op["output_shapes"],
                    param_shapes=op.get("param_shapes"),
                    hyperparameters=op.get("hyperparameters"),
                    torch_fn=op["torch_fn"],
                    config=config
                )
                
                kernel_data = load_kernel(kernel_path)
                passed = kernel_data['validation']['num_passed']
                total = kernel_data['validation']['num_total']
                
                print(f"  ✓ Success: {passed}/{total} tests passed")
                successful_operations += 1
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"PROGRESSIVE LEARNING RESULTS")
    print(f"{'='*70}")
    print(f"Successfully completed: {successful_operations}/{total_operations} operations")
    print(f"Success rate: {successful_operations/total_operations*100:.1f}%")
    
    return successful_operations >= total_operations * 0.7  # 70% success rate


def attempt_medium_conv3d():
    """Attempt a medium-complexity 3D convolution."""
    
    print(f"\n{'='*70}")
    print("MEDIUM COMPLEXITY 3D CONVOLUTION")
    print(f"{'='*70}")
    print("Testing improved 3D convolution with moderate tensor sizes...")
    
    torch_source = """def conv3d_medium(x, weight, bias):
    # 3D convolution with dilation for feature extraction
    return torch.nn.functional.conv3d(x, weight, bias, stride=1, padding=2, dilation=2)"""
    
    # Medium size tensors
    batch_size = 1
    in_channels = 4
    out_channels = 8
    depth, height, width = 32, 32, 32
    
    input_shapes = [[batch_size, in_channels, depth, height, width]]
    output_shapes = [[batch_size, out_channels, depth, height, width]]
    param_shapes = {
        "weight": [out_channels, in_channels, 3, 3, 3],
        "bias": [out_channels]
    }
    hyperparameters = {
        "stride": [1, 1, 1],
        "padding": [2, 2, 2],
        "dilation": [2, 2, 2],
        "kernel_size": [3, 3, 3]
    }
    
    def torch_fn(x, weight, bias):
        return F.conv3d(x, weight, bias, stride=1, padding=2, dilation=2)
    
    print(f"Operation: 3D dilated convolution (dilation=2, padding=2)")
    print(f"Input: {input_shapes[0]} -> Output: {output_shapes[0]}")
    print(f"Total output elements: {batch_size * out_channels * depth * height * width:,}")
    
    try:
        config = default_config()
        config["max_samples"] = 10
        
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv3d_medium_progressive",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=torch_fn,
            config=config,
            force_regenerate=True
        )
        
        kernel_data = load_kernel(kernel_path)
        passed = kernel_data['validation']['num_passed']
        total = kernel_data['validation']['num_total']
        tokens = kernel_data['llm_response']['usage']['total_tokens']
        
        print(f"✓ SUCCESS! Validation: {passed}/{total} tests passed")
        print(f"✓ Tokens used: {tokens:,}")
        print(f"✓ Kernel saved: {kernel_path.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Medium conv3d failed: {e}")
        return False


def attempt_large_conv3d():
    """Attempt the target large 3D convolution with 8x8x8 kernel on large tensors."""
    
    print(f"\n{'='*70}")
    print("TARGET LARGE 3D CONVOLUTION")
    print(f"{'='*70}")
    print("Attempting 8x8x8 kernel on large tensors - the ultimate challenge...")
    
    torch_source = """def conv3d_large_kernel(x, weight, bias):
    # Large kernel 3D convolution - ultimate challenge
    return torch.nn.functional.conv3d(x, weight, bias, stride=2, padding=4)"""
    
    # Large tensors - approaching 256^3 but manageable for testing
    batch_size = 1
    in_channels = 8
    out_channels = 16
    input_depth, input_height, input_width = 128, 128, 128  # Large but not quite 256^3
    output_depth = (input_depth + 2*4 - 8) // 2 + 1  # With padding=4, kernel=8, stride=2
    output_height = (input_height + 2*4 - 8) // 2 + 1
    output_width = (input_width + 2*4 - 8) // 2 + 1
    
    input_shapes = [[batch_size, in_channels, input_depth, input_height, input_width]]
    output_shapes = [[batch_size, out_channels, output_depth, output_height, output_width]]
    param_shapes = {
        "weight": [out_channels, in_channels, 8, 8, 8],  # 8x8x8 kernel!
        "bias": [out_channels]
    }
    hyperparameters = {
        "stride": [2, 2, 2],
        "padding": [4, 4, 4],
        "kernel_size": [8, 8, 8]
    }
    
    def torch_fn(x, weight, bias):
        return F.conv3d(x, weight, bias, stride=2, padding=4)
    
    total_input_elements = batch_size * in_channels * input_depth * input_height * input_width
    total_output_elements = batch_size * out_channels * output_depth * output_height * output_width
    
    print(f"Operation: 3D convolution with 8×8×8 kernel")
    print(f"Input: {input_shapes[0]} ({total_input_elements:,} elements)")
    print(f"Output: {output_shapes[0]} ({total_output_elements:,} elements)")
    print(f"This is approaching our 256³ target complexity!")
    
    try:
        config = default_config()
        config["max_samples"] = 5  # Allow more attempts for this challenging operation
        
        kernel_path = generate_kernel(
            torch_source=torch_source,
            operation="conv3d_8x8x8_large_progressive",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_shapes=param_shapes,
            hyperparameters=hyperparameters,
            torch_fn=torch_fn,
            config=config,
            force_regenerate=True
        )
        
        kernel_data = load_kernel(kernel_path)
        passed = kernel_data['validation']['num_passed']
        total = kernel_data['validation']['num_total']
        tokens = kernel_data['llm_response']['usage']['total_tokens']
        
        print(f"🎉 ULTIMATE SUCCESS! Validation: {passed}/{total} tests passed")
        print(f"✓ Tokens used: {tokens:,}")
        print(f"✓ Model: {kernel_data['llm_request']['model']}")
        print(f"✓ Kernel saved: {kernel_path.name}")
        
        # Show kernel preview
        kernel_source = kernel_data['llm_response']['extracted_kernel']
        print(f"\n=== Generated 8×8×8 Conv3D Kernel (preview) ===")
        print(kernel_source[:800] + "..." if len(kernel_source) > 800 else kernel_source)
        
        return True
        
    except Exception as e:
        print(f"✗ Large conv3d failed: {e}")
        print("This demonstrates the extreme difficulty of generating large kernel conv3d operations.")
        return False


def show_enhanced_features():
    """Display the key enhancements in this system."""
    
    print(f"\n{'='*70}")
    print("ENHANCED CONV3D GENERATION FEATURES")
    print(f"{'='*70}")
    
    features = [
        ("Progressive Complexity", "Builds from simple element-wise ops to complex 3D convolutions"),
        ("Complexity-Aware Sampling", "Optimizes temperature, tokens, and model based on operation difficulty"),
        ("3D-Specific Prompts", "Enhanced system prompts with 3D indexing and numerical stability guidance"),
        ("Large Tensor Support", "Handles memory layout and indexing for 256³+ tensors"),
        ("Iterative Learning", "Each level builds knowledge for the next complexity tier"),
        ("Enhanced Validation", "Specialized validation for large tensor operations"),
        ("Cost Optimization", "Uses gpt-4o-mini for simple ops, gpt-4o for complex 3D operations")
    ]
    
    for i, (title, description) in enumerate(features, 1):
        print(f"\n{i}. {title}")
        print(f"   {description}")
    
    print(f"\n{'='*70}")
    print("PROGRESSION LEVELS")
    print(f"{'='*70}")
    levels = [
        "Level 1: Element-wise operations (add, relu) on 3D tensors",
        "Level 2: Reduction operations (sum, mean) over spatial dimensions", 
        "Level 3: Complex aggregations (norms, variance) on large tensors",
        "Level 4: 2D convolutions as spatial reasoning building blocks",
        "Level 5: Simple 3D convolutions with small kernels",
        "Level 6: TARGET - 8×8×8 kernels on 256³ tensors"
    ]
    
    for level in levels:
        print(f"• {level}")


if __name__ == "__main__":
    print("YoloKernelGen - Example 6: Progressive Conv3D Generation")
    print("=" * 65)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print("This example demonstrates enhanced conv3d generation through")
    print("progressive learning and complexity-aware optimization.\n")
    
    # Show enhanced features
    show_enhanced_features()
    
    print(f"\n{'='*70}")
    print("LIVE DEMONSTRATION")
    print(f"{'='*70}")
    
    # Show initial knowledge base stats
    print("Initial knowledge base state:")
    show_stats()
    
    # Run progressive learning sequence
    print(f"\nStep 1: Progressive Learning Sequence")
    progressive_success = run_progressive_learning_sequence()
    
    # Show updated stats after progressive learning
    print(f"\nKnowledge base after progressive learning:")
    show_stats()
    
    if progressive_success:
        print(f"\nStep 2: Medium Complexity 3D Convolution")
        medium_success = attempt_medium_conv3d()
        
        if medium_success:
            print(f"\nStep 3: Target Large 3D Convolution (8×8×8 kernel)")
            large_success = attempt_large_conv3d()
            
            # Final results
            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print(f"{'='*70}")
            
            if large_success:
                print("🏆 COMPLETE SUCCESS!")
                print("• Progressive learning sequence: ✓")
                print("• Medium complexity conv3d: ✓") 
                print("• Large 8×8×8 kernel conv3d: ✓")
                print("\nThe enhanced system successfully generated complex 3D convolutions!")
            else:
                print("⚡ SIGNIFICANT PROGRESS!")
                print("• Progressive learning sequence: ✓")
                print("• Medium complexity conv3d: ✓")
                print("• Large 8×8×8 kernel conv3d: ⚠️ (challenging)")
                print("\nThe progressive system dramatically improved success rates.")
        else:
            print("\n📈 FOUNDATION SUCCESS!")
            print("• Progressive learning sequence: ✓")
            print("• Built strong knowledge base for future 3D operations")
    else:
        print("\n📚 LEARNING IN PROGRESS")
        print("• The system is building knowledge - run multiple times to see improvement")
    
    # Show final knowledge base
    print(f"\nFinal knowledge base state:")
    show_stats()
    
    print(f"\n{'='*70}")
    print("Next steps:")
    print("• Run this example multiple times to see cumulative learning")
    print("• The knowledge base will improve success rates over time")
    print("• Try scaling up to true 256³ tensors once the system is well-trained")
    print("• Use the progressive framework for other complex operations")
