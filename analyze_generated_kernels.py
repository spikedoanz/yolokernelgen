"""Analyze the quality of generated kernels without validation dependency."""

import os
import json
from pathlib import Path
from yolokernelgen import list_kernels, load_kernel


def analyze_kernel_structure(kernel_source: str, operation: str) -> dict:
    """Analyze WGSL kernel structure and quality."""

    analysis = {
        "basic_structure": {},
        "operation_specific": {},
        "quality_score": 0
    }

    # Basic WGSL structure
    analysis["basic_structure"] = {
        "has_group_binding": "@group" in kernel_source and "@binding" in kernel_source,
        "has_compute_shader": "@compute" in kernel_source,
        "has_workgroup_size": "@workgroup_size" in kernel_source,
        "has_main_function": "fn main" in kernel_source,
        "has_bounds_check": "if" in kernel_source and ">=" in kernel_source and "return" in kernel_source,
        "has_total_elements": "TOTAL_ELEMENTS" in kernel_source,
        "proper_variable_types": "var<storage" in kernel_source,
        "uses_global_id": "global_invocation_id" in kernel_source,
    }

    # Operation-specific analysis
    if operation in ["conv2d", "conv2d_padded"]:
        analysis["operation_specific"] = {
            "has_nested_loops": kernel_source.count("for") >= 3,
            "has_tensor_indexing": "n *" in kernel_source or "batch" in kernel_source.lower(),
            "has_channel_loops": "ic" in kernel_source or "in_channels" in kernel_source.lower(),
            "has_spatial_loops": ("kh" in kernel_source and "kw" in kernel_source) or "kernel" in kernel_source.lower(),
            "has_bias_handling": "bias[" in kernel_source,
            "proper_weight_access": "weight[" in kernel_source,
            "nchw_awareness": any(x in kernel_source.lower() for x in ["nchw", "height", "width", "channel"])
        }

    elif operation in ["matmul"]:
        analysis["operation_specific"] = {
            "has_dot_product": "sum" in kernel_source and "+=" in kernel_source,
            "has_row_col_calc": ("row" in kernel_source and "col" in kernel_source) or ("/ N" in kernel_source and "% N" in kernel_source),
            "has_inner_loop": kernel_source.count("for") >= 1,
            "proper_matrix_indexing": "*" in kernel_source and "+" in kernel_source,
            "accumulates_properly": "var sum" in kernel_source or "+=" in kernel_source
        }

    elif operation in ["relu"]:
        analysis["operation_specific"] = {
            "uses_max_function": "max(" in kernel_source,
            "clamps_to_zero": "0.0" in kernel_source,
            "simple_elementwise": kernel_source.count("for") == 0,  # Should be element-wise, no loops
        }

    elif operation in ["add_one", "add"]:
        analysis["operation_specific"] = {
            "simple_arithmetic": "+" in kernel_source,
            "element_wise": kernel_source.count("for") == 0,
            "direct_indexing": "input[index]" in kernel_source or "input_a[index]" in kernel_source
        }

    # Calculate quality score
    basic_score = sum(analysis["basic_structure"].values())
    specific_score = sum(analysis["operation_specific"].values()) if analysis["operation_specific"] else 0
    analysis["quality_score"] = (basic_score + specific_score) / max(1, len(analysis["basic_structure"]) + len(analysis["operation_specific"]))

    return analysis


def analyze_all_generated_kernels():
    """Analyze all kernels in the cache."""

    print("=== Generated Kernel Analysis ===")

    # List all kernels
    all_kernels = list_kernels()

    if not all_kernels:
        print("No kernels found in cache.")
        return

    print(f"Found {len(all_kernels)} kernels in cache\n")

    results = {}

    for kernel_info in all_kernels:
        filepath = Path(kernel_info["filepath"])
        operation = kernel_info["operation"]
        status = kernel_info["status"]

        try:
            kernel_data = load_kernel(filepath)
            kernel_source = kernel_data["llm_response"]["extracted_kernel"]

            # Analyze the kernel
            analysis = analyze_kernel_structure(kernel_source, operation)

            # Get token usage
            usage = kernel_data["llm_response"].get("usage", {})

            results[filepath.name] = {
                "operation": operation,
                "status": status,
                "analysis": analysis,
                "usage": usage,
                "shapes": {
                    "input": kernel_info.get("input_shape"),
                    "output": kernel_info.get("output_shape")
                }
            }

            print(f"🔍 {operation.upper()}")
            print(f"   File: {filepath.name}")
            print(f"   Status: {status}")
            print(f"   Shapes: {kernel_info.get('input_shape')} → {kernel_info.get('output_shape')}")
            print(f"   Quality Score: {analysis['quality_score']:.2f}")
            print(f"   Tokens: {usage.get('total_tokens', 'N/A')}")

            # Show key quality indicators
            if analysis["operation_specific"]:
                good_features = [k for k, v in analysis["operation_specific"].items() if v]
                print(f"   ✓ Features: {', '.join(good_features)}")

            print()

        except Exception as e:
            print(f"   ❌ Error analyzing {filepath.name}: {e}")
            continue

    # Summary
    if results:
        avg_quality = sum(r["analysis"]["quality_score"] for r in results.values()) / len(results)
        total_tokens = sum(r["usage"].get("total_tokens", 0) for r in results.values())
        operations = set(r["operation"] for r in results.values())

        print(f"=== Summary ===")
        print(f"Operations tested: {', '.join(sorted(operations))}")
        print(f"Average quality score: {avg_quality:.2f}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Complex operations working: {len([r for r in results.values() if r['operation'] in ['conv2d', 'matmul']])}")


def show_best_kernel():
    """Show the highest quality kernel generated."""

    all_kernels = list_kernels()
    if not all_kernels:
        return

    best_score = 0
    best_kernel = None
    best_info = None

    for kernel_info in all_kernels:
        try:
            kernel_data = load_kernel(Path(kernel_info["filepath"]))
            kernel_source = kernel_data["llm_response"]["extracted_kernel"]
            analysis = analyze_kernel_structure(kernel_source, kernel_info["operation"])

            if analysis["quality_score"] > best_score:
                best_score = analysis["quality_score"]
                best_kernel = kernel_source
                best_info = kernel_info

        except:
            continue

    if best_kernel:
        print(f"\n=== Best Generated Kernel ===")
        print(f"Operation: {best_info['operation']}")
        print(f"Quality Score: {best_score:.2f}")
        print(f"Code:")
        print("-" * 60)
        print(best_kernel)
        print("-" * 60)


if __name__ == "__main__":
    analyze_all_generated_kernels()
    show_best_kernel()