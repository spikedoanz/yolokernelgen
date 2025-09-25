"""Knowledge base for successful kernel patterns and learning."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


def get_cache_dir() -> str:
    """Get default cache directory."""
    return ".cache/yolokernelgen"


def get_knowledge_base_path(cache_dir: Optional[str] = None) -> Path:
    """Get path to knowledge base file."""
    if cache_dir is None:
        cache_dir = get_cache_dir()

    kb_dir = Path(cache_dir) / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir / "success_patterns.json"


def load_knowledge_base(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load the knowledge base of successful patterns."""
    kb_path = get_knowledge_base_path(cache_dir)

    if not kb_path.exists():
        return {
            "successful_kernels": [],
            "operation_patterns": {},
            "common_solutions": {},
            "version": "1.0"
        }

    try:
        with open(kb_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {
            "successful_kernels": [],
            "operation_patterns": {},
            "common_solutions": {},
            "version": "1.0"
        }


def save_knowledge_base(kb_data: Dict[str, Any], cache_dir: Optional[str] = None):
    """Save the knowledge base."""
    kb_path = get_knowledge_base_path(cache_dir)

    with open(kb_path, 'w') as f:
        json.dump(kb_data, f, indent=2)


def extract_kernel_patterns(kernel_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract patterns from a successful kernel for learning."""
    kernel_source = kernel_data.get("llm_response", {}).get("extracted_kernel", "")

    patterns = {
        "operation": kernel_data.get("operation", ""),
        "input_shapes": kernel_data.get("metadata", {}).get("input_shapes", []),
        "output_shapes": kernel_data.get("metadata", {}).get("output_shapes", []),
        "hyperparameters": kernel_data.get("metadata", {}).get("hyperparameters", {}),
        "validation_score": kernel_data.get("validation", {}).get("num_passed", 0),
        "total_tests": kernel_data.get("validation", {}).get("num_total", 0),
        "tokens_used": kernel_data.get("llm_response", {}).get("usage", {}).get("total_tokens", 0)
    }

    # Extract key code patterns
    if kernel_source:
        patterns["code_features"] = analyze_kernel_code(kernel_source)

    return patterns


def analyze_kernel_code(kernel_source: str) -> Dict[str, Any]:
    """Analyze kernel source code to extract useful patterns."""
    features = {
        "has_nested_loops": kernel_source.count("for") >= 2,
        "uses_boundaries_check": "if" in kernel_source and (">" in kernel_source or "<" in kernel_source),
        "has_padding_logic": "padding" in kernel_source.lower(),
        "uses_dilation": "dilation" in kernel_source.lower(),
        "workgroup_size": extract_workgroup_size(kernel_source),
        "binding_count": kernel_source.count("@binding("),
        "uses_constants": "const" in kernel_source,
        "operation_type": classify_operation_type(kernel_source)
    }

    return features


def extract_workgroup_size(kernel_source: str) -> Optional[int]:
    """Extract workgroup size from kernel source."""
    import re

    # Look for @workgroup_size(N) pattern
    match = re.search(r'@workgroup_size\((\d+)\)', kernel_source)
    if match:
        return int(match.group(1))

    # Look for @workgroup_size(N, M, K) pattern and return first dimension
    match = re.search(r'@workgroup_size\((\d+),\s*\d+,\s*\d+\)', kernel_source)
    if match:
        return int(match.group(1))

    return None


def classify_operation_type(kernel_source: str) -> str:
    """Classify the type of operation based on kernel patterns."""
    source_lower = kernel_source.lower()

    if "conv" in source_lower and kernel_source.count("for") >= 3:
        return "convolution"
    elif "matrix" in source_lower or "matmul" in source_lower:
        return "matrix_operation"
    elif "relu" in source_lower or "max(0" in source_lower:
        return "activation"
    elif kernel_source.count("for") == 0 and "index" in source_lower:
        return "elementwise"
    else:
        return "unknown"


def add_successful_kernel(kernel_data: Dict[str, Any], cache_dir: Optional[str] = None):
    """Add a successful kernel to the knowledge base."""
    kb = load_knowledge_base(cache_dir)

    # Extract patterns
    patterns = extract_kernel_patterns(kernel_data)

    # Add to successful kernels list
    kb["successful_kernels"].append({
        "patterns": patterns,
        "kernel_source": kernel_data.get("llm_response", {}).get("extracted_kernel", ""),
        "timestamp": kernel_data.get("metadata", {}).get("timestamp"),
        "operation": patterns["operation"]
    })

    # Update operation-specific patterns
    operation = patterns["operation"]
    if operation not in kb["operation_patterns"]:
        kb["operation_patterns"][operation] = {
            "count": 0,
            "avg_tokens": 0,
            "successful_features": {},
            "common_shapes": []
        }

    op_patterns = kb["operation_patterns"][operation]
    op_patterns["count"] += 1

    # Update average tokens
    tokens = patterns.get("tokens_used", 0)
    if tokens > 0:
        total_tokens = op_patterns["avg_tokens"] * (op_patterns["count"] - 1) + tokens
        op_patterns["avg_tokens"] = total_tokens // op_patterns["count"]

    # Track successful features
    code_features = patterns.get("code_features", {})
    for feature, value in code_features.items():
        if feature not in op_patterns["successful_features"]:
            op_patterns["successful_features"][feature] = {}

        feature_key = str(value)
        op_patterns["successful_features"][feature][feature_key] = op_patterns["successful_features"][feature].get(feature_key, 0) + 1

    # Track common shapes
    input_shape = patterns.get("input_shapes", [])
    if input_shape and input_shape not in op_patterns["common_shapes"]:
        op_patterns["common_shapes"].append(input_shape)

    # Keep only the last 50 successful kernels to avoid bloat
    if len(kb["successful_kernels"]) > 50:
        kb["successful_kernels"] = kb["successful_kernels"][-50:]

    save_knowledge_base(kb, cache_dir)


def get_relevant_success_examples(
    operation: str,
    input_shapes: List[List[int]],
    cache_dir: Optional[str] = None,
    max_examples: int = 2
) -> List[Dict[str, Any]]:
    """Get relevant successful examples for the given operation."""
    kb = load_knowledge_base(cache_dir)

    relevant_examples = []

    for kernel_entry in kb["successful_kernels"]:
        patterns = kernel_entry["patterns"]

        # Calculate relevance score
        relevance_score = 0

        # Exact operation match gets high score
        if patterns["operation"] == operation:
            relevance_score += 10

        # Similar operation (e.g., conv3d_d1_p1_k3_l0 vs conv3d_d2_p2_k3_l2)
        elif operation.split("_")[0] == patterns["operation"].split("_")[0]:
            relevance_score += 7

        # Operation type match (e.g., both convolutions)
        elif patterns.get("code_features", {}).get("operation_type") == classify_operation_from_name(operation):
            relevance_score += 5

        # Shape similarity
        if patterns.get("input_shapes") and input_shapes:
            if len(patterns["input_shapes"]) == len(input_shapes):
                relevance_score += 2
                # Similar dimensions
                if patterns["input_shapes"][0] == input_shapes[0]:
                    relevance_score += 5

        # High validation score
        validation_score = patterns.get("validation_score", 0)
        if validation_score >= 10:  # Perfect score
            relevance_score += 3

        if relevance_score > 0:
            relevant_examples.append({
                "relevance_score": relevance_score,
                "operation": patterns["operation"],
                "llm_response": {"extracted_kernel": kernel_entry["kernel_source"]},
                "patterns": patterns
            })

    # Sort by relevance and return top examples
    relevant_examples.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant_examples[:max_examples]


def classify_operation_from_name(operation_name: str) -> str:
    """Classify operation type from its name."""
    name_lower = operation_name.lower()

    if "conv" in name_lower:
        return "convolution"
    elif "relu" in name_lower:
        return "activation"
    elif "matmul" in name_lower or "matrix" in name_lower:
        return "matrix_operation"
    elif "add" in name_lower or "sub" in name_lower or "mul" in name_lower:
        return "elementwise"
    else:
        return "unknown"


def get_knowledge_base_stats(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics about the knowledge base."""
    kb = load_knowledge_base(cache_dir)

    stats = {
        "total_successful_kernels": len(kb["successful_kernels"]),
        "operations_learned": len(kb["operation_patterns"]),
        "operation_breakdown": {}
    }

    # Count kernels per operation type
    for operation, patterns in kb["operation_patterns"].items():
        stats["operation_breakdown"][operation] = {
            "count": patterns["count"],
            "avg_tokens": patterns.get("avg_tokens", 0)
        }

    return stats