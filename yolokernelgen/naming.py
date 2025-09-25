"""Kernel naming and filename utilities."""

import hashlib
from typing import Dict, List, Any, Optional


def shape_to_str(shape: List[int]) -> str:
    """Convert shape list to string representation."""
    return "x".join(map(str, shape))


def params_to_str(operation: str, params: Dict[str, Any]) -> str:
    """Convert operation parameters to string representation."""
    if operation in ["conv2d", "conv3d"]:
        parts = []
        if "kernel" in params:
            parts.append(f"k{shape_to_str(params['kernel'])}")
        if "stride" in params:
            parts.append(f"s{shape_to_str(params['stride'])}")
        if "padding" in params:
            parts.append(f"p{shape_to_str(params['padding'])}")
        if "dilation" in params:
            parts.append(f"d{shape_to_str(params['dilation'])}")
        if "groups" in params:
            parts.append(f"g{params['groups']}")
        return "_".join(parts)

    elif operation in ["maxpool2d", "maxpool3d", "avgpool2d", "avgpool3d"]:
        parts = []
        if "kernel" in params:
            parts.append(f"k{shape_to_str(params['kernel'])}")
        if "stride" in params:
            parts.append(f"s{shape_to_str(params['stride'])}")
        if "padding" in params:
            parts.append(f"p{shape_to_str(params['padding'])}")
        return "_".join(parts)

    elif operation in ["batchnorm2d", "batchnorm3d", "layernorm", "groupnorm"]:
        parts = []
        if "epsilon" in params:
            parts.append(f"eps{params['epsilon']}")
        if "momentum" in params:
            parts.append(f"m{params['momentum']}")
        if "num_groups" in params:
            parts.append(f"g{params['num_groups']}")
        return "_".join(parts)

    return ""


def compute_hash(torch_source: str) -> str:
    """Compute SHA256 hash of torch source code."""
    return hashlib.sha256(torch_source.encode()).hexdigest()[:8]


def generate_filename(
    operation: str,
    input_shape: List[int],
    output_shape: List[int],
    torch_source: str,
    params: Optional[Dict[str, Any]] = None,
    status: str = "c"
) -> str:
    """Generate kernel filename from components."""
    parts = [
        f"{status}",
        operation,
        f"i{shape_to_str(input_shape)}",
        f"o{shape_to_str(output_shape)}"
    ]

    if params:
        param_str = params_to_str(operation, params)
        if param_str:
            parts.append(param_str)

    hash_str = compute_hash(torch_source)
    parts.append(f"h{hash_str}")

    return "_".join(parts) + ".json"


def parse_filename(filename: str) -> Dict[str, Any]:
    """Parse kernel filename to extract components."""
    if not filename.endswith(".json"):
        raise ValueError(f"Invalid kernel filename: {filename}")

    base = filename[:-5]
    parts = base.split("_")

    result = {
        "status": "correct" if parts[0] == "c" else "rejected",
        "operation": parts[1] if len(parts) > 1 else None,
        "input_shape": None,
        "output_shape": None,
        "params": {},
        "hash": None
    }

    for part in parts[2:]:
        if part.startswith("i"):
            result["input_shape"] = list(map(int, part[1:].split("x")))
        elif part.startswith("o"):
            result["output_shape"] = list(map(int, part[1:].split("x")))
        elif part.startswith("h"):
            result["hash"] = part[1:]
        elif part.startswith("k"):
            result["params"]["kernel"] = list(map(int, part[1:].split("x")))
        elif part.startswith("s"):
            result["params"]["stride"] = list(map(int, part[1:].split("x")))
        elif part.startswith("p"):
            result["params"]["padding"] = list(map(int, part[1:].split("x")))
        elif part.startswith("d"):
            result["params"]["dilation"] = list(map(int, part[1:].split("x")))
        elif part.startswith("g"):
            try:
                result["params"]["groups"] = int(part[1:])
            except ValueError:
                pass
        elif part.startswith("eps"):
            result["params"]["epsilon"] = float(part[3:])
        elif part.startswith("m"):
            try:
                result["params"]["momentum"] = float(part[1:])
            except ValueError:
                pass

    return result