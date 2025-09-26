"""Memory management utilities for large tensor operations."""

import math
from typing import List, Dict, Any, Tuple


def estimate_memory_usage(
    input_shapes: List[List[int]], 
    output_shapes: List[List[int]],
    param_shapes: Dict[str, List[int]] = None,
    dtype_size: int = 4  # float32 = 4 bytes
) -> Dict[str, Any]:
    """Estimate memory usage for tensor operations."""
    
    # Calculate input memory
    input_memory = 0
    for shape in input_shapes:
        elements = 1
        for dim in shape:
            elements *= dim
        input_memory += elements * dtype_size
    
    # Calculate output memory
    output_memory = 0
    for shape in output_shapes:
        elements = 1
        for dim in shape:
            elements *= dim
        output_memory += elements * dtype_size
    
    # Calculate parameter memory
    param_memory = 0
    if param_shapes:
        for shape in param_shapes.values():
            elements = 1
            for dim in shape:
                elements *= dim
            param_memory += elements * dtype_size
    
    total_memory = input_memory + output_memory + param_memory
    
    return {
        "input_memory_bytes": input_memory,
        "output_memory_bytes": output_memory,
        "param_memory_bytes": param_memory,
        "total_memory_bytes": total_memory,
        "input_memory_mb": input_memory / (1024 * 1024),
        "output_memory_mb": output_memory / (1024 * 1024),
        "param_memory_mb": param_memory / (1024 * 1024),
        "total_memory_mb": total_memory / (1024 * 1024)
    }


def optimize_workgroup_size_for_large_tensors(
    output_shape: List[int],
    operation_type: str = "elementwise"
) -> Tuple[List[int], int]:
    """Optimize workgroup size and dispatch for large tensor operations."""
    
    total_elements = 1
    for dim in output_shape:
        total_elements *= dim
    
    # For 256^3 tensors (~16M elements), we need efficient dispatch
    if total_elements > 10_000_000:  # > 10M elements
        if operation_type == "conv3d" and len(output_shape) >= 5:
            # 3D convolution: use 3D workgroups for spatial dimensions
            # output_shape: [batch, channels, depth, height, width]
            depth = output_shape[-3] if len(output_shape) >= 3 else 1
            height = output_shape[-2] if len(output_shape) >= 2 else 1
            width = output_shape[-1]
            
            # Optimize for GPU architecture (prefer powers of 2, max 1024 total)
            wg_x = min(16, width)
            wg_y = min(16, height) 
            wg_z = min(4, depth)
            
            # Ensure total workgroup size <= 1024
            total_wg = wg_x * wg_y * wg_z
            if total_wg > 1024:
                wg_z = max(1, 1024 // (wg_x * wg_y))
            
            workgroup_size = [wg_x, wg_y, wg_z]
            
            # Calculate dispatch size
            dispatch_x = math.ceil(width / wg_x)
            dispatch_y = math.ceil(height / wg_y)  
            dispatch_z = math.ceil(depth / wg_z)
            dispatch_total = dispatch_x * dispatch_y * dispatch_z
            
        else:
            # 1D dispatch for other operations
            workgroup_size = [256, 1, 1]  # Standard workgroup size
            dispatch_total = math.ceil(total_elements / 256)
    
    elif total_elements > 1_000_000:  # 1M - 10M elements
        if operation_type == "conv3d":
            workgroup_size = [8, 8, 4]
            dispatch_total = math.ceil(total_elements / (8 * 8 * 4))
        else:
            workgroup_size = [256, 1, 1]
            dispatch_total = math.ceil(total_elements / 256)
    
    else:
        # Smaller tensors - standard approach
        workgroup_size = [256, 1, 1]
        dispatch_total = math.ceil(total_elements / 256)
    
    return workgroup_size, dispatch_total


def get_memory_management_hints(memory_estimate: Dict[str, Any]) -> List[str]:
    """Get memory management hints based on estimated usage."""
    
    hints = []
    total_mb = memory_estimate["total_memory_mb"]
    
    if total_mb > 1000:  # > 1GB
        hints.append("Use u32 indices to handle large tensor addressing")
        hints.append("Consider tiled processing for memory-bound operations")
        hints.append("Minimize intermediate value storage in registers")
        hints.append("Use shared memory efficiently for workgroup-level reductions")
    
    if total_mb > 100:  # > 100MB
        hints.append("Ensure coalesced memory access patterns")
        hints.append("Consider reducing workgroup size to fit in cache")
    
    # Check for specific patterns
    param_mb = memory_estimate["param_memory_mb"]
    if param_mb > 50:  # Large kernels like 8x8x8 conv3d
        hints.append("Large kernel detected - cache kernel weights in shared memory")
        hints.append("Consider spatial tiling to reuse kernel weights")
    
    return hints


def check_index_overflow_risk(shapes: List[List[int]]) -> Dict[str, Any]:
    """Check if tensor shapes risk 32-bit index overflow."""
    
    max_elements = 0
    risky_shapes = []
    
    for i, shape in enumerate(shapes):
        elements = 1
        for dim in shape:
            elements *= dim
        
        if elements > max_elements:
            max_elements = elements
        
        # Check if approaching 32-bit limit (2^32 = ~4.3B)
        if elements > 2**30:  # > 1B elements (getting close to limit)
            risky_shapes.append((i, shape, elements))
    
    # 32-bit signed int max is 2^31 - 1 = 2,147,483,647
    # 32-bit unsigned int max is 2^32 - 1 = 4,294,967,295
    overflow_risk = max_elements > 2**31
    
    return {
        "max_elements": max_elements,
        "overflow_risk": overflow_risk,
        "risky_shapes": risky_shapes,
        "recommendation": "Use u32 indices" if overflow_risk else "i32 indices OK"
    }


def generate_memory_management_code(
    output_shape: List[int],
    operation_type: str = "elementwise"
) -> str:
    """Generate WGSL code snippets for efficient memory management."""
    
    total_elements = 1
    for dim in output_shape:
        total_elements *= dim
    
    # Get optimized workgroup configuration
    workgroup_size, dispatch_total = optimize_workgroup_size_for_large_tensors(
        output_shape, operation_type
    )
    
    # Generate constants
    code_lines = [
        f"// Memory management for {total_elements:,} elements",
        f"const TOTAL_ELEMENTS: u32 = {total_elements}u;",
    ]
    
    # Add shape constants for multi-dimensional tensors
    if len(output_shape) > 1:
        for i, dim in enumerate(output_shape):
            dim_name = ["BATCH_SIZE", "CHANNELS", "DEPTH", "HEIGHT", "WIDTH"][i] if i < 5 else f"DIM_{i}"
            code_lines.append(f"const {dim_name}: u32 = {dim}u;")
    
    # Add workgroup size
    if len(workgroup_size) == 3 and workgroup_size[1] > 1:
        code_lines.append(f"@compute @workgroup_size({workgroup_size[0]}, {workgroup_size[1]}, {workgroup_size[2]})")
    else:
        code_lines.append(f"@compute @workgroup_size({workgroup_size[0]})")
    
    # Add bounds checking template
    if operation_type == "conv3d" and len(output_shape) >= 3:
        code_lines.extend([
            "",
            "// 3D spatial bounds checking template:",
            "// let out_d = global_id.z;",
            "// let out_h = global_id.y;", 
            "// let out_w = global_id.x;",
            "// if (out_d >= DEPTH || out_h >= HEIGHT || out_w >= WIDTH) { return; }"
        ])
    else:
        code_lines.extend([
            "",
            "// Standard bounds checking template:",
            "// let index = global_id.x;",
            "// if (index >= TOTAL_ELEMENTS) { return; }"
        ])
    
    return "\n".join(code_lines)