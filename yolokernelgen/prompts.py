"""LLM prompt generation for kernel translation."""

import re
from typing import Dict, List, Optional, Any
from .memory_utils import estimate_memory_usage, get_memory_management_hints, generate_memory_management_code


def build_system_prompt(complexity_level: str = "basic") -> str:
    """Build system prompt for LLM with complexity-specific guidance."""
    
    base_prompt = """You are an expert at translating PyTorch operations to WebGPU compute shaders using WGSL (WebGPU Shading Language).
Generate a single, complete kernel that exactly replicates the PyTorch behavior.
Use einsum notation in comments where applicable.
Follow WebGPU/WGSL best practices and ensure memory coalescing where possible.
IMPORTANT: Generate valid WGSL syntax, not GLSL or HLSL."""

    if complexity_level == "conv3d":
        return base_prompt + """

=== 3D CONVOLUTION EXPERTISE ===
You are specifically optimized for 3D convolution operations. Pay special attention to:

INDEXING PATTERNS:
- 5D tensor indexing: [batch, channel, depth, height, width] (NCDHW format)
- Kernel indexing: [out_channels, in_channels, kd, kh, kw]
- Output position calculation: out[b][oc][od][oh][ow]
- Input sampling with dilation: in[b][ic][od*stride_d + kd*dilation_d - pad_d][...]

NUMERICAL STABILITY:
- Use proper bounds checking: if (input_d >= 0 && input_d < depth_size)
- Initialize accumulation variables to 0.0 before convolution loops
- Ensure all array accesses are within bounds
- Handle padding regions by skipping out-of-bounds reads

MEMORY LAYOUT:
- WebGPU uses row-major layout: rightmost dimension is most contiguous
- Linear index calculation: idx = ((((b*C + c)*D + d)*H + h)*W + w)
- For 256^3 tensors, ensure 32-bit index arithmetic doesn't overflow

OPTIMIZATION FOR LARGE TENSORS:
- For 256^3 inputs, total elements = 16,777,216+ per channel
- Use u32 for all indices and size constants
- Consider workgroup.xy dispatch for spatial dimensions when possible
- Minimize register pressure in inner loops"""

    elif complexity_level == "aggregation":
        return base_prompt + """

=== LARGE TENSOR AGGREGATION EXPERTISE ===
You specialize in operations on very large tensors that require aggregation patterns:

CHUNKED PROCESSING:
- Break large reductions into smaller chunks
- Use shared memory for workgroup-level reductions
- Implement hierarchical reduction patterns

MEMORY PATTERNS:
- Coalesced memory access for large tensors
- Minimize global memory bandwidth
- Use appropriate data types (f32 vs f16) based on precision needs"""

    elif complexity_level == "simple":
        return base_prompt + """

=== SIMPLE OPERATION EXPERTISE ===
Focus on generating clean, correct kernels for basic operations:

BEST PRACTICES:
- Clear, readable indexing patterns
- Proper bounds checking
- Efficient memory access patterns
- Use these as building blocks for more complex operations"""

    return base_prompt


def build_user_prompt(
    torch_source: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    example_kernel: Optional[str] = None
) -> str:
    """Build user prompt for kernel generation."""

    # Calculate total elements for the kernel
    total_elements = 1
    for dim in output_shapes[0]:
        total_elements *= dim

    prompt = f"""Translate this PyTorch operation to WebGPU WGSL:

```python
{torch_source}
```

Input shapes: {input_shapes}
Output shapes: {output_shapes}
Total output elements: {total_elements}"""

    if param_shapes:
        prompt += f"\nParameter shapes: {param_shapes}"

    if example_kernel:
        prompt += f"""

Example WGSL kernel structure for reference:
```wgsl
{example_kernel}
```"""

    # Add memory management guidance for large tensors
    memory_estimate = estimate_memory_usage(input_shapes, output_shapes, param_shapes)
    memory_hints = get_memory_management_hints(memory_estimate)
    
    if memory_estimate["total_memory_mb"] > 50:  # Large tensor operation
        prompt += f"\n\n=== MEMORY MANAGEMENT GUIDANCE ==="
        prompt += f"\nEstimated memory usage: {memory_estimate['total_memory_mb']:.1f} MB"
        
        if memory_hints:
            prompt += f"\nSpecific memory optimization hints:"
            for hint in memory_hints:
                prompt += f"\n- {hint}"
        
        # Add memory management code template
        memory_code = generate_memory_management_code(output_shapes[0])
        if memory_code:
            prompt += f"\n\nMemory-optimized template:\n```\n{memory_code}\n```"

    prompt += f"""

=== REQUIREMENTS ===
- Generate a complete, valid WGSL compute shader
- Use static shapes hardcoded as constants
- Total elements constant should be {total_elements}u
- Match PyTorch's memory layout (row-major, last dimension is contiguous)
- Include proper workgroup size (optimized for tensor size)
- Handle workgroup dispatch for {total_elements} elements
- Use global_invocation_id for thread indexing
- Include bounds checking (index < total_elements)
- Use @group(0) @binding(N) for buffer bindings
- Input buffers should be var<storage, read>
- Output buffer should be var<storage, read_write>
- Use array<f32> for buffer types"""

    if memory_estimate["total_memory_mb"] > 100:
        prompt += f"""
- CRITICAL for large tensors: Use u32 for all indices and size constants
- Ensure proper bounds checking to prevent out-of-bounds access
- Consider memory coalescing for optimal performance"""

    prompt += f"""

Return ONLY the WGSL kernel code enclosed in ```wgsl ... ``` tags. Do not include any explanation."""

    return prompt


def extract_kernel_from_response(llm_response: str) -> Optional[str]:
    """Extract WGSL kernel from LLM response."""
    # Try to find code blocks with wgsl or webgpu markers
    patterns = [
        r'```wgsl\n(.*?)```',
        r'```webgpu\n(.*?)```',
        r'```glsl\n(.*?)```',  # Sometimes LLMs confuse WGSL with GLSL
        r'```\n(@group.*?)```'  # Fallback: look for @group directive
    ]

    for pattern in patterns:
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: try to find anything that looks like a WGSL kernel
    if '@group' in llm_response and '@compute' in llm_response:
        # Extract from first @group to end of likely kernel
        start = llm_response.index('@group')
        # Find the end - usually after the last closing brace
        kernel_candidate = llm_response[start:]

        # Find balanced braces to get complete function
        brace_count = 0
        end_pos = 0
        in_kernel = False

        for i, char in enumerate(kernel_candidate):
            if char == '{':
                brace_count += 1
                in_kernel = True
            elif char == '}':
                brace_count -= 1
                if in_kernel and brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos > 0:
            return kernel_candidate[:end_pos].strip()

    return None


def get_progressive_example_kernels() -> Dict[str, Dict[str, str]]:
    """Get progressive example kernels organized by complexity level."""
    return {
        "level_1_elementwise": {
            "add": """@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 65536u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    output[index] = input_a[index] + input_b[index];
}""",

            "relu": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 65536u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    output[index] = max(0.0, input[index]);
}"""
        },

        "level_2_reductions": {
            "sum": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Sum over last dimension: [B, C, H, W] -> [B, C, H]
const BATCH_SIZE: u32 = 2u;
const CHANNELS: u32 = 4u;
const HEIGHT: u32 = 32u;
const WIDTH: u32 = 32u;
const TOTAL_OUTPUT: u32 = 256u; // B*C*H

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    
    if (out_idx >= TOTAL_OUTPUT) {
        return;
    }
    
    // Decode output coordinates
    let b = out_idx / (CHANNELS * HEIGHT);
    let remainder = out_idx % (CHANNELS * HEIGHT);
    let c = remainder / HEIGHT;
    let h = remainder % HEIGHT;
    
    var sum_val: f32 = 0.0;
    for (var w: u32 = 0u; w < WIDTH; w = w + 1u) {
        let in_idx = ((b * CHANNELS + c) * HEIGHT + h) * WIDTH + w;
        sum_val = sum_val + input[in_idx];
    }
    
    output[out_idx] = sum_val;
}""",

            "mean": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const BATCH_SIZE: u32 = 2u;
const CHANNELS: u32 = 4u;
const SIZE: u32 = 16u;
const REDUCTION_DIM: u32 = 16u;
const TOTAL_OUTPUT: u32 = 128u; // B*C*SIZE

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    
    if (out_idx >= TOTAL_OUTPUT) {
        return;
    }
    
    var sum_val: f32 = 0.0;
    let base_input_idx = out_idx * REDUCTION_DIM;
    
    for (var i: u32 = 0u; i < REDUCTION_DIM; i = i + 1u) {
        sum_val = sum_val + input[base_input_idx + i];
    }
    
    output[out_idx] = sum_val / f32(REDUCTION_DIM);
}"""
        },

        "level_3_matrix": {
            "matmul": """@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

// Matrix multiplication: [M, K] x [K, N] -> [M, N]
const M: u32 = 64u;
const K: u32 = 64u;
const N: u32 = 64u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= M || col >= N) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a_val = input_a[row * K + k];
        let b_val = input_b[k * N + col];
        sum = sum + a_val * b_val;
    }
    
    output[row * N + col] = sum;
}""",

            "norm": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// L2 norm over last dimension
const BATCH_SIZE: u32 = 8u;
const FEATURE_SIZE: u32 = 128u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= BATCH_SIZE) {
        return;
    }
    
    var sum_squares: f32 = 0.0;
    let base_idx = batch_idx * FEATURE_SIZE;
    
    for (var i: u32 = 0u; i < FEATURE_SIZE; i = i + 1u) {
        let val = input[base_idx + i];
        sum_squares = sum_squares + val * val;
    }
    
    output[batch_idx] = sqrt(sum_squares);
}"""
        },

        "level_4_aggregation": {
            "large_tensor_sum": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Chunk-wise sum for very large tensors
const CHUNK_SIZE: u32 = 1024u;
const TOTAL_CHUNKS: u32 = 16384u; // For 16M+ elements

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk_idx = global_id.x;
    
    if (chunk_idx >= TOTAL_CHUNKS) {
        return;
    }
    
    var chunk_sum: f32 = 0.0;
    let start_idx = chunk_idx * CHUNK_SIZE;
    let end_idx = min(start_idx + CHUNK_SIZE, TOTAL_CHUNKS * CHUNK_SIZE);
    
    for (var i: u32 = start_idx; i < end_idx; i = i + 1u) {
        chunk_sum = chunk_sum + input[i];
    }
    
    output[chunk_idx] = chunk_sum;
}"""
        },

        "level_5_conv2d": {
            "conv2d_simple": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read> bias: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

// Conv2D: [N, C_in, H, W] -> [N, C_out, H_out, W_out]
const BATCH_SIZE: u32 = 1u;
const IN_CHANNELS: u32 = 3u;
const OUT_CHANNELS: u32 = 16u;
const INPUT_HEIGHT: u32 = 32u;
const INPUT_WIDTH: u32 = 32u;
const OUTPUT_HEIGHT: u32 = 32u;
const OUTPUT_WIDTH: u32 = 32u;
const KERNEL_SIZE: u32 = 3u;
const PADDING: u32 = 1u;
const STRIDE: u32 = 1u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_h = global_id.y;
    let out_w = global_id.x;
    
    if (out_h >= OUTPUT_HEIGHT || out_w >= OUTPUT_WIDTH) {
        return;
    }
    
    for (var b: u32 = 0u; b < BATCH_SIZE; b = b + 1u) {
        for (var oc: u32 = 0u; oc < OUT_CHANNELS; oc = oc + 1u) {
            var conv_sum: f32 = 0.0;
            
            // Convolution over input channels and kernel
            for (var ic: u32 = 0u; ic < IN_CHANNELS; ic = ic + 1u) {
                for (var kh: u32 = 0u; kh < KERNEL_SIZE; kh = kh + 1u) {
                    for (var kw: u32 = 0u; kw < KERNEL_SIZE; kw = kw + 1u) {
                        let in_h = out_h * STRIDE + kh - PADDING;
                        let in_w = out_w * STRIDE + kw - PADDING;
                        
                        if (in_h >= 0u && in_h < INPUT_HEIGHT && in_w >= 0u && in_w < INPUT_WIDTH) {
                            let input_idx = ((b * IN_CHANNELS + ic) * INPUT_HEIGHT + in_h) * INPUT_WIDTH + in_w;
                            let weight_idx = ((oc * IN_CHANNELS + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw;
                            conv_sum = conv_sum + input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            let output_idx = ((b * OUT_CHANNELS + oc) * OUTPUT_HEIGHT + out_h) * OUTPUT_WIDTH + out_w;
            output[output_idx] = conv_sum + bias[oc];
        }
    }
}"""
        },

        "level_6_conv3d": {
            "conv3d_template": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read> bias: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

// Conv3D: [N, C_in, D, H, W] -> [N, C_out, D_out, H_out, W_out]
const BATCH_SIZE: u32 = 1u;
const IN_CHANNELS: u32 = 4u;
const OUT_CHANNELS: u32 = 4u;
const INPUT_DEPTH: u32 = 16u;
const INPUT_HEIGHT: u32 = 16u;
const INPUT_WIDTH: u32 = 16u;
const OUTPUT_DEPTH: u32 = 16u;
const OUTPUT_HEIGHT: u32 = 16u;
const OUTPUT_WIDTH: u32 = 16u;
const KERNEL_DEPTH: u32 = 3u;
const KERNEL_HEIGHT: u32 = 3u;
const KERNEL_WIDTH: u32 = 3u;
const PADDING_D: u32 = 1u;
const PADDING_H: u32 = 1u;
const PADDING_W: u32 = 1u;
const STRIDE_D: u32 = 1u;
const STRIDE_H: u32 = 1u;
const STRIDE_W: u32 = 1u;
const DILATION_D: u32 = 1u;
const DILATION_H: u32 = 1u;
const DILATION_W: u32 = 1u;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_d = global_id.z;
    let out_h = global_id.y;
    let out_w = global_id.x;
    
    if (out_d >= OUTPUT_DEPTH || out_h >= OUTPUT_HEIGHT || out_w >= OUTPUT_WIDTH) {
        return;
    }
    
    for (var b: u32 = 0u; b < BATCH_SIZE; b = b + 1u) {
        for (var oc: u32 = 0u; oc < OUT_CHANNELS; oc = oc + 1u) {
            var conv_sum: f32 = 0.0;
            
            // 3D convolution over input channels and 3D kernel
            for (var ic: u32 = 0u; ic < IN_CHANNELS; ic = ic + 1u) {
                for (var kd: u32 = 0u; kd < KERNEL_DEPTH; kd = kd + 1u) {
                    for (var kh: u32 = 0u; kh < KERNEL_HEIGHT; kh = kh + 1u) {
                        for (var kw: u32 = 0u; kw < KERNEL_WIDTH; kw = kw + 1u) {
                            let in_d = out_d * STRIDE_D + kd * DILATION_D - PADDING_D;
                            let in_h = out_h * STRIDE_H + kh * DILATION_H - PADDING_H;
                            let in_w = out_w * STRIDE_W + kw * DILATION_W - PADDING_W;
                            
                            if (in_d >= 0u && in_d < INPUT_DEPTH && 
                                in_h >= 0u && in_h < INPUT_HEIGHT && 
                                in_w >= 0u && in_w < INPUT_WIDTH) {
                                
                                let input_idx = ((((b * IN_CHANNELS + ic) * INPUT_DEPTH + in_d) * INPUT_HEIGHT + in_h) * INPUT_WIDTH + in_w);
                                let weight_idx = ((((oc * IN_CHANNELS + ic) * KERNEL_DEPTH + kd) * KERNEL_HEIGHT + kh) * KERNEL_WIDTH + kw);
                                conv_sum = conv_sum + input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            
            let output_idx = ((((b * OUT_CHANNELS + oc) * OUTPUT_DEPTH + out_d) * OUTPUT_HEIGHT + out_h) * OUTPUT_WIDTH + out_w);
            output[output_idx] = conv_sum + bias[oc];
        }
    }
}"""
        }
    }


def get_example_kernels() -> Dict[str, str]:
    """Get example kernels for different operations - legacy compatibility."""
    progressive = get_progressive_example_kernels()
    
    # Flatten progressive examples for backward compatibility
    flattened = {}
    for level_dict in progressive.values():
        flattened.update(level_dict)
    
    # Add legacy examples
    flattened.update({
        "add_one": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 96u;  // 2*3*4*4

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    // Add 1.0 to each element
    output[index] = input[index] + 1.0;
}""",

        "simple": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 1024u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    // Simple operation
    output[index] = input[index];
}"""
    })
    
    return flattened


def determine_complexity_level(operation: str, input_shapes: List[List[int]], torch_source: str) -> str:
    """Determine complexity level for operation-specific guidance."""
    
    # Check total tensor size for large tensor handling
    total_elements = 1
    for shape in input_shapes:
        shape_elements = 1
        for dim in shape:
            shape_elements *= dim
        total_elements = max(total_elements, shape_elements)
    
    # Very large tensors (256^3 = ~16M elements)
    if total_elements > 10_000_000:
        if "conv3d" in operation.lower() or "conv3d" in torch_source.lower():
            return "conv3d"
        else:
            return "aggregation"
    
    # 3D convolution patterns
    if "conv3d" in operation.lower() or "conv3d" in torch_source.lower():
        return "conv3d"
    
    # Aggregation operations
    if any(keyword in operation.lower() for keyword in ["sum", "mean", "reduce", "norm", "aggregate"]):
        return "aggregation"
    
    # Matrix operations
    if any(keyword in operation.lower() for keyword in ["matmul", "mm", "bmm", "dot"]):
        return "matrix"
    
    # Convolution operations
    if "conv" in operation.lower() or "conv" in torch_source.lower():
        return "conv2d"
    
    # Simple operations
    if any(keyword in operation.lower() for keyword in ["add", "mul", "relu", "sigmoid", "tanh"]):
        return "simple"
    
    # Default to basic
    return "basic"


def get_sampling_config_for_complexity(complexity_level: str) -> Dict[str, Any]:
    """Get optimized sampling configuration for different complexity levels."""
    
    configs = {
        "simple": {
            "temperature": 0.3,
            "max_tokens": 2000,
            "model": "gpt-4o-mini"  # Cost efficient for simple ops
        },
        "aggregation": {
            "temperature": 0.5,
            "max_tokens": 4000,
            "model": "gpt-4o"
        },
        "matrix": {
            "temperature": 0.4,
            "max_tokens": 3000,
            "model": "gpt-4o"
        },
        "conv2d": {
            "temperature": 0.6,
            "max_tokens": 5000,
            "model": "gpt-4o"
        },
        "conv3d": {
            "temperature": 0.7,
            "max_tokens": 8000,
            "model": "gpt-4o"  # Use most capable model for complex 3D ops
        },
        "basic": {
            "temperature": 0.7,
            "max_tokens": 4000,
            "model": "gpt-4o"
        }
    }
    
    return configs.get(complexity_level, configs["basic"])


def get_example_kernels() -> Dict[str, str]:
    """Get example kernels for different operations - legacy compatibility."""
    progressive = get_progressive_example_kernels()
    
    # Flatten progressive examples for backward compatibility
    flattened = {}
    for level_dict in progressive.values():
        flattened.update(level_dict)
    
    # Add legacy examples
    flattened.update({
        "add_one": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 96u;  // 2*3*4*4

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    // Add 1.0 to each element
    output[index] = input[index] + 1.0;
}""",

        "simple": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 1024u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    // Simple operation
    output[index] = input[index];
}"""
    })
    
    return flattened


def format_shapes_for_prompt(shapes: List[List[int]]) -> str:
    """Format shapes for readable prompt."""
    return ", ".join([f"[{', '.join(map(str, shape))}]" for shape in shapes])


def build_feedback_aware_prompt(
    torch_source: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    attempt_analysis: Optional[Dict[str, Any]] = None,
    success_examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build feedback-aware prompt using previous attempt analysis."""

    # Calculate total elements
    total_elements = 1
    for dim in output_shapes[0]:
        total_elements *= dim

    # Start with base prompt
    prompt = f"""Translate this PyTorch operation to WebGPU WGSL:

```python
{torch_source}
```

Input shapes: {input_shapes}
Output shapes: {output_shapes}
Total output elements: {total_elements}"""

    if param_shapes:
        prompt += f"\nParameter shapes: {param_shapes}"

    if hyperparameters:
        prompt += f"\nHyperparameters: {hyperparameters}"

    # Add feedback from previous attempts
    if attempt_analysis:
        prompt += f"\n\n=== PREVIOUS ATTEMPT ANALYSIS ==="

        attempts = attempt_analysis.get("total_attempts", 0)
        if attempts > 0:
            prompt += f"\nPrevious {attempts} attempts failed. Learn from these issues:"

            # Add common failure patterns
            common_failures = attempt_analysis.get("common_failures", {})
            if common_failures:
                prompt += f"\n\nCommon failure patterns:"
                for error_type, count in common_failures.items():
                    prompt += f"\n- {error_type}: {count} failures"

            # Add specific suggestions
            suggestions = attempt_analysis.get("suggestions", [])
            if suggestions:
                prompt += f"\n\nSpecific guidance for this attempt:"
                for suggestion in suggestions:
                    prompt += f"\n- {suggestion}"

            # Add near-miss information
            near_misses = attempt_analysis.get("near_misses", [])
            if near_misses:
                prompt += f"\n\nPrevious attempts were close to success:"
                for miss in near_misses[:2]:  # Show top 2
                    prompt += f"\n- {miss['passed']}/{miss['total']} tests passed"
                    failure_summary = miss.get("failure_summary", {})
                    if failure_summary:
                        issues = failure_summary.get("common_issues", [])
                        if issues:
                            prompt += f" (failed on: {', '.join(issues)})"

    # Add successful examples
    if success_examples:
        prompt += f"\n\n=== SUCCESSFUL EXAMPLES ==="
        prompt += f"\nHere are similar successful kernels for reference:"

        for i, example in enumerate(success_examples[:2]):  # Limit to 2
            prompt += f"\n\nExample {i+1} - {example.get('operation', 'unknown')}:"
            kernel_source = example.get("llm_response", {}).get("extracted_kernel", "")
            if kernel_source:
                # Truncate if too long
                if len(kernel_source) > 1000:
                    kernel_source = kernel_source[:1000] + "\n// ... (truncated)"
                prompt += f"\n```wgsl\n{kernel_source}\n```"

    # Add operation-specific guidance
    operation = torch_source.lower()
    if "conv" in operation:
        prompt += f"\n\n=== CONVOLUTION-SPECIFIC GUIDANCE ==="
        prompt += f"\nFor convolution operations, pay special attention to:"
        prompt += f"\n- Proper tensor indexing for NCHW layout"
        prompt += f"\n- Bounds checking for padding regions"
        prompt += f"\n- Correct dilation and stride calculations"
        prompt += f"\n- Channel-wise iteration loops"
        if "3d" in operation:
            prompt += f"\n- 3D spatial indexing (depth, height, width)"

    # Standard requirements
    prompt += f"""

=== REQUIREMENTS ===
- Generate a complete, valid WGSL compute shader
- Use static shapes hardcoded as constants
- Total elements constant should be {total_elements}u
- Match PyTorch's memory layout (row-major, last dimension is contiguous)
- Include proper workgroup size (typically 256)
- Handle workgroup dispatch for {total_elements} elements
- Use global_invocation_id for thread indexing
- Include bounds checking (index < total_elements)
- Use @group(0) @binding(N) for buffer bindings
- Input buffers should be var<storage, read>
- Output buffer should be var<storage, read_write>
- Use array<f32> for buffer types

CRITICAL: Learn from the previous failures listed above. Address the specific issues mentioned.

Return ONLY the WGSL kernel code enclosed in ```wgsl ... ``` tags. Do not include any explanation."""

    return prompt