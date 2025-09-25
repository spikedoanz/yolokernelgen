"""Enhanced prompt generation addressing Opus feedback."""

from typing import Dict, List, Any, Optional


def build_enhanced_system_prompt() -> str:
    """Enhanced system prompt with better guidance."""
    return """You are an expert at translating PyTorch operations to WebGPU compute shaders using WGSL (WebGPU Shading Language).

Key Requirements:
- Generate a single, complete kernel that exactly replicates PyTorch behavior
- Use WGSL syntax (not GLSL or HLSL)
- Handle tensor memory layout correctly (NCHW for convolutions)
- Implement proper bounds checking and memory safety
- Use einsum notation in comments where helpful
- Follow WebGPU best practices for memory coalescing"""


def build_enhanced_user_prompt(
    torch_source: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    example_kernel: Optional[str] = None
) -> str:
    """Enhanced user prompt with better layout documentation."""

    # Calculate total elements
    total_elements = 1
    for dim in output_shapes[0]:
        total_elements *= dim

    # Build buffer binding documentation
    binding_docs = []
    binding_idx = 0

    # Input tensors
    for i, shape in enumerate(input_shapes):
        binding_docs.append(f"@binding({binding_idx}): input tensor {i} - shape {shape}")
        binding_idx += 1

    # Parameter tensors
    if param_shapes:
        for name, shape in param_shapes.items():
            binding_docs.append(f"@binding({binding_idx}): {name} tensor - shape {shape}")
            binding_idx += 1

    # Output tensor (always last)
    binding_docs.append(f"@binding({binding_idx}): output tensor - shape {output_shapes[0]}")

    prompt = f"""Translate this PyTorch operation to WebGPU WGSL:

```python
{torch_source}
```

**Tensor Information:**
- Input shapes: {input_shapes}
- Output shapes: {output_shapes}
- Total output elements: {total_elements}"""

    if param_shapes:
        prompt += f"\n- Parameter shapes: {param_shapes}"

    if hyperparameters:
        prompt += f"\n- Hyperparameters: {hyperparameters}"

    prompt += f"""

**Memory Layout (Critical):**
- Tensors use C-contiguous (row-major) flattening
- For NCHW tensors: flat_index = n*C*H*W + c*H*W + h*W + w
- For matrices: flat_index = row*width + col
- Last dimension changes fastest in memory

**Buffer Bindings:**
{chr(10).join(binding_docs)}"""

    if example_kernel:
        prompt += f"""

**Example WGSL structure:**
```wgsl
{example_kernel}
```"""

    prompt += f"""

**Requirements:**
- Generate complete, valid WGSL compute shader
- Use static shapes hardcoded as constants
- Set TOTAL_ELEMENTS constant to {total_elements}u
- Include proper @workgroup_size (typically 256)
- Use global_invocation_id.x for thread indexing
- Include bounds checking: if (index >= TOTAL_ELEMENTS) return;
- Use var<storage, read> for inputs, var<storage, read_write> for output
- Handle tensor indexing correctly for the operation's dimensionality
- For convolutions: implement proper spatial indexing with stride/padding
- For matrix ops: implement proper row/column indexing
- Include comments explaining index calculations

**WGSL Syntax Reminders:**
- Use 'let' for immutable values, 'var' for mutable
- Loop syntax: for (var i: u32 = 0u; i < N; i = i + 1u)
- Array indexing: array_name[index]
- Arithmetic: standard operators (+, -, *, /, %)

Return ONLY the WGSL kernel code in ```wgsl ... ``` tags."""

    return prompt


def get_enhanced_example_kernels() -> Dict[str, str]:
    """Enhanced example kernels with better documentation."""
    return {
        "conv2d": """@group(0) @binding(0)
var<storage, read> input: array<f32>;  // Input tensor [N, C_in, H, W]

@group(0) @binding(1)
var<storage, read> weight: array<f32>; // Weight tensor [C_out, C_in, K_h, K_w]

@group(0) @binding(2)
var<storage, read> bias: array<f32>;   // Bias tensor [C_out]

@group(0) @binding(3)
var<storage, read_write> output: array<f32>; // Output tensor [N, C_out, H_out, W_out]

// Static shape constants
const BATCH_SIZE: u32 = 1u;
const IN_CHANNELS: u32 = 3u;
const OUT_CHANNELS: u32 = 16u;
const INPUT_HEIGHT: u32 = 32u;
const INPUT_WIDTH: u32 = 32u;
const KERNEL_HEIGHT: u32 = 3u;
const KERNEL_WIDTH: u32 = 3u;
const OUTPUT_HEIGHT: u32 = 30u;  // (32 - 3 + 0*2)/1 + 1 = 30 (no padding, stride=1)
const OUTPUT_WIDTH: u32 = 30u;
const TOTAL_ELEMENTS: u32 = 14400u; // 1*16*30*30

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    // Convert flat index to NCHW coordinates
    // index = n*C*H*W + c*H*W + h*W + w
    let n = index / (OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH);
    let c = (index / (OUTPUT_HEIGHT * OUTPUT_WIDTH)) % OUT_CHANNELS;
    let h = (index / OUTPUT_WIDTH) % OUTPUT_HEIGHT;
    let w = index % OUTPUT_WIDTH;

    // Initialize with bias
    var sum = bias[c];

    // Convolution loop: sum over input channels and kernel
    for (var ic: u32 = 0u; ic < IN_CHANNELS; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < KERNEL_HEIGHT; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < KERNEL_WIDTH; kw = kw + 1u) {
                // Input spatial coordinates
                let ih = h + kh;  // No padding
                let iw = w + kw;  // No padding

                // Calculate flat indices
                let input_idx = (((n * IN_CHANNELS + ic) * INPUT_HEIGHT) + ih) * INPUT_WIDTH + iw;
                let weight_idx = (((c * IN_CHANNELS + ic) * KERNEL_HEIGHT) + kh) * KERNEL_WIDTH + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[index] = sum;
}""",

        "matmul": """@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>; // Shape [M, K]

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>; // Shape [K, N]

@group(0) @binding(2)
var<storage, read_write> output: array<f32>; // Shape [M, N]

const M: u32 = 64u;
const K: u32 = 32u;
const N: u32 = 16u;
const TOTAL_ELEMENTS: u32 = 1024u; // M * N

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= TOTAL_ELEMENTS) {
        return;
    }

    // Convert flat index to 2D coordinates
    let row = index / N;
    let col = index % N;

    // Dot product: sum over K dimension
    var sum = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a_idx = row * K + k;     // A[row, k]
        let b_idx = k * N + col;     // B[k, col]
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }

    output[index] = sum;
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

    // ReLU: max(0, x)
    output[index] = max(0.0, input[index]);
}""",

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
}"""
    }