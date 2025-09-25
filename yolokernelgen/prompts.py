"""LLM prompt generation for kernel translation."""

import re
from typing import Dict, List, Optional


def build_system_prompt() -> str:
    """Build system prompt for LLM."""
    return """You are an expert at translating PyTorch operations to WebGPU compute shaders using WGSL (WebGPU Shading Language).
Generate a single, complete kernel that exactly replicates the PyTorch behavior.
Use einsum notation in comments where applicable.
Follow WebGPU/WGSL best practices and ensure memory coalescing where possible.
IMPORTANT: Generate valid WGSL syntax, not GLSL or HLSL."""


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

    prompt += f"""

Requirements:
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


def get_example_kernels() -> Dict[str, str]:
    """Get example kernels for different operations."""
    return {
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

        "relu": """@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const TOTAL_ELEMENTS: u32 = 65536u;  // Example: 8*16*32*32

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
    }


def format_shapes_for_prompt(shapes: List[List[int]]) -> str:
    """Format shapes for readable prompt."""
    return ", ".join([f"[{', '.join(map(str, shape))}]" for shape in shapes])