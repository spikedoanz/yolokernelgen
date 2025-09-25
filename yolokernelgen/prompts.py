"""LLM prompt generation for kernel translation."""

import re
from typing import Dict, List, Optional, Any


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