# YoloKernelGen

Functional-style framework for rejection sampling WebGPU kernels from PyTorch
operations using LLMs with automated numerical validation.

## Overview

YoloKernelGen translates PyTorch operations into  WebGPU kernels using GPT-4o.
Each generated kernel is numerically validated against PyTorch ground truth to
ensure correctness.

## Quick Start

### Prerequisites

```bash
# Install dependencies
uv pip install -e .

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### Basic Usage

```python
from yolokernelgen import generate_kernel
import torch.nn.functional as F

# Generate a ReLU kernel
kernel_path = generate_kernel(
    torch_source=\"\"\"def relu(x):
        return torch.nn.functional.relu(x)\"\"\",
    operation=\"relu\",
    input_shapes=[[4, 8, 16, 16]],
    output_shapes=[[4, 8, 16, 16]],
    torch_fn=lambda x: F.relu(x)  # For validation
)

print(f"Generated kernel: {kernel_path}")
```

## Examples (Progressive Complexity)

```bash
# 1. Simple element-wise operations
python examples/example_01_simple_operations.py

# 2. Matrix and tensor operations
python examples/example_02_tensor_operations.py

# 3. Neural network convolutions
python examples/example_03_convolutions.py

# 4. Kernel fusion and optimization
python examples/example_04_fusion_optimization.py
```

## Project Structure

```
yolokernelgen/
├── yolokernelgen/          # Core framework (functional modules)
│   ├── generation.py       # Main generation pipeline
│   ├── validation.py       # PyTorch validation suite
│   ├── storage.py          # Kernel caching system
│   ├── prompts.py         # LLM prompt engineering
│   ├── webgpu_executor.py # WebGPU kernel execution
│   ├── naming.py          # Deterministic filename generation
│   └── config.py          # Configuration management
├── examples/              # Progressive learning examples
│   ├── example_01_simple_operations.py
│   ├── example_02_tensor_operations.py
│   ├── example_03_convolutions.py
│   └── example_04_fusion_optimization.py
├── tests/                 # Test suite
└── docs/                  # Documentation and utilities
```

### Validation Process
Each kernel undergoes rigorous testing:
- **5 Random tests**: Uniform, normal, sparse distributions
- **5 Edge cases**: Zeros, ones, extreme values, patterns
- **Mathematical verification**: Max difference < 1e-5 tolerance
- **Only validated kernels** get `c_` (correct) prefix

## Cache System

Generated kernels are cached with deterministic naming:

```
.cache/yolokernelgen/generated/
├── c_relu_i4x8x16x16_o4x8x16x16_h12c3b1c0.json             # Validated
├── c_conv2d_i1x3x8x8_o1x4x6x6_s1x1_p0x0_hf15d83cf.json     # Validated
└── r_matmul_i8x4_o8x6_h236222a9.json                       # Failed validation
```

**Filename format**: `{status}_{operation}_i{input_shape}_o{output_shape}_{params}_h{hash}.json`

- `c_` = Correct (validated)
- `r_` = Rejected (failed validation)

## Configuration

Customize generation behavior:

```python
from yolokernelgen.config import default_config

config = default_config()
config["max_samples"] = 5          # Retry attempts
config["llm"]["temperature"] = 0.5  # More deterministic
config["llm"]["max_tokens"] = 6000  # Longer kernels
config["validation"]["tolerance"]["float32"] = 1e-6  # Stricter validation

generate_kernel(..., config=config)
```

## Advanced Features

### Kernel Fusion
Combine operations for performance:

```python
torch_source = \"\"\"def conv_relu_fused(x, weight, bias):
    conv_out = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)
    return torch.nn.functional.relu(conv_out)\"\"\"
```

### Custom Validation
Provide your own PyTorch reference:

```python
def torch_fn(x, weight, bias):
    return F.relu(F.conv2d(x, weight, bias, stride=1, padding=1))

generate_kernel(..., torch_fn=torch_fn)
```

## Testing

Run the test suite:

```bash
# Basic functionality
python tests/test_llm_only.py

# Full validation pipeline
python tests/test_full_validation.py

# Complex operations
python tests/test_conv2d.py
python tests/test_enhanced.py
```

## API Reference

### Core Functions

```python
generate_kernel(
    torch_source: str,           # PyTorch function source
    operation: str,              # Operation name
    input_shapes: List[List[int]], # Input tensor shapes
    output_shapes: List[List[int]], # Output tensor shapes
    param_shapes: Optional[Dict] = None, # Parameter shapes (weights, bias)
    hyperparameters: Optional[Dict] = None, # Op hyperparams (stride, padding)
    torch_fn: Optional[Callable] = None, # PyTorch function for validation
    config: Optional[Dict] = None, # Custom configuration
    force_regenerate: bool = False # Skip cache lookup
) -> Path
```

```python
load_kernel(filepath: Path) -> Dict[str, Any]  # Load cached kernel
list_kernels(status_filter: str = None) -> List[Dict]  # List cached kernels
execute_kernel(kernel_source: str, inputs: List[np.ndarray]) -> np.ndarray  # Run kernel
