# YoloKernelGen - Functional Style MVP

A functional-style framework for translating PyTorch operations to WebGPU kernels using LLM generation with rigorous validation.

## Structure

Pure functional modules without classes:

- `config.py` - Configuration management
- `naming.py` - Kernel filename generation and parsing
- `storage.py` - Kernel storage and retrieval
- `validation.py` - Test suite generation and validation
- `prompts.py` - LLM prompt building and response parsing
- `webgpu_executor.py` - WebGPU kernel execution
- `generation.py` - Main generation pipeline
- `example_usage.py` - Usage examples

## Key Features

- **Functional design** - Pure functions, immutable data, explicit I/O
- **Deterministic naming** - Kernels named by operation, shapes, and parameters
- **Comprehensive validation** - 10 test cases per kernel (random + edge cases)
- **Caching** - Reuses previously generated kernels
- **WebGPU execution** - Direct kernel execution via pydawn

## Usage

```python
from generation import generate_kernel
import torch.nn.functional as F

# Define PyTorch operation
torch_source = """def relu(x):
    return torch.nn.functional.relu(x)"""

# Generate and validate kernel
kernel_path = generate_kernel(
    torch_source=torch_source,
    operation="relu",
    input_shapes=[[8, 16, 32, 32]],
    output_shapes=[[8, 16, 32, 32]],
    torch_fn=lambda x: F.relu(x)
)
```

## Configuration

Default config in `config.py`:
- Cache directory: `.cache/yolokernelgen/`
- Max generation attempts: 5
- Float32 tolerance: 1e-5
- LLM: GPT-4, temperature 0.7

## File Naming

Pattern: `{status}_{operation}_i{input_shape}_o{output_shape}_{params}_h{hash8}.json`

Example: `c_relu_i8x16x32x32_o8x16x32x32_h1a2b3c4d.json`

## TODO

- [ ] Replace mock LLM with actual API calls
- [ ] Add more operation types
- [ ] Parallel generation support
- [ ] Better error handling
- [ ] Performance benchmarking