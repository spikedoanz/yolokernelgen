# YoloKernelGen Architecture

## Design Philosophy

YoloKernelGen follows **functional programming principles** with a clean separation of concerns:

- **Pure functions** with explicit inputs/outputs
- **Immutable data structures** (dicts, tuples, lists)
- **No hidden state** or global variables
- **Function composition** over inheritance
- **Side effects isolated** to I/O operations

## Module Overview

### Core Framework (`yolokernelgen/`)

```
yolokernelgen/
├── __init__.py           # Public API exports
├── generation.py         # Main generation pipeline
├── validation.py         # PyTorch validation suite
├── storage.py           # Kernel persistence
├── prompts.py           # LLM prompt engineering
├── webgpu_executor.py   # WebGPU kernel execution
├── naming.py            # Deterministic filename generation
└── config.py            # Configuration management
```

### Key Data Flows

#### 1. Kernel Generation Pipeline
```
User Request → Prompt Building → LLM API → WGSL Extraction → Validation → Storage
```

#### 2. Validation Pipeline
```
Test Suite Generation → PyTorch Reference → WebGPU Execution → Comparison → Results
```

#### 3. Caching System
```
Operation Spec → Filename Generation → Cache Lookup → Load/Generate → Store
```

## Functional Design Patterns

### Configuration Management
- **Immutable config objects** passed explicitly
- **Default config** as pure function
- **Config merging** with functional composition

### Error Handling
- **Exceptions for unrecoverable errors**
- **Return values** for expected failures
- **Explicit error types** in validation results

### Data Structures

#### Kernel Data Schema
```python
{
    "uuid": str,
    "timestamp": str,
    "status": "correct" | "rejected",
    "operation": str,
    "torch_source": str,
    "torch_hash": str,
    "llm_request": {...},
    "llm_response": {...},
    "validation": {...},
    "metadata": {...}
}
```

#### Validation Results Schema
```python
{
    "tolerance": float,
    "dtype": str,
    "test_cases": [...],
    "all_passed": bool,
    "num_passed": int,
    "num_total": int
}
```

## LLM Integration Architecture

### Prompt Engineering Strategy
1. **System prompt** sets expertise and constraints
2. **User prompt** provides specific requirements
3. **Examples** guide output format
4. **Requirements** ensure completeness

### Response Processing
1. **Raw response** captured for debugging
2. **WGSL extraction** via regex patterns
3. **Fallback parsing** for edge cases
4. **Usage tracking** for cost analysis

## Validation Architecture

### Test Case Generation
- **Deterministic seeds** for reproducibility
- **Multiple distributions** (uniform, normal, sparse)
- **Edge cases** (zeros, ones, extremes)
- **Shape-aware** generation

### Comparison Strategy
- **Numerical tolerance** (1e-5 for float32)
- **Shape validation** before comparison
- **Element-wise difference** calculation
- **Statistical metrics** (max, mean difference)

## Storage Architecture

### Filename Convention
```
{status}_{operation}_i{input_shape}_o{output_shape}_{params}_h{hash}.json
```

### Cache Strategy
- **Content-based hashing** of PyTorch source
- **Shape and parameter encoding** in filename
- **Status prefix** for quick filtering
- **Atomic writes** to prevent corruption

## WebGPU Integration

### Buffer Management
- **16-byte alignment** for WebGPU requirements
- **Multiple input support** via binding indices
- **Workgroup dispatch limits** handling
- **Memory layout** preservation (NCHW)

### Execution Pipeline
1. **Buffer creation** and data upload
2. **Shader compilation** and pipeline setup
3. **Workgroup dispatch** with bounds checking
4. **Result retrieval** and format conversion

## Extension Points

### Adding New Operations
1. **Torch function** implementation
2. **Parameter shape** specification
3. **Validation function** creation
4. **Example kernel** (optional)

### Custom Validation
- **Test suite** customization
- **Tolerance** adjustment
- **Custom comparison** functions

### LLM Backend
- **Model selection** via config
- **Temperature** and token tuning
- **Alternative providers** support

## Performance Considerations

### Token Efficiency
- **Operation complexity** → token usage
- **Prompt optimization** for clarity
- **Example selection** for guidance

### Cache Efficiency
- **Deterministic naming** prevents duplicates
- **Fast lookup** by filename pattern
- **Atomic operations** prevent race conditions

### Memory Usage
- **Streaming** for large kernels
- **Lazy loading** of cached data
- **Minimal retention** of intermediate results

## Production Deployment

### Integration Patterns
```python
# Batch generation
kernels = batch_generate_kernels(operations, config)

# Pipeline integration
kernel_path = generate_kernel(**op_spec)
kernel_data = load_kernel(kernel_path)
wgsl_code = kernel_data["llm_response"]["extracted_kernel"]
```

### Error Recovery
- **Multiple attempts** with backoff
- **Fallback strategies** for failed generation
- **Graceful degradation** for validation failures

### Monitoring
- **Success rate** tracking
- **Token usage** analysis
- **Performance metrics** collection

This architecture enables **production-scale kernel generation** with mathematical correctness guarantees.