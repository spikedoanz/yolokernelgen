"""Main kernel generation pipeline."""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .config import default_config, load_config
from .naming import generate_filename, compute_hash
from .storage import save_kernel, find_kernel
from .validation import create_test_suite, validate_kernel
from .prompts import build_system_prompt, build_user_prompt, extract_kernel_from_response, get_example_kernels
from .webgpu_executor import execute_kernel


def sample_from_llm(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4000
) -> Dict[str, Any]:
    """Sample from OpenAI LLM."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "raw_completion": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}")


def attempt_generation(
    torch_source: str,
    operation: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    torch_fn: Optional[Callable] = None,
    llm_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Attempt to generate and validate a kernel."""

    if llm_config is None:
        llm_config = default_config()["llm"]

    # Get example kernel if available
    example_kernels = get_example_kernels()
    example_kernel = example_kernels.get(operation, example_kernels.get("add", None))

    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        torch_source,
        input_shapes,
        output_shapes,
        param_shapes,
        example_kernel
    )

    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Sample from LLM
    llm_response = sample_from_llm(
        messages,
        model=llm_config.get("model", "gpt-4o"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4000)
    )

    # Extract kernel from response
    kernel_source = extract_kernel_from_response(llm_response["raw_completion"])

    if kernel_source is None:
        raise ValueError("Failed to extract kernel from LLM response")

    # Prepare kernel data structure
    kernel_data = {
        "operation": operation,
        "torch_source": torch_source,
        "torch_hash": hashlib.sha256(torch_source.encode()).hexdigest(),
        "llm_request": {
            "model": llm_config.get("model", "gpt-4o"),
            "messages": messages,
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 4000),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        },
        "llm_response": {
            "raw_completion": llm_response["raw_completion"],
            "extracted_kernel": kernel_source,
            "extraction_method": "regex",
            "usage": llm_response.get("usage", {})
        },
        "metadata": {
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "parameter_shapes": param_shapes or {},
            "hyperparameters": hyperparameters or {}
        }
    }

    # Validate if torch_fn provided
    if torch_fn:
        test_suite = create_test_suite(
            input_shapes,
            param_shapes,
            num_random=5,
            num_edge=5
        )

        # Create executor wrapper
        def webgpu_executor(kernel_src: str, inputs: List) -> Any:
            return execute_kernel(kernel_src, inputs, output_shapes[0])

        validation_result = validate_kernel(
            kernel_source,
            torch_fn,
            test_suite,
            webgpu_executor,
            tolerance=1e-5,
            dtype="float32"
        )

        kernel_data["validation"] = validation_result
    else:
        # No validation possible without torch_fn
        kernel_data["validation"] = {
            "tolerance": 1e-5,
            "dtype": "float32",
            "test_cases": [],
            "all_passed": False,
            "num_passed": 0,
            "num_total": 0,
            "note": "No validation performed - torch_fn not provided"
        }

    return kernel_data


def generate_kernel(
    torch_source: str,
    operation: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    torch_fn: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    force_regenerate: bool = False
) -> Path:
    """Main kernel generation pipeline."""

    if config is None:
        config = default_config()

    # Check if kernel already exists
    if not force_regenerate:
        existing = find_kernel(
            operation,
            input_shapes[0],
            output_shapes[0],
            torch_source,
            hyperparameters,
            cache_dir=config["cache_dir"],
            prefer_correct=True
        )

        if existing:
            print(f"Found existing kernel: {existing}")
            return existing

    # Attempt generation up to max_samples times
    max_samples = config.get("max_samples", 5)
    last_error = None

    for attempt in range(max_samples):
        print(f"Generation attempt {attempt + 1}/{max_samples}")

        try:
            kernel_data = attempt_generation(
                torch_source,
                operation,
                input_shapes,
                output_shapes,
                param_shapes,
                hyperparameters,
                torch_fn,
                config.get("llm", {})
            )

            # Save kernel
            kernel_path = save_kernel(kernel_data, config["cache_dir"])
            print(f"Kernel saved: {kernel_path}")

            # Return path if validation passed
            if kernel_data["validation"]["all_passed"]:
                print(f"✓ Kernel validated successfully!")
                return kernel_path
            else:
                print(f"✗ Validation failed: {kernel_data['validation']['num_passed']}/{kernel_data['validation']['num_total']} tests passed")
                last_error = "Validation failed"

        except Exception as e:
            print(f"Generation attempt failed: {e}")
            last_error = str(e)
            continue

    # All attempts failed
    raise RuntimeError(f"Failed to generate valid kernel after {max_samples} attempts. Last error: {last_error}")


def batch_generate_kernels(
    operations: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = False
) -> List[Path]:
    """Generate multiple kernels in batch."""
    results = []

    for op_spec in operations:
        try:
            path = generate_kernel(
                torch_source=op_spec["torch_source"],
                operation=op_spec["operation"],
                input_shapes=op_spec["input_shapes"],
                output_shapes=op_spec["output_shapes"],
                param_shapes=op_spec.get("param_shapes"),
                hyperparameters=op_spec.get("hyperparameters"),
                torch_fn=op_spec.get("torch_fn"),
                config=config
            )
            results.append(path)
        except Exception as e:
            print(f"Failed to generate kernel for {op_spec['operation']}: {e}")
            results.append(None)

    return results