"""Main kernel generation pipeline."""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, cast

from .config import default_config
from .storage import save_kernel, find_kernel
from .validation import create_test_suite, validate_kernel
from .prompts import build_system_prompt, build_user_prompt, extract_kernel_from_response, get_example_kernels, build_feedback_aware_prompt
from .webgpu_executor import execute_kernel
from .knowledge_base import add_successful_kernel, get_relevant_success_examples
from datetime import datetime


def analyze_previous_attempts(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in previous failed attempts."""
    if not attempts:
        return {}

    analysis = {
        "total_attempts": len(attempts),
        "common_failures": {},
        "error_patterns": [],
        "suggestions": []
    }

    # Count failure types
    failure_types = {}
    all_failure_summaries = []

    for attempt in attempts:
        validation = attempt.get("validation", {})
        failure_summary = validation.get("failure_summary")

        if failure_summary:
            all_failure_summaries.append(failure_summary)

            # Count error patterns
            for error_type, count in failure_summary.get("error_patterns", {}).items():
                failure_types[error_type] = failure_types.get(error_type, 0) + count

    analysis["common_failures"] = failure_types

    # Generate suggestions based on patterns
    if "boundary" in failure_types:
        analysis["suggestions"].append("Focus on boundary condition handling and padding logic")
    if "overflow" in failure_types:
        analysis["suggestions"].append("Check array indexing bounds - likely out-of-bounds access")
    if "indexing" in failure_types:
        analysis["suggestions"].append("Review tensor indexing formulas, especially for multi-dimensional arrays")
    if "zero_handling" in failure_types:
        analysis["suggestions"].append("Ensure kernel correctly handles zero/empty inputs")

    # Find near-misses (high success rate)
    near_misses = []
    for attempt in attempts:
        validation = attempt.get("validation", {})
        if validation.get("num_passed", 0) >= 8:  # 8/10 or better
            near_misses.append({
                "passed": validation.get("num_passed", 0),
                "total": validation.get("num_total", 10),
                "failure_summary": validation.get("failure_summary")
            })

    if near_misses:
        analysis["near_misses"] = near_misses
        analysis["suggestions"].append("Previous attempts were very close - focus on specific edge cases")

    return analysis


def select_relevant_examples(
    operation: str,
    success_examples: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Select most relevant success examples for the current operation."""
    if not success_examples:
        return []

    # For now, return up to 2 most relevant examples
    # In the future, could use more sophisticated matching (operation type, shapes, etc.)
    relevant = []

    for example in success_examples:
        example_op = example.get("operation", "")

        # Exact match gets highest priority
        if example_op == operation:
            relevant.append(example)
        # Similar operations (e.g., conv3d variants)
        elif operation.startswith("conv") and example_op.startswith("conv"):
            relevant.append(example)
        # Generic operations that might be helpful
        elif len(relevant) < 2:
            relevant.append(example)

    return relevant[:2]  # Limit to 2 examples to avoid prompt bloat


def sample_from_llm(
    messages: List[Dict[str, Any]],
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
            messages=cast(Any, messages),  # Cast to avoid OpenAI typing issues
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "raw_completion": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
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
    llm_config: Optional[Dict[str, Any]] = None,
    previous_attempts: Optional[List[Dict[str, Any]]] = None,
    success_examples: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Attempt to generate and validate a kernel."""

    if llm_config is None:
        llm_config = default_config()["llm"]

    assert llm_config is not None  # Help type checker

    # Analyze previous attempts and select examples
    attempt_analysis = analyze_previous_attempts(previous_attempts) if previous_attempts else {}
    relevant_examples = select_relevant_examples(operation, success_examples)

    # Build prompts with feedback
    system_prompt = build_system_prompt()

    if previous_attempts or success_examples:
        # Use feedback-aware prompt for subsequent attempts
        user_prompt = build_feedback_aware_prompt(
            torch_source,
            input_shapes,
            output_shapes,
            param_shapes,
            hyperparameters,
            attempt_analysis,
            relevant_examples
        )
    else:
        # Use standard prompt for first attempt
        example_kernels = get_example_kernels()
        example_kernel = example_kernels.get(operation, example_kernels.get("add", None))
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
            "hyperparameters": hyperparameters or {},
            "timestamp": datetime.now().isoformat()
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

    # Attempt generation up to max_samples times with learning
    max_samples = config.get("max_samples", 5)
    last_error = None
    previous_attempts = []

    # Load relevant success examples from knowledge base
    success_examples = get_relevant_success_examples(
        operation=operation,
        input_shapes=input_shapes,
        cache_dir=config["cache_dir"],
        max_examples=2
    )

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
                config.get("llm", {}),
                previous_attempts=previous_attempts if attempt > 0 else None,
                success_examples=success_examples
            )

            # Save kernel
            kernel_path = save_kernel(kernel_data, config["cache_dir"])
            print(f"Kernel saved: {kernel_path}")

            # Check validation results
            validation_passed = kernel_data["validation"]["all_passed"]
            has_validation = kernel_data["validation"]["num_total"] > 0

            if validation_passed or not has_validation:
                if has_validation:
                    print("✓ Kernel validated successfully!")
                else:
                    print("✓ Kernel generated successfully (validation skipped)")

                # Add successful kernel to knowledge base (only if validated)
                if has_validation and validation_passed:
                    try:
                        add_successful_kernel(kernel_data, config["cache_dir"])
                        print("✓ Added to knowledge base for future learning")
                    except Exception as e:
                        print(f"Warning: Failed to add kernel to knowledge base: {e}")

                return kernel_path
            else:
                print(f"✗ Validation failed: {kernel_data['validation']['num_passed']}/{kernel_data['validation']['num_total']} tests passed")
                last_error = "Validation failed"

                # Store failed attempt for learning
                previous_attempts.append({
                    "attempt_number": attempt + 1,
                    "operation": operation,
                    "validation": kernel_data["validation"],
                    "llm_response": kernel_data["llm_response"],
                    "kernel_source": kernel_data["llm_response"]["extracted_kernel"]
                })

                # Show feedback summary if available
                failure_summary = kernel_data["validation"].get("failure_summary")
                if failure_summary:
                    print(f"  Common issues: {', '.join(failure_summary.get('common_issues', []))}")
                    for feedback in failure_summary.get("specific_feedback", [])[:2]:  # Show top 2
                        print(f"  • {feedback}")
                    if failure_summary.get("performance_note"):
                        print(f"  Note: {failure_summary['performance_note']}")

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