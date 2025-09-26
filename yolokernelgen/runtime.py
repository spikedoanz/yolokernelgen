"""Kernel generation runtime orchestration (was generation.py)."""

import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from .types import KernelData, KernelMetadata, Config
from .exceptions import KernelGenerationError, ValidationError, LLMError
from .llm import LLMClient, build_prompts
from .validator import create_test_suite, validate_kernel
from .storage import save_kernel, find_kernel
from .webgpu_executor import execute_kernel
from .knowledge_base import add_successful_kernel, get_relevant_success_examples
from .logging_config import get_runtime_logger

logger = get_runtime_logger()


async def attempt_generation(
    torch_source: str,
    operation: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    torch_fn: Optional[Callable] = None,
    config: Optional[Config] = None,
    previous_attempts: Optional[List[Dict[str, Any]]] = None,
    success_examples: Optional[List[Dict[str, Any]]] = None,
    semaphore: Optional[asyncio.Semaphore] = None
) -> KernelData:
    """Attempt to generate and validate a kernel asynchronously."""

    if config is None:
        config = Config()

    async with semaphore or asyncio.Semaphore(1):
        try:
            # Initialize LLM client
            llm_client = LLMClient()

            # Build prompts with feedback awareness
            llm_request = build_prompts(
                torch_source,
                input_shapes,
                output_shapes,
                param_shapes,
                hyperparameters,
                operation,
                previous_attempts,
                success_examples
            )

            # Override with config values
            llm_request.model = config.llm.get("model", "gpt-4o")
            llm_request.temperature = config.llm.get("temperature", 0.7)
            llm_request.max_tokens = config.llm.get("max_tokens", 4000)

            # Sample from LLM
            llm_response = await asyncio.to_thread(
                llm_client.sample_from_llm,
                llm_request.messages,
                llm_request.model,
                llm_request.temperature,
                llm_request.max_tokens
            )

            # Create metadata
            metadata = KernelMetadata(
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                parameter_shapes=param_shapes or {},
                hyperparameters=hyperparameters or {},
                timestamp=datetime.now().isoformat()
            )

            # Validate if torch_fn provided
            if torch_fn:
                test_suite = create_test_suite(
                    input_shapes,
                    param_shapes,
                    num_random=config.validation.get("num_random_tests", 5),
                    num_edge=config.validation.get("num_edge_tests", 5),
                    seeds=config.validation.get("test_seeds", [42, 123, 456, 789, 1011])
                )

                # Create executor wrapper
                def webgpu_executor(kernel_src: str, inputs: List) -> Any:
                    return execute_kernel(kernel_src, inputs, output_shapes[0])

                # Run validation (CPU-bound, use thread pool)
                validation_result = await validate_kernel(
                    llm_response.extracted_kernel,
                    torch_fn,
                    test_suite,
                    webgpu_executor,
                    tolerance=config.tolerance.get("float32", 1e-5),
                    dtype="float32"
                )
            else:
                # No validation possible without torch_fn
                from .types import ValidationResult, TestCase
                validation_result = ValidationResult(
                    tolerance=config.tolerance.get("float32", 1e-5),
                    dtype="float32",
                    test_cases=[],
                    all_passed=False,
                    num_passed=0,
                    num_total=0,
                    note="No validation performed - torch_fn not provided"
                )

            # Determine status
            status = "correct" if validation_result.all_passed else "rejected"

            # Create kernel data
            kernel_data = KernelData(
                operation=operation,
                torch_source=torch_source,
                torch_hash=hashlib.sha256(torch_source.encode()).hexdigest(),
                llm_request=llm_request,
                llm_response=llm_response,
                metadata=metadata,
                validation=validation_result,
                status=status
            )

            return kernel_data

        except Exception as e:
            if isinstance(e, (LLMError, ValidationError)):
                raise e
            else:
                raise KernelGenerationError(f"Kernel generation failed: {e}")


async def generate_kernel(
    torch_source: str,
    operation: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    torch_fn: Optional[Callable] = None,
    config: Optional[Config] = None,
    force_regenerate: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None
) -> Path:
    """Main kernel generation pipeline with async support."""

    if config is None:
        config = Config()

    # Check if kernel already exists
    if not force_regenerate:
        existing = find_kernel(
            operation,
            input_shapes[0],
            output_shapes[0],
            torch_source,
            hyperparameters,
            cache_dir=config.cache_dir,
            prefer_correct=True
        )

        if existing:
            logger.info(f"Found existing kernel: {existing}")
            return existing

    # Attempt generation up to max_samples times with learning
    max_samples = config.max_samples
    last_error = None
    previous_attempts = []

    # Load relevant success examples from knowledge base
    success_examples = get_relevant_success_examples(
        operation=operation,
        input_shapes=input_shapes,
        cache_dir=config.cache_dir,
        max_examples=2
    )

    for attempt in range(max_samples):
        logger.info(f"Generation attempt {attempt + 1}/{max_samples}")

        try:
            kernel_data = await attempt_generation(
                torch_source,
                operation,
                input_shapes,
                output_shapes,
                param_shapes,
                hyperparameters,
                torch_fn,
                config,
                previous_attempts=previous_attempts if attempt > 0 else None,
                success_examples=success_examples,
                semaphore=semaphore
            )

            # Save kernel
            kernel_path = save_kernel(kernel_data.to_dict(), config.cache_dir)
            logger.info(f"Kernel saved: {kernel_path}")

            # Check validation results
            validation_passed = kernel_data.validation.all_passed
            has_validation = kernel_data.validation.num_total > 0

            if validation_passed or not has_validation:
                if has_validation:
                    logger.info("✓ Kernel validated successfully!")
                else:
                    logger.info("✓ Kernel generated successfully (validation skipped)")

                # Add successful kernel to knowledge base (only if validated)
                if has_validation and validation_passed:
                    try:
                        add_successful_kernel(kernel_data.to_dict(), config.cache_dir)
                        logger.info("✓ Added to knowledge base for future learning")
                    except Exception as e:
                        logger.warning(f"Failed to add kernel to knowledge base: {e}")

                return kernel_path
            else:
                logger.warning(f"✗ Validation failed: {kernel_data.validation.num_passed}/{kernel_data.validation.num_total} tests passed")
                last_error = "Validation failed"

                # Store failed attempt for learning
                previous_attempts.append({
                    "attempt_number": attempt + 1,
                    "operation": operation,
                    "validation": kernel_data.validation.to_dict(),
                    "llm_response": kernel_data.llm_response.to_dict(),
                    "kernel_source": kernel_data.llm_response.extracted_kernel
                })

                # Show feedback summary if available
                failure_summary = kernel_data.validation.failure_summary
                if failure_summary:
                    logger.info(f"  Common issues: {', '.join(failure_summary.get('common_issues', []))}")
                    for feedback in failure_summary.get("specific_feedback", [])[:2]:  # Show top 2
                        logger.info(f"  • {feedback}")
                    if failure_summary.get("performance_note"):
                        logger.info(f"  Note: {failure_summary['performance_note']}")

        except Exception as e:
            logger.error(f"Generation attempt failed: {e}")
            last_error = str(e)
            continue

    # All attempts failed
    raise KernelGenerationError(f"Failed to generate valid kernel after {max_samples} attempts. Last error: {last_error}")


async def batch_generate_kernels(
    operations: List[Dict[str, Any]],
    config: Optional[Config] = None,
    max_concurrent: Optional[int] = None
) -> List[Path]:
    """Generate multiple kernels in batch with controlled concurrency."""

    if config is None:
        config = Config()

    if max_concurrent is None:
        max_concurrent = config.max_concurrent_llm

    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all operations
    tasks = []
    for op_spec in operations:
        task = generate_kernel(
            torch_source=op_spec["torch_source"],
            operation=op_spec["operation"],
            input_shapes=op_spec["input_shapes"],
            output_shapes=op_spec["output_shapes"],
            param_shapes=op_spec.get("param_shapes"),
            hyperparameters=op_spec.get("hyperparameters"),
            torch_fn=op_spec.get("torch_fn"),
            config=config,
            semaphore=semaphore
        )
        tasks.append(task)

    # Execute tasks with controlled concurrency
    results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        try:
            path = await task
            results.append((i, path))
            logger.info(f"Completed kernel {i + 1}/{len(operations)}")
        except Exception as e:
            logger.error(f"Failed to generate kernel {i + 1}: {e}")
            results.append((i, None))

    # Sort results by original order
    results.sort(key=lambda x: x[0])
    return [result[1] for result in results]