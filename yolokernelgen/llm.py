"""LLM interaction and prompt handling."""

import os
from typing import Dict, List, Any, Optional
from .types import LLMRequest, LLMResponse
from .exceptions import LLMError
from .prompts import build_system_prompt, build_user_prompt, extract_kernel_from_response, get_example_kernels, build_feedback_aware_prompt
from .logging_config import get_llm_logger

logger = get_llm_logger()


class LLMClient:
    """Client for interacting with LLM APIs with retry logic."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("Please set OPENAI_API_KEY environment variable")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMError("Please install openai: pip install openai")

    def sample_from_llm(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> LLMResponse:
        """Sample from OpenAI LLM and return structured response."""
        logger.debug(f"Calling LLM with model={model}, temp={temperature}, max_tokens={max_tokens}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            raw_completion = response.choices[0].message.content
            extracted_kernel = extract_kernel_from_response(raw_completion)

            if extracted_kernel is None:
                logger.error("Failed to extract kernel from LLM response")
                raise LLMError("Failed to extract kernel from LLM response")

            return LLMResponse(
                raw_completion=raw_completion,
                extracted_kernel=extracted_kernel,
                extraction_method="regex",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            logger.debug(f"LLM response received, tokens used: {response.usage.total_tokens}")
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise LLMError(f"LLM API call failed: {e}")


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


def build_prompts(
    torch_source: str,
    input_shapes: List[List[int]],
    output_shapes: List[List[int]],
    param_shapes: Optional[Dict[str, List[int]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    operation: str = "",
    previous_attempts: Optional[List[Dict[str, Any]]] = None,
    success_examples: Optional[List[Dict[str, Any]]] = None
) -> LLMRequest:
    """Build system and user prompts with feedback awareness."""

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

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return LLMRequest(
        model="gpt-4o",  # Will be overridden by config
        messages=messages,
        temperature=0.7,  # Will be overridden by config
        max_tokens=4000,  # Will be overridden by config
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )