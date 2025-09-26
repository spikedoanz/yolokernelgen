"""YoloKernelGen - Functional-style kernel generation framework."""

from .runtime import generate_kernel, batch_generate_kernels
from .llm import LLMClient
from .validator import create_test_suite, validate_kernel
from .storage import save_kernel, load_kernel, find_kernel, list_kernels
from .config import default_config, load_config
from .webgpu_executor import execute_kernel

__version__ = "0.1.0"

__all__ = [
    # Main generation functions
    "generate_kernel",
    "batch_generate_kernels",

    # LLM client
    "LLMClient",

    # Validation
    "create_test_suite",
    "validate_kernel",

    # Storage
    "save_kernel",
    "load_kernel",
    "find_kernel",
    "list_kernels",

    # Config
    "default_config",
    "load_config",

    # Execution
    "execute_kernel",
]