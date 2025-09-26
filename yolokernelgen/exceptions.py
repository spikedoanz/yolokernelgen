"""Exception hierarchy for kernel generation errors."""


class KernelGenerationError(Exception):
    """Base exception for all kernel generation errors."""
    pass


class ValidationError(KernelGenerationError):
    """Error during kernel validation."""
    pass


class LLMError(KernelGenerationError):
    """Error during LLM interaction."""
    pass


class StorageError(KernelGenerationError):
    """Error during kernel storage/retrieval operations."""
    pass


class WebGPUExecutionError(KernelGenerationError):
    """Error during WebGPU kernel execution."""
    pass


class ConfigurationError(KernelGenerationError):
    """Error in configuration validation or setup."""
    pass


class MigrationError(KernelGenerationError):
    """Error during data migration from old formats."""
    pass