"""Logging configuration for yolokernelgen."""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    cache_dir: str = ".cache/yolokernelgen"
) -> None:
    """Setup logging configuration for the package."""

    # Determine logging level
    if level is None:
        if os.getenv("VERBOSE") == "1":
            level = logging.DEBUG
        elif os.getenv("QUIET") == "1":
            level = logging.WARNING
        else:
            level = logging.INFO

    # Setup log file path
    if log_file is None:
        log_dir = Path(cache_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "yolokernelgen.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger for this package
    logger = logging.getLogger('yolokernelgen')
    logger.setLevel(logging.DEBUG)  # Let handlers control the level

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    logger.info(f"Logging configured - Console: {logging.getLevelName(level)}, File: DEBUG")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module name."""
    # Ensure the name is under yolokernelgen package
    if not name.startswith('yolokernelgen'):
        name = f'yolokernelgen.{name}'

    return logging.getLogger(name)


# Module-level loggers for convenience
def get_runtime_logger() -> logging.Logger:
    """Get logger for runtime module."""
    return get_logger('yolokernelgen.runtime')


def get_llm_logger() -> logging.Logger:
    """Get logger for LLM module."""
    return get_logger('yolokernelgen.llm')


def get_validator_logger() -> logging.Logger:
    """Get logger for validator module."""
    return get_logger('yolokernelgen.validator')


def get_storage_logger() -> logging.Logger:
    """Get logger for storage module."""
    return get_logger('yolokernelgen.storage')


def get_webgpu_logger() -> logging.Logger:
    """Get logger for WebGPU module."""
    return get_logger('yolokernelgen.webgpu_executor')