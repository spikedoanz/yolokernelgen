"""Configuration management with type safety and validation."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .types import Config
from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger('yolokernelgen.config')


def default_config() -> Config:
    """Return default configuration as a validated Config object."""
    return Config()


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file, falling back to defaults with validation."""
    try:
        base_config = default_config()

        if path is None:
            logger.debug("No config path provided, using defaults")
            return base_config

        config_path = Path(path)
        if not config_path.exists():
            logger.debug(f"Config file not found: {config_path}, using defaults")
            return base_config

        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            override_dict = json.load(f)

        # Merge with base config
        merged_dict = merge_config_dicts(base_config.to_dict(), override_dict)

        # Create and validate new Config object
        config = Config.from_dict(merged_dict)
        logger.info("Configuration loaded and validated successfully")
        return config

    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file {path}: {e}")
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise e
        raise ConfigurationError(f"Failed to load config from {path}: {e}")


def merge_config_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config_dicts(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Config, path: Union[str, Path]) -> None:
    """Save configuration to file with validation."""
    try:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        raise ConfigurationError(f"Failed to save config to {path}: {e}")


def validate_config_dict(config_dict: Dict[str, Any]) -> None:
    """Validate a configuration dictionary without creating Config object."""
    try:
        # Try to create Config object to validate
        Config.from_dict(config_dict)
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")


def create_default_config_file(path: Union[str, Path]) -> None:
    """Create a default configuration file at the specified path."""
    config = default_config()
    save_config(config, path)
    logger.info(f"Created default configuration file at {path}")


# Backward compatibility functions for existing code
def load_config_dict(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration as a dictionary (backward compatibility)."""
    config = load_config(path)
    return config.to_dict()


def default_config_dict() -> Dict[str, Any]:
    """Return default configuration as a dictionary (backward compatibility)."""
    config = default_config()
    return config.to_dict()


# Environment-based configuration helpers
def load_config_from_env() -> Config:
    """Load configuration with environment variable overrides."""
    import os

    config = default_config()

    # Override from environment variables
    if os.getenv("YOLO_CACHE_DIR"):
        config.cache_dir = os.getenv("YOLO_CACHE_DIR")

    if os.getenv("YOLO_MAX_SAMPLES"):
        try:
            config.max_samples = int(os.getenv("YOLO_MAX_SAMPLES"))
        except ValueError:
            logger.warning("Invalid YOLO_MAX_SAMPLES value, using default")

    if os.getenv("YOLO_MAX_CONCURRENT_LLM"):
        try:
            config.max_concurrent_llm = int(os.getenv("YOLO_MAX_CONCURRENT_LLM"))
        except ValueError:
            logger.warning("Invalid YOLO_MAX_CONCURRENT_LLM value, using default")

    if os.getenv("YOLO_LLM_MODEL"):
        config.llm["model"] = os.getenv("YOLO_LLM_MODEL")

    if os.getenv("YOLO_LLM_TEMPERATURE"):
        try:
            config.llm["temperature"] = float(os.getenv("YOLO_LLM_TEMPERATURE"))
        except ValueError:
            logger.warning("Invalid YOLO_LLM_TEMPERATURE value, using default")

    logger.debug("Configuration loaded from environment variables")
    return config