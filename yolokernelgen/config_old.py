"""Configuration management for kernel generation."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "cache_dir": ".cache/yolokernelgen",
        "max_samples": 5,
        "tolerance": {
            "float32": 1e-5,
            "float16": 1e-3
        },
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "validation": {
            "num_random_tests": 5,
            "num_edge_tests": 5,
            "test_seeds": [42, 123, 456, 789, 1011]
        },
        "webgpu": {
            "workgroup_size": 256,
            "max_workgroups_per_dim": 65535
        }
    }


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from file, falling back to defaults."""
    base = default_config()

    if path and path.exists():
        with open(path, 'r') as f:
            override = json.load(f)
        return merge_configs(base, override)

    return base


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Save configuration to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)