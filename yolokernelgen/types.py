"""Type-safe data models for kernel generation using SerialDataclass."""

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Type, TypeVar, Union, Literal

try:
    import dacite
except ImportError:
    raise ImportError("Please install dacite: pip install dacite")

T = TypeVar('T', bound='SerialDataclass')


class SerialDataclass:
    """Base class for dataclasses with JSON serialization support"""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)  # type:ignore

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        return dacite.from_dict(data_class=cls, data=json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return dacite.from_dict(data_class=cls, data=data)

    def to_path(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_path(cls: Type[T], path: Union[str, Path]) -> T:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def hash(self) -> int:
        return hash(self.to_json())

    def __repr__(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)  # type:ignore


@dataclass
class LLMRequest(SerialDataclass):
    """LLM request data structure."""
    model: str
    messages: List[Dict[str, str]]
    temperature: float
    max_tokens: int
    system_prompt: str
    user_prompt: str


@dataclass
class LLMResponse(SerialDataclass):
    """LLM response data structure."""
    raw_completion: str
    extracted_kernel: str
    extraction_method: str
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class TestCase(SerialDataclass):
    """Individual test case data structure."""
    inputs: List[List[float]]
    expected_output: List[float]
    test_type: str
    seed: int
    tolerance: float
    passed: bool
    error_message: Optional[str] = None
    actual_output: Optional[List[float]] = None


@dataclass
class ValidationResult(SerialDataclass):
    """Validation results for a kernel."""
    tolerance: float
    dtype: str
    test_cases: List[TestCase]
    all_passed: bool
    num_passed: int
    num_total: int
    failure_summary: Optional[Dict[str, Any]] = None
    note: Optional[str] = None


@dataclass
class KernelMetadata(SerialDataclass):
    """Metadata about kernel shapes and hyperparameters."""
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    parameter_shapes: Dict[str, List[int]] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KernelData(SerialDataclass):
    """Main kernel data structure replacing the kernel dict."""
    operation: str
    torch_source: str
    torch_hash: str
    llm_request: LLMRequest
    llm_response: LLMResponse
    metadata: KernelMetadata
    validation: ValidationResult
    status: Literal["correct", "rejected"]
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1

    def __post_init__(self):
        """Ensure torch_hash is computed from torch_source."""
        if not self.torch_hash:
            self.torch_hash = hashlib.sha256(self.torch_source.encode()).hexdigest()

    @classmethod
    def migrate_from_dict(cls, data: Dict[str, Any]) -> "KernelData":
        """Migrate old cache file format to new KernelData structure."""
        try:
            # Handle missing version field (legacy data)
            version = data.get("version", 0)

            if version == 0:
                # Legacy format migration
                llm_request = LLMRequest(
                    model=data.get("llm_request", {}).get("model", "gpt-4o"),
                    messages=data.get("llm_request", {}).get("messages", []),
                    temperature=data.get("llm_request", {}).get("temperature", 0.7),
                    max_tokens=data.get("llm_request", {}).get("max_tokens", 4000),
                    system_prompt=data.get("llm_request", {}).get("system_prompt", ""),
                    user_prompt=data.get("llm_request", {}).get("user_prompt", "")
                )

                llm_response = LLMResponse(
                    raw_completion=data.get("llm_response", {}).get("raw_completion", ""),
                    extracted_kernel=data.get("llm_response", {}).get("extracted_kernel", ""),
                    extraction_method=data.get("llm_response", {}).get("extraction_method", "regex"),
                    usage=data.get("llm_response", {}).get("usage", {})
                )

                # Convert test_cases from old format
                old_test_cases = data.get("validation", {}).get("test_cases", [])
                test_cases = []
                for case in old_test_cases:
                    test_case = TestCase(
                        inputs=case.get("inputs", []),
                        expected_output=case.get("expected_output", []),
                        test_type=case.get("test_type", "unknown"),
                        seed=case.get("seed", 0),
                        tolerance=case.get("tolerance", 1e-5),
                        passed=case.get("passed", False),
                        error_message=case.get("error_message"),
                        actual_output=case.get("actual_output")
                    )
                    test_cases.append(test_case)

                validation = ValidationResult(
                    tolerance=data.get("validation", {}).get("tolerance", 1e-5),
                    dtype=data.get("validation", {}).get("dtype", "float32"),
                    test_cases=test_cases,
                    all_passed=data.get("validation", {}).get("all_passed", False),
                    num_passed=data.get("validation", {}).get("num_passed", 0),
                    num_total=data.get("validation", {}).get("num_total", 0),
                    failure_summary=data.get("validation", {}).get("failure_summary"),
                    note=data.get("validation", {}).get("note")
                )

                metadata = KernelMetadata(
                    input_shapes=data.get("metadata", {}).get("input_shapes", []),
                    output_shapes=data.get("metadata", {}).get("output_shapes", []),
                    parameter_shapes=data.get("metadata", {}).get("parameter_shapes", {}),
                    hyperparameters=data.get("metadata", {}).get("hyperparameters", {}),
                    timestamp=data.get("metadata", {}).get("timestamp", data.get("timestamp", datetime.now().isoformat()))
                )

                # Determine status from validation
                status = "correct" if validation.all_passed else "rejected"

                return cls(
                    operation=data.get("operation", "unknown"),
                    torch_source=data.get("torch_source", ""),
                    torch_hash=data.get("torch_hash", ""),
                    llm_request=llm_request,
                    llm_response=llm_response,
                    metadata=metadata,
                    validation=validation,
                    status=status,
                    uuid=data.get("uuid", str(uuid.uuid4())),
                    timestamp=data.get("timestamp", datetime.now().isoformat()),
                    version=1
                )
            else:
                # Current version, use normal deserialization
                return cls.from_dict(data)

        except Exception as e:
            raise ValueError(f"Failed to migrate legacy kernel data: {e}")


@dataclass
class Config(SerialDataclass):
    """Configuration for kernel generation with validation."""
    cache_dir: str = ".cache/yolokernelgen"
    max_samples: int = 5
    max_concurrent_llm: int = 5
    max_concurrent_validation: int = 10
    approaches: List[str] = field(default_factory=lambda: ["standard"])
    tolerance: Dict[str, float] = field(default_factory=lambda: {"float32": 1e-5, "float16": 1e-3})
    llm: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4000
    })
    validation: Dict[str, Any] = field(default_factory=lambda: {
        "num_random_tests": 5,
        "num_edge_tests": 5,
        "test_seeds": [42, 123, 456, 789, 1011]
    })
    webgpu: Dict[str, Any] = field(default_factory=lambda: {
        "workgroup_size": 256,
        "max_workgroups_per_dim": 65535
    })

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.max_concurrent_llm <= 0:
            raise ValueError("max_concurrent_llm must be positive")
        if self.max_concurrent_validation <= 0:
            raise ValueError("max_concurrent_validation must be positive")

        cache_parent = Path(self.cache_dir).parent
        if not cache_parent.exists():
            raise ValueError(f"Parent directory of cache_dir must exist: {cache_parent}")