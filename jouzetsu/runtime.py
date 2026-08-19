"""Typed runtime information shared by the LM Studio client, state, and UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias


class RuntimePhase(Enum):
    """Mutually exclusive phases shown in the active chat status line."""

    CHECKING = "checking"
    OFFLINE = "offline"
    NO_MODEL_SELECTED = "no_model_selected"
    MODEL_UNLOADED = "model_unloaded"
    STARTING = "starting"
    LOADING_MODEL = "loading_model"
    PROCESSING_PROMPT = "processing_prompt"
    REASONING = "reasoning"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    STOPPING = "stopping"
    READY = "ready"
    ERROR = "error"


class ConnectionState(Enum):
    """Reachability of the configured LM Studio instance."""

    CHECKING = "checking"
    ONLINE = "online"
    OFFLINE = "offline"


ReasoningOption = Literal["off", "on", "low", "medium", "high"]
ModelFormat = Literal["gguf", "mlx"]


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Public capabilities advertised by one model."""

    vision: bool = False
    trained_for_tool_use: bool = False
    reasoning: bool = False
    reasoning_options: tuple[ReasoningOption, ...] = ()
    reasoning_default: ReasoningOption | None = None


@dataclass(frozen=True, slots=True)
class ModelInstanceDescriptor:
    """Runtime configuration for one loaded model instance."""

    id: str
    context_length: int | None = None
    eval_batch_size: int | None = None
    parallel: int | None = None
    flash_attention: bool | None = None
    num_experts: int | None = None
    offload_kv_cache_to_gpu: bool | None = None
    auto_unload_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Downloaded model metadata and its currently loaded instances."""

    key: str
    display_name: str
    loaded_instance_ids: tuple[str, ...] = ()
    loaded_instances: tuple[ModelInstanceDescriptor, ...] = ()
    context_length: int | None = None
    max_context_length: int | None = None
    auto_unload_minutes: int | None = None
    publisher: str = ""
    architecture: str = ""
    quantization_name: str = ""
    bits_per_weight: float | None = None
    size_bytes: int | None = None
    params_string: str = ""
    format: ModelFormat | None = None
    capabilities: ModelCapabilities = ModelCapabilities()
    description: str = ""
    variants: tuple[str, ...] = ()
    selected_variant: str = ""

    @property
    def is_loaded(self) -> bool:
        return bool(self.loaded_instance_ids)


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Exact statistics reported after a completed prediction."""

    output_tokens: int | None = None
    prompt_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_seconds: float | None = None
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Complete status-line state for one chat."""

    phase: RuntimePhase
    model_key: str = ""
    model_name: str = ""
    progress: float | None = None
    started_at: float | None = None
    phase_started_at: float | None = None
    output_tokens: int = 0
    metrics: GenerationMetrics | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.progress is not None and not 0.0 <= self.progress <= 1.0:
            raise ValueError("runtime progress must be between 0 and 1")
        if self.output_tokens < 0:
            raise ValueError("output token count cannot be negative")


@dataclass(frozen=True, slots=True)
class ModelLoadProgress:
    progress: float | None


@dataclass(frozen=True, slots=True)
class ModelReady:
    model_key: str
    model_name: str
    context_length: int


@dataclass(frozen=True, slots=True)
class PromptProcessingProgress:
    progress: float | None


@dataclass(frozen=True, slots=True)
class FirstToken:
    pass


@dataclass(frozen=True, slots=True)
class PredictionFragment:
    """One streamed content fragment with an optional exact token delta."""

    content: str
    token_count: int | None
    is_reasoning: bool


@dataclass(frozen=True, slots=True)
class PredictionComplete:
    metrics: GenerationMetrics


ChatStreamEvent: TypeAlias = (
    ModelLoadProgress | ModelReady | PromptProcessingProgress | FirstToken | PredictionFragment | PredictionComplete
)
