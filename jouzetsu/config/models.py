"""Pure domain models for application configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import ClassVar

from ..colors import HEX_COLOR_PATTERN as _HEX_COLOR_PATTERN
from .paths import AppPaths, default_paths

CONTINUE_PROMPT_CONTENT: str = (
    "Continue the previous assistant response exactly from where it stopped. "
    "Do not repeat earlier text and do not add prefatory wording."
)


def _are_booleans(*values: object) -> bool:
    """Return whether every runtime-supplied value is a boolean."""

    return all(isinstance(value, bool) for value in values)


def _builtin_spelling_replacements() -> list[SpellingReplacement]:
    """Defer packaged-data access until the domain model is constructed."""

    from .defaults import builtin_spelling_replacements

    return list(builtin_spelling_replacements())


def _builtin_theme_value(name: str) -> str:
    """Defer one packaged theme value until a default model is constructed."""

    from .defaults import builtin_theme_values

    return builtin_theme_values()[name]


@dataclass
class ServerSettings:
    """Connection details for the LM Studio API server."""

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    default_model: str = ""
    model_aliases: dict[str, str] = field(default_factory=dict)
    auto_unload_minutes: int | None = None

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url cannot be blank")
        if self.auto_unload_minutes is not None and (
            type(self.auto_unload_minutes) is not int or self.auto_unload_minutes < 1
        ):
            raise ValueError("auto_unload_minutes must be an integer of at least 1")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.model_aliases.items()
        ):
            raise ValueError("model aliases must use non-empty keys and values")

    def apply(self, settings: ServerSettings) -> None:
        """Copy one validated value while retaining this instance's identity."""

        settings.validate()
        self.base_url = settings.base_url
        self.api_key = settings.api_key
        self.default_model = settings.default_model
        self.model_aliases = dict(settings.model_aliases)
        self.auto_unload_minutes = settings.auto_unload_minutes


@dataclass(frozen=True, slots=True)
class SpellingReplacement:
    """One case-insensitive whole-word spelling replacement."""

    source: str
    replacement: str

    def validate(self) -> None:
        if not self.source.strip() or not self.replacement.strip():
            raise ValueError(
                "spelling replacement source and replacement cannot be empty"
            )
        if (
            self.source != self.source.strip()
            or self.replacement != self.replacement.strip()
        ):
            raise ValueError("spelling replacements cannot have surrounding whitespace")
        if any(character.isspace() for character in self.source + self.replacement):
            raise ValueError("spelling replacements must each be one word")


@dataclass
class GenerationSettings:
    """Default sampling parameters sent on every generation request."""

    MIN_TEMPERATURE: ClassVar[float] = 0.0
    MAX_TEMPERATURE: ClassVar[float] = 2.0
    MIN_TOP_P: ClassVar[float] = 0.0
    MAX_TOP_P: ClassVar[float] = 1.0
    MIN_MAX_TOKENS: ClassVar[int] = 1

    temperature: float = 0.75
    top_p: float = 0.95
    max_tokens: int = 4096
    system_prompt: str = "You are a helpful assistant."
    continuity_review: bool = True
    british_english: bool = False
    british_spelling_replacements: list[SpellingReplacement] = field(
        default_factory=_builtin_spelling_replacements
    )

    def validate(self) -> None:
        if isinstance(self.temperature, bool) or not math.isfinite(self.temperature):
            raise ValueError("temperature must be a finite number")
        if not self.MIN_TEMPERATURE <= self.temperature <= self.MAX_TEMPERATURE:
            raise ValueError(
                f"temperature must be between {self.MIN_TEMPERATURE:g} and {self.MAX_TEMPERATURE:g}"
            )
        if isinstance(self.top_p, bool) or not math.isfinite(self.top_p):
            raise ValueError("top_p must be a finite number")
        if not self.MIN_TOP_P <= self.top_p <= self.MAX_TOP_P:
            raise ValueError(
                f"top_p must be between {self.MIN_TOP_P:g} and {self.MAX_TOP_P:g}"
            )
        if type(self.max_tokens) is not int or self.max_tokens < self.MIN_MAX_TOKENS:
            raise ValueError(
                f"max_tokens must be an integer of at least {self.MIN_MAX_TOKENS}"
            )
        if not _are_booleans(self.continuity_review, self.british_english):
            raise ValueError("continuity_review and british_english must be booleans")
        seen_sources: set[str] = set()
        for replacement in self.british_spelling_replacements:
            replacement.validate()
            source_key: str = replacement.source.casefold()
            if source_key in seen_sources:
                raise ValueError(
                    f"duplicate spelling replacement source: {replacement.source}"
                )
            seen_sources.add(source_key)


@dataclass(frozen=True, slots=True)
class StarterPrompt:
    """A labelled composer shortcut."""

    label: str
    content: str

    def validate(self) -> None:
        if not self.label.strip() or not self.content.strip():
            raise ValueError("starter prompts need non-empty labels and content")


DEFAULT_STARTER_PROMPTS: tuple[StarterPrompt, ...] = (
    StarterPrompt(label="Think it through", content="Help me think through this"),
    StarterPrompt(label="Explain", content="Explain a concept"),
    StarterPrompt(label="Make a plan", content="Make a plan"),
    StarterPrompt(label="Summarise", content="Summarise something"),
    StarterPrompt(label="Continue", content=CONTINUE_PROMPT_CONTENT),
)


@dataclass
class IconColorSettings:
    """The independently configurable colours in the app icon."""

    linework_color: str = "#000000"
    accent_color: str = "#7439b0"
    surface_color: str = "#f1eaf8"

    def validate(self) -> None:
        for field_name, value in (
            ("linework_color", self.linework_color),
            ("accent_color", self.accent_color),
            ("surface_color", self.surface_color),
        ):
            if not _HEX_COLOR_PATTERN.fullmatch(value):
                raise ValueError(f"{field_name} must be a six-digit hexadecimal color")


class MessageActionStyle(str, Enum):
    """How message action controls draw their colour."""

    UNIFORM = "uniform"
    SEMANTIC = "semantic"

    @property
    def label(self) -> str:
        """Return the user-facing option label."""

        match self:
            case MessageActionStyle.UNIFORM:
                return "Uniform App colour"
            case MessageActionStyle.SEMANTIC:
                return "Semantic colours"


class MessageAction(str, Enum):
    """The independently colourable message actions."""

    DELETE = "delete"
    REGENERATE = "regenerate"
    MERGE = "merge"
    EDIT = "edit"
    CONTINUE = "continue"

    @property
    def label(self) -> str:
        """Return the user-facing action label."""

        match self:
            case MessageAction.DELETE:
                return "Delete"
            case MessageAction.REGENERATE:
                return "Regenerate"
            case MessageAction.MERGE:
                return "Merge"
            case MessageAction.EDIT:
                return "Edit"
            case MessageAction.CONTINUE:
                return "Continue and resend"

    @property
    def color_field(self) -> str:
        """Return this action's App colourway setting name."""

        return f"message_action_{self.value}"


class ThemeColorway(str, Enum):
    """The four independent owners of UI colour."""

    APP = "app"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    @property
    def label(self) -> str:
        return self.value.capitalize()


@dataclass(frozen=True, slots=True)
class ColorwaySettings:
    """One main colour with optional overrides for its automatic shades."""

    accent: str
    muted: str | None = field(default=None, metadata={"label": "Muted colour"})
    subtle: str | None = field(default=None, metadata={"label": "Subtle colour"})
    hover: str | None = field(default=None, metadata={"label": "Hover colour"})

    def values(self) -> dict[str, str | None]:
        """Return the declared settings, including unset automatic overrides."""

        return {item.name: getattr(self, item.name) for item in fields(self)}

    def validate(self) -> None:
        for name, value in self.values().items():
            if value is None and name != "accent":
                continue
            if not isinstance(value, str) or not _HEX_COLOR_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a six-digit hexadecimal color")


@dataclass(frozen=True, slots=True)
class AppColorwaySettings(ColorwaySettings):
    """App accents and optional overrides for the shared light/dark surfaces."""

    key_visual: str | None = field(
        default=None, metadata={"label": "Key visual colour"}
    )
    message_actions: str | None = field(
        default=None, metadata={"label": "Message action colour"}
    )
    message_action_delete: str | None = field(
        default=None, metadata={"label": "Delete colour"}
    )
    message_action_regenerate: str | None = field(
        default=None, metadata={"label": "Regenerate colour"}
    )
    message_action_merge: str | None = field(
        default=None, metadata={"label": "Merge colour"}
    )
    message_action_edit: str | None = field(
        default=None, metadata={"label": "Edit colour"}
    )
    message_action_continue: str | None = field(
        default=None, metadata={"label": "Continue and resend colour"}
    )
    canvas: str | None = field(default=None, metadata={"label": "Page background"})
    surface: str | None = field(default=None, metadata={"label": "Panel background"})
    surface_raised: str | None = field(
        default=None, metadata={"label": "Raised background"}
    )
    border: str | None = field(default=None, metadata={"label": "Surface border"})
    border_strong: str | None = field(
        default=None, metadata={"label": "Strong surface border"}
    )
    text: str | None = field(default=None, metadata={"label": "Body text"})
    text_muted: str | None = field(default=None, metadata={"label": "Secondary text"})


@dataclass(frozen=True, slots=True)
class ThemeSettings:
    """Independent App, User, Assistant, and System colourways."""

    app: AppColorwaySettings = field(
        default_factory=lambda: AppColorwaySettings(_builtin_theme_value("app"))
    )
    user: ColorwaySettings = field(
        default_factory=lambda: ColorwaySettings(_builtin_theme_value("user"))
    )
    assistant: ColorwaySettings = field(
        default_factory=lambda: ColorwaySettings(_builtin_theme_value("assistant"))
    )
    system: ColorwaySettings = field(
        default_factory=lambda: ColorwaySettings(_builtin_theme_value("system"))
    )

    def colorways(self) -> tuple[tuple[ThemeColorway, ColorwaySettings], ...]:
        """Return each colourway once in the editor and stylesheet order."""

        return (
            (ThemeColorway.APP, self.app),
            (ThemeColorway.USER, self.user),
            (ThemeColorway.ASSISTANT, self.assistant),
            (ThemeColorway.SYSTEM, self.system),
        )

    def values(self) -> dict[str, dict[str, str]]:
        """Return only configured colours; omitted shades remain automatic."""

        return {
            owner.value: {
                name: value
                for name, value in settings.values().items()
                if value is not None
            }
            for owner, settings in self.colorways()
        }

    def validate(self) -> None:
        for owner, settings in self.colorways():
            try:
                settings.validate()
            except ValueError as exc:
                raise ValueError(f"theme.{owner.value}.{exc}") from exc


@dataclass
class UiSettings:
    """Presentation and web-server settings."""

    host: str = "0.0.0.0"
    port: int = 8080
    dark_mode: bool = True
    auto_open_browser: bool = False
    active_chat_id: str = ""
    message_action_style: MessageActionStyle = MessageActionStyle.UNIFORM
    icon_colors: IconColorSettings = field(default_factory=IconColorSettings)
    starter_prompts: list[StarterPrompt] = field(
        default_factory=lambda: list(DEFAULT_STARTER_PROMPTS)
    )

    def validate(self) -> None:
        if not self.host.strip():
            raise ValueError("host cannot be blank")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        if not _are_booleans(self.dark_mode, self.auto_open_browser):
            raise ValueError("dark_mode and auto_open_browser must be booleans")
        if type(self.message_action_style) is not MessageActionStyle:
            raise TypeError("message_action_style must be a MessageActionStyle")
        self.icon_colors.validate()
        for prompt in self.starter_prompts:
            prompt.validate()


@dataclass
class LoggingSettings:
    """Optional logging-directory override."""

    enabled: bool = True
    directory: Path | None = None

    def validate(self) -> None:
        if not _are_booleans(self.enabled):
            raise ValueError("logging.enabled must be a boolean")


@dataclass
class HostStatsDeviceSettings:
    """Per-device host-statistics presentation settings."""

    visible: bool = True
    label: str = ""

    def validate(self) -> None:
        if not _are_booleans(self.visible):
            raise ValueError("device visibility must be a boolean")


@dataclass
class HostStatsSettings:
    """Visibility controls for the host diagnostics dialog."""

    system: bool = True
    cpu: bool = True
    gpu: bool = True
    network: bool = True
    gpus: dict[str, HostStatsDeviceSettings] = field(default_factory=dict)
    interfaces: dict[str, HostStatsDeviceSettings] = field(default_factory=dict)
    activity_start_color: str = "#1266d6"
    activity_end_color: str = "#ff3048"

    def validate(self) -> None:
        if not _are_booleans(self.system, self.cpu, self.gpu, self.network):
            raise ValueError("host-stat visibility values must be booleans")
        for field_name, value in (
            ("activity_start_color", self.activity_start_color),
            ("activity_end_color", self.activity_end_color),
        ):
            if not _HEX_COLOR_PATTERN.fullmatch(value):
                raise ValueError(f"{field_name} must be a six-digit hexadecimal color")
        for device_id, settings in {**self.gpus, **self.interfaces}.items():
            if not device_id.strip():
                raise ValueError("host-stat device IDs cannot be blank")
            settings.validate()


@dataclass
class DeviceAccessSettings:
    """Access state for one browser/device identifier."""

    access_allowed: bool = False
    label: str = ""
    last_ip: str = ""
    hostname: str = ""
    first_seen_at: str = ""
    last_seen_at: str = ""

    def validate(self) -> None:
        if not _are_booleans(self.access_allowed):
            raise ValueError("device access_allowed must be a boolean")


@dataclass
class AccessSettings:
    """Default-private browser/device access controls."""

    default_private: bool = True
    allow_localhost_without_approval: bool = True
    global_settings_for_approved: bool = False
    allow_network_device_reassociation: bool = False
    approval_phrase: str = ""
    devices: dict[str, DeviceAccessSettings] = field(default_factory=dict)

    def validate(self) -> None:
        flags: tuple[bool, ...] = (
            self.default_private,
            self.allow_localhost_without_approval,
            self.global_settings_for_approved,
            self.allow_network_device_reassociation,
        )
        if not _are_booleans(*flags):
            raise ValueError("access flags must be booleans")
        for device_id, device in self.devices.items():
            if not device_id.strip():
                raise ValueError("access device IDs cannot be blank")
            device.validate()


@dataclass
class AppConfig:
    """Complete runtime configuration with one authoritative path set."""

    paths: AppPaths = field(default_factory=default_paths)
    server: ServerSettings = field(default_factory=ServerSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    theme: ThemeSettings = field(default_factory=ThemeSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    host_stats: HostStatsSettings = field(default_factory=HostStatsSettings)
    access: AccessSettings = field(default_factory=AccessSettings)

    @property
    def config_file(self) -> Path:
        return self.paths.config_file

    @property
    def data_dir(self) -> Path:
        return self.paths.data_dir

    @property
    def chats_file(self) -> Path:
        return self.paths.chats_file

    @property
    def characters_directory(self) -> Path:
        return self.paths.characters_directory

    @property
    def character_presets_directory(self) -> Path:
        return self.paths.character_presets_directory

    @property
    def character_name_suggestions_file(self) -> Path:
        return self.paths.character_name_suggestions_file

    @property
    def log_directory(self) -> Path:
        return self.logging.directory or self.paths.log_directory

    def validate(self) -> None:
        self.server.validate()
        self.generation.validate()
        self.ui.validate()
        self.theme.validate()
        self.logging.validate()
        log_directory: Path = self.log_directory.resolve()
        protected_paths: tuple[Path, ...] = (
            self.paths.home,
            self.paths.data_dir,
            self.paths.chats_file.parent / "chats",
            self.paths.characters_directory,
            self.paths.character_presets_directory,
        )
        if any(
            protected_path.is_relative_to(log_directory)
            for protected_path in protected_paths
        ):
            raise ValueError("logging.directory must not contain application data")
        self.host_stats.validate()
        self.access.validate()
