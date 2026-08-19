"""Configuration loading and defaults for Jouzetsu.

Reads ``config.json`` from the configured application home, falling back to
defaults if the file is missing or invalid. Runtime-safe generation defaults
are also editable through the UI.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias, TypedDict, cast

from .atomic_write import atomic_write_text

log: Logger = logging.getLogger(__name__)
APP_HOME_ENVIRONMENT_VARIABLE: Final[str] = "JOUZETSU_HOME"
_SOURCE_CHECKOUT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def _normalise_app_home(app_home: Path) -> Path:
    """Return one absolute writable-directory target without creating it."""

    resolved_home: Path = app_home.expanduser().resolve()
    if resolved_home.exists() and not resolved_home.is_dir():
        raise ValueError(f"application home must be a directory: {resolved_home}")
    return resolved_home


def resolve_app_home(app_home: Path | None = None) -> Path:
    """Resolve the explicit, environment-selected, or development application home."""

    if app_home is not None:
        return _normalise_app_home(app_home)
    configured_home: str | None = os.environ.get(APP_HOME_ENVIRONMENT_VARIABLE)
    if configured_home is not None:
        if not configured_home.strip():
            raise ValueError(f"{APP_HOME_ENVIRONMENT_VARIABLE} must name a directory")
        return _normalise_app_home(Path(configured_home))
    if (_SOURCE_CHECKOUT_ROOT / "pyproject.toml").is_file():
        return _SOURCE_CHECKOUT_ROOT
    return Path.cwd().resolve()


@dataclass(frozen=True, slots=True)
class AppPaths:
    """The mutable paths contained by one self-contained application home."""

    home: Path
    config_file: Path
    data_dir: Path
    chats_file: Path
    characters_directory: Path
    character_presets_directory: Path
    log_directory: Path

    @classmethod
    def for_home(cls, app_home: Path) -> AppPaths:
        """Build every default mutable path from one validated home directory."""

        home: Path = _normalise_app_home(app_home)
        data_dir: Path = home / "data"
        return cls(
            home=home,
            config_file=home / "config.json",
            data_dir=data_dir,
            chats_file=data_dir / "chats.json",
            characters_directory=data_dir / "characters",
            character_presets_directory=data_dir / "character-presets",
            log_directory=data_dir / "logs",
        )


DEFAULT_APP_HOME: Path = resolve_app_home()
DEFAULT_PATHS: AppPaths = AppPaths.for_home(DEFAULT_APP_HOME)
CONFIG_PATH: Path = DEFAULT_PATHS.config_file
DEFAULT_DATA_DIR: Path = DEFAULT_PATHS.data_dir
CHATS_FILE: Path = DEFAULT_PATHS.chats_file
CHARACTERS_DIRECTORY: Path = DEFAULT_PATHS.characters_directory
CHARACTER_PRESETS_DIRECTORY: Path = DEFAULT_PATHS.character_presets_directory
DEFAULT_LOG_DIR: Path = DEFAULT_PATHS.log_directory
SYSTEM_LOG_FILE_NAME: str = "system.log"
ERROR_LOG_FILE_NAME: str = "error.log"
CHAT_LOG_FILE_NAME: str = "chat.log"
MessageActionIconStyle: TypeAlias = Literal["monochrome", "muted_color"]
MESSAGE_ACTION_ICON_STYLES: tuple[MessageActionIconStyle, ...] = ("monochrome", "muted_color")
_HEX_COLOR_PATTERN: re.Pattern[str] = re.compile(r"#[0-9a-fA-F]{6}")


def _path_to_config_string(path: Path, *, app_home: Path) -> str:
    try:
        return str(path.relative_to(app_home))
    except ValueError:
        return str(path)


class ServerConfigJson(TypedDict):
    base_url: str
    api_key: str
    default_model: str
    model_aliases: dict[str, str]
    auto_unload_minutes: int | None


class SpellingReplacementJson(TypedDict):
    source: str
    replacement: str


class GenerationConfigJson(TypedDict):
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt: str
    continuity_review: bool
    british_english: bool
    british_spelling_replacements: list[SpellingReplacementJson]


class StarterPromptConfigJson(TypedDict):
    label: str
    content: str


class UiConfigJson(TypedDict):
    host: str
    port: int
    dark_mode: bool
    auto_open_browser: bool
    active_chat_id: str
    message_action_icon_style: MessageActionIconStyle
    icon_colors: IconColorConfigJson
    starter_prompts: list[StarterPromptConfigJson]


class IconColorConfigJson(TypedDict):
    """Persisted palette for the app icon artwork."""

    linework_color: str
    accent_color: str
    surface_color: str


class ThemeConfigJson(TypedDict):
    """The complete semantic colour palette consumed by the web client."""

    canvas: str
    surface: str
    surface_raised: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_inverse: str
    primary: str
    primary_hover: str
    primary_muted: str
    primary_subtle: str
    on_accent: str
    secondary: str
    secondary_muted: str
    secondary_subtle: str
    edit: str
    streaming_highlight: str
    action_delete: str
    action_regenerate: str
    action_delete_muted: str
    action_regenerate_muted: str
    action_merge_muted: str
    action_edit_muted: str
    action_continue_muted: str


class LoggingConfigJson(TypedDict):
    enabled: bool
    directory: str


class HostStatsDeviceConfigJson(TypedDict):
    visible: bool
    label: str


class HostStatsConfigJson(TypedDict):
    system: bool
    cpu: bool
    gpu: bool
    network: bool
    gpus: dict[str, HostStatsDeviceConfigJson]
    interfaces: dict[str, HostStatsDeviceConfigJson]
    activity_start_color: str
    activity_end_color: str


class DeviceAccessConfigJson(TypedDict):
    access_allowed: bool
    label: str
    last_ip: str
    hostname: str
    first_seen_at: str
    last_seen_at: str


class AccessConfigJson(TypedDict):
    default_private: bool
    allow_localhost_without_approval: bool
    global_settings_for_approved: bool
    allow_network_device_reassociation: bool
    approval_phrase: str
    devices: dict[str, DeviceAccessConfigJson]


class AppConfigJson(TypedDict):
    server: ServerConfigJson
    generation: GenerationConfigJson
    ui: UiConfigJson
    theme: ThemeConfigJson
    logging: LoggingConfigJson
    host_stats: HostStatsConfigJson
    access: AccessConfigJson


class PartialDeviceAccessConfigJson(TypedDict, total=False):
    access_allowed: bool
    label: str
    last_ip: str
    hostname: str
    first_seen_at: str
    last_seen_at: str


class PartialAccessConfigJson(TypedDict, total=False):
    default_private: bool
    allow_localhost_without_approval: bool
    global_settings_for_approved: bool
    allow_network_device_reassociation: bool
    approval_phrase: str
    devices: dict[str, PartialDeviceAccessConfigJson]


@dataclass
class ServerSettings:
    """Connection details for the LM Studio API server."""

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    default_model: str = ""
    model_aliases: dict[str, str] = field(default_factory=dict)
    auto_unload_minutes: int | None = None

    def validate(self) -> None:
        if self.auto_unload_minutes is None:
            return
        if isinstance(self.auto_unload_minutes, bool):
            raise ValueError("auto_unload_minutes must be an integer")
        if self.auto_unload_minutes < 1:
            raise ValueError("auto_unload_minutes must be at least 1")

    def apply(self, settings: "ServerSettings") -> None:
        """Replace these live settings with one validated configuration value.

        The application owns this instance and the LM Studio client retains a
        reference to it, so retaining its identity keeps configuration as the
        single source of truth for both components.
        """

        settings.validate()
        self.base_url = settings.base_url
        self.api_key = settings.api_key
        self.default_model = settings.default_model
        self.model_aliases = dict[str, str](settings.model_aliases)
        self.auto_unload_minutes = settings.auto_unload_minutes


@dataclass
class SpellingReplacement:
    """One case-insensitive whole-word spelling replacement."""

    source: str
    replacement: str

    def validate(self) -> None:
        if not self.source.strip():
            raise ValueError("spelling replacement source cannot be empty")
        if not self.replacement.strip():
            raise ValueError("spelling replacement replacement cannot be empty")
        if self.source != self.source.strip():
            raise ValueError("spelling replacement source cannot have surrounding whitespace")
        if self.replacement != self.replacement.strip():
            raise ValueError("spelling replacement replacement cannot have surrounding whitespace")
        if any(char.isspace() for char in self.source):
            raise ValueError("spelling replacement source must be one word")
        if any(char.isspace() for char in self.replacement):
            raise ValueError("spelling replacement replacement must be one word")

    def to_dict(self) -> SpellingReplacementJson:
        return {"source": self.source, "replacement": self.replacement}


DEFAULT_BRITISH_SPELLING_REPLACEMENTS: tuple[SpellingReplacement, ...] = (
    SpellingReplacement("mom", "mum"),
    SpellingReplacement("mommy", "mummy"),
    SpellingReplacement("momma", "mumma"),
    SpellingReplacement("color", "colour"),
    SpellingReplacement("colors", "colours"),
    SpellingReplacement("colored", "coloured"),
    SpellingReplacement("coloring", "colouring"),
    SpellingReplacement("colorful", "colourful"),
    SpellingReplacement("discolor", "discolour"),
    SpellingReplacement("discolored", "discoloured"),
    SpellingReplacement("discoloring", "discolouring"),
    SpellingReplacement("discoloration", "discolouration"),
    SpellingReplacement("favorite", "favourite"),
    SpellingReplacement("favorites", "favourites"),
    SpellingReplacement("favor", "favour"),
    SpellingReplacement("favors", "favours"),
    SpellingReplacement("favored", "favoured"),
    SpellingReplacement("favoring", "favouring"),
    SpellingReplacement("favorable", "favourable"),
    SpellingReplacement("unfavorable", "unfavourable"),
    SpellingReplacement("favoritism", "favouritism"),
    SpellingReplacement("honor", "honour"),
    SpellingReplacement("honors", "honours"),
    SpellingReplacement("honored", "honoured"),
    SpellingReplacement("honoring", "honouring"),
    SpellingReplacement("honorable", "honourable"),
    SpellingReplacement("dishonor", "dishonour"),
    SpellingReplacement("dishonored", "dishonoured"),
    SpellingReplacement("behavior", "behaviour"),
    SpellingReplacement("behaviors", "behaviours"),
    SpellingReplacement("behavioral", "behavioural"),
    SpellingReplacement("labor", "labour"),
    SpellingReplacement("labors", "labours"),
    SpellingReplacement("labored", "laboured"),
    SpellingReplacement("laboring", "labouring"),
    SpellingReplacement("neighbor", "neighbour"),
    SpellingReplacement("neighbors", "neighbours"),
    SpellingReplacement("neighborhood", "neighbourhood"),
    SpellingReplacement("neighboring", "neighbouring"),
    SpellingReplacement("center", "centre"),
    SpellingReplacement("centers", "centres"),
    SpellingReplacement("centered", "centred"),
    SpellingReplacement("centering", "centring"),
    SpellingReplacement("kilometer", "kilometre"),
    SpellingReplacement("kilometers", "kilometres"),
    SpellingReplacement("centimeter", "centimetre"),
    SpellingReplacement("centimeters", "centimetres"),
    SpellingReplacement("millimeter", "millimetre"),
    SpellingReplacement("millimeters", "millimetres"),
    SpellingReplacement("liter", "litre"),
    SpellingReplacement("liters", "litres"),
    SpellingReplacement("fiber", "fibre"),
    SpellingReplacement("fibers", "fibres"),
    SpellingReplacement("theater", "theatre"),
    SpellingReplacement("theaters", "theatres"),
    SpellingReplacement("gray", "grey"),
    SpellingReplacement("grays", "greys"),
    SpellingReplacement("grayed", "greyed"),
    SpellingReplacement("graying", "greying"),
    SpellingReplacement("catalog", "catalogue"),
    SpellingReplacement("catalogs", "catalogues"),
    SpellingReplacement("cataloged", "catalogued"),
    SpellingReplacement("cataloging", "cataloguing"),
    SpellingReplacement("dialog", "dialogue"),
    SpellingReplacement("dialogs", "dialogues"),
    SpellingReplacement("organize", "organise"),
    SpellingReplacement("organizes", "organises"),
    SpellingReplacement("organized", "organised"),
    SpellingReplacement("organizing", "organising"),
    SpellingReplacement("organization", "organisation"),
    SpellingReplacement("organizations", "organisations"),
    SpellingReplacement("organizational", "organisational"),
    SpellingReplacement("realize", "realise"),
    SpellingReplacement("realizes", "realises"),
    SpellingReplacement("realized", "realised"),
    SpellingReplacement("realizing", "realising"),
    SpellingReplacement("recognize", "recognise"),
    SpellingReplacement("recognizes", "recognises"),
    SpellingReplacement("recognized", "recognised"),
    SpellingReplacement("recognizing", "recognising"),
    SpellingReplacement("analyze", "analyse"),
    SpellingReplacement("analyzes", "analyses"),
    SpellingReplacement("analyzed", "analysed"),
    SpellingReplacement("analyzing", "analysing"),
    SpellingReplacement("analyzer", "analyser"),
    SpellingReplacement("analyzers", "analysers"),
    SpellingReplacement("paralyze", "paralyse"),
    SpellingReplacement("paralyzed", "paralysed"),
    SpellingReplacement("paralyzing", "paralysing"),
    SpellingReplacement("defense", "defence"),
    SpellingReplacement("defenses", "defences"),
    SpellingReplacement("offense", "offence"),
    SpellingReplacement("offenses", "offences"),
    SpellingReplacement("pretense", "pretence"),
    SpellingReplacement("traveling", "travelling"),
    SpellingReplacement("traveled", "travelled"),
    SpellingReplacement("traveler", "traveller"),
    SpellingReplacement("travelers", "travellers"),
    SpellingReplacement("canceled", "cancelled"),
    SpellingReplacement("canceling", "cancelling"),
    SpellingReplacement("modeled", "modelled"),
    SpellingReplacement("modeling", "modelling"),
    SpellingReplacement("jewelry", "jewellery"),
    SpellingReplacement("plow", "plough"),
    SpellingReplacement("plowed", "ploughed"),
    SpellingReplacement("plowing", "ploughing"),
    SpellingReplacement("mold", "mould"),
    SpellingReplacement("molds", "moulds"),
    SpellingReplacement("molded", "moulded"),
    SpellingReplacement("molding", "moulding"),
    SpellingReplacement("smolder", "smoulder"),
    SpellingReplacement("smoldering", "smouldering"),
    SpellingReplacement("sulfur", "sulphur"),
    SpellingReplacement("aluminum", "aluminium"),
    SpellingReplacement("airplane", "aeroplane"),
    SpellingReplacement("airplanes", "aeroplanes"),
    SpellingReplacement("tire", "tyre"),
    SpellingReplacement("tires", "tyres"),
    SpellingReplacement("curb", "kerb"),
    SpellingReplacement("specialty", "speciality"),
    SpellingReplacement("specialties", "specialities"),
    SpellingReplacement("cozy", "cosy"),
    SpellingReplacement("mustache", "moustache"),
    SpellingReplacement("pajamas", "pyjamas"),
    SpellingReplacement("esthetic", "aesthetic"),
    SpellingReplacement("estrogen", "oestrogen"),
    SpellingReplacement("pediatric", "paediatric"),
    SpellingReplacement("pediatrics", "paediatrics"),
    SpellingReplacement("fetus", "foetus"),
    SpellingReplacement("fetuses", "foetuses"),
    SpellingReplacement("anemia", "anaemia"),
    SpellingReplacement("anemic", "anaemic"),
    SpellingReplacement("leukemia", "leukaemia"),
    SpellingReplacement("diarrhea", "diarrhoea"),
    SpellingReplacement("hemorrhoid", "haemorrhoid"),
    SpellingReplacement("hemorrhoids", "haemorrhoids"),
    SpellingReplacement("orthopedic", "orthopaedic"),
    SpellingReplacement("maneuver", "manoeuvre"),
    SpellingReplacement("maneuvers", "manoeuvres"),
    SpellingReplacement("maneuvered", "manoeuvred"),
    SpellingReplacement("maneuvering", "manoeuvring"),
    SpellingReplacement("toward", "towards"),
    SpellingReplacement("enroll", "enrol"),
    SpellingReplacement("enrollment", "enrolment"),
    SpellingReplacement("installment", "instalment"),
    SpellingReplacement("installments", "instalments"),
    SpellingReplacement("fulfill", "fulfil"),
    SpellingReplacement("fulfillment", "fulfilment"),
    SpellingReplacement("skillful", "skilful"),
    SpellingReplacement("willful", "wilful"),
    SpellingReplacement("judgment", "judgement"),
    SpellingReplacement("acknowledgment", "acknowledgement"),
    SpellingReplacement("aging", "ageing"),
    SpellingReplacement("ax", "axe"),
)


@dataclass
class GenerationSettings:
    """Default sampling parameters sent on every request."""

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
        default_factory=lambda: list(DEFAULT_BRITISH_SPELLING_REPLACEMENTS)
    )

    def validate(self) -> None:
        """Raise when generation defaults cannot form a valid request."""
        if isinstance(self.temperature, bool):
            raise ValueError("temperature must be a number")
        if not math.isfinite(self.temperature) or not (
            self.MIN_TEMPERATURE <= self.temperature <= self.MAX_TEMPERATURE
        ):
            raise ValueError(f"temperature must be between {self.MIN_TEMPERATURE:g} and {self.MAX_TEMPERATURE:g}")
        if isinstance(self.top_p, bool):
            raise ValueError("top_p must be a number")
        if not math.isfinite(self.top_p) or not self.MIN_TOP_P <= self.top_p <= self.MAX_TOP_P:
            raise ValueError(f"top_p must be between {self.MIN_TOP_P:g} and {self.MAX_TOP_P:g}")
        if isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens < self.MIN_MAX_TOKENS:
            raise ValueError(f"max_tokens must be at least {self.MIN_MAX_TOKENS}")
        continuity_review: object = cast(object, self.continuity_review)
        if not isinstance(continuity_review, bool):
            raise ValueError("continuity_review must be a boolean")
        seen_sources: set[str] = set()
        for replacement in self.british_spelling_replacements:
            replacement.validate()
            source_key: str = replacement.source.casefold()
            if source_key in seen_sources:
                raise ValueError(f"duplicate spelling replacement source: {replacement.source}")
            seen_sources.add(source_key)


CONTINUE_PROMPT_CONTENT: str = (
    "Continue the previous assistant response exactly from where it stopped. "
    "Do not repeat earlier text and do not add prefatory wording."
)


@dataclass(frozen=True)
class StarterPrompt:
    """A labelled composer shortcut stored in UI configuration."""

    label: str
    content: str


DEFAULT_STARTER_PROMPTS: tuple[StarterPrompt, ...] = (
    StarterPrompt(label="Think it through", content="Help me think through this"),
    StarterPrompt(label="Explain", content="Explain a concept"),
    StarterPrompt(label="Make a plan", content="Make a plan"),
    StarterPrompt(label="Summarise", content="Summarise something"),
    StarterPrompt(label="Continue", content=CONTINUE_PROMPT_CONTENT),
)


@dataclass
class IconColorSettings:
    """The three independently configurable colours in the app icon."""

    linework_color: str = "#000000"
    accent_color: str = "#D60000"
    surface_color: str = "#F9DED7"

    def validate(self) -> None:
        """Raise when an icon colour cannot be safely embedded in SVG."""

        for field_name, value in (
            ("linework_color", self.linework_color),
            ("accent_color", self.accent_color),
            ("surface_color", self.surface_color),
        ):
            if not _HEX_COLOR_PATTERN.fullmatch(value):
                raise ValueError(f"{field_name} must be a six-digit hexadecimal color")


@dataclass(frozen=True, slots=True)
class ThemeSettings:
    """Semantic colour tokens for every non-user-generated UI surface."""

    canvas: str = "#000000"
    surface: str = "#050505"
    surface_raised: str = "#0b090d"
    border: str = "#241a29"
    border_strong: str = "#38253f"
    text: str = "#f2edf5"
    text_muted: str = "#d0c7d5"
    text_inverse: str = "#ffffff"
    primary: str = "#ff3048"
    primary_hover: str = "#ff5367"
    primary_muted: str = "#741628"
    primary_subtle: str = "#21070d"
    on_accent: str = "#070107"
    secondary: str = "#ad6cff"
    secondary_muted: str = "#57357a"
    secondary_subtle: str = "#180d22"
    edit: str = "#4ea3ff"
    streaming_highlight: str = "#eadcff"
    action_delete: str = "#b9626b"
    action_regenerate: str = "#bd7d4d"
    action_delete_muted: str = "#e68e97"
    action_regenerate_muted: str = "#e3a26b"
    action_merge_muted: str = "#75a7df"
    action_edit_muted: str = "#8fbe9b"
    action_continue_muted: str = "#b89ae0"

    def validate(self) -> None:
        """Raise when a palette entry cannot be safely embedded in CSS."""

        for field_name, value in cast(dict[str, str], self.to_dict()).items():
            if not _HEX_COLOR_PATTERN.fullmatch(value):
                raise ValueError(f"theme.{field_name} must be a six-digit hexadecimal color")

    def to_dict(self) -> ThemeConfigJson:
        """Return this immutable palette as its JSON representation."""

        return {
            "canvas": self.canvas,
            "surface": self.surface,
            "surface_raised": self.surface_raised,
            "border": self.border,
            "border_strong": self.border_strong,
            "text": self.text,
            "text_muted": self.text_muted,
            "text_inverse": self.text_inverse,
            "primary": self.primary,
            "primary_hover": self.primary_hover,
            "primary_muted": self.primary_muted,
            "primary_subtle": self.primary_subtle,
            "on_accent": self.on_accent,
            "secondary": self.secondary,
            "secondary_muted": self.secondary_muted,
            "secondary_subtle": self.secondary_subtle,
            "edit": self.edit,
            "streaming_highlight": self.streaming_highlight,
            "action_delete": self.action_delete,
            "action_regenerate": self.action_regenerate,
            "action_delete_muted": self.action_delete_muted,
            "action_regenerate_muted": self.action_regenerate_muted,
            "action_merge_muted": self.action_merge_muted,
            "action_edit_muted": self.action_edit_muted,
            "action_continue_muted": self.action_continue_muted,
        }


@dataclass
class UiSettings:
    """UI-level knobs."""

    host: str = "0.0.0.0"
    port: int = 8080
    dark_mode: bool = True
    auto_open_browser: bool = False
    active_chat_id: str = ""
    message_action_icon_style: MessageActionIconStyle = "monochrome"
    icon_colors: IconColorSettings = field(default_factory=IconColorSettings)
    starter_prompts: list[StarterPrompt] = field(default_factory=lambda: list(DEFAULT_STARTER_PROMPTS))


@dataclass
class LoggingSettings:
    """File logging destinations."""

    enabled: bool = True
    directory: Path = DEFAULT_LOG_DIR

    @property
    def system_log_file(self) -> Path:
        return self.directory / SYSTEM_LOG_FILE_NAME

    @property
    def error_log_file(self) -> Path:
        return self.directory / ERROR_LOG_FILE_NAME

    @property
    def chat_log_file(self) -> Path:
        return self.directory / CHAT_LOG_FILE_NAME

    def ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)


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


@dataclass
class HostStatsDeviceSettings:
    """Per-device Host Stats presentation settings."""

    visible: bool = True
    label: str = ""


@dataclass
class DeviceAccessSettings:
    """Access state for one browser/device identifier."""

    access_allowed: bool = False
    label: str = ""
    last_ip: str = ""
    hostname: str = ""
    first_seen_at: str = ""
    last_seen_at: str = ""


@dataclass
class AccessSettings:
    """Default-private browser/device access controls."""

    default_private: bool = True
    allow_localhost_without_approval: bool = True
    global_settings_for_approved: bool = False
    allow_network_device_reassociation: bool = False
    approval_phrase: str = ""
    devices: dict[str, DeviceAccessSettings] = field(default_factory=dict)


@dataclass
class AppConfig:
    server: ServerSettings = field(default_factory=ServerSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    theme: ThemeSettings = field(default_factory=ThemeSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    host_stats: HostStatsSettings = field(default_factory=HostStatsSettings)
    access: AccessSettings = field(default_factory=AccessSettings)
    data_dir: Path = DEFAULT_DATA_DIR
    chats_file: Path = CHATS_FILE
    characters_directory: Path = CHARACTERS_DIRECTORY
    character_presets_directory: Path = CHARACTER_PRESETS_DIRECTORY
    config_file: Path = CONFIG_PATH

    def __post_init__(self) -> None:
        """Keep default character files beside a caller-provided data directory."""

        if self.data_dir != DEFAULT_DATA_DIR and self.characters_directory == CHARACTERS_DIRECTORY:
            self.characters_directory = self.data_dir / "characters"
        if self.data_dir != DEFAULT_DATA_DIR and self.character_presets_directory == CHARACTER_PRESETS_DIRECTORY:
            self.character_presets_directory = self.data_dir / "character-presets"

    def to_dict(self) -> AppConfigJson:
        return {
            "server": {
                "base_url": self.server.base_url,
                "api_key": self.server.api_key,
                "default_model": self.server.default_model,
                "model_aliases": self.server.model_aliases,
                "auto_unload_minutes": self.server.auto_unload_minutes,
            },
            "generation": {
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "max_tokens": self.generation.max_tokens,
                "system_prompt": self.generation.system_prompt,
                "continuity_review": self.generation.continuity_review,
                "british_english": self.generation.british_english,
                "british_spelling_replacements": [
                    replacement.to_dict() for replacement in self.generation.british_spelling_replacements
                ],
            },
            "ui": {
                "host": self.ui.host,
                "port": self.ui.port,
                "dark_mode": self.ui.dark_mode,
                "auto_open_browser": self.ui.auto_open_browser,
                "active_chat_id": self.ui.active_chat_id,
                "message_action_icon_style": self.ui.message_action_icon_style,
                "icon_colors": {
                    "linework_color": self.ui.icon_colors.linework_color,
                    "accent_color": self.ui.icon_colors.accent_color,
                    "surface_color": self.ui.icon_colors.surface_color,
                },
                "starter_prompts": [
                    {"label": prompt.label, "content": prompt.content} for prompt in self.ui.starter_prompts
                ],
            },
            "theme": self.theme.to_dict(),
            "logging": {
                "enabled": self.logging.enabled,
                "directory": _path_to_config_string(self.logging.directory, app_home=self.config_file.parent),
            },
            "host_stats": {
                "system": self.host_stats.system,
                "cpu": self.host_stats.cpu,
                "gpu": self.host_stats.gpu,
                "network": self.host_stats.network,
                "gpus": {
                    device_id: {"visible": settings.visible, "label": settings.label}
                    for device_id, settings in self.host_stats.gpus.items()
                },
                "interfaces": {
                    name: {"visible": settings.visible, "label": settings.label}
                    for name, settings in self.host_stats.interfaces.items()
                },
                "activity_start_color": self.host_stats.activity_start_color,
                "activity_end_color": self.host_stats.activity_end_color,
            },
            "access": {
                "default_private": self.access.default_private,
                "allow_localhost_without_approval": self.access.allow_localhost_without_approval,
                "global_settings_for_approved": (self.access.global_settings_for_approved),
                "allow_network_device_reassociation": self.access.allow_network_device_reassociation,
                "approval_phrase": self.access.approval_phrase,
                "devices": {
                    device_id: {
                        "access_allowed": device.access_allowed,
                        "label": device.label,
                        "last_ip": device.last_ip,
                        "hostname": device.hostname,
                        "first_seen_at": device.first_seen_at,
                        "last_seen_at": device.last_seen_at,
                    }
                    for device_id, device in self.access.devices.items()
                },
            },
        }

    def ensure_directories(self) -> None:
        """Make sure runtime directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chats_file.parent.mkdir(parents=True, exist_ok=True)
        self.characters_directory.mkdir(parents=True, exist_ok=True)
        self.character_presets_directory.mkdir(parents=True, exist_ok=True)
        if self.logging.enabled:
            self.logging.ensure_directory()

    def save(self) -> None:
        """Persist current config to disk as pretty JSON."""
        atomic_write_text(self.config_file, json.dumps(self.to_dict(), indent=4) + "\n")
        log.info("saved configuration path=%s", self.config_file)


def _default_config_json() -> AppConfigJson:
    """Build the serialisable default config from dataclass defaults."""
    return AppConfig().to_dict()


DEFAULT_CONFIG_JSON: AppConfigJson = _default_config_json()


def write_default_config(path: Path = CONFIG_PATH) -> None:
    """Write a fresh `config.json` with default values."""
    atomic_write_text(path, json.dumps(DEFAULT_CONFIG_JSON, indent=4) + "\n")


def _as_section(raw: object, *, name: str) -> dict[str, object]:
    """Return a config section mapping or an empty one on invalid input."""
    if isinstance(raw, dict):
        mapping: dict[object, object] = cast(dict[object, object], raw)
        return {key: value for key, value in mapping.items() if isinstance(key, str)}
    if raw is not None:
        log.warning("config.%s must be an object; using defaults for that section", name)
    return {}


def _coerce_str(raw: object, *, default: str, path: str) -> str:
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw
    log.warning("%s must be a string; using default %r", path, default)
    return default


def _coerce_bool(raw: object, *, default: bool, path: str) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    log.warning("%s must be a boolean; using default %r", path, default)
    return default


def _coerce_hex_color(raw: object, *, default: str, path: str) -> str:
    if raw is None:
        return default
    if isinstance(raw, str) and _HEX_COLOR_PATTERN.fullmatch(raw):
        return raw.lower()
    log.warning("%s must be a six-digit hexadecimal color; using default %r", path, default)
    return default


def _coerce_int(raw: object, *, default: int, path: str) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        log.warning("%s must be an integer; using default %r", path, default)
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            pass
    log.warning("%s must be an integer; using default %r", path, default)
    return default


def _coerce_optional_int(raw: object, *, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        log.warning("%s must be an integer or null; using null", path)
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            pass
    log.warning("%s must be an integer or null; using null", path)
    return None


def _coerce_float(raw: object, *, default: float, path: str) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool):
        log.warning("%s must be a number; using default %r", path, default)
        return default
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            pass
    log.warning("%s must be a number; using default %r", path, default)
    return default


def _coerce_string_mapping(raw: object, *, path: str) -> dict[str, str]:
    """Validate a JSON object whose keys and values are non-empty strings."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        log.warning("%s must be an object containing string aliases; using an empty mapping", path)
        return {}

    aliases: dict[str, str] = {}
    for raw_key, raw_alias in cast(dict[object, object], raw).items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            log.warning("%s contains an invalid model key; ignoring it", path)
            continue
        if not isinstance(raw_alias, str) or not raw_alias.strip():
            log.warning("%s.%s must be a non-empty string; ignoring it", path, raw_key)
            continue
        aliases[raw_key.strip()] = raw_alias.strip()
    return aliases


def _coerce_spelling_replacements(
    raw: object,
    *,
    defaults: list[SpellingReplacement],
    path: str,
) -> list[SpellingReplacement]:
    """Load valid whole-word replacement pairs, preserving an intentionally empty list."""
    if raw is None:
        return list(defaults)
    if not isinstance(raw, list):
        log.warning("%s must be a list of replacement pairs; using defaults", path)
        return list(defaults)

    replacements: list[SpellingReplacement] = []
    seen_sources: set[str] = set()
    for index, item in enumerate(cast(list[object], raw)):
        if not isinstance(item, dict):
            log.warning("%s[%d] must be an object; ignoring it", path, index)
            continue
        values: dict[object, object] = cast(dict[object, object], item)
        raw_source: object = values.get("source")
        raw_replacement: object = values.get("replacement")
        if not isinstance(raw_source, str) or not isinstance(raw_replacement, str):
            log.warning("%s[%d] must contain source and replacement strings; ignoring it", path, index)
            continue
        replacement = SpellingReplacement(source=raw_source.strip(), replacement=raw_replacement.strip())
        try:
            replacement.validate()
        except ValueError as exc:
            log.warning("%s[%d] is invalid (%s); ignoring it", path, index, exc)
            continue
        source_key: str = replacement.source.casefold()
        if source_key in seen_sources:
            log.warning("%s[%d] duplicates source %r; ignoring it", path, index, replacement.source)
            continue
        seen_sources.add(source_key)
        replacements.append(replacement)
    return replacements


def _coerce_device_access_mapping(raw: object, *, path: str) -> dict[str, DeviceAccessSettings]:
    """Validate a JSON object of device ids to access records."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        log.warning("%s must be an object containing device access records; using an empty mapping", path)
        return {}

    devices: dict[str, DeviceAccessSettings] = {}
    for raw_device_id, raw_device in cast(dict[object, object], raw).items():
        if not isinstance(raw_device_id, str) or not raw_device_id.strip():
            log.warning("%s contains an invalid device id; ignoring it", path)
            continue
        section: dict[str, object] = _as_section(raw_device, name=f"access.devices.{raw_device_id}")
        device_id: str = raw_device_id.strip()
        devices[device_id] = DeviceAccessSettings(
            access_allowed=_coerce_bool(
                section.get("access_allowed"),
                default=False,
                path=f"{path}.{device_id}.access_allowed",
            ),
            label=_coerce_str(section.get("label"), default="", path=f"{path}.{device_id}.label"),
            last_ip=_coerce_str(section.get("last_ip"), default="", path=f"{path}.{device_id}.last_ip"),
            hostname=_coerce_str(section.get("hostname"), default="", path=f"{path}.{device_id}.hostname"),
            first_seen_at=_coerce_str(
                section.get("first_seen_at"),
                default="",
                path=f"{path}.{device_id}.first_seen_at",
            ),
            last_seen_at=_coerce_str(
                section.get("last_seen_at"),
                default="",
                path=f"{path}.{device_id}.last_seen_at",
            ),
        )
    return devices


def _load_server_settings(raw: object) -> ServerSettings:
    section: dict[str, object] = _as_section(raw, name="server")
    defaults: ServerSettings = ServerSettings()
    settings: ServerSettings = ServerSettings(
        base_url=_coerce_str(section.get("base_url"), default=defaults.base_url, path="config.server.base_url"),
        api_key=_coerce_str(section.get("api_key"), default=defaults.api_key, path="config.server.api_key"),
        default_model=_coerce_str(
            section.get("default_model"),
            default=defaults.default_model,
            path="config.server.default_model",
        ),
        model_aliases=_coerce_string_mapping(
            section.get("model_aliases"),
            path="config.server.model_aliases",
        ),
        auto_unload_minutes=_coerce_optional_int(
            section.get("auto_unload_minutes"),
            path="config.server.auto_unload_minutes",
        ),
    )
    try:
        settings.validate()
    except ValueError as exc:
        log.warning("config.server contains invalid values (%s); using server defaults", exc)
        return defaults
    return settings


def _load_generation_settings(raw: object) -> GenerationSettings:
    section: dict[str, object] = _as_section(raw, name="generation")
    defaults: GenerationSettings = GenerationSettings()
    settings: GenerationSettings = GenerationSettings(
        temperature=_coerce_float(
            section.get("temperature"),
            default=defaults.temperature,
            path="config.generation.temperature",
        ),
        top_p=_coerce_float(section.get("top_p"), default=defaults.top_p, path="config.generation.top_p"),
        max_tokens=_coerce_int(
            section.get("max_tokens"),
            default=defaults.max_tokens,
            path="config.generation.max_tokens",
        ),
        system_prompt=_coerce_str(
            section.get("system_prompt"),
            default=defaults.system_prompt,
            path="config.generation.system_prompt",
        ),
        continuity_review=_coerce_bool(
            section.get("continuity_review"),
            default=defaults.continuity_review,
            path="config.generation.continuity_review",
        ),
        british_english=_coerce_bool(
            section.get("british_english"),
            default=defaults.british_english,
            path="config.generation.british_english",
        ),
        british_spelling_replacements=_coerce_spelling_replacements(
            section.get("british_spelling_replacements"),
            defaults=defaults.british_spelling_replacements,
            path="config.generation.british_spelling_replacements",
        ),
    )
    try:
        settings.validate()
    except ValueError as exc:
        log.warning("config.generation contains invalid values (%s); using generation defaults", exc)
        return defaults
    return settings


def _load_ui_settings(raw: object) -> UiSettings:
    section: dict[str, object] = _as_section(raw, name="ui")
    defaults: UiSettings = UiSettings()
    return UiSettings(
        host=_coerce_str(section.get("host"), default=defaults.host, path="config.ui.host"),
        port=_coerce_int(section.get("port"), default=defaults.port, path="config.ui.port"),
        dark_mode=_coerce_bool(section.get("dark_mode"), default=defaults.dark_mode, path="config.ui.dark_mode"),
        auto_open_browser=_coerce_bool(
            section.get("auto_open_browser"),
            default=defaults.auto_open_browser,
            path="config.ui.auto_open_browser",
        ),
        active_chat_id=_coerce_str(
            section.get("active_chat_id"),
            default=defaults.active_chat_id,
            path="config.ui.active_chat_id",
        ),
        message_action_icon_style=_coerce_message_action_icon_style(
            section.get("message_action_icon_style"),
            default=defaults.message_action_icon_style,
            path="config.ui.message_action_icon_style",
        ),
        icon_colors=_load_icon_color_settings(
            section.get("icon_colors"),
            defaults=defaults.icon_colors,
        ),
        starter_prompts=_coerce_starter_prompts(
            section.get("starter_prompts"),
            defaults=defaults.starter_prompts,
            path="config.ui.starter_prompts",
        ),
    )


def _load_theme_settings(raw: object) -> ThemeSettings:
    """Load the complete palette, independently repairing invalid tokens."""

    section: dict[str, object] = _as_section(raw, name="theme")
    defaults: ThemeSettings = ThemeSettings()
    default_values: dict[str, str] = cast(dict[str, str], defaults.to_dict())
    settings: ThemeSettings = ThemeSettings(
        **{
            field_name: _coerce_hex_color(
                section.get(field_name),
                default=default,
                path=f"config.theme.{field_name}",
            )
            for field_name, default in default_values.items()
        }
    )
    settings.validate()
    return settings


def _load_icon_color_settings(raw: object, *, defaults: IconColorSettings) -> IconColorSettings:
    """Load the icon palette while independently falling back invalid fields."""

    section: dict[str, object] = _as_section(raw, name="ui.icon_colors")
    return IconColorSettings(
        linework_color=_coerce_hex_color(
            section.get("linework_color"),
            default=defaults.linework_color,
            path="config.ui.icon_colors.linework_color",
        ),
        accent_color=_coerce_hex_color(
            section.get("accent_color"),
            default=defaults.accent_color,
            path="config.ui.icon_colors.accent_color",
        ),
        surface_color=_coerce_hex_color(
            section.get("surface_color"),
            default=defaults.surface_color,
            path="config.ui.icon_colors.surface_color",
        ),
    )


def _coerce_message_action_icon_style(
    raw: object,
    *,
    default: MessageActionIconStyle,
    path: str,
) -> MessageActionIconStyle:
    if raw is None:
        return default
    if isinstance(raw, str) and raw in MESSAGE_ACTION_ICON_STYLES:
        return raw
    log.warning("%s must be one of %s; using default", path, ", ".join(MESSAGE_ACTION_ICON_STYLES))
    return default


def _coerce_starter_prompts(
    raw: object,
    *,
    defaults: list[StarterPrompt],
    path: str,
) -> list[StarterPrompt]:
    """Load valid labelled starter prompts, preserving an intentionally empty list."""
    if raw is None:
        return list(defaults)
    if not isinstance(raw, list):
        log.warning("%s must be a list of labelled prompts; using defaults", path)
        return list(defaults)

    prompts: list[StarterPrompt] = []
    for index, item in enumerate(cast(list[object], raw)):
        if not isinstance(item, dict):
            log.warning("%s[%d] must be an object; ignoring it", path, index)
            continue
        values: dict[object, object] = cast(dict[object, object], item)
        raw_label: object = values.get("label")
        raw_content: object = values.get("content")
        if not isinstance(raw_label, str) or not raw_label.strip():
            log.warning("%s[%d].label must be a non-empty string; ignoring it", path, index)
            continue
        if not isinstance(raw_content, str) or not raw_content.strip():
            log.warning("%s[%d].content must be a non-empty string; ignoring it", path, index)
            continue
        prompts.append(StarterPrompt(label=raw_label.strip(), content=raw_content))
    return prompts


def _path_from_config(raw: object, *, app_home: Path, default: Path, path: str) -> Path:
    if raw is None:
        return default
    if not isinstance(raw, str) or not raw.strip():
        log.warning("%s must be a non-empty string; using default %s", path, default)
        return default
    configured_path = Path(raw).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return app_home / configured_path


def _load_logging_settings(raw: object, *, app_paths: AppPaths) -> LoggingSettings:
    section: dict[str, object] = _as_section(raw, name="logging")
    defaults: LoggingSettings = LoggingSettings(directory=app_paths.log_directory)
    return LoggingSettings(
        enabled=_coerce_bool(section.get("enabled"), default=defaults.enabled, path="config.logging.enabled"),
        directory=_path_from_config(
            section.get("directory"),
            app_home=app_paths.home,
            default=defaults.directory,
            path="config.logging.directory",
        ),
    )


def _load_host_stats_settings(raw: object) -> HostStatsSettings:
    section: dict[str, object] = _as_section(raw, name="host_stats")
    defaults: HostStatsSettings = HostStatsSettings()
    def device_mapping(name: str) -> dict[str, HostStatsDeviceSettings]:
        raw_values: object = section.get(name, {})
        values: dict[str, HostStatsDeviceSettings] = {}
        if isinstance(raw_values, dict):
            for key, value in cast(dict[object, object], raw_values).items():
                if not isinstance(key, str):
                    log.warning("config.host_stats.%s must use string device IDs; ignoring invalid entry", name)
                    continue
                if isinstance(value, bool):
                    values[key] = HostStatsDeviceSettings(visible=value)
                    continue
                if isinstance(value, dict):
                    device_config: dict[object, object] = cast(dict[object, object], value)
                    visible: object = device_config.get("visible")
                    if not isinstance(visible, bool):
                        log.warning("config.host_stats.%s.%s must contain visible=true or false; ignoring invalid entry", name, key)
                        continue
                    label: object = device_config.get("label", "")
                    values[key] = HostStatsDeviceSettings(
                        visible=visible,
                        label=label if isinstance(label, str) else "",
                    )
                    continue
                log.warning("config.host_stats.%s.%s must contain visible=true or false; ignoring invalid entry", name, key)
        elif raw_values is not None:
            log.warning("config.host_stats.%s must be an object; using automatic device visibility", name)
        return values

    gpus: dict[str, HostStatsDeviceSettings] = device_mapping("gpus")
    interfaces: dict[str, HostStatsDeviceSettings] = device_mapping("interfaces")
    return HostStatsSettings(
        system=_coerce_bool(section.get("system"), default=defaults.system, path="config.host_stats.system"),
        cpu=_coerce_bool(section.get("cpu"), default=defaults.cpu, path="config.host_stats.cpu"),
        gpu=_coerce_bool(section.get("gpu"), default=defaults.gpu, path="config.host_stats.gpu"),
        network=_coerce_bool(section.get("network"), default=defaults.network, path="config.host_stats.network"),
        gpus=gpus,
        interfaces=interfaces,
        activity_start_color=_coerce_hex_color(
            section.get("activity_start_color"),
            default=defaults.activity_start_color,
            path="config.host_stats.activity_start_color",
        ),
        activity_end_color=_coerce_hex_color(
            section.get("activity_end_color"),
            default=defaults.activity_end_color,
            path="config.host_stats.activity_end_color",
        ),
    )


def _load_access_settings(raw: object) -> AccessSettings:
    section: dict[str, object] = _as_section(raw, name="access")
    defaults: AccessSettings = AccessSettings()
    return AccessSettings(
        default_private=_coerce_bool(
            section.get("default_private"),
            default=defaults.default_private,
            path="config.access.default_private",
        ),
        allow_localhost_without_approval=_coerce_bool(
            section.get("allow_localhost_without_approval"),
            default=defaults.allow_localhost_without_approval,
            path="config.access.allow_localhost_without_approval",
        ),
        global_settings_for_approved=_coerce_bool(
            section.get("global_settings_for_approved"),
            default=defaults.global_settings_for_approved,
            path="config.access.global_settings_for_approved",
        ),
        allow_network_device_reassociation=_coerce_bool(
            section.get("allow_network_device_reassociation"),
            default=defaults.allow_network_device_reassociation,
            path="config.access.allow_network_device_reassociation",
        ),
        approval_phrase=_coerce_str(
            section.get("approval_phrase"),
            default=defaults.approval_phrase,
            path="config.access.approval_phrase",
        ),
        devices=_coerce_device_access_mapping(section.get("devices"), path="config.access.devices"),
    )


def _apply_environment_overrides(config: AppConfig) -> None:
    """Apply documented environment overrides after file parsing."""
    host_override: str | None = os.environ.get("JOUZETSU_HOST")
    if host_override is not None:
        config.ui.host = host_override

    port_override: str | None = os.environ.get("JOUZETSU_PORT")
    if port_override is not None:
        try:
            config.ui.port = int(port_override)
        except ValueError as exc:
            raise ValueError("JOUZETSU_PORT must be an integer") from exc


def load_config(path: Path | None = None, *, app_home: Path | None = None) -> AppConfig:
    """Load config from one explicit or environment-selected application home.

    - Missing file: write defaults, return them.
    - Invalid JSON or schema: log a warning, return defaults in memory
      (the on-disk file is *not* overwritten; user can fix it manually).
    - Environment variables override file values after parsing.
    """
    if path is not None and app_home is not None:
        raise ValueError("specify either a config path or an application home, not both")
    config_path: Path
    if path is None:
        app_paths: AppPaths = AppPaths.for_home(resolve_app_home(app_home))
        config_path = app_paths.config_file
    else:
        config_path = path.expanduser().resolve()
        app_paths = AppPaths.for_home(config_path.parent)

    if not config_path.exists():
        log.info("No config.json found, writing defaults to %s", config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_default_config(config_path)

    try:
        raw: object = cast(object, json.loads(config_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        log.warning("config.json is not valid JSON (%s); using in-memory defaults", exc)
        raw = {}

    if not isinstance(raw, dict):
        log.warning("config.json root must be an object; using in-memory defaults")
        raw = {}
    root: dict[str, object] = {
        key: value for key, value in cast(dict[object, object], raw).items() if isinstance(key, str)
    }

    cfg: AppConfig = AppConfig(
        server=_load_server_settings(root.get("server")),
        generation=_load_generation_settings(root.get("generation")),
        ui=_load_ui_settings(root.get("ui")),
        theme=_load_theme_settings(root.get("theme")),
        logging=_load_logging_settings(root.get("logging"), app_paths=app_paths),
        host_stats=_load_host_stats_settings(root.get("host_stats")),
        access=_load_access_settings(root.get("access")),
        data_dir=app_paths.data_dir,
        chats_file=app_paths.chats_file,
        characters_directory=app_paths.characters_directory,
        character_presets_directory=app_paths.character_presets_directory,
        config_file=config_path,
    )
    _apply_environment_overrides(cfg)
    cfg.ensure_directories()
    log.info("loaded configuration path=%s app_home=%s", config_path, app_paths.home)
    return cfg
