"""Browser-owned navigation preferences, never persisted in application config."""

from typing import Final

LAST_CHAT_COOKIE_NAME: Final[str] = "jouzetsu_last_chat"
LAST_CHAT_COOKIE_MAX_AGE: Final[int] = 365 * 24 * 60 * 60
