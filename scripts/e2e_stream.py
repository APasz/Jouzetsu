"""End-to-end test: drive the streaming pipeline against a real LM Studio.

The script uses the configured LM Studio connection but creates all runtime
state in a temporary directory, leaving the project config and chat history
untouched.

Run from the project root:

    uv run python -m scripts.e2e_stream
"""

import asyncio
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Literal

from jouzetsu.config import AppConfig, load_config
from jouzetsu.lmstudio import LMStudioClient
from jouzetsu.models import Message
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage

_GENERATION_TIMEOUT_SECONDS: float = 300.0


async def _wait_for_generation(state: AppState, *, timeout_seconds: float) -> bool:
    """Wait for the active generation using only the public state API."""
    deadline: float = asyncio.get_running_loop().time() + timeout_seconds
    while state.is_generating():
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.1)
    return True


async def main() -> int:
    source_config: AppConfig = load_config()
    print(f"Using base_url={source_config.server.base_url}")

    with tempfile.TemporaryDirectory(prefix="jouzetsu-e2e-") as runtime_dir:
        runtime_path: Path = Path(runtime_dir)
        cfg: AppConfig = replace(
            source_config,
            data_dir=runtime_path,
            chats_file=runtime_path / "chats.json",
            config_file=runtime_path / "config.json",
        )
        state: AppState = AppState(cfg, ChatStorage(cfg.chats_file), LMStudioClient(cfg.server))

        models: list[str] = await state.client.list_models()
        print(f"models downloaded: {models}")
        if not models:
            print("No models downloaded in LMStudio — aborting.")
            _ = await state.shutdown()
            return 2

        state.active_chat.model = models[0]
        print(f"using model: {state.active_chat.model}")

        print("Sending user message...")
        _ = await state.send_user_message("In one sentence, what is the capital of France?")

        if not await _wait_for_generation(state, timeout_seconds=_GENERATION_TIMEOUT_SECONDS):
            print(f"FAIL: generation did not finish within {_GENERATION_TIMEOUT_SECONDS:.0f}s")
            _ = await state.shutdown()
            return 1

        msgs: list[Message] = state.active_chat.messages
        print("--- final messages ---")
        for m in msgs:
            print(f"[{m.role}] {m.content[:200]}")
        print("--- end ---")

        assistant: Message | None = next((m for m in msgs if m.role == "assistant"), None)
        ok: bool | Literal[""] = assistant is not None and assistant.content and not assistant.content.startswith("⚠")
        _ = await state.shutdown()
        if not ok:
            print("FAIL: no valid assistant response")
            return 1
        print("OK: assistant response received and persisted")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
