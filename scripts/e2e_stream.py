"""Exercise the configured default backend in a temporary application home.

Run: uv run python -m scripts.e2e_stream
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from jouzetsu.config import AppPaths, ConfigStore
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage


async def main() -> int:
    source = ConfigStore.for_home().load()
    if source.backends.default_backend_id is None:
        print("Configure a default backend and model before running this check.")
        return 2
    with tempfile.TemporaryDirectory(prefix="jouzetsu-e2e-") as directory:
        config = replace(source, paths=AppPaths.for_home(Path(directory)))
        state = AppState(config, ChatStorage(config.chats_file))
        try:
            chat = await state.new_chat()
            sent = await state.inference.send_message(
                chat.id,
                chat.revision,
                str(uuid4()),
                "In one sentence, what is the capital of France?",
            )
            if sent.job is None:
                print(f"Generation rejected: {state.session(chat.id).latest_error}")
                return 1
            try:
                await asyncio.wait_for(sent.job.wait(), timeout=300.0)
            except TimeoutError:
                print("Generation did not finish within 300 seconds.")
                return 1
            if sent.job.error or sent.job.save_error:
                print(f"Generation failed: {sent.job.error or sent.job.save_error}")
                return 1
            assistant = chat.messages[-1]
            print(assistant.content[:200])
            assert state.storage.load_all()[0].to_dict() == chat.to_dict()
            return 0 if assistant.role == "assistant" and assistant.content else 1
        finally:
            if not await state.shutdown():
                print("Runtime shutdown was incomplete.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
