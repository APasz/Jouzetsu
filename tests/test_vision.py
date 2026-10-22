"""Image persistence, history, model compatibility, and protected upload coverage."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import httpx
import pytest
from image_support import image_bytes
from PIL import Image
from test_lmstudio import RecordingLMStudioClient
from web_support import close_web_application, csrf_headers, noop, transport

from jouzetsu.config import (
    AccessSettings,
    AppConfig,
    AppPaths,
    GenerationSettings,
    LoggingSettings,
    ServerSettings,
)
from jouzetsu.image_storage import ImageLimits, ImageStorage
from jouzetsu.models import CHAT_SCHEMA_VERSION, Chat, ImageMediaType, Message
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage
from jouzetsu.web.application import WebApplication


@dataclass(frozen=True, slots=True)
class VisionApp:
    state: AppState
    client: RecordingLMStudioClient


@pytest.fixture()
def vision_app(tmp_path: Path) -> Iterator[VisionApp]:
    config = AppConfig(
        paths=AppPaths.for_home(tmp_path),
        server=ServerSettings(default_model="gemma-test"),
        generation=GenerationSettings(continuity_review=False),
        logging=LoggingSettings(enabled=False),
        access=AccessSettings(
            default_private=False, allow_localhost_without_approval=False
        ),
    )
    images = ImageStorage(config.paths.images_directory)
    client = RecordingLMStudioClient(config.server, image_storage=images)
    client.recording_session.get_body = json.dumps(
        {
            "models": [
                {
                    "type": "llm",
                    "key": key,
                    "display_name": key,
                    "capabilities": {"vision": vision},
                    "loaded_instances": [{"id": f"{key}-instance"}],
                }
                for key, vision in (("gemma-test", True), ("text-model", False))
            ]
        }
    )
    client.recording_session.stream_data = (
        b'data: {"choices":[{"delta":{"content":"Image description"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    state = AppState(
        config, ChatStorage(config.chats_file), client, image_storage=images
    )
    try:
        yield VisionApp(state, client)
    finally:
        assert asyncio.run(state.shutdown())


async def wait_for_generation(state: AppState) -> None:
    async with asyncio.timeout(2):
        while state.is_generating():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize(
    "format,media_type",
    (
        ("PNG", ImageMediaType.PNG),
        ("JPEG", ImageMediaType.JPEG),
        ("WEBP", ImageMediaType.WEBP),
    ),
)
def test_image_storage_preserves_originals_and_decodes_actual_format(
    tmp_path: Path,
    format: Literal["PNG", "JPEG", "WEBP"],
    media_type: ImageMediaType,
) -> None:
    data = image_bytes(format=format)
    storage = ImageStorage(tmp_path)
    image = storage.store(data, "../../picture.wrong-extension")

    assert image.media_type is media_type
    assert image.name == "picture.wrong-extension"
    assert storage.path_for(image).read_bytes() == data
    transport_type = (
        ImageMediaType.PNG if media_type is ImageMediaType.WEBP else media_type
    )
    data_url = storage.data_url(image)
    assert data_url.startswith(f"data:{transport_type.value};base64,")
    with (
        Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as transported,
        Image.open(BytesIO(data)) as original,
    ):
        assert transported.size == original.size
        assert (
            transported.convert("RGBA").tobytes() == original.convert("RGBA").tobytes()
        )
    with Image.open(storage.path_for(image, thumbnail=True)) as preview:
        assert preview.width <= 320 and preview.height <= 240
    assert storage.store(data, image.name) == image
    assert len(list(tmp_path.iterdir())) == 2


def test_webp_transport_preserves_transparency(vision_app: VisionApp) -> None:
    data = BytesIO()
    Image.new("RGBA", (37, 29), (20, 40, 60, 128)).save(
        data, format="WEBP", lossless=True
    )
    storage = vision_app.state.images
    image = storage.store(data.getvalue(), "transparent.webp")
    data_url = storage.data_url(image)
    assert data_url.startswith("data:image/png;base64,")
    with Image.open(
        BytesIO(base64.b64decode(data_url.split(",", 1)[1]))
    ) as transported:
        assert transported.size == (37, 29)
        assert transported.convert("RGBA").getpixel((0, 0)) == (20, 40, 60, 128)
    assert storage.path_for(image).read_bytes() == data.getvalue()


@pytest.mark.parametrize(
    "data", (b"not an image", b"<svg></svg>", image_bytes(format="GIF"))
)
def test_invalid_and_unsupported_images_never_reach_storage(
    tmp_path: Path, data: bytes
) -> None:
    with pytest.raises(ValueError, match="image|JPEG"):
        _ = ImageStorage(tmp_path).store(data, "image.png")
    assert list(tmp_path.iterdir()) == []


def test_image_limits_and_missing_originals_fail_explicitly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _ = ImageLimits(max_images_per_message=0)
    with pytest.raises(ValueError, match="each image"):
        _ = ImageStorage(tmp_path, ImageLimits(max_image_bytes=2)).store(
            image_bytes(), "large.png"
        )
    with pytest.raises(ValueError, match="pixel limit"):
        _ = ImageStorage(tmp_path, ImageLimits(max_image_pixels=2)).store(
            image_bytes(), "large.png"
        )
    storage = ImageStorage(tmp_path)
    image = storage.store(image_bytes(), "original.png")
    _ = storage.path_for(image).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="damaged"):
        _ = storage.data_url(image)
    with pytest.raises(ValueError, match="missing or damaged"):
        storage.validate_available((image,))


def test_reupload_repairs_damaged_originals_and_previews(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path)
    data = image_bytes()
    image = storage.store(data, "original.png")
    preview_path = storage.path_for(image, thumbnail=True)
    preview_data = preview_path.read_bytes()
    _ = storage.path_for(image).write_bytes(b"x" * len(data))
    _ = preview_path.write_bytes(b"damaged preview")

    assert storage.store(data, image.name) == image
    assert storage.path_for(image).read_bytes() == data
    assert preview_path.read_bytes() == preview_data
    storage.validate_available((image,))


def test_images_round_trip_and_old_chats_load_without_attachments(
    vision_app: VisionApp,
) -> None:
    state = vision_app.state
    image = state.images.store(image_bytes(), "saved.png")
    chat = Chat(draft_images=(image,), messages=[Message(role="user", images=(image,))])
    assert Chat.from_dict(chat.to_dict()) == chat
    assert chat.to_dict()["schema_version"] == CHAT_SCHEMA_VERSION
    old: dict[str, object] = {
        "schema_version": 1,
        "messages": [{"role": "user", "content": "Old message"}],
    }
    loaded = Chat.from_dict(old)
    assert loaded.draft_images == () and loaded.messages[0].images == ()
    with pytest.raises(ValueError, match="only user"):
        _ = Message(role="assistant", images=(image,))
    with pytest.raises(ValueError, match="SHA-256"):
        _ = replace(image, id="../escape")


@pytest.mark.parametrize("caption", ("", "Compare these images"))
def test_images_follow_their_original_turn_through_generation_and_regeneration(
    vision_app: VisionApp,
    caption: str,
) -> None:
    async def scenario() -> None:
        state, client = vision_app.state, vision_app.client
        chat = state.active_chat
        images = (
            state.images.store(image_bytes(), "red.png"),
            state.images.store(image_bytes(color="blue"), "blue.png"),
        )
        await state.add_chat_images(chat.id, images)
        first = await state.send_user_message(caption)
        await wait_for_generation(state)
        assert first.images == images and first.content == caption
        assert chat.draft_images == ()
        first_payload = client.recording_session.posts[-1]["messages"]
        expected_content: list[dict[str, object]] = (
            [{"type": "text", "text": caption}] if caption else []
        )
        expected_content.extend(
            {"type": "image_url", "image_url": {"url": state.images.data_url(image)}}
            for image in images
        )
        expected_first: dict[str, object] = {
            "role": "user",
            "content": expected_content,
        }
        assert isinstance(first_payload, list) and first_payload[-1] == expected_first
        _ = await state.send_user_message("Now compare their colours")
        await wait_for_generation(state)
        previous_reply = chat.messages[-1]
        await state.regenerate_last()
        await wait_for_generation(state)
        payload = client.recording_session.posts[-1]
        assert payload["messages"] == [
            {"role": "system", "content": state.config.generation.system_prompt},
            expected_first,
            {"role": "assistant", "content": "Image description"},
            {"role": "user", "content": "Now compare their colours"},
        ]
        assert chat.messages[-1].id != previous_reply.id
        await state.continue_last_response()
        await wait_for_generation(state)
        assert state.storage.load_all()[0].messages[0].images == images

    asyncio.run(scenario())


def test_image_context_is_retained_for_the_continuity_review(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state, client = vision_app.state, vision_app.client
        state.config.generation.continuity_review = True
        image = state.images.store(image_bytes(), "review.png")
        await state.add_chat_images(state.active_chat.id, (image,))
        _ = await state.send_user_message("Describe this")
        await wait_for_generation(state)
        assert len(client.recording_session.posts) == 2
        for payload in client.recording_session.posts:
            assert state.images.data_url(image) in json.dumps(payload["messages"])
        assert len(state.active_chat.messages) == 2

    asyncio.run(scenario())


def test_repeated_images_are_encoded_once_per_request(vision_app: VisionApp) -> None:
    async def scenario() -> None:
        state, client = vision_app.state, vision_app.client
        image = state.images.store(image_bytes(format="WEBP"), "scene.webp")
        chat = state.active_chat
        chat.messages = [
            Message(role="user", content="First look", images=(image,)),
            Message(role="assistant", content="First description"),
            Message(role="user", content="Look again", images=(image,)),
        ]
        with patch.object(
            state.images, "data_url", wraps=state.images.data_url
        ) as encode:
            _ = await state.resend_user_message(chat.messages[-1].id)
            await wait_for_generation(state)
        assert encode.call_count == 1
        assert (
            json.dumps(client.recording_session.posts[-1]["messages"]).count(
                state.images.data_url(image)
            )
            == 2
        )

    asyncio.run(scenario())


def test_non_vision_models_block_every_generation_action_without_changing_history(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state = vision_app.state
        image = state.images.store(image_bytes(), "scene.png")
        await state.add_chat_images(state.active_chat.id, (image,))
        _ = await state.send_user_message("")
        await wait_for_generation(state)
        await state.set_active_chat_model("text-model")
        original = state.active_chat.to_dict()
        for operation in (
            lambda: state.send_user_message("Another question"),
            state.regenerate_last,
            state.continue_last_response,
        ):
            with pytest.raises(ValueError, match="vision-capable"):
                await operation()
            assert state.active_chat.to_dict() == original
        await state.delete_message(state.active_chat.messages[-1].id)
        original = state.active_chat.to_dict()
        with pytest.raises(ValueError, match="vision-capable"):
            _ = await state.resend_user_message(state.active_chat.messages[-1].id)
        assert state.active_chat.to_dict() == original
        await state.add_chat_images(state.active_chat.id, (image,))
        original = state.active_chat.to_dict()
        with pytest.raises(ValueError, match="vision-capable"):
            _ = await state.send_user_message("")
        assert state.active_chat.to_dict() == original

    asyncio.run(scenario())


def test_image_count_limits_and_duplicate_draft_uploads(vision_app: VisionApp) -> None:
    async def scenario() -> None:
        state = vision_app.state
        state.images.limits = ImageLimits(max_images_per_message=1)
        first = state.images.store(image_bytes(), "first.png")
        second = state.images.store(image_bytes(color="blue"), "second.png")
        await state.add_chat_images(state.active_chat.id, (first, first))
        assert state.active_chat.draft_images == (first,)
        with pytest.raises(ValueError, match="at most 1"):
            await state.add_chat_images(state.active_chat.id, (second,))
        assert state.active_chat.draft_images == (first,)
        assert state.storage.load_all()[0].draft_images == (first,)
        source = state.active_chat.add_message("user", "First")
        source.images = (first,)
        target = state.active_chat.add_message("user", "Second")
        target.images = (second,)
        original = state.active_chat.to_dict()
        with pytest.raises(ValueError, match="at most 1"):
            _ = await state.merge_message_with_previous(target.id)
        assert state.active_chat.to_dict() == original

    asyncio.run(scenario())


def test_damaged_original_does_not_delete_the_reply_being_regenerated(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state = vision_app.state
        image = state.images.store(image_bytes(), "scene.png")
        await state.add_chat_images(state.active_chat.id, (image,))
        _ = await state.send_user_message("Describe this")
        await wait_for_generation(state)
        original = state.active_chat.to_dict()
        path = state.images.path_for(image)
        _ = path.write_bytes(b"x" * image.size_bytes)
        with pytest.raises(ValueError, match="missing or damaged"):
            await state.regenerate_last()
        assert state.active_chat.to_dict() == original

    asyncio.run(scenario())


def test_image_preflight_rejects_history_changed_while_checking_the_model(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state = vision_app.state
        image = state.images.store(image_bytes(), "scene.png")
        await state.add_chat_images(state.active_chat.id, (image,))
        source = await state.send_user_message("Original question")
        await wait_for_generation(state)
        reply = state.active_chat.messages[-1]
        checking = asyncio.Event()
        resume = asyncio.Event()
        original_refresh = state.refresh_runtime_status

        async def delayed_refresh() -> bool:
            checking.set()
            await resume.wait()
            return await original_refresh()

        with patch.object(state, "refresh_runtime_status", delayed_refresh):
            regeneration = asyncio.create_task(state.regenerate_last())
            await checking.wait()
            await state.edit_message(source.id, "Edited during model validation")
            resume.set()
            with pytest.raises(ValueError, match="conversation changed"):
                await regeneration
        assert state.active_chat.messages[-1] is reply
        assert len(vision_app.client.recording_session.posts) == 1

    asyncio.run(scenario())


def test_continue_uses_the_prompt_selected_during_image_preflight(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state, client = vision_app.state, vision_app.client
        chat = state.active_chat
        image = state.images.store(image_bytes(), "scene.png")
        await state.add_chat_images(chat.id, (image,))
        _ = await state.send_user_message("Describe this")
        await wait_for_generation(state)
        chat.system_prompt = "Previous system prompt"
        original_refresh = state.refresh_runtime_status

        async def change_prompt_during_refresh() -> bool:
            chat.system_prompt = "Current system prompt"
            return await original_refresh()

        with patch.object(
            state, "refresh_runtime_status", change_prompt_during_refresh
        ):
            await state.continue_last_response()
            await wait_for_generation(state)
        messages = client.recording_session.posts[-1]["messages"]
        assert isinstance(messages, list)
        assert messages[0] == {
            "role": "system",
            "content": "Current system prompt",
        }

    asyncio.run(scenario())


def test_images_survive_editing_merging_forking_and_delete_undo(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state = vision_app.state
        image = state.images.store(image_bytes(), "original.png")
        await state.add_chat_images(state.active_chat.id, (image,))
        source = await state.send_user_message("Caption")
        await wait_for_generation(state)
        await state.edit_message(source.id, "")
        source_chat = state.active_chat
        fork = await state.fork_chat_at_message(source.id)
        assert fork.messages[0].images == (image,)
        assert fork.messages[0].id != source.id
        undo = await state.delete_message_with_undo(fork.messages[0].id)
        assert undo is not None
        await state.restore_message_undo(undo)
        assert fork.messages[0].images == (image,)
        second = fork.add_message("user", "Added caption")
        second.images = (image,)
        _ = await state.merge_message_with_previous(second.id)
        assert fork.messages[0].images == (image, image)
        assert source_chat.messages[0].images == (image,)
        assert state.images.path_for(image).is_file()

    asyncio.run(scenario())


def test_image_uploads_and_originals_require_browser_access_and_csrf(
    vision_app: VisionApp,
) -> None:
    async def scenario() -> None:
        state = vision_app.state
        await state.refresh_runtime_status()
        web = WebApplication(state.config, state, on_startup=noop, on_shutdown=noop)
        try:
            async with httpx.AsyncClient(
                transport=transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                headers = csrf_headers(await client.get("/chats"))
                upload_url = f"/chats/{state.active_chat.id}/images"
                denied = await client.post(
                    upload_url,
                    files={"images": ("red.png", image_bytes(), "image/png")},
                )
                assert denied.status_code == 403 and not state.active_chat.draft_images
                uploaded = await client.post(
                    upload_url,
                    files=[
                        (
                            "images",
                            ("red.png", image_bytes(), "application/octet-stream"),
                        ),
                        (
                            "images",
                            (
                                "blue.webp",
                                image_bytes(format="WEBP", color="blue"),
                                "image/webp",
                            ),
                        ),
                    ],
                    headers=headers,
                )
                assert (
                    uploaded.status_code == 303
                    and "error=" not in uploaded.headers["location"]
                )
                assert len(state.active_chat.draft_images) == 2
                image = state.active_chat.draft_images[0]
                page = await client.get("/chats")
                assert (
                    'data-has-images="true"' in page.text
                    and 'data-testid="vision-model-required"' not in page.text
                )
                original = await client.get(f"/images/{image.id}")
                assert original.status_code == 200 and original.content == image_bytes()
                assert original.headers["content-type"] == "image/png"
                assert (await client.get(f"/images/{image.id}?thumbnail=1")).headers[
                    "content-type"
                ] == "image/jpeg"
                assert (await client.get(f"/images/{'0' * 64}")).status_code == 404
                invalid = await client.post(
                    upload_url,
                    files={"images": ("fake.png", b"garbage", "image/png")},
                    headers=headers,
                )
                assert (
                    "error=" in invalid.headers["location"]
                    and len(state.active_chat.draft_images) == 2
                )
                removed = await client.post(
                    f"{upload_url}/{image.id}/remove", headers=headers
                )
                assert (
                    removed.status_code == 303
                    and len(state.active_chat.draft_images) == 1
                )
                state.config.access.default_private = True
                assert (
                    await client.get(f"/images/{state.active_chat.draft_images[0].id}")
                ).status_code == 403
        finally:
            await close_web_application(web, state)

    asyncio.run(scenario())
