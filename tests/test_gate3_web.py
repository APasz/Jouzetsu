"""Gate 3 HTTP contracts: explicit documents, shared revisions and access."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from app_support import WebHarness
from image_support import image_bytes
from inference_support import Plan, backend
from web_support import csrf_headers, transport

from jouzetsu.access import DEVICE_ID_COOKIE_NAME
from jouzetsu.config import ConfigStore, DeviceAccessSettings, encode_config
from jouzetsu.models import Chat
from jouzetsu.runtime import ModelCapabilities
from jouzetsu.web.ui_events import stream_events


def client(h: WebHarness, ip: str = "127.0.0.1") -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport(h.web, client_ip=ip), base_url="http://testserver"
    )


def snapshot(chat: Chat) -> dict[str, str]:
    return {
        "chat_id": chat.id,
        "revision": str(chat.revision),
        "draft_revision": str(chat.draft_revision),
    }


def test_fresh_install_has_no_backend_or_persisted_navigation(tmp_path: Path) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        async with client(h) as c:
            root = await c.get("/")
            chats = await c.get("/chats")
            assert root.status_code == chats.status_code == 200
            assert 'data-testid="new-chat-button"' in chats.text
            assert 'data-testid="add-backend-form"' in chats.text
            assert not h.state.chats
            config = encode_config(h.state.config)
            assert not {"server", "host_stats"} & config.keys()
            assert "active_chat_id" not in json.dumps(config)
            assert not h.state.inference.configuration.backends
            created = await c.post("/chats/new", headers=csrf_headers(chats))
            assert created.status_code == 303
            assert created.headers["location"].startswith("/chats/")
        await h.close()

    asyncio.run(exercise())


def test_navigation_is_read_only_and_explicit_per_browser(tmp_path: Path) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        first, second = await h.state.new_chat(), await h.state.new_chat()
        config = h.state.config.config_file.read_bytes()
        async with client(h) as a, client(h) as b:
            page_a = await a.get(f"/chats/{first.id}")
            for _ in range(2):
                page_a, page_b = await asyncio.gather(
                    a.get(f"/chats/{first.id}"), b.get(f"/chats/{second.id}")
                )
                assert f'data-chat-id="{first.id}"' in page_a.text
                assert f'data-chat-id="{second.id}"' in page_b.text
            edited = await a.post(
                f"/chats/{first.id}/rename",
                data={**snapshot(first), "title": "Only first"},
                headers=csrf_headers(page_a),
            )
            assert edited.status_code == 303
            assert edited.headers["location"] == f"/chats/{first.id}"
            assert second.title == "New chat"
            assert h.state.config.config_file.read_bytes() == config
            assert (await a.get("/fragments/chat")).status_code == 400
            assert (await a.get("/chats/missing")).status_code == 404
        await h.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "action",
    (
        "rename",
        "edit",
        "delete",
        "truncate",
        "merge",
        "fork",
        "regenerate",
        "continue",
        "resend",
    ),
)
def test_stale_committed_mutations_are_rejected_without_notifications(
    tmp_path: Path, action: str
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        await h.state.inference.send_message(chat.id, 0, "first", "One", generate=False)
        page_snapshot = snapshot(chat)
        await h.state.inference.edit_chat(
            chat.id,
            chat.revision,
            lambda candidate: candidate.set_manual_title("Current"),
        )
        saved = chat.to_dict()
        queue = h.web._event_broker.subscribe()  # pyright: ignore[reportPrivateUsage]
        async with client(h) as c:
            page = await c.get(f"/chats/{chat.id}")
            path = (
                f"/chats/{chat.id}/rename"
                if action == "rename"
                else "/messages/edit"
                if action == "edit"
                else f"/messages/{chat.messages[0].id}/{action}"
            )
            result = await c.post(
                path,
                data={
                    **page_snapshot,
                    "message_id": chat.messages[0].id,
                    "content": "Rejected",
                    "title": "Rejected",
                },
                headers=csrf_headers(page),
            )
            # Wrong-role generation targets fail validation before revision checking.
            assert result.status_code in {400, 409}
            assert chat.to_dict() == saved
            assert queue.empty()
        await h.close()

    asyncio.run(exercise())


def test_shared_draft_has_separate_revision_and_preserves_images(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        image = h.state.images.store(image_bytes(), "one.png")
        await h.state.inference.update_draft(chat.id, 0, "Initial", (image,))
        async with client(h) as a, client(h) as b:
            page = await a.get(f"/chats/{chat.id}")
            page_b = await b.get(f"/chats/{chat.id}")
            draft = {
                "chat_id": chat.id,
                "draft_revision": "1",
                "draft": "First browser",
            }
            accepted = await a.post(
                "/chat/draft", data=draft, headers=csrf_headers(page)
            )
            stale = await b.post(
                "/chat/draft",
                data={**draft, "draft": "Stale browser"},
                headers=csrf_headers(page_b),
            )
            assert accepted.status_code == 200
            assert accepted.json() == {"draft_revision": 2}
            assert stale.status_code == 409
            assert chat.revision == 0 and chat.draft == "First browser"
            assert chat.draft_images == (image,)
            shared = await b.get(f"/fragments/chat?chat_id={chat.id}")
            assert "First browser" in shared.text
        await h.close()
        assert h.state.storage.load_all()[0].draft == "First browser"

    asyncio.run(exercise())


def test_send_submission_is_idempotent_with_original_image_snapshot(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        configured = backend()
        configured = replace(
            configured,
            models=(
                replace(
                    configured.models[0],
                    descriptor=replace(
                        configured.models[0].descriptor,
                        capabilities=ModelCapabilities(vision=True),
                    ),
                ),
            ),
        )
        h = WebHarness(tmp_path, (configured,))
        chat = await h.state.new_chat()
        image = h.state.images.store(image_bytes(), "image.png")
        await h.state.inference.update_draft(chat.id, 0, "Caption", (image,))
        plan = Plan(hold=True)
        h.adapters[configured.id].plans.append(plan)
        async with client(h) as c:
            page = await c.get(f"/chats/{chat.id}")
            body = {
                **snapshot(chat),
                "submission_id": "same-attempt",
                "content": "Caption",
                "images": json.dumps([image.to_dict()]),
            }
            first = await c.post("/chat/send", data=body, headers=csrf_headers(page))
            assert first.status_code == 303
            await asyncio.wait_for(plan.produced.wait(), 2)
            duplicate = await c.post(
                "/chat/send", data=body, headers=csrf_headers(page)
            )
            assert duplicate.status_code == 303
            assert len(h.adapters[configured.id].requests) == 1
            assert len(chat.messages) == 1 and chat.messages[0].images == (image,)
            assert chat.draft == "" and chat.draft_images == ()
            plan.release.set()
            job = h.state.inference.job(chat.id)
            assert job is not None
            await job.wait()
            assert len(chat.messages) == 2
            await c.get(f"/chats/{chat.id}")
            assert len(h.adapters[configured.id].requests) == 1
        await h.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("send_only", (True, False))
def test_offline_send_saves_and_errors_never_enter_transcript(
    tmp_path: Path, send_only: bool
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        async with client(h) as c:
            page = await c.get(f"/chats/{chat.id}")
            result = await c.post(
                "/chat/send",
                data={
                    **snapshot(chat),
                    "submission_id": "offline",
                    "content": "Saved",
                    "generate": "false" if send_only else "true",
                },
                headers=csrf_headers(page),
            )
            assert result.status_code == 303
            assert [message.content for message in chat.messages] == ["Saved"]
            session = h.state.session(chat.id)
            assert (session.latest_error is None) is send_only
            updated = await c.get(f"/chats/{chat.id}")
            assert (
                'data-message-id="generation-error"' in updated.text
            ) is not send_only
            assert "generation-error" not in json.dumps(chat.to_dict())
            assert h.state.storage.load_all()[0].to_dict() == chat.to_dict()
        await h.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "route",
    (
        "/chat/send",
        "/chat/stop",
        "/messages/edit",
        "/chats/new",
        "/settings/backends/new",
    ),
)
def test_mutations_enforce_access_and_csrf(tmp_path: Path, route: str) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        async with client(h) as local, client(h, "192.0.2.10") as remote:
            assert (await local.post(route)).status_code == 403
            page = await remote.get("/")
            result = await remote.post(route, headers=csrf_headers(page))
            assert result.status_code == 403
        await h.close()

    asyncio.run(exercise())


def test_remote_approved_browser_has_settings_but_cannot_administer_access(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        device = "dvc_approvedbrowser"
        h.state.config.access.devices[device] = DeviceAccessSettings(
            access_allowed=True
        )
        async with client(h, "192.0.2.20") as remote:
            remote.cookies.set(DEVICE_ID_COOKIE_NAME, device)
            page = await remote.get("/chats")
            assert 'data-testid="add-backend-form"' in page.text
            headers = csrf_headers(page)
            assert (
                await remote.post(
                    "/settings/global/prompt",
                    data={"system_prompt": "Approved edit"},
                    headers=headers,
                )
            ).status_code == 303
            for path in ("/access/phrase", f"/access/devices/{device}"):
                result = await remote.post(
                    path,
                    headers={**headers, "X-Forwarded-For": "127.0.0.1"},
                    data={"approval_phrase": "forged"},
                )
                assert result.status_code == 403
            assert h.state.config.generation.system_prompt == "Approved edit"
        await h.close()

    asyncio.run(exercise())


def test_phrase_is_sole_enrollment_path_and_cooldown_is_per_peer(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        await h.state.set_approval_phrase("correct phrase")
        async with client(h, "192.0.2.30") as a, client(h, "192.0.2.31") as b:
            a.cookies.set(DEVICE_ID_COOKIE_NAME, "dvc_unapproved0001")
            b.cookies.set(DEVICE_ID_COOKIE_NAME, "dvc_unapproved0002")
            page_a, page_b = await a.get("/"), await b.get("/")
            headers = csrf_headers(page_a)
            with patch(
                "jouzetsu.web.request_access.time.monotonic", return_value=100.0
            ):
                for _ in range(5):
                    assert (
                        await a.post(
                            "/access/current-device/pending",
                            data={"approval_phrase": "wrong"},
                            headers=headers,
                        )
                    ).status_code == 403
                assert (
                    await a.post(
                        "/access/current-device/pending",
                        data={"approval_phrase": "correct phrase"},
                        headers=headers,
                    )
                ).status_code == 429
                assert (
                    await b.post(
                        "/access/current-device/pending",
                        data={"approval_phrase": "correct phrase"},
                        headers=csrf_headers(page_b),
                    )
                ).status_code == 303
            with patch(
                "jouzetsu.web.request_access.time.monotonic", return_value=161.0
            ):
                assert (
                    await a.post(
                        "/access/current-device/pending",
                        data={"approval_phrase": "correct phrase"},
                        headers=headers,
                    )
                ).status_code == 303
            assert (
                await a.post("/access/current-device/approve", headers=headers)
            ).status_code in {404, 405}
            assert (
                ConfigStore.for_home(tmp_path)
                .load()
                .access.devices["dvc_unapproved0001"]
                .access_allowed
            )
        await h.close()

    asyncio.run(exercise())


def test_revocation_stops_sse_and_next_request(tmp_path: Path) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        device = "dvc_revokeexample"
        h.state.config.access.devices[device] = DeviceAccessSettings(
            access_allowed=True
        )
        broker = h.web._event_broker  # pyright: ignore[reportPrivateUsage]
        queue = broker.subscribe()
        frames = stream_events(
            broker, queue, lambda: h.state.config.access.devices[device].access_allowed
        )
        assert "event: state" in await anext(frames)
        async with client(h) as local, client(h, "192.0.2.40") as remote:
            remote.cookies.set(DEVICE_ID_COOKIE_NAME, device)
            page = await local.get("/chats")
            result = await local.post(
                f"/access/devices/{device}", headers=csrf_headers(page)
            )
            assert result.status_code == 303
            assert "event: revoked" in await anext(frames)
            with pytest.raises(StopAsyncIteration):
                await anext(frames)
            assert (
                await remote.get("/fragments/navigation?chat_id=anything")
            ).status_code == 403
        await h.close()

    asyncio.run(exercise())


def test_backend_management_snapshots_defaults_and_retains_missing_selection(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        async with client(h) as c:
            page = await c.get("/chats")
            headers = csrf_headers(page)
            created = await c.post(
                "/settings/backends/new",
                data={
                    "kind": "openai_compatible",
                    "name": "Local",
                    "base_url": "http://127.0.0.1:9000/v1",
                },
                headers=headers,
            )
            assert created.status_code == 303
            configured = h.state.config.backends.backends[0]
            registered = await c.post(
                f"/settings/backends/{configured.id}/register-model",
                data={"model_id": "manual", "alias": "Friendly"},
                headers=headers,
            )
            assert registered.status_code == 303
            saved = await c.post(
                f"/settings/backends/{configured.id}/save",
                data={
                    "name": "Local",
                    "base_url": configured.base_url,
                    "enabled": "true",
                    "concurrency_limit": "2",
                    "model": "manual",
                    "temperature": "0.2",
                    "top_p": "0.7",
                    "max_tokens": "100",
                },
                headers=headers,
            )
            assert saved.status_code == 303
            assert (
                await c.post(
                    "/settings/backends/default",
                    data={"backend_id": configured.id},
                    headers=headers,
                )
            ).status_code == 303
            chat = await h.state.new_chat()
            profile = chat.inference.profiles[configured.id]
            assert (
                profile.model_id == "manual" and profile.parameters.temperature == 0.2
            )
            assert (
                await c.post(
                    "/settings/backends/default",
                    data={"backend_id": ""},
                    headers=headers,
                )
            ).status_code == 303
            assert (
                await c.post(
                    f"/settings/backends/{configured.id}/remove", headers=headers
                )
            ).status_code == 303
            assert chat.inference.profiles[configured.id] == profile
            missing = await c.get(f"/chats/{chat.id}")
            assert "Missing backend" in missing.text and "manual" in missing.text
        await h.close()

    asyncio.run(exercise())


def test_preview_repair_preserves_original_and_missing_original_keeps_reference(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        original = image_bytes()
        image = h.state.images.store(original, "retained.png")
        await h.state.inference.update_draft(chat.id, 0, "Caption", (image,))
        preview = h.state.images.path_for(image, thumbnail=True)
        preview.write_bytes(b"damaged preview")
        async with client(h) as c:
            response = await c.get(f"/images/{image.id}?thumbnail=1")
            assert response.status_code == 200
            assert preview.read_bytes() != b"damaged preview"
            assert h.state.images.path_for(image).read_bytes() == original
            h.state.images.path_for(image).unlink()
            assert (await c.get(f"/images/{image.id}?thumbnail=1")).status_code == 404
            assert chat.draft_images == (image,)
        await h.close()
        assert h.state.storage.load_all()[0].draft_images == (image,)

    asyncio.run(exercise())


def test_deferred_draft_disk_failure_is_visible_and_retry_uses_latest_draft(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        await h.state.inference.update_draft(chat.id, 0, "Retain me")
        with (
            patch.object(
                h.state.storage,
                "save_records",
                side_effect=OSError("controlled disk failure"),
            ),
            pytest.raises(OSError),
        ):
            await h.state.persistence.flush_chats()
        async with client(h) as c:
            page = await c.get(f"/chats/{chat.id}")
            assert 'data-testid="retry-draft-save"' in page.text
            result = await c.post(
                "/chat/retry-draft-save",
                data=snapshot(chat),
                headers=csrf_headers(page),
            )
            assert result.status_code == 303
            assert h.state.storage.load_all()[0].draft == "Retain me"
            assert h.state.persistence.chat_write_error is None
        await h.close()

    asyncio.run(exercise())


def test_pending_registration_limit_cannot_block_valid_phrase_enrollment(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        await h.state.set_approval_phrase("enroll despite pending spam")
        h.state.config.access.devices.update(
            {f"dvc_pending{i:08d}": DeviceAccessSettings() for i in range(64)}
        )
        async with client(h, "192.0.2.60") as c:
            c.cookies.set(DEVICE_ID_COOKIE_NAME, "dvc_legitimate001")
            page = await c.get("/")
            assert "dvc_legitimate001" not in h.state.config.access.devices
            result = await c.post(
                "/access/current-device/pending",
                data={"approval_phrase": "enroll despite pending spam"},
                headers=csrf_headers(page),
            )
            assert result.status_code == 303
            assert h.state.config.access.devices["dvc_legitimate001"].access_allowed
        await h.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("current", ("stream", "characters", "status"))
def test_coalescing_cannot_drop_draft_refresh_during_generation(
    tmp_path: Path, current: str
) -> None:
    from jouzetsu.events import StateChangeKind

    async def exercise() -> None:
        h = WebHarness(tmp_path)
        broker = h.web._event_broker  # pyright: ignore[reportPrivateUsage]
        queue = broker.subscribe()
        await broker._on_state_change(StateChangeKind.DRAFT)  # pyright: ignore[reportPrivateUsage]
        await broker._on_state_change(StateChangeKind(current))  # pyright: ignore[reportPrivateUsage]
        assert queue.get_nowait().kind is StateChangeKind.FULL
        await h.close()

    asyncio.run(exercise())


def test_isolated_documents_are_visible_without_changing_their_files(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        await h.close()
        damaged_chat = h.state.storage.directory / "broken.json"
        damaged_character = h.state.character_storage.directory / "broken.json"
        for path in (damaged_chat, damaged_character):
            path.write_bytes(b"{damaged JSON")
        h = WebHarness(tmp_path)
        assert h.state.storage.isolated_count == 1
        assert h.state.character_storage.isolated_count == 1
        async with client(h) as c:
            for url in ("/", f"/chats/{chat.id}", "/characters"):
                page = await c.get(url)
                assert page.status_code == 200
                assert 'data-storage-warning="true"' in page.text
                assert "1 chat files, 1 character files" in page.text
            page = await c.get(f"/chats/{chat.id}")
            deleted = await c.post(
                f"/chats/{chat.id}/delete",
                data=snapshot(h.state.chats[0]),
                headers=csrf_headers(page),
            )
            assert deleted.status_code == 303
            empty = await c.get("/chats")
            assert 'data-storage-warning="true"' in empty.text
        await h.close()
        for path in (damaged_chat, damaged_character):
            assert path.read_bytes() == b"{damaged JSON"

    asyncio.run(exercise())
