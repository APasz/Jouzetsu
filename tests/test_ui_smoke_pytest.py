"""HTTP-level smoke and security tests for the FastHTML application."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Final

import httpx

from jouzetsu.access import DEVICE_ID_COOKIE_NAME
from jouzetsu.config import (
    AccessSettings,
    AppConfig,
    DeviceAccessSettings,
    GenerationSettings,
    LoggingSettings,
    ServerSettings,
    UiSettings,
)
from jouzetsu.events import StateChangeKind
from jouzetsu.models import Chat, Message
from jouzetsu.runtime import (
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelDescriptor,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
)
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage
from jouzetsu.web.application import WebApplication
from jouzetsu.web.security import CSRF_HEADER_NAME
from jouzetsu.web.ui_events import UiEventBroker

_CSRF_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r'data-csrf-token="([A-Za-z0-9_-]{32,128})"')
_LOCAL_DEVICE_ID: Final[str] = "dvc_localhost-device-0001"
_REMOTE_DEVICE_ID: Final[str] = "dvc_remote-device-0001"
_CHAT_CLIENT_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "jouzetsu" / "web" / "static"
_THEME_DIRECTORY: Final[Path] = _CHAT_CLIENT_DIRECTORY / "theme"
_THEME_STYLESHEETS: Final[tuple[str, ...]] = (
    "foundation.css",
    "chat.css",
    "composer.css",
    "panels.css",
    "workspace.css",
    "diagnostics.css",
    "notices.css",
    "responsive.css",
)


def _chat_client_source() -> str:
    """Return every application module so assertions follow the module graph."""

    scripts: tuple[Path, ...] = (_CHAT_CLIENT_DIRECTORY / "app.js", *sorted((_CHAT_CLIENT_DIRECTORY / "app").glob("*.js")))
    return "\n".join(script.read_text(encoding="utf-8") for script in scripts)


def _theme_styles() -> str:
    """Return the complete stylesheet in its cascade order."""

    return "\n".join(
        (_THEME_DIRECTORY / stylesheet).read_text(encoding="utf-8")
        for stylesheet in _THEME_STYLESHEETS
    )


class FakeLMStudioClient:
    """Predictable local runtime used to exercise request-to-state behaviour."""

    async def list_models(self) -> list[str]:
        return ["demo-model"]

    async def list_model_inventory(self) -> tuple[ModelDescriptor, ...]:
        return (ModelDescriptor(key="demo-model", display_name="Demo model"),)

    async def unload_model_instance(self, instance_id: str) -> None:
        _ = instance_id

    async def stream_chat(
        self,
        chat: Chat,
        model: str,
        generation: GenerationSettings,
    ) -> AsyncIterator[ChatStreamEvent]:
        _ = (chat, model, generation)
        yield ModelReady("demo-model", "Demo model", 4096)
        yield FirstToken()
        yield PredictionFragment("Fast", 1, False)
        yield PredictionFragment("HTML reply", 1, False)
        yield PredictionComplete(GenerationMetrics(output_tokens=2, tokens_per_second=20.0))

    async def close(self) -> None:
        return None


async def _noop() -> None:
    return None


def _build_state(
    root: Path,
    *,
    private: bool,
    approval_phrase: str = "",
    global_settings_for_approved: bool = False,
) -> AppState:
    chats_file: Path = root / "chats.json"
    config: AppConfig = AppConfig(
        server=ServerSettings(default_model="demo-model"),
        ui=UiSettings(auto_open_browser=False),
        logging=LoggingSettings(enabled=False, directory=root / "logs"),
        access=AccessSettings(
            default_private=private,
            allow_localhost_without_approval=False,
            global_settings_for_approved=global_settings_for_approved,
            approval_phrase=approval_phrase,
        ),
        data_dir=root,
        chats_file=chats_file,
        config_file=root / "config.json",
    )
    return AppState(config, ChatStorage(chats_file), FakeLMStudioClient())


def _csrf_headers(page: httpx.Response) -> dict[str, str]:
    """Extract the token issued for a rendered page for one same-browser POST."""

    match: re.Match[str] | None = _CSRF_TOKEN_PATTERN.search(page.text)
    assert match is not None, "rendered page did not include a CSRF token"
    return {CSRF_HEADER_NAME: match.group(1)}


def _set_device_cookie(client: httpx.AsyncClient, device_id: str) -> None:
    client.cookies.set(DEVICE_ID_COOKIE_NAME, device_id, path="/")


async def _wait_for_generation(state: AppState) -> None:
    for _ in range(100):
        if not state.is_generating():
            return
        await asyncio.sleep(0)
    raise AssertionError("fake generation did not finish")


async def _close_web_application(web: WebApplication, state: AppState) -> None:
    web._event_broker.close()  # pyright: ignore[reportPrivateUsage]
    _ = await state.shutdown()


def _transport(web: WebApplication, *, client_ip: str) -> httpx.ASGITransport:
    return httpx.ASGITransport(
        app=web.app,
        raise_app_exceptions=True,
        client=(client_ip, 43123),
    )


def test_ui_event_broker_preserves_a_pending_full_refresh() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        broker: UiEventBroker = UiEventBroker(state)
        queue = broker.subscribe()
        try:
            await broker._on_state_change(StateChangeKind.FULL)  # pyright: ignore[reportPrivateUsage]
            await broker._on_state_change(StateChangeKind.STREAM)  # pyright: ignore[reportPrivateUsage]

            event = queue.get_nowait()
            assert event.revision == 2
            assert event.kind is StateChangeKind.FULL
            assert not event.characters_changed
        finally:
            broker.close()
            _ = await state.shutdown()

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_ui_event_broker_preserves_character_changes_when_events_are_coalesced() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        broker: UiEventBroker = UiEventBroker(state)
        queue = broker.subscribe()
        try:
            await broker._on_state_change(StateChangeKind.CHARACTERS)  # pyright: ignore[reportPrivateUsage]
            await broker._on_state_change(StateChangeKind.FULL)  # pyright: ignore[reportPrivateUsage]

            event = queue.get_nowait()
            assert event.revision == 2
            assert event.kind is StateChangeKind.FULL
            assert event.characters_changed
        finally:
            broker.close()
            _ = await state.shutdown()

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_client_refreshes_only_for_character_events() -> None:
    script: str = _chat_client_source()

    assert "payload?.characters_changed === true && characters.refreshIfNeeded(notices.announce.bind(notices))" in script
    assert "payload?.kind === 'characters'" in script


def test_character_client_supports_manual_fields_and_layered_field_presets() -> None:
    script: str = _chat_client_source()

    assert "export class CharacterEditorController" in script
    assert "handleClick(target)" in script
    assert "#applyPresets(button)" in script
    assert "const appliedBaseIds" in script
    assert "removePresetRows(fields, 'extra', selectedIds);" in script


def test_chat_client_anchors_new_and_live_chats_to_bottom() -> None:
    script: str = _chat_client_source()

    assert "isNearBottom(messages, threshold = BOTTOM_THRESHOLD_PX)" in script
    assert "scrollListToBottom(messages)" in script
    assert "if (followLatest || !sameChat || stickToBottom) this.#messages.scrollListToBottom(currentMessages);" in script
    assert "scrollStreamingMessageUntilTop(messages, message)" in script
    assert "const scrollRoomBeforeMessageTop" in script
    assert "const currentMessage = currentContent.closest('[data-message-id]');" in script
    assert "if (stickToBottom) this.#messages.scrollStreamingMessageUntilTop(messageList, currentMessage);" in script
    assert "if (!messages.restoreScrollAfterNavigation()) messages.scrollListToBottom(byId('message-list'));" in script


def test_chat_client_follows_explicit_generation_actions_without_disrupting_passive_updates() -> None:
    script: str = _chat_client_source()

    assert "LATEST_MESSAGE_ACTION_PATTERN" in script
    assert "const followLatest = action === '/chat/send' || LATEST_MESSAGE_ACTION_PATTERN.test(action);" in script
    assert "this.replaceFragment({ followLatest })" in script
    assert "if (followLatest || !sameChat || stickToBottom)" in script


def test_chat_client_ignores_stale_full_and_live_fragment_responses() -> None:
    script: str = _chat_client_source()

    full_response_body: int = script.index("const markup = await response.text();\n            if (refreshRequest !== this.#latestFullRefreshRequest) return;")
    live_response_body: int = script.index("const markup = await response.text();\n            if (fullRefreshAtRequestStart !== this.#latestFullRefreshRequest) return;")

    assert full_response_body < script.index("const next = parseFragment(markup, 'chat-fragment');", full_response_body)
    assert live_response_body < script.index("const next = parseFragment(markup, 'chat-live-fragment');", live_response_body)


def test_chat_client_keeps_the_stop_generation_control_enabled() -> None:
    script: str = _chat_client_source()

    assert "const isGenerating = form.dataset.composerGenerating === 'true';" in script
    assert "primary.disabled = !isGenerating && !input.value.trim();" in script


def test_chat_client_stops_streaming_scroll_when_the_message_reaches_the_top() -> None:
    script: str = _chat_client_source()

    assert "if (this.#streamingScrollReachedMessageTop) return;" in script
    assert "messages.scrollTop = nextScrollTop;" in script
    assert "if (nextScrollTop < desiredScrollTop) this.#streamingScrollReachedMessageTop = true;" in script


def test_chat_client_preserves_message_scroll_when_generation_completion_replaces_the_composer() -> None:
    script: str = _chat_client_source()

    scroll_capture: int = script.index("const messageScrollTop = sameChat ? currentMessages.scrollTop : null;")
    composer_replacement: int = script.index("currentComposer.replaceWith(nextComposer);")
    scroll_restore: int = script.index("this.#messages.restoreScroll(currentMessages, messageScrollTop);")

    assert "const stickToBottom = sameChat && this.#messages.isNearBottom(currentMessages);" in script
    assert scroll_capture < composer_replacement < scroll_restore


def test_chat_client_respects_reduced_motion_for_message_navigation() -> None:
    script: str = _chat_client_source()

    assert "prefers-reduced-motion: reduce" in script
    assert "behavior: reducedMotion ? 'auto' : 'smooth'" in script


def test_message_list_contains_overscroll_to_the_chat_surface() -> None:
    styles: str = _theme_styles()

    message_list_styles: str = styles.split(".jouzetsu-messages {", maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "overscroll-behavior: contain;" in message_list_styles


def test_chat_client_continues_the_last_message_after_a_double_touch_swipe() -> None:
    script: str = _chat_client_source()

    assert "const DOUBLE_SWIPE_DELAY_MS = 1000;" in script
    assert "const SWIPE_MIN_DISTANCE_PX = 40;" in script
    assert "const SWIPE_MAX_HORIZONTAL_DISTANCE_PX = 72;" in script
    assert "const CONTINUE_SWIPE_BOTTOM_THRESHOLD_PX = 160;" in script
    assert "#continueControl(message)" in script
    assert "#invokeContinue(message)" in script
    assert "document.addEventListener('touchstart'" in script
    assert "verticalDistance >= SWIPE_MIN_DISTANCE_PX" in script
    assert "this.isNearBottom(messages, CONTINUE_SWIPE_BOTTOM_THRESHOLD_PX)" in script
    assert "#armContinueSwipe(control)" in script
    assert "control.classList.add('is-swipe-armed');" in script
    assert "if (this.#invokeContinue(message)) event.preventDefault();" in script


def test_chat_client_submits_the_composer_with_shift_enter() -> None:
    script: str = _chat_client_source()

    assert "messageInput.matches('.jouzetsu-message-input')" in script
    assert "&& (event.shiftKey || event.metaKey)" in script
    assert "form.requestSubmit(primary);" in script


def test_chat_client_keeps_the_chat_shell_inside_the_visual_viewport() -> None:
    script: str = _chat_client_source()
    styles: str = _theme_styles()

    assert "#viewportMetrics()" in script
    assert "KEYBOARD_BOTTOM_CLEARANCE_PX" in script
    assert "#syncAppViewportHeight()" in script
    assert "app.style.setProperty('--jouzetsu-app-height'" in script
    assert "app.classList.toggle('is-keyboard-open', keyboardOpen);" in script
    assert "window.visualViewport?.addEventListener('scroll', () => composer.scheduleViewportReconciliation());" in script
    assert "this.#syncAppViewportHeight();" in script
    assert ".jouzetsu-app.is-keyboard-open .jouzetsu-composer-footer" in styles


def test_chat_client_ticks_the_loading_status_elapsed_value() -> None:
    script: str = _chat_client_source()

    assert "const COMPOSER_STATUS_ELAPSED_INTERVAL_MS = 100;" in script
    assert "#updateStatusDuration(" in script
    assert "#syncStatusDuration(" in script
    assert "syncStatusElapsedTimer()" in script
    assert "composerStatusStageElapsedSeconds" in script
    assert "composerStatusOverallElapsedSeconds" in script
    assert "this.#composer.syncLiveStatus(" in script


def test_chat_client_updates_the_live_reasoning_panel_without_replacing_the_composer() -> None:
    script: str = _chat_client_source()
    styles: str = _theme_styles()

    assert "syncGenerationReasoning(currentReasoning, nextReasoning)" in script
    assert "data-live-generation-reasoning" in script
    assert "this.scheduleViewportReconciliation();" in script
    assert ".jouzetsu-generation-reasoning" in styles
    assert ".jouzetsu-generation-reasoning-content" in styles


def test_composer_primary_icon_has_a_centred_layout() -> None:
    styles: str = _theme_styles()

    assert ".jouzetsu-composer-primary.is-icon {" in styles
    assert "place-items: center;" in styles
    assert ".jouzetsu-composer-primary.is-icon .jouzetsu-svg-icon" in styles


def test_armed_continue_gesture_emboldens_its_icon() -> None:
    styles: str = _theme_styles()

    assert ".jouzetsu-message-action.is-continue.is-swipe-armed .jouzetsu-svg-icon" in styles
    assert "stroke-width: 2.5;" in styles


def test_streaming_feedback_uses_a_rail_scan_and_pending_indicator() -> None:
    script: str = _chat_client_source()
    styles: str = _theme_styles()

    assert ".jouzetsu-message.is-streaming .jouzetsu-message-surface::before" in styles
    assert "@keyframes jouzetsu-streaming-rail-scan" in styles
    assert ".jouzetsu-streaming-pending-dot" in styles
    assert "message.querySelector('[data-streaming-pending]')" in script
    assert "form.dataset.composerGenerating !== 'true'" in script


def test_chat_client_formats_message_update_times_in_the_browser_timezone() -> None:
    script: str = _chat_client_source()

    assert "const formatTimestamp" in script
    assert "date.getMonth() + 1" in script
    assert "date.getHours()" in script
    assert "export const localizeMessageUpdatedTimes" in script
    assert "localizeMessageUpdatedTimes();" in script


def test_message_context_menu_opens_a_details_dialog_with_reasoning() -> None:
    script: str = _chat_client_source()
    styles: str = _theme_styles()

    assert "openContextMenu(message, clientX, clientY)" in script
    assert "document.addEventListener('contextmenu'" in script
    assert "openDetails()" in script
    assert "async #loadDetails(dialog, messageId)" in script
    assert "#syncDetailsForkAction(dialog, action)" in script
    assert "formatTimestamp" in script
    assert "data-message-details-timestamp" in script
    assert ".jouzetsu-message-context-menu" in styles
    assert ".jouzetsu-message-details-reasoning" in styles


def test_chat_page_renders_the_message_details_context_menu_and_dialog() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        assistant: Message = state.active_chat.add_message("assistant", "A complete response")
        assistant.set_model("demo-model")
        assistant.set_reasoning("Reviewed the available context.")
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")

                assert 'data-testid="message-context-menu"' in page.text
                assert 'data-testid="message-details-context-action"' in page.text
                assert 'data-testid="message-details-dialog"' in page.text
                assert 'data-testid="message-details-fork-action"' in page.text
                assert 'data-message-details-actions="true"' in page.text
                assert f'data-message-id="{assistant.id}"' in page.text
                assert "Reviewed the available context." not in page.text

                details: httpx.Response = await client.get(f"/fragments/messages/{assistant.id}/details")
                assert details.status_code == 200
                assert 'data-testid="message-details-content"' in details.text
                assert 'data-testid="message-details-model"' in details.text
                assert "demo-model" in details.text
                assert 'data-testid="message-details-reasoning"' in details.text
                assert "Reviewed the available context." in details.text
                assert f'data-message-details-fork-action="/messages/{assistant.id}/fork"' in details.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_chat_client_preserves_drafts_and_focuses_the_edit_composer() -> None:
    script: str = _chat_client_source()

    assert "async preserve()" in script
    assert "if (!await this.#drafts.preserve()) return;" in script
    assert "const enteringEditMode" in script
    assert "this.#composer.focusInput(renderedInput);" in script
    assert "#syncEditValidation(form, input)" in script


def test_cancelled_character_submission_does_not_clear_unsaved_editor_state() -> None:
    script: str = _chat_client_source()

    confirmation: int = script.index("if (confirmation && !window.confirm(confirmation))")
    submitting: int = script.index("characters.beginSubmit(form);")

    assert "validateForm(form)" in script
    assert confirmation < submitting


def test_chat_client_exits_edit_mode_after_saving_or_cancelling() -> None:
    script: str = _chat_client_source()

    assert "#exitEditMode()" in script
    assert "url.searchParams.delete('edit');" in script
    assert "currentUrl.searchParams.has('edit') && !url.searchParams.has('edit')" in script
    assert "if (action === '/messages/edit') this.#exitEditMode();" in script


def test_chat_client_focuses_an_open_panel_without_scrolling_the_chat_surface() -> None:
    script: str = _chat_client_source()

    assert "focusTarget.focus({ preventScroll: true })" in script


def test_chat_settings_drawer_hides_its_scrollbar_and_uses_wide_desktop_fields() -> None:
    styles: str = _theme_styles()

    assert ".jouzetsu-panel-content::-webkit-scrollbar" in styles
    assert "scrollbar-width: none;" in styles
    assert ".jouzetsu-form-grid.is-chat-sampling-grid" in styles
    assert ".jouzetsu-chat-metadata-item:last-child" in styles


def test_mobile_ui_hides_scrollbars_on_every_scrollable_surface() -> None:
    styles: str = _theme_styles()

    assert "@media (max-width: 640px) {\n    * {\n        scrollbar-width: none;" in styles
    assert "*::-webkit-scrollbar {\n        display: none;" in styles


def test_root_renders_the_workspace_launch_page() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/")
                assert page.status_code == 200
                assert 'data-testid="launch-chats-link"' in page.text
                assert 'data-testid="launch-characters-link"' in page.text
                assert 'data-testid="active-chat-title"' not in page.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_fast_html_routes_render_and_mutate_chat_state() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert page.status_code == 200
                assert 'href="/icon.svg?linework=%23000000&amp;accent=%23D60000&amp;surface=%23F9DED7"' in page.text
                for stylesheet in _THEME_STYLESHEETS:
                    assert re.search(
                        rf'href="/static/theme/{re.escape(stylesheet)}\?v=\d+"',
                        page.text,
                    ) is not None
                assert re.search(r'src="/static/app\.js\?v=\d+"', page.text) is not None
                assert 'type="module"' in page.text
                assert 'data-testid="navigation-brand-icon"' in page.text
                assert page.text.index('data-testid="navigation-brand-icon"') < page.text.index(">Jouzetsu</span>")
                assert 'data-testid="active-chat-title"' in page.text
                assert 'data-testid="message-input"' in page.text

                icon: httpx.Response = await client.get("/icon.svg")
                assert icon.status_code == 200
                assert icon.headers["content-type"].startswith("image/svg+xml")
                assert 'stroke="#000000"' in icon.text
                assert 'fill="#D60000"' in icon.text
                assert 'fill="#F9DED7"' in icon.text

                settings_fragment: httpx.Response = await client.get("/fragments/settings")
                assert settings_fragment.status_code == 200
                assert 'action="/chats/' in settings_fragment.text
                assert 'data-testid="delete-chat-confirm-button"' in settings_fragment.text
                assert 'data-confirm="Delete this chat permanently?"' in settings_fragment.text
                assert 'data-testid="delete-chat-confirm-input"' not in settings_fragment.text
                assert 'action="/chat/sampling"' in settings_fragment.text
                assert 'class="jouzetsu-form-grid is-chat-sampling-grid"' in settings_fragment.text
                assert 'data-live-submit="true"' in settings_fragment.text
                assert 'data-live-autosave="true"' in settings_fragment.text
                for marker in (
                    "active-chat-temperature",
                    "active-chat-top-p",
                    "active-chat-max-tokens",
                    "active-chat-sampling-source",
                    "active-chat-save-reasoning",
                    "chat-metadata",
                    "chat-created-at",
                    "chat-last-active-at",
                    "chat-message-count",
                ):
                    assert f'data-testid="{marker}"' in settings_fragment.text

                saved_sampling: httpx.Response = await client.post(
                    "/chat/sampling",
                    data={"temperature": "0.2", "top_p": "0.8", "max_tokens": "512"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_sampling.status_code == 303
                assert saved_sampling.headers["location"] == "/chats?notice=Sampling+settings+saved"
                assert state.active_chat.sampling_overrides.temperature == 0.2
                assert state.active_chat.sampling_overrides.top_p == 0.8
                assert state.active_chat.sampling_overrides.max_tokens == 512

                disabled_reasoning: httpx.Response = await client.post(
                    "/chat/reasoning",
                    data={},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert disabled_reasoning.status_code == 303
                assert disabled_reasoning.headers["location"] == "/chats?notice=Reasoning+setting+saved"
                assert not state.active_chat.save_reasoning

                app_script: httpx.Response = await client.get("/static/app.js")
                assert app_script.status_code == 200
                assert "import { startApplication } from './app/application.js';" in app_script.text
                for module_name in ("application", "chat", "composer", "host-stats", "messages"):
                    module: httpx.Response = await client.get(f"/static/app/{module_name}.js")
                    assert module.status_code == 200

                active_chat_id: str = state.active_chat.id
                renamed: httpx.Response = await client.post(
                    f"/chats/{active_chat_id}/rename",
                    data={"title": "FastHTML smoke chat"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert renamed.status_code == 303
                assert renamed.headers["location"] == "/chats?notice=Chat+renamed"
                assert state.active_chat.title == "FastHTML smoke chat"

                sent: httpx.Response = await client.post(
                    "/chat/send",
                    data={"content": "Hello from the route"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert sent.status_code == 303
                assert sent.headers["location"] == "/chats"
                await _wait_for_generation(state)

                fragment: httpx.Response = await client.get("/fragments/chat")
                assert fragment.status_code == 200
                assert "Hello from the route" in fragment.text
                assert "FastHTML reply" in fragment.text
                assert 'data-testid="continue-message-' in fragment.text

                previous_assistant_id: str = state.active_chat.messages[-1].id
                regenerated: httpx.Response = await client.post(
                    f"/messages/{previous_assistant_id}/regenerate",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert regenerated.status_code == 303
                assert regenerated.headers["location"] == "/chats"
                await _wait_for_generation(state)
                assert [message.role for message in state.active_chat.messages] == ["user", "assistant"]
                assert state.active_chat.messages[-1].id != previous_assistant_id
                assert state.active_chat.messages[-1].content == "FastHTML reply"

                live_fragment: httpx.Response = await client.get("/fragments/chat/live")
                assert live_fragment.status_code == 200
                assert 'id="chat-live-fragment"' in live_fragment.text
                assert 'data-live-composer-status="true"' in live_fragment.text
                assert "jouzetsu-chat-shell" not in live_fragment.text

                user_message_id: str = state.active_chat.messages[0].id
                edited: httpx.Response = await client.post(
                    "/messages/edit",
                    data={"message_id": user_message_id, "content": "Edited through FastHTML"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert edited.status_code == 303
                assert edited.headers["location"] == "/chats?notice=Message+updated"
                assert state.active_chat.messages[0].content == "Edited through FastHTML"

                drafted: httpx.Response = await client.post(
                    "/chat/draft",
                    data={"chat_id": active_chat_id, "draft": "Saved browser draft"},
                    headers=csrf_headers,
                )
                assert drafted.status_code == 204
                assert state.active_chat.draft == "Saved browser draft"

                new_chat: httpx.Response = await client.post(
                    "/chats/new",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert new_chat.status_code == 303
                assert new_chat.headers["location"] == "/chats?notice=New+chat+created"
                assert state.active_chat.id != active_chat_id

                delayed_draft: httpx.Response = await client.post(
                    "/chat/draft",
                    data={"chat_id": active_chat_id, "draft": "Late draft for the original chat"},
                    headers=csrf_headers,
                )
                original_chat: Chat = next(chat for chat in state.chats if chat.id == active_chat_id)
                assert delayed_draft.status_code == 204
                assert original_chat.draft == "Late draft for the original chat"
                assert state.active_chat.draft == ""

                deleted_chat_id: str = state.active_chat.id
                deleted: httpx.Response = await client.post(
                    f"/chats/{deleted_chat_id}/delete",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert deleted.status_code == 303
                assert deleted.headers["location"] == "/chats?notice=Chat+deleted"
                assert deleted_chat_id not in {chat.id for chat in state.chats}
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_workspace_persists_a_custom_field_schema_and_starts_a_snapshot_chat() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                library: httpx.Response = await client.get("/characters")
                assert library.status_code == 200
                assert 'data-testid="character-library"' in library.text

                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                assert created.status_code == 303
                character_url: str = created.headers["location"]
                assert character_url.startswith("/characters/")
                character_id: str = character_url.split("?", maxsplit=1)[0].rsplit("/", maxsplit=1)[1]

                editor: httpx.Response = await client.get(character_url)
                assert 'data-testid="character-base-preset"' in editor.text
                assert 'data-testid="character-apply-presets"' in editor.text
                assert 'data-character-preset-editor="true"' in editor.text
                assert 'data-character-fields-editor="true"' in editor.text
                assert 'data-character-extra-preset="true"' in editor.text
                saved: httpx.Response = await client.post(
                    character_url.split("?", maxsplit=1)[0],
                    data={
                        "name": "Mira",
                        "revision": "1",
                        "field_id": ["personality", "occupation"],
                        "field_label": ["Personality", "Occupation"],
                        "field_kind": ["long_text", "short_text"],
                        "field_value": ["Warm and observant.", "Pilot"],
                    },
                    headers=_csrf_headers(editor),
                    follow_redirects=False,
                )
                assert saved.headers["location"].startswith("/characters/")
                character = state.character(character_id)
                assert [(field.label, field.kind) for field in character.fields] == [
                    ("Personality", "long_text"),
                    ("Occupation", "short_text"),
                ]
                assert (root / "characters" / f"{character_id}.json").is_file()

                character_page: httpx.Response = await client.get(saved.headers["location"])
                started: httpx.Response = await client.post(
                    f"/characters/{character_id}/chat",
                    data={
                        "name": "Mira",
                        "revision": str(character.revision),
                        "field_id": ["personality", "occupation"],
                        "field_label": ["Personality", "Occupation"],
                        "field_kind": ["long_text", "short_text"],
                        "field_value": ["Warm and observant.", "Pilot and cartographer"],
                    },
                    headers=_csrf_headers(character_page),
                    follow_redirects=False,
                )
                assert started.headers["location"].startswith("/chats?notice=Chat+started+with+Mira")
                assert state.active_chat.character is not None
                assert state.active_chat.character.id == character_id
                assert state.active_chat.character.revision == character.revision + 1
                assert "Occupation: Pilot and cartographer" in state.active_chat.system_prompt
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_field_actions_have_server_rendered_fallbacks() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                library: httpx.Response = await client.get("/characters")
                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                character_url: str = created.headers["location"].split("?", maxsplit=1)[0]
                character_id: str = character_url.rsplit("/", maxsplit=1)[1]

                editor: httpx.Response = await client.get(character_url)
                csrf_token: str = _csrf_headers(editor)[CSRF_HEADER_NAME]
                added: httpx.Response = await client.post(
                    f"{character_url}/fields/add",
                    data={"name": "Mira", "revision": "1", "_jouzetsu_csrf": csrf_token},
                    follow_redirects=False,
                )
                assert added.status_code == 303
                character = state.character(character_id)
                assert [(field.label, field.value) for field in character.fields] == [("New field", "")]

                editor = await client.get(added.headers["location"])
                csrf_token = _csrf_headers(editor)[CSRF_HEADER_NAME]
                applied: httpx.Response = await client.post(
                    f"{character_url}/presets/apply",
                    data={
                        "name": "Mira",
                        "revision": str(character.revision),
                        "field_id": [field.id for field in character.fields],
                        "field_label": [field.label for field in character.fields],
                        "field_kind": [field.kind for field in character.fields],
                        "field_value": [field.value for field in character.fields],
                        "base_preset": "builtin:humanoid",
                        "extra_preset": ["builtin:identity", "builtin:voice"],
                        "_jouzetsu_csrf": csrf_token,
                    },
                    follow_redirects=False,
                )
                assert applied.status_code == 303
                labels: set[str] = {field.label for field in state.character(character_id).fields}
                assert {"New field", "Species", "Role", "Speaking style"} <= labels
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_workspace_loads_and_applies_a_private_preset_pack() -> None:
    async def scenario(root: Path) -> None:
        presets_directory: Path = root / "character-presets"
        presets_directory.mkdir()
        pack_path: Path = presets_directory / "private.json"
        _ = pack_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": "private",
                    "label": "Private presets",
                    "presets": [
                        {
                            "id": "notes",
                            "label": "Private notes",
                            "description": "Personal fields stored only in this local pack.",
                            "layer": "extra",
                            "fields": [{"label": "Private note", "kind": "long_text"}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                library: httpx.Response = await client.get("/characters")
                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                character_url: str = created.headers["location"].split("?", maxsplit=1)[0]
                character_id: str = character_url.rsplit("/", maxsplit=1)[1]
                editor: httpx.Response = await client.get(character_url)
                assert "Private notes" in editor.text

                applied: httpx.Response = await client.post(
                    f"{character_url}/presets/apply",
                    data={
                        "name": "Mira",
                        "revision": "1",
                        "extra_preset": "private:notes",
                        "_jouzetsu_csrf": _csrf_headers(editor)[CSRF_HEADER_NAME],
                    },
                    follow_redirects=False,
                )
                assert applied.status_code == 303
                assert [(field.label, field.kind) for field in state.character(character_id).fields] == [
                    ("Private note", "long_text")
                ]
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_mutations_require_a_same_origin_csrf_token() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                chat_id: str = state.active_chat.id

                missing_token: httpx.Response = await client.post(
                    f"/chats/{chat_id}/rename",
                    data={"title": "Missing token"},
                    follow_redirects=False,
                )
                assert missing_token.status_code == 403
                assert state.active_chat.title == "New chat"

                cross_origin_headers: dict[str, str] = {**csrf_headers, "Origin": "https://attacker.example"}
                cross_origin: httpx.Response = await client.post(
                    f"/chats/{chat_id}/rename",
                    data={"title": "Cross origin"},
                    headers=cross_origin_headers,
                    follow_redirects=False,
                )
                assert cross_origin.status_code == 403
                assert state.active_chat.title == "New chat"

                accepted: httpx.Response = await client.post(
                    f"/chats/{chat_id}/rename",
                    data={"title": "Token accepted"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert accepted.status_code == 303
                assert state.active_chat.title == "Token accepted"
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_global_and_host_dialogs_preserve_context_and_expose_live_assets() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        state.config.theme = replace(state.config.theme, primary="#123456")
        state.config.host_stats.activity_start_color = "#123456"
        state.config.host_stats.activity_end_color = "#fedcba"
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert 'id="global-settings-dialog"' in page.text
                assert 'data-dialog-open="global-settings-dialog"' in page.text
                for action, marker in (
                    ("/settings/global/prompt", "save-global-prompt-button"),
                    ("/settings/global/model-defaults", "save-global-model-defaults-button"),
                    ("/settings/global/interface", "save-global-interface-button"),
                    ("/settings/global/generation", "save-global-generation-button"),
                ):
                    assert f'action="{action}"' in page.text
                    assert f'data-testid="{marker}"' in page.text
                for marker in ("icon-linework-color", "icon-accent-color", "icon-surface-color"):
                    assert f'data-testid="{marker}"' in page.text
                assert 'data-testid="global-continuity-review"' in page.text

                saved_prompt: httpx.Response = await client.post(
                    "/settings/global/prompt",
                    data={"system_prompt": "Saved as its own section"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_prompt.status_code == 303
                assert saved_prompt.headers["location"] == "/chats?dialog=global&notice=Global+system+prompt+saved"
                assert state.config.generation.system_prompt == "Saved as its own section"

                saved_model_defaults: httpx.Response = await client.post(
                    "/settings/global/model-defaults",
                    data={"default_model": "demo-model", "model_alias": "Demo", "auto_unload_minutes": "15"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_model_defaults.status_code == 303
                assert saved_model_defaults.headers["location"] == "/chats?dialog=global&notice=Model+defaults+saved"
                assert state.config.server.auto_unload_minutes == 15

                saved_interface: httpx.Response = await client.post(
                    "/settings/global/interface",
                    data={
                        "muted_color_icons": "true",
                        "icon_linework_color": "#112233",
                        "icon_accent_color": "#445566",
                        "icon_surface_color": "#778899",
                    },
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_interface.status_code == 303
                assert saved_interface.headers["location"] == "/chats?dialog=global&notice=Interface+settings+saved"
                assert state.config.ui.message_action_icon_style == "muted_color"
                assert state.config.ui.icon_colors.linework_color == "#112233"
                assert state.config.ui.icon_colors.accent_color == "#445566"
                assert state.config.ui.icon_colors.surface_color == "#778899"

                configured_icon: httpx.Response = await client.get("/icon.svg")
                assert 'stroke="#112233"' in configured_icon.text
                assert 'fill="#445566"' in configured_icon.text
                assert 'fill="#778899"' in configured_icon.text

                saved_generation: httpx.Response = await client.post(
                    "/settings/global/generation",
                    data={
                        "temperature": "0.7",
                        "top_p": "0.9",
                        "max_tokens": "128",
                        "continuity_review": "true",
                        "british_spelling_replacements": "color\tcolour",
                    },
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_generation.status_code == 303
                assert saved_generation.headers["location"] == "/chats?dialog=global&notice=Generation+defaults+saved"
                assert state.config.generation.continuity_review is True
                assert state.config.generation.british_spelling_replacements[0].replacement == "colour"

                saved: httpx.Response = await client.post(
                    "/settings/global",
                    data={
                        "system_prompt": "Global FastHTML prompt",
                        "model_alias": "",
                        "auto_unload_minutes": "",
                        "temperature": "0.7",
                        "top_p": "0.9",
                        "max_tokens": "128",
                        "continuity_review": "true",
                        "british_spelling_replacements": "color\tcolour",
                        "muted_color_icons": "true",
                    },
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved.status_code == 303
                assert saved.headers["location"] == "/chats?dialog=global&notice=Global+settings+saved"
                assert state.config.ui.message_action_icon_style == "muted_color"
                assert state.config.server.auto_unload_minutes is None

                saved_page: httpx.Response = await client.get(saved.headers["location"])
                assert 'data-open-dialog="global-settings-dialog"' in saved_page.text
                assert "Global settings saved" in saved_page.text

                invalid: httpx.Response = await client.post(
                    "/settings/global",
                    data={
                        "system_prompt": "Global FastHTML prompt",
                        "model_alias": "",
                        "auto_unload_minutes": "",
                        "temperature": "0.7",
                        "top_p": "0.9",
                        "max_tokens": "0",
                        "british_spelling_replacements": "color\tcolour",
                    },
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert invalid.status_code == 303
                assert invalid.headers["location"].startswith("/chats?dialog=global&error=")
                invalid_page: httpx.Response = await client.get(invalid.headers["location"])
                assert 'data-open-dialog="global-settings-dialog"' in invalid_page.text
                assert 'class="jouzetsu-dialog-feedback is-error"' in invalid_page.text

                themed_css: httpx.Response = await client.get("/theme.css")
                assert themed_css.status_code == 200
                assert "--jouzetsu-primary: #123456" in themed_css.text
                assert "--jouzetsu-stat-start: #123456" in themed_css.text
                assert "--jouzetsu-stat-end: #fedcba" in themed_css.text
                assert ".jouzetsu-host-stat-meter-100 { width: 100%;" in themed_css.text

                static_theme: str = _theme_styles()
                assert "inset: 0 auto 0 0;" in static_theme
                assert "mix-blend-mode: difference;" not in static_theme
                assert "transition: width 140ms ease-out;" in static_theme
                assert re.search(r"#[0-9a-fA-F]{3,8}\b", static_theme) is None
                assert "rgb(" not in static_theme

                host_page: httpx.Response = await client.get("/chats?dialog=host")
                assert 'data-host-stats-dialog="true"' in host_page.text
                assert 'data-open-dialog="host-stats-dialog"' in host_page.text
                assert 'data-host-stats-refresh="true"' in host_page.text
                assert "1-minute load average" in host_page.text
                assert "Logical CPU cores" not in host_page.text
                assert 'role="progressbar"' in host_page.text
                assert 'style="width: ' not in host_page.text

                host_fragment: httpx.Response = await client.get("/fragments/host-stats")
                assert host_fragment.status_code == 200
                assert 'id="host-stats-content"' in host_fragment.text
                assert 'data-sampled-at="' in host_fragment.text
                assert 'data-host-stat-row="' in host_fragment.text
                assert "Logical CPU cores" not in host_fragment.text
                assert " of " not in host_fragment.text

                missing_fragment_refresh: httpx.Response = await client.post("/fragments/host-stats/refresh")
                assert missing_fragment_refresh.status_code == 403

                fragment_refresh: httpx.Response = await client.post(
                    "/fragments/host-stats/refresh",
                    headers=csrf_headers,
                )
                assert fragment_refresh.status_code == 200
                assert 'id="host-stats-content"' in fragment_refresh.text

                refreshed: httpx.Response = await client.post(
                    "/host-stats/refresh",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert refreshed.status_code == 303
                assert refreshed.headers["location"] == "/chats?dialog=host&notice=Host+stats+refreshed"

                unknown_dialog: httpx.Response = await client.get("/chats?dialog=unknown&notice=Visible+feedback")
                assert "Visible feedback" in unknown_dialog.text
                assert 'data-open-dialog="unknown"' not in unknown_dialog.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_continuity_rewrite_dialog_allows_applying_or_discarding_a_proposal() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        assistant: Message = state.active_chat.add_message("assistant", "Original reply")
        assistant.set_continuity_rewrite("Proposed rewrite")
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get(f"/chats?continuity_rewrite={assistant.id}")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert 'data-open-dialog="continuity-rewrite-dialog"' in page.text
                assert 'data-testid="continuity-rewrite-dialog"' in page.text
                assert 'data-testid="continuity-rewrite-preview"' in page.text
                assert "Proposed rewrite" in page.text
                assert f'action="/messages/{assistant.id}/continuity-rewrite/apply"' in page.text
                assert f'action="/messages/{assistant.id}/continuity-rewrite/discard"' in page.text

                applied: httpx.Response = await client.post(
                    f"/messages/{assistant.id}/continuity-rewrite/apply",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert applied.status_code == 303
                assert applied.headers["location"] == "/chats?notice=Continuity+rewrite+applied"
                assert assistant.content == "Proposed rewrite"
                assert assistant.continuity_rewrite == ""

                assistant.set_continuity_rewrite("Discarded rewrite")
                discarded: httpx.Response = await client.post(
                    f"/messages/{assistant.id}/continuity-rewrite/discard",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert discarded.status_code == 303
                assert discarded.headers["location"] == "/chats?notice=Continuity+rewrite+discarded"
                assert assistant.content == "Proposed rewrite"
                assert assistant.continuity_rewrite == ""
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_access_and_logs_dialog_separates_access_and_log_file_tabs() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        log_directory: Path = state.config.logging.directory
        log_directory.mkdir()
        _ = (log_directory / "error.log").write_text("error log contents", encoding="utf-8")
        _ = (log_directory / "system.log").write_text("system log contents", encoding="utf-8")
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")
                assert "Access &amp; Logs" in page.text
                assert 'data-testid="access-dialog-access-tab"' in page.text
                assert 'data-testid="access-dialog-logs-tab"' in page.text
                assert 'data-tab-list="true"' in page.text
                assert 'data-testid="access-log-tab-0"' in page.text
                assert 'data-testid="access-log-tab-1"' in page.text
                assert 'data-testid="access-log-panel-0"' in page.text
                assert 'data-testid="access-log-panel-1"' in page.text
                assert "error log contents" in page.text
                assert "system log contents" in page.text

                models_page: httpx.Response = await client.get("/chats?dialog=models")
                assert 'data-open-dialog="models-dialog"' in models_page.text
                assert 'data-testid="models-tab"' in models_page.text
                assert 'data-testid="models-stdout-tab"' not in models_page.text
                assert "Models &amp; logs" not in models_page.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_localhost_can_approve_its_pending_device_when_bypass_is_disabled() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _LOCAL_DEVICE_ID)
                pending_page: httpx.Response = await client.get("/")
                csrf_headers: dict[str, str] = _csrf_headers(pending_page)
                assert 'data-testid="access-locked-page"' in pending_page.text
                assert 'data-testid="approve-current-device-button"' in pending_page.text
                assert not state.config.access.devices[_LOCAL_DEVICE_ID].access_allowed

                approved: httpx.Response = await client.post(
                    "/access/current-device/approve",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert approved.status_code == 303
                assert approved.headers["location"] == "/chats?notice=Browser+approved"
                assert state.config.access.devices[_LOCAL_DEVICE_ID].access_allowed

                main_page: httpx.Response = await client.get("/chats")
                assert 'data-testid="active-chat-title"' in main_page.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_pending_device_can_self_approve_only_with_the_configured_phrase() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True, approval_phrase="open-sesame")
        state.config.access.devices[_REMOTE_DEVICE_ID] = DeviceAccessSettings(label="Remote browser")
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="192.0.2.10"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _REMOTE_DEVICE_ID)
                pending_page: httpx.Response = await client.get("/")
                csrf_headers: dict[str, str] = _csrf_headers(pending_page)
                assert 'data-testid="access-approval-phrase-input"' in pending_page.text

                rejected_phrase: httpx.Response = await client.post(
                    "/access/current-device/pending",
                    data={"label": "Remote browser", "approval_phrase": "wrong"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert rejected_phrase.status_code == 303
                assert rejected_phrase.headers["location"] == "/chats?notice=Browser+details+saved"
                assert not state.config.access.devices[_REMOTE_DEVICE_ID].access_allowed

                approved_phrase: httpx.Response = await client.post(
                    "/access/current-device/pending",
                    data={"label": "Remote browser", "approval_phrase": "open-sesame"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert approved_phrase.status_code == 303
                assert approved_phrase.headers["location"] == "/chats?notice=Browser+approved"
                assert state.config.access.devices[_REMOTE_DEVICE_ID].access_allowed
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_approved_non_global_device_cannot_view_or_use_model_and_host_controls() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        state.config.access.devices[_REMOTE_DEVICE_ID] = DeviceAccessSettings(access_allowed=True, label="Remote browser")
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="192.0.2.10"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _REMOTE_DEVICE_ID)
                page: httpx.Response = await client.get("/chats")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert 'data-testid="active-chat-title"' in page.text
                assert 'data-testid="loaded-models-button"' not in page.text
                assert 'data-testid="host-stats-button"' not in page.text

                requested_dialogs: httpx.Response = await client.get("/chats?dialog=models")
                assert 'data-testid="models-tab"' not in requested_dialogs.text
                assert 'data-testid="host-stats-dialog"' not in requested_dialogs.text

                model_refresh: httpx.Response = await client.post("/models/refresh", headers=csrf_headers)
                host_refresh: httpx.Response = await client.post("/host-stats/refresh", headers=csrf_headers)
                assert model_refresh.status_code == 403
                assert host_refresh.status_code == 403
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_private_routes_lock_unknown_devices_and_reject_mutations() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="192.0.2.10"),
                base_url="http://testserver",
            ) as client:
                locked_page: httpx.Response = await client.get("/")
                csrf_headers: dict[str, str] = _csrf_headers(locked_page)
                assert locked_page.status_code == 200
                assert 'data-testid="access-locked-page"' in locked_page.text
                assert 'data-device-cookie-bootstrap="true"' in locked_page.text

                fragment: httpx.Response = await client.get("/fragments/chat")
                assert fragment.status_code == 403

                draft: httpx.Response = await client.post(
                    "/chat/draft",
                    data={"chat_id": state.active_chat.id, "draft": "Must not persist"},
                    headers=csrf_headers,
                )
                assert draft.status_code == 403
                assert state.active_chat.draft == ""
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_system_message_routes_reject_crafted_mutation_targets() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        system_message: Message = Message(id="system-message", role="system", content="Do not expose this")
        state.active_chat.messages.append(system_message)
        web: WebApplication = WebApplication(state.config, state, on_startup=_noop, on_shutdown=_noop)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/")
                csrf_headers: dict[str, str] = _csrf_headers(page)

                rejected: httpx.Response = await client.post(
                    "/messages/system-message/delete",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert rejected.status_code == 303
                assert "error=message+action+requires+a+visible+user+or+assistant+message" in rejected.headers["location"]
                assert state.active_chat.find_message(system_message.id) is system_message
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))
