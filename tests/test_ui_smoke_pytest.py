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
from unittest.mock import patch

import httpx

from jouzetsu.access import DEVICE_ID_COOKIE_NAME
from jouzetsu.config import (
    AccessSettings,
    AppConfig,
    AppPaths,
    DeviceAccessSettings,
    GenerationSettings,
    LoggingSettings,
    ServerSettings,
    UiSettings,
)
from jouzetsu.events import StateChangeKind
from jouzetsu.models import CharacterField, CharacterName, Chat, ChatPromptMode, Message
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

_CSRF_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'data-csrf-token="([A-Za-z0-9_-]{32,128})"'
)
_LOCAL_DEVICE_ID: Final[str] = "dvc_localhost-device-0001"
_REMOTE_DEVICE_ID: Final[str] = "dvc_remote-device-0001"
_CHAT_CLIENT_DIRECTORY: Final[Path] = (
    Path(__file__).parents[1] / "jouzetsu" / "web" / "static"
)
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

    scripts: tuple[Path, ...] = (
        _CHAT_CLIENT_DIRECTORY / "app.js",
        *sorted((_CHAT_CLIENT_DIRECTORY / "app").glob("*.js")),
    )
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
        yield PredictionComplete(
            GenerationMetrics(output_tokens=2, tokens_per_second=20.0)
        )

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
    paths = AppPaths.for_home(root)
    chats_file: Path = paths.chats_file
    config: AppConfig = AppConfig(
        paths=paths,
        server=ServerSettings(default_model="demo-model"),
        ui=UiSettings(auto_open_browser=False),
        logging=LoggingSettings(enabled=False, directory=root / "logs"),
        access=AccessSettings(
            default_private=private,
            allow_localhost_without_approval=False,
            global_settings_for_approved=global_settings_for_approved,
            approval_phrase=approval_phrase,
        ),
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


def test_ui_event_broker_preserves_character_changes_when_events_are_coalesced() -> (
    None
):
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

    assert (
        "payload?.characters_changed === true && characters.refreshIfNeeded(notices.announce.bind(notices))"
        in script
    )
    assert "payload?.kind === 'characters'" in script


def test_device_activity_client_uses_visible_tab_heartbeats() -> None:
    script: str = _chat_client_source()

    assert "DeviceActivityController" in script
    assert "document.visibilityState !== 'visible'" in script
    assert "window.addEventListener('focus', heartbeatIfVisible)" in script
    assert "5 * 60 * 1000" in script
    assert "/access/current-device/activity" in script


def test_character_client_adds_template_fields_without_duplicate_or_destructive_updates() -> (
    None
):
    script: str = _chat_client_source()

    assert "export class CharacterEditorController" in script
    assert "handleClick(target)" in script
    assert "#applyPresets(button)" in script
    assert "const normalizedFieldLabel" in script
    assert (
        "const normalizedFieldLabel = (value) => value.trim().toLowerCase();" in script
    )
    assert (
        "addedRows.push(...addPresetRows(fields, presetFields(selectedBase)));"
        in script
    )
    assert "removePresetRows" not in script
    assert "#syncPromptPreview(form)" in script
    assert "#syncPromptMode(form)" in script
    assert "customPrompt.hidden = !isCustom;" in script
    assert "#validateCustomInstruction(form, submitter)" in script
    assert "customInput.required = false;" in script
    assert "storyDirection.hidden = !isStory;" in script
    assert "storyDirectionHeading" in script
    assert "const selectedCastProfiles" in script
    assert "form.querySelectorAll('[data-character-cast-member]:checked')" in script
    assert "const compiledCastProfiles" in script


def test_character_client_randomizes_an_unsaved_name() -> None:
    script: str = _chat_client_source()

    assert "const randomCharacterNameSuggestions = (control)" in script
    assert "JSON.parse(control.dataset.characterRandomNames || '{}')" in script
    assert "const randomCharacterNamePart = (names, currentName)" in script
    assert "const characterDisplayName = (givenName, familyName)" in script
    assert "return given ? (family ? `${given} ${family}` : given) : '';" in script
    assert "button.matches('[data-character-randomize-name-part]')" in script
    assert "const namePart = button.dataset.characterRandomizeNamePart;" in script
    assert "const suggestions = randomCharacterNameSuggestions(nameControl);" in script
    assert "input.value = nextName;" in script


def test_character_name_parts_have_independent_randomizer_rows() -> None:
    workspace_styles: str = (_THEME_DIRECTORY / "workspace.css").read_text(
        encoding="utf-8"
    )

    assert (
        ".jouzetsu-character-name-control {\n"
        "    display: flex;\n"
        "    flex-direction: column;"
    ) in workspace_styles
    assert (
        ".jouzetsu-character-name-row {\n"
        "    display: grid;\n"
        "    grid-template-columns: minmax(0, 1fr);"
    ) in workspace_styles
    assert (
        ".jouzetsu-character-name-row.is-randomizable {\n"
        "    grid-template-columns: minmax(0, 1fr) 34px;"
    ) in workspace_styles


def test_character_client_preserves_literal_prompt_names_and_only_focuses_open_field_menus() -> (
    None
):
    script: str = _chat_client_source()

    assert "nameTemplate.replace('{name}', () => primaryName)" in script
    assert (
        "if (!menu.hidden && style instanceof HTMLSelectElement) style.focus();"
        in script
    )


def test_character_client_resolves_field_menus_from_their_field_rows() -> None:
    script: str = _chat_client_source()

    assert "const fieldMenuForToggle" in script
    assert "const fieldToggleForMenu" in script
    assert "const row = toggle.closest('[data-character-field]');" in script
    assert "const row = menu.closest('[data-character-field]');" in script
    assert "data-character-field-actions" not in script
    assert "[data-character-field-menu], [data-character-field-menu-toggle]" in script


def test_character_client_keeps_template_fields_responsive() -> None:
    script: str = _chat_client_source()

    assert "#syncResponsiveState()" in script


def test_character_library_can_shrink_to_the_mobile_viewport() -> None:
    workspace_styles: str = (_THEME_DIRECTORY / "workspace.css").read_text(
        encoding="utf-8"
    )

    for selector in (
        ".jouzetsu-character-layout",
        ".jouzetsu-character-library",
        ".jouzetsu-character-library-list",
    ):
        assert re.search(
            rf"{re.escape(selector)} \{{[^}}]*min-width: 0;",
            workspace_styles,
        )


def test_character_field_actions_fill_desktop_rows_and_share_mobile_field_rows() -> (
    None
):
    workspace_styles: str = (_THEME_DIRECTORY / "workspace.css").read_text(
        encoding="utf-8"
    )
    responsive_styles: str = (_THEME_DIRECTORY / "responsive.css").read_text(
        encoding="utf-8"
    )

    assert 'grid-template-areas: "field value actions";' in workspace_styles
    assert (
        ".jouzetsu-character-field-name {\n    grid-area: field;\n}" in workspace_styles
    )
    assert (
        ".jouzetsu-character-field-value {\n    grid-area: value;\n}"
        in workspace_styles
    )
    assert (
        "grid-area: actions;\n    position: relative;\n    align-self: stretch;"
        in workspace_styles
    )
    assert "height: 100%;" in workspace_styles
    assert (
        'grid-template-areas:\n            "field actions"\n            "value value";'
        in responsive_styles
    )
    assert (
        "height: var(--jouzetsu-character-field-menu-toggle-size);" in responsive_styles
    )


def test_character_field_labels_use_a_fixed_vertical_rail() -> None:
    workspace_styles: str = (_THEME_DIRECTORY / "workspace.css").read_text(
        encoding="utf-8"
    )

    assert "--jouzetsu-character-field-label-rail-size: 18px;" in workspace_styles
    assert "display: block;" in workspace_styles
    assert (
        "padding-left: calc(var(--jouzetsu-character-field-label-rail-size) + 4px);"
        in workspace_styles
    )
    assert "position: absolute;" in workspace_styles
    assert "transform: translate(-50%, -50%) rotate(-90deg);" in workspace_styles


def test_chat_client_anchors_new_and_live_chats_to_bottom() -> None:
    script: str = _chat_client_source()

    assert "isNearBottom(messages, threshold = BOTTOM_THRESHOLD_PX)" in script
    assert "scrollListToBottom(messages)" in script
    assert (
        "if (followLatest || !sameChat || stickToBottom) this.#messages.scrollListToBottom(currentMessages);"
        in script
    )
    assert "scrollStreamingMessageUntilTop(messages, message)" in script
    assert "const scrollRoomBeforeMessageTop" in script
    assert (
        "const currentMessage = currentContent.closest('[data-message-id]');" in script
    )
    assert (
        "if (stickToBottom) this.#messages.scrollStreamingMessageUntilTop(messageList, currentMessage);"
        in script
    )
    assert (
        "if (!messages.restoreScrollAfterNavigation()) messages.scrollListToBottom(byId('message-list'));"
        in script
    )


def test_chat_client_follows_explicit_generation_actions_without_disrupting_passive_updates() -> (
    None
):
    script: str = _chat_client_source()

    assert "LATEST_MESSAGE_ACTION_PATTERN" in script
    assert (
        "const followLatest = action === '/chat/send' || LATEST_MESSAGE_ACTION_PATTERN.test(action);"
        in script
    )
    assert "this.replaceFragment({ followLatest })" in script
    assert "if (followLatest || !sameChat || stickToBottom)" in script


def test_chat_client_ignores_stale_full_and_live_fragment_responses() -> None:
    script: str = _chat_client_source()

    full_response_body: int = script.index(
        "const markup = await response.text();\n            if (refreshRequest !== this.#latestFullRefreshRequest) return;"
    )
    live_response_body: int = script.index(
        "const markup = await response.text();\n            if (fullRefreshAtRequestStart !== this.#latestFullRefreshRequest) return;"
    )

    assert full_response_body < script.index(
        "const next = parseFragment(markup, 'chat-fragment');", full_response_body
    )
    assert live_response_body < script.index(
        "const next = parseFragment(markup, 'chat-live-fragment');", live_response_body
    )


def test_chat_client_keeps_the_stop_generation_control_enabled() -> None:
    script: str = _chat_client_source()

    assert "const isGenerating = form.dataset.composerGenerating === 'true';" in script
    assert "primary.disabled = !isGenerating && !input.value.trim();" in script


def test_delete_dialog_hides_the_tail_delete_choice_for_the_last_message() -> None:
    script: str = _chat_client_source()

    assert (
        "const hasFollowing = target.dataset.deleteChoiceHasFollowing === 'true';"
        in script
    )
    assert "following.hidden = !hasFollowing;" in script


def test_chat_client_stops_streaming_scroll_when_the_message_reaches_the_top() -> None:
    script: str = _chat_client_source()

    assert "const messageId = message.dataset.messageId || '';" in script
    assert "if (this.#streamingScrollReachedMessageTop) return;" in script
    assert "messages.scrollTop = nextScrollTop;" in script
    assert (
        "if (nextScrollTop < desiredScrollTop) this.#streamingScrollReachedMessageTop = true;"
        in script
    )


def test_chat_client_preserves_message_scroll_when_generation_completion_replaces_the_composer() -> (
    None
):
    script: str = _chat_client_source()

    scroll_capture: int = script.index(
        "const messageScrollTop = sameChat ? currentMessages.scrollTop : null;"
    )
    composer_replacement: int = script.index(
        "currentComposer.replaceWith(nextComposer);"
    )
    scroll_restore: int = script.index(
        "this.#messages.restoreScroll(currentMessages, messageScrollTop);"
    )

    assert (
        "const stickToBottom = sameChat && this.#messages.isNearBottom(currentMessages);"
        in script
    )
    assert scroll_capture < composer_replacement < scroll_restore


def test_chat_client_cancels_stale_scroll_restoration_before_explicit_navigation() -> (
    None
):
    script: str = _chat_client_source()

    assert "#scrollRestoreFrame = 0;" in script
    assert "this.#cancelScrollRestore();" in script
    assert "window.cancelAnimationFrame(this.#scrollRestoreFrame);" in script


def test_chat_client_keeps_a_pinned_message_list_at_bottom_when_its_viewport_resizes() -> (
    None
):
    script: str = _chat_client_source()

    assert "watchScrollContainer()" in script
    assert "const ResizeObserverConstructor = window.ResizeObserver;" in script
    assert "new ResizeObserverConstructor(() => {" in script
    assert "this.#scrollContainerPinned && this.#scrollContainer === messages" in script
    assert (
        "messages.addEventListener('scroll', this.#handleScrollContainerScroll, { passive: true });"
        in script
    )
    assert "this.#messages.watchScrollContainer();" in script


def test_chat_client_respects_reduced_motion_for_message_navigation() -> None:
    script: str = _chat_client_source()

    assert "prefers-reduced-motion: reduce" in script
    assert "behavior: reducedMotion ? 'auto' : 'smooth'" in script


def test_message_list_contains_overscroll_to_the_chat_surface() -> None:
    styles: str = _theme_styles()

    message_list_styles: str = styles.split(".jouzetsu-messages {", maxsplit=1)[
        1
    ].split("}", maxsplit=1)[0]
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
    assert (
        "window.visualViewport?.addEventListener('scroll', () => composer.scheduleViewportReconciliation());"
        in script
    )
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


def test_chat_client_updates_the_live_reasoning_panel_without_replacing_the_composer() -> (
    None
):
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

    assert (
        ".jouzetsu-message-action.is-continue.is-swipe-armed .jouzetsu-svg-icon"
        in styles
    )
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
    assert "async copySelectedMessage()" in script
    assert "navigator.clipboard?.writeText" in script
    assert "#syncContextMenuForkAction(menu, messageId)" in script
    assert "formatTimestamp" in script
    assert "data-message-details-timestamp" in script
    assert ".jouzetsu-message-context-menu" in styles
    assert ".jouzetsu-message-details-reasoning" in styles


def test_chat_page_renders_the_message_details_context_menu_and_dialog() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        assistant: Message = state.active_chat.add_message(
            "assistant", "A complete response"
        )
        assistant.set_model("demo-model")
        assistant.set_reasoning("Reviewed the available context.")
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")

                assert 'data-testid="message-context-menu"' in page.text
                assert 'data-testid="message-copy-context-action"' in page.text
                assert 'data-testid="message-fork-context-action"' in page.text
                assert 'data-testid="message-details-context-action"' in page.text
                assert 'data-testid="message-details-dialog"' in page.text
                assert 'data-testid="message-details-fork-action"' not in page.text
                assert f'data-message-id="{assistant.id}"' in page.text
                assert "Reviewed the available context." not in page.text

                details: httpx.Response = await client.get(
                    f"/fragments/messages/{assistant.id}/details"
                )
                assert details.status_code == 200
                assert 'data-testid="message-details-content"' in details.text
                assert 'data-testid="message-details-model"' in details.text
                assert "demo-model" in details.text
                assert 'data-testid="message-details-reasoning"' in details.text
                assert "Reviewed the available context." in details.text
                assert "data-message-details-fork-action" not in details.text
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

    submitter: int = script.index("const submitter = event.submitter;")
    validation: int = script.index("if (!characters.validateForm(form, submitter))")
    confirmation: int = script.index(
        "if (confirmation && !window.confirm(confirmation))"
    )
    submitting: int = script.index("characters.beginSubmit(form);")

    assert submitter < validation
    assert "validateForm(form, submitter)" in script
    assert confirmation < submitting


def test_chat_client_exits_edit_mode_after_saving_or_cancelling() -> None:
    script: str = _chat_client_source()

    assert "#exitEditMode()" in script
    assert "url.searchParams.delete('edit');" in script
    assert (
        "currentUrl.searchParams.has('edit') && !url.searchParams.has('edit')" in script
    )
    assert "if (action === '/messages/edit') this.#exitEditMode();" in script


def test_chat_client_focuses_an_open_panel_without_scrolling_the_chat_surface() -> None:
    script: str = _chat_client_source()

    assert "focusTarget.focus({ preventScroll: true })" in script


def test_chat_settings_drawer_hides_its_scrollbar_and_uses_wide_desktop_fields() -> (
    None
):
    styles: str = _theme_styles()

    assert ".jouzetsu-panel-content::-webkit-scrollbar" in styles
    assert "scrollbar-width: none;" in styles
    assert ".jouzetsu-form-grid.is-chat-sampling-grid" in styles
    assert ".jouzetsu-chat-metadata-item:last-child" in styles


def test_mobile_ui_hides_scrollbars_on_every_scrollable_surface() -> None:
    styles: str = _theme_styles()

    assert (
        "@media (max-width: 640px) {\n    * {\n        scrollbar-width: none;" in styles
    )
    assert "*::-webkit-scrollbar {\n        display: none;" in styles


def test_root_renders_the_workspace_launch_page() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get("/chats")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert page.status_code == 200
                assert (
                    'href="/icon.svg?linework=%23000000&amp;accent=%23D60000&amp;surface=%23F9DED7"'
                    in page.text
                )
                for stylesheet in _THEME_STYLESHEETS:
                    assert (
                        re.search(
                            rf'href="/static/theme/{re.escape(stylesheet)}\?v=\d+"',
                            page.text,
                        )
                        is not None
                    )
                assert re.search(r'src="/static/app\.js\?v=\d+"', page.text) is not None
                assert 'type="module"' in page.text
                assert 'data-testid="navigation-brand-icon"' in page.text
                assert page.text.index(
                    'data-testid="navigation-brand-icon"'
                ) < page.text.index(">Jouzetsu</span>")
                assert 'data-testid="active-chat-title"' in page.text
                assert 'data-testid="message-input"' in page.text

                icon: httpx.Response = await client.get("/icon.svg")
                assert icon.status_code == 200
                assert icon.headers["content-type"].startswith("image/svg+xml")
                assert 'stroke="#000000"' in icon.text
                assert 'fill="#D60000"' in icon.text
                assert 'fill="#F9DED7"' in icon.text

                settings_fragment: httpx.Response = await client.get(
                    "/fragments/settings"
                )
                assert settings_fragment.status_code == 200
                assert 'action="/chats/' in settings_fragment.text
                assert (
                    'data-testid="delete-chat-confirm-button"' in settings_fragment.text
                )
                assert (
                    'data-confirm="Delete this chat permanently?"'
                    in settings_fragment.text
                )
                assert (
                    'data-testid="delete-chat-confirm-input"'
                    not in settings_fragment.text
                )
                assert 'action="/chat/sampling"' in settings_fragment.text
                assert (
                    'class="jouzetsu-form-grid is-chat-sampling-grid"'
                    in settings_fragment.text
                )
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
                assert (
                    saved_sampling.headers["location"]
                    == "/chats?notice=Sampling+settings+saved"
                )
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
                assert (
                    disabled_reasoning.headers["location"]
                    == "/chats?notice=Reasoning+setting+saved"
                )
                assert not state.active_chat.save_reasoning

                app_script: httpx.Response = await client.get("/static/app.js")
                assert app_script.status_code == 200
                assert (
                    "import { startApplication } from './app/application.js';"
                    in app_script.text
                )
                missing_static_asset: httpx.Response = await client.get(
                    "/static/missing.js"
                )
                assert missing_static_asset.status_code == 404
                for module_name in (
                    "application",
                    "chat",
                    "composer",
                    "host-stats",
                    "messages",
                ):
                    module: httpx.Response = await client.get(
                        f"/static/app/{module_name}.js"
                    )
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
                assert [message.role for message in state.active_chat.messages] == [
                    "user",
                    "assistant",
                ]
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
                    data={
                        "message_id": user_message_id,
                        "content": "Edited through FastHTML",
                    },
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert edited.status_code == 303
                assert edited.headers["location"] == "/chats?notice=Message+updated"
                assert (
                    state.active_chat.messages[0].content == "Edited through FastHTML"
                )

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
                    data={
                        "chat_id": active_chat_id,
                        "draft": "Late draft for the original chat",
                    },
                    headers=csrf_headers,
                )
                original_chat: Chat = next(
                    chat for chat in state.chats if chat.id == active_chat_id
                )
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


def test_character_workspace_persists_a_custom_field_schema_and_starts_a_snapshot_chat() -> (
    None
):
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
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
                character_id: str = character_url.split("?", maxsplit=1)[0].rsplit(
                    "/", maxsplit=1
                )[1]

                editor: httpx.Response = await client.get(character_url)
                assert 'data-testid="character-base-preset"' in editor.text
                assert 'data-testid="character-apply-presets"' in editor.text
                assert 'data-character-preset-editor="true"' in editor.text
                assert 'data-character-fields-editor="true"' in editor.text
                assert 'data-character-extra-preset="true"' in editor.text
                assert 'data-character-prompt-preview="true"' in editor.text
                assert "Create character" in editor.text
                saved: httpx.Response = await client.post(
                    character_url.split("?", maxsplit=1)[0],
                    data={
                        "given_name": "Mira",
                        "family_name": "Ash",
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
                assert character.name == "Mira Ash"
                assert [(field.label, field.kind) for field in character.fields] == [
                    ("Personality", "long_text"),
                    ("Occupation", "short_text"),
                ]
                assert (root / "data" / "characters" / f"{character_id}.json").is_file()

                character_page: httpx.Response = await client.get(
                    saved.headers["location"]
                )
                assert 'data-character-field="true"' in character_page.text
                assert "jouzetsu-character-field-name" in character_page.text
                assert "jouzetsu-character-field-value" in character_page.text
                assert 'data-character-field-menu-toggle="true"' in character_page.text
                assert 'data-character-field-menu="true"' in character_page.text
                started: httpx.Response = await client.post(
                    f"/characters/{character_id}/chat",
                    data={
                        "given_name": "Mira",
                        "family_name": "Ash",
                        "revision": str(character.revision),
                        "field_id": ["personality", "occupation"],
                        "field_label": ["Personality", "Occupation"],
                        "field_kind": ["long_text", "short_text"],
                        "field_value": [
                            "Warm and observant.",
                            "Pilot and cartographer",
                        ],
                    },
                    headers=_csrf_headers(character_page),
                    follow_redirects=False,
                )
                assert started.headers["location"].startswith(
                    "/chats?notice=Chat+started+with+Mira+Ash"
                )
                assert state.active_chat.character is not None
                assert state.active_chat.character.id == character_id
                assert state.active_chat.character.revision == character.revision + 1
                assert (
                    "Occupation: Pilot and cartographer"
                    in state.active_chat.system_prompt
                )
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_editor_starts_an_ensemble_chat_from_selected_cast() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        mira = await state.new_character()
        ren = await state.new_character()
        mira = await state.update_character(
            mira.id,
            expected_revision=mira.revision,
            name=CharacterName("Mira"),
            fields=[],
        )
        ren = await state.update_character(
            ren.id,
            expected_revision=ren.revision,
            name=CharacterName("Ren"),
            fields=[CharacterField(label="Role", value="Pilot")],
        )
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                editor: httpx.Response = await client.get(f"/characters/{mira.id}")

                assert 'data-testid="character-cast-selector"' in editor.text
                assert 'data-testid="character-cast-primary"' in editor.text
                assert f'data-testid="character-cast-member-{ren.id}"' in editor.text
                assert 'data-character-cast-member="true"' in editor.text
                assert 'data-character-cast-member-name="Ren"' in editor.text
                assert 'data-character-cast-member-profile="Role: Pilot"' in editor.text
                assert 'data-testid="character-prompt-mode"' in editor.text
                assert 'data-testid="character-custom-prompt"' in editor.text
                assert 'data-testid="character-story-direction-panel"' in editor.text
                assert 'data-character-start-chat="true"' in editor.text
                assert "Roleplay" in editor.text
                assert "Assistant" in editor.text
                assert "Story" in editor.text
                assert "Custom" in editor.text

                started: httpx.Response = await client.post(
                    f"/characters/{mira.id}/chat",
                    data={
                        "name": mira.name,
                        "revision": str(mira.revision),
                        "cast_member_id": ren.id,
                        "prompt_mode": "custom",
                        "custom_instruction": "Use concise mission briefings.",
                    },
                    headers=_csrf_headers(editor),
                    follow_redirects=False,
                )

                assert started.status_code == 303
                assert started.headers["location"].startswith("/chats?notice=")
                assert state.active_chat.title == "Mira & Ren"
                assert state.active_chat.prompt_mode is ChatPromptMode.CUSTOM
                assert tuple(
                    binding.id for binding in state.active_chat.character_cast
                ) == (mira.id, ren.id)
                assert state.active_chat.system_prompt.startswith(
                    "Use concise mission briefings."
                )
                assert "Mira:" in state.active_chat.system_prompt
                assert "Ren:" in state.active_chat.system_prompt

                story_started: httpx.Response = await client.post(
                    f"/characters/{mira.id}/chat",
                    data={
                        "name": mira.name,
                        "revision": str(mira.revision),
                        "cast_member_id": ren.id,
                        "prompt_mode": "story",
                        "story_direction": "A tense rescue on an ocean moon.",
                    },
                    headers=_csrf_headers(editor),
                    follow_redirects=False,
                )

                assert story_started.status_code == 303
                assert state.active_chat.prompt_mode is ChatPromptMode.STORY
                assert (
                    "A tense rescue on an ocean moon."
                    in state.active_chat.system_prompt
                )
                assert "Write in third person" in state.active_chat.system_prompt
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_onboarding_ends_after_the_initial_profile_is_saved() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                library: httpx.Response = await client.get("/characters")
                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                character_url: str = created.headers["location"].split("?", maxsplit=1)[
                    0
                ]

                editor: httpx.Response = await client.get(character_url)
                assert "Create character" in editor.text
                assert "Quick start" in editor.text
                assert 'data-testid="character-given-name-input"' in editor.text
                assert 'data-testid="character-family-name-input"' in editor.text
                assert re.search(
                    r'<input[^>]*name="given_name"[^>]*value=""', editor.text
                )
                assert re.search(
                    r'<input[^>]*name="family_name"[^>]*value=""', editor.text
                )
                assert 'data-testid="character-given-name-randomize"' in editor.text
                assert 'data-testid="character-family-name-randomize"' in editor.text
                assert 'data-character-randomize-name-part="given_name"' in editor.text
                assert 'data-character-randomize-name-part="family_name"' in editor.text
                assert "data-character-random-names=" in editor.text
                assert 'aria-label="Generate a random given name"' in editor.text
                assert 'aria-label="Generate a random family name"' in editor.text

                saved: httpx.Response = await client.post(
                    character_url,
                    data={
                        "given_name": "New",
                        "family_name": "character",
                        "revision": "1",
                    },
                    headers=_csrf_headers(editor),
                    follow_redirects=False,
                )
                assert saved.status_code == 303
                character_id: str = character_url.rsplit("/", maxsplit=1)[1]
                assert state.character(character_id).is_draft is False
                assert state.character(character_id).name == "New character"
                assert (
                    state.character(character_id)
                    .compiled_system_prompt()
                    .startswith("You are roleplaying as New character.\n\n")
                )

                updated_editor: httpx.Response = await client.get(
                    saved.headers["location"]
                )
                assert "Create character" not in updated_editor.text
                assert "New character" in updated_editor.text
                assert "Profile templates" in updated_editor.text
                assert 'value="New"' in updated_editor.text
                assert 'value="character"' in updated_editor.text
                assert "data-character-randomize-name-part=" not in updated_editor.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_field_actions_have_server_rendered_fallbacks() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                library: httpx.Response = await client.get("/characters")
                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                character_url: str = created.headers["location"].split("?", maxsplit=1)[
                    0
                ]
                character_id: str = character_url.rsplit("/", maxsplit=1)[1]

                editor: httpx.Response = await client.get(character_url)
                csrf_token: str = _csrf_headers(editor)[CSRF_HEADER_NAME]
                added: httpx.Response = await client.post(
                    f"{character_url}/fields/add",
                    data={
                        "given_name": "",
                        "family_name": "",
                        "revision": "1",
                        "_jouzetsu_csrf": csrf_token,
                    },
                    follow_redirects=False,
                )
                assert added.status_code == 303
                character = state.character(character_id)
                assert [(field.label, field.value) for field in character.fields] == [
                    ("New field", "")
                ]
                assert character.is_draft is True
                assert character.name_parts is None

                editor = await client.get(added.headers["location"])
                assert "Create character" in editor.text
                assert "Quick start" in editor.text
                assert re.search(
                    r'<input[^>]*name="given_name"[^>]*value=""', editor.text
                )
                assert re.search(
                    r'<input[^>]*name="family_name"[^>]*value=""', editor.text
                )
                assert 'data-testid="character-given-name-randomize"' in editor.text
                assert 'data-testid="character-family-name-randomize"' in editor.text
                csrf_token = _csrf_headers(editor)[CSRF_HEADER_NAME]
                applied: httpx.Response = await client.post(
                    f"{character_url}/presets/apply",
                    data={
                        "given_name": "",
                        "family_name": "",
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
                character = state.character(character_id)
                labels: set[str] = {field.label for field in character.fields}
                assert {"New field", "Species", "Role", "Speaking style"} <= labels
                assert character.presets.base_id == "builtin:humanoid"
                assert character.presets.extra_ids == (
                    "builtin:identity",
                    "builtin:voice",
                )
                assert character.is_draft is True
                assert character.name_parts is None

                persisted_editor: httpx.Response = await client.get(
                    applied.headers["location"]
                )
                assert "Create character" in persisted_editor.text
                assert re.search(
                    r'<input[^>]*name="given_name"[^>]*value=""',
                    persisted_editor.text,
                )
                assert re.search(
                    r'<input[^>]*name="family_name"[^>]*value=""',
                    persisted_editor.text,
                )
                assert re.search(
                    r'<option[^>]*value="builtin:humanoid"[^>]*selected',
                    persisted_editor.text,
                )
                assert re.search(
                    r'<input[^>]*value="builtin:identity"[^>]*checked',
                    persisted_editor.text,
                )
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_workspace_loads_and_applies_a_private_preset_pack() -> None:
    async def scenario(root: Path) -> None:
        presets_directory: Path = root / "data" / "character-presets"
        presets_directory.mkdir(parents=True)
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
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                library: httpx.Response = await client.get("/characters")
                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                character_url: str = created.headers["location"].split("?", maxsplit=1)[
                    0
                ]
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
                character = state.character(character_id)
                assert [(field.label, field.kind) for field in character.fields] == [
                    ("Private note", "long_text")
                ]
                assert character.presets.extra_ids == ("private:notes",)
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_character_workspace_loads_user_supplied_name_suggestions() -> None:
    async def scenario(root: Path) -> None:
        names_path: Path = root / "data" / "character-names.json"
        names_path.parent.mkdir(parents=True)
        _ = names_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "disable_vanilla": True,
                    "given_names": ["Ayla"],
                    "family_names": ["Khan"],
                }
            ),
            encoding="utf-8",
        )
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        transport: httpx.ASGITransport = _transport(web, client_ip="127.0.0.1")
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                assert web.character_name_suggestions.suggestions.given_names == (
                    "Ayla",
                )
                assert web.character_name_suggestions.suggestions.family_names == (
                    "Khan",
                )

                library: httpx.Response = await client.get("/characters")
                created: httpx.Response = await client.post(
                    "/characters/new",
                    headers=_csrf_headers(library),
                    follow_redirects=False,
                )
                editor: httpx.Response = await client.get(created.headers["location"])

                assert "Ayla" in editor.text
                assert "Khan" in editor.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_mutations_require_a_same_origin_csrf_token() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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

                cross_origin_headers: dict[str, str] = {
                    **csrf_headers,
                    "Origin": "https://attacker.example",
                }
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
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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
                    (
                        "/settings/global/model-defaults",
                        "save-global-model-defaults-button",
                    ),
                    ("/settings/global/interface", "save-global-interface-button"),
                    ("/settings/global/generation", "save-global-generation-button"),
                ):
                    assert f'action="{action}"' in page.text
                    assert f'data-testid="{marker}"' in page.text
                for marker in (
                    "icon-linework-color",
                    "icon-accent-color",
                    "icon-surface-color",
                ):
                    assert f'data-testid="{marker}"' in page.text
                assert 'data-testid="global-continuity-review"' in page.text

                saved_prompt: httpx.Response = await client.post(
                    "/settings/global/prompt",
                    data={"system_prompt": "Saved as its own section"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_prompt.status_code == 303
                assert (
                    saved_prompt.headers["location"]
                    == "/chats?dialog=global&notice=Global+system+prompt+saved"
                )
                assert (
                    state.config.generation.system_prompt == "Saved as its own section"
                )

                saved_model_defaults: httpx.Response = await client.post(
                    "/settings/global/model-defaults",
                    data={
                        "default_model": "demo-model",
                        "model_alias": "Demo",
                        "auto_unload_minutes": "15",
                    },
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert saved_model_defaults.status_code == 303
                assert (
                    saved_model_defaults.headers["location"]
                    == "/chats?dialog=global&notice=Model+defaults+saved"
                )
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
                assert (
                    saved_interface.headers["location"]
                    == "/chats?dialog=global&notice=Interface+settings+saved"
                )
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
                assert (
                    saved_generation.headers["location"]
                    == "/chats?dialog=global&notice=Generation+defaults+saved"
                )
                assert state.config.generation.continuity_review is True
                assert (
                    state.config.generation.british_spelling_replacements[0].replacement
                    == "colour"
                )

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
                assert (
                    saved.headers["location"]
                    == "/chats?dialog=global&notice=Global+settings+saved"
                )
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
                assert invalid.headers["location"].startswith(
                    "/chats?dialog=global&error="
                )
                invalid_page: httpx.Response = await client.get(
                    invalid.headers["location"]
                )
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

                host_fragment: httpx.Response = await client.get(
                    "/fragments/host-stats"
                )
                assert host_fragment.status_code == 200
                assert 'id="host-stats-content"' in host_fragment.text
                assert 'data-sampled-at="' in host_fragment.text
                assert 'data-host-stat-row="' in host_fragment.text
                assert "Logical CPU cores" not in host_fragment.text
                assert " of " not in host_fragment.text

                missing_fragment_refresh: httpx.Response = await client.post(
                    "/fragments/host-stats/refresh"
                )
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
                assert (
                    refreshed.headers["location"]
                    == "/chats?dialog=host&notice=Host+stats+refreshed"
                )

                unknown_dialog: httpx.Response = await client.get(
                    "/chats?dialog=unknown&notice=Visible+feedback"
                )
                assert "Visible feedback" in unknown_dialog.text
                assert 'data-open-dialog="unknown"' not in unknown_dialog.text
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_continuity_rewrite_dialog_allows_applying_or_discarding_a_proposal() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        assistant: Message = state.active_chat.add_message(
            "assistant", "Original reply"
        )
        assistant.set_continuity_rewrite("Proposed rewrite")
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get(
                    f"/chats?continuity_rewrite={assistant.id}"
                )
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert 'data-open-dialog="continuity-rewrite-dialog"' in page.text
                assert 'data-testid="continuity-rewrite-dialog"' in page.text
                assert 'data-testid="continuity-rewrite-preview"' in page.text
                assert "Proposed rewrite" in page.text
                assert (
                    f'action="/messages/{assistant.id}/continuity-rewrite/apply"'
                    in page.text
                )
                assert (
                    f'action="/messages/{assistant.id}/continuity-rewrite/discard"'
                    in page.text
                )

                applied: httpx.Response = await client.post(
                    f"/messages/{assistant.id}/continuity-rewrite/apply",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert applied.status_code == 303
                assert (
                    applied.headers["location"]
                    == "/chats?notice=Continuity+rewrite+applied"
                )
                assert assistant.content == "Proposed rewrite"
                assert assistant.continuity_rewrite == ""

                assistant.set_continuity_rewrite("Discarded rewrite")
                discarded: httpx.Response = await client.post(
                    f"/messages/{assistant.id}/continuity-rewrite/discard",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert discarded.status_code == 303
                assert (
                    discarded.headers["location"]
                    == "/chats?notice=Continuity+rewrite+discarded"
                )
                assert assistant.content == "Proposed rewrite"
                assert assistant.continuity_rewrite == ""
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_access_and_logs_dialog_separates_access_and_log_file_tabs() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        log_directory: Path = state.config.log_directory
        log_directory.mkdir()
        _ = (log_directory / "error.log").write_text(
            "error log contents", encoding="utf-8"
        )
        _ = (log_directory / "system.log").write_text(
            "system log contents", encoding="utf-8"
        )
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _LOCAL_DEVICE_ID)
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
                assert 'data-testid="access-denied-page-button"' in page.text
                assert 'href="/access/denied"' in page.text
                assert 'target="_blank"' in page.text

                access_denied_page: httpx.Response = await client.get("/access/denied")
                assert access_denied_page.status_code == 200
                assert 'data-testid="access-locked-page"' in access_denied_page.text
                assert (
                    "This is the page shown to browsers awaiting approval."
                    in access_denied_page.text
                )
                assert (
                    'data-testid="approve-current-device-button"'
                    not in access_denied_page.text
                )
                assert (
                    'data-testid="access-current-device-label-input"'
                    not in access_denied_page.text
                )

                models_page: httpx.Response = await client.get("/chats?dialog=models")
                assert 'data-open-dialog="models-dialog"' in models_page.text
                assert 'data-testid="models-tab"' in models_page.text
                assert 'data-testid="models-stdout-tab"' not in models_page.text
                assert "Models &amp; logs" not in models_page.text

            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="192.0.2.10"),
                base_url="http://testserver",
            ) as remote_client:
                forbidden: httpx.Response = await remote_client.get("/access/denied")
                assert forbidden.status_code == 403
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_access_device_list_groups_devices_and_uses_explicit_actions() -> None:
    pending_device_id: str = "dvc_pending-device-0001"
    forgotten_device_id: str = "dvc_forgotten-device-0001"
    approved_device_id: str = "dvc_approved-device-0001"

    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=False)
        state.config.access.devices[_LOCAL_DEVICE_ID] = DeviceAccessSettings(
            access_allowed=True,
            label="Local Mac",
            last_ip="127.0.0.1",
            hostname="localhost",
            first_seen_at="2026-08-24T08:00:00Z",
            last_seen_at="2026-08-25T08:00:00Z",
        )
        state.config.access.devices[pending_device_id] = DeviceAccessSettings(
            label="Kitchen tablet",
            last_ip="192.0.2.10",
            hostname="tablet.local",
            first_seen_at="2026-08-23T08:00:00Z",
            last_seen_at="2026-08-25T09:00:00Z",
        )
        state.config.access.devices[forgotten_device_id] = DeviceAccessSettings(
            label="Old phone",
            last_ip="192.0.2.12",
            hostname="phone.local",
            first_seen_at="2026-08-21T08:00:00Z",
            last_seen_at="2026-08-25T06:00:00Z",
        )
        state.config.access.devices[approved_device_id] = DeviceAccessSettings(
            access_allowed=True,
            label="Office laptop",
            last_ip="192.0.2.11",
            hostname="laptop.local",
            first_seen_at="2026-08-22T08:00:00Z",
            last_seen_at="2026-08-25T07:00:00Z",
        )
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _LOCAL_DEVICE_ID)
                page: httpx.Response = await client.get("/chats?dialog=access")
                csrf_headers: dict[str, str] = _csrf_headers(page)

                assert 'data-testid="access-current-device"' in page.text
                assert 'data-testid="access-pending-devices"' in page.text
                assert 'data-testid="access-approved-devices"' in page.text
                assert page.text.index(
                    'data-testid="access-pending-devices"'
                ) < page.text.index('data-testid="access-approved-devices"')
                assert "Kitchen tablet" in page.text
                assert "tablet.local · 192.0.2.10" in page.text
                assert "Last active 2026-08-25T09:00:00Z" in page.text
                assert (
                    f'data-testid="device-card-toggle-{pending_device_id}"' in page.text
                )
                assert "Manage device" not in page.text
                assert f'data-testid="approve-device-{pending_device_id}"' in page.text
                assert f'data-testid="forget-device-{pending_device_id}"' in page.text
                assert page.text.index(
                    f'data-testid="approve-device-{pending_device_id}"'
                ) < page.text.index(f'data-testid="forget-device-{pending_device_id}"')
                assert f'data-testid="revoke-device-{approved_device_id}"' in page.text
                assert (
                    f'data-testid="forget-device-{approved_device_id}"' not in page.text
                )
                assert (
                    f'data-testid="revoke-device-{_LOCAL_DEVICE_ID}"' not in page.text
                )
                assert 'data-confirm="Revoke this device' in page.text
                assert 'data-confirm="Forget this pending device' in page.text

                forgotten: httpx.Response = await client.post(
                    f"/access/devices/{forgotten_device_id}/forget",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert forgotten.status_code == 303
                assert (
                    forgotten.headers["location"]
                    == "/chats?dialog=access&notice=Pending+device+forgotten"
                )
                assert forgotten_device_id not in state.config.access.devices

                current_forget: httpx.Response = await client.post(
                    f"/access/devices/{_LOCAL_DEVICE_ID}/forget",
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert current_forget.status_code == 303
                assert current_forget.headers["location"].startswith(
                    "/chats?dialog=access&error=the+current+browser+cannot+be+forgotten"
                )
                assert _LOCAL_DEVICE_ID in state.config.access.devices

                approved: httpx.Response = await client.post(
                    f"/access/devices/{pending_device_id}",
                    data={"access_allowed": "true"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert approved.status_code == 303
                assert (
                    approved.headers["location"]
                    == "/chats?dialog=access&notice=Device+access+saved"
                )
                assert state.config.access.devices[pending_device_id].access_allowed

                revoked: httpx.Response = await client.post(
                    f"/access/devices/{approved_device_id}",
                    data={"access_allowed": "false"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert revoked.status_code == 303
                assert (
                    revoked.headers["location"]
                    == "/chats?dialog=access&notice=Device+access+saved"
                )
                assert not state.config.access.devices[
                    approved_device_id
                ].access_allowed
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_device_activity_heartbeat_updates_known_pending_devices() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        state.config.access.devices[_REMOTE_DEVICE_ID] = DeviceAccessSettings(
            label="Remote browser",
            last_ip="192.0.2.10",
            hostname="remote.local",
            first_seen_at="2026-08-25T08:00:00Z",
            last_seen_at="2026-08-25T08:00:00Z",
        )
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        state_change_kinds: list[StateChangeKind] = []

        async def record_state_change(kind: StateChangeKind) -> None:
            state_change_kinds.append(kind)

        remove_state_change_listener = state.add_listener(record_state_change)
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="192.0.2.10"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _REMOTE_DEVICE_ID)
                with patch(
                    "jouzetsu.access.utc_timestamp",
                    return_value="2026-08-26T12:00:00Z",
                ):
                    page: httpx.Response = await client.get("/")
                csrf_headers: dict[str, str] = _csrf_headers(page)
                assert page.status_code == 200
                assert (
                    state.config.access.devices[_REMOTE_DEVICE_ID].last_seen_at
                    == "2026-08-26T12:00:00Z"
                )

                with patch(
                    "jouzetsu.access.utc_timestamp",
                    return_value="2026-08-26T12:01:00Z",
                ):
                    throttled: httpx.Response = await client.post(
                        "/access/current-device/activity",
                        headers=csrf_headers,
                    )
                assert throttled.status_code == 204
                assert (
                    state.config.access.devices[_REMOTE_DEVICE_ID].last_seen_at
                    == "2026-08-26T12:00:00Z"
                )

                with patch(
                    "jouzetsu.access.utc_timestamp",
                    return_value="2026-08-26T12:02:00Z",
                ):
                    refreshed: httpx.Response = await client.post(
                        "/access/current-device/activity",
                        headers=csrf_headers,
                    )
                assert refreshed.status_code == 204
                assert (
                    state.config.access.devices[_REMOTE_DEVICE_ID].last_seen_at
                    == "2026-08-26T12:02:00Z"
                )
                assert not state_change_kinds

                await state.forget_pending_access_device(_REMOTE_DEVICE_ID)
                forgotten: httpx.Response = await client.post(
                    "/access/current-device/activity",
                    headers=csrf_headers,
                )
                assert forgotten.status_code == 204
                assert _REMOTE_DEVICE_ID not in state.config.access.devices
        finally:
            remove_state_change_listener()
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_localhost_can_approve_its_pending_device_when_bypass_is_disabled() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                _set_device_cookie(client, _LOCAL_DEVICE_ID)
                pending_page: httpx.Response = await client.get("/")
                csrf_headers: dict[str, str] = _csrf_headers(pending_page)
                assert 'data-testid="access-locked-page"' in pending_page.text
                assert (
                    'data-testid="approve-current-device-button"' in pending_page.text
                )
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
        state: AppState = _build_state(
            root, private=True, approval_phrase="open-sesame"
        )
        state.config.access.devices[_REMOTE_DEVICE_ID] = DeviceAccessSettings(
            label="Remote browser"
        )
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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
                assert (
                    rejected_phrase.headers["location"]
                    == "/chats?notice=Browser+details+saved"
                )
                assert not state.config.access.devices[_REMOTE_DEVICE_ID].access_allowed

                approved_phrase: httpx.Response = await client.post(
                    "/access/current-device/pending",
                    data={"label": "Remote browser", "approval_phrase": "open-sesame"},
                    headers=csrf_headers,
                    follow_redirects=False,
                )
                assert approved_phrase.status_code == 303
                assert (
                    approved_phrase.headers["location"]
                    == "/chats?notice=Browser+approved"
                )
                assert state.config.access.devices[_REMOTE_DEVICE_ID].access_allowed
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_approved_non_global_device_cannot_view_or_use_model_and_host_controls() -> (
    None
):
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        state.config.access.devices[_REMOTE_DEVICE_ID] = DeviceAccessSettings(
            access_allowed=True, label="Remote browser"
        )
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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

                requested_dialogs: httpx.Response = await client.get(
                    "/chats?dialog=models"
                )
                assert 'data-testid="models-tab"' not in requested_dialogs.text
                assert 'data-testid="host-stats-dialog"' not in requested_dialogs.text

                model_refresh: httpx.Response = await client.post(
                    "/models/refresh", headers=csrf_headers
                )
                host_refresh: httpx.Response = await client.post(
                    "/host-stats/refresh", headers=csrf_headers
                )
                assert model_refresh.status_code == 403
                assert host_refresh.status_code == 403
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))


def test_private_routes_lock_unknown_devices_and_reject_mutations() -> None:
    async def scenario(root: Path) -> None:
        state: AppState = _build_state(root, private=True)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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
        system_message: Message = Message(
            id="system-message", role="system", content="Do not expose this"
        )
        state.active_chat.messages.append(system_message)
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
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
                assert (
                    "error=message+action+requires+a+visible+user+or+assistant+message"
                    in rejected.headers["location"]
                )
                assert (
                    state.active_chat.find_message(system_message.id) is system_message
                )
        finally:
            await _close_web_application(web, state)

    with tempfile.TemporaryDirectory() as temporary_directory:
        asyncio.run(scenario(Path(temporary_directory)))
