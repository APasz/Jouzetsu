"""HTTP-level smoke and security tests for the FastHTML application."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Final

import httpx
from web_support import (
    close_web_application as _close_web_application,
    csrf_headers as _csrf_headers,
    noop as _noop,
    transport as _transport,
)

from jouzetsu.config import (
    AccessSettings,
    AppConfig,
    AppPaths,
    LoggingSettings,
    UiSettings,
)
from jouzetsu.events import StateChangeKind
from jouzetsu.models import (
    CharacterField,
    CharacterName,
    ChatPromptMode,
    Message,
)
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage
from jouzetsu.web.application import WebApplication
from jouzetsu.web.security import CSRF_HEADER_NAME
from jouzetsu.web.ui_events import UiEventBroker

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


def _build_state(
    root: Path,
    *,
    private: bool,
    approval_phrase: str = "",
    global_settings_for_approved: bool = False,
) -> AppState:
    paths = AppPaths.for_home(root)
    chats_file: Path = paths.chats_file
    config = AppConfig(
        paths=paths,
        ui=UiSettings(auto_open_browser=False),
        logging=LoggingSettings(enabled=False, directory=root / "logs"),
        access=AccessSettings(approval_phrase=approval_phrase),
    )
    return AppState(config, ChatStorage(chats_file))


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
    assert "snapListToBottom(messages)" in script
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
    assert "primary.disabled = !isGenerating && (" in script


def test_chat_client_settles_message_snaps_before_streaming_follows() -> None:
    script: str = _chat_client_source()
    streaming_scroll: int = script.index(
        "scrollStreamingMessageUntilTop(messages, message)"
    )

    assert "snapListToBottom(messages)" in script
    assert "#bottomSnapFrame = 0;" in script
    assert "if (!messages.isConnected) return;" in script
    assert script.index("this.#cancelBottomSnap();", streaming_scroll) < script.index(
        "messages.scrollTop = nextScrollTop;", streaming_scroll
    )


def test_chat_client_preserves_message_scroll_when_generation_completion_replaces_the_composer() -> (
    None
):
    script: str = _chat_client_source()

    scroll_capture: int = script.index(
        "const messageScrollTop = sameChat ? currentMessages.scrollTop : null;"
    )
    composer_replacement: int = script.index(
        "this.#reconcileComposer(currentComposer, nextComposer, next.dataset.draftRevision);"
    )
    scroll_restore: int = script.index("this.#messages.restoreScroll(")

    assert "const streamingMessageAnchor = sameChat" in script
    assert "captureStreamingMessageAnchor(currentMessages)" in script
    assert "streamingMessageAnchor," in script
    assert "#restoreScrollPosition(messages, scrollTop, messageAnchor)" in script
    assert "&& !streamingMessageAnchor" in script
    assert (
        "#recordScrollContainerPosition(messages, suppressBottomPinning = false)"
        in script
    )
    assert "!suppressBottomPinning" in script
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


def test_composer_layout_uses_shared_rails_and_reserves_progress_space() -> None:
    styles: str = _theme_styles()

    composer_styles = (_THEME_DIRECTORY / "composer.css").read_text(encoding="utf-8")
    assert "top: -5px;" not in composer_styles
    assert "--jouzetsu-composer-resize-handle-height: 10px;" in styles
    assert "padding: 6px 0 0;" in styles
    assert "padding: 6px 6px 0;" in styles
    assert "--jouzetsu-composer-status-side-rail: clamp(88px, 16vw, 160px);" in styles
    assert "min-height: 38px;" in styles
    assert "max-width: var(--jouzetsu-composer-status-side-rail);" in styles
    assert "grid-template-rows: auto 2px;" in styles
    assert "grid-column: 1 / -1;" in styles
    assert "--jouzetsu-composer-status-side-rail: 72px;" in styles
    assert "position: static;" in styles


def test_generation_panels_use_assistant_accents_and_app_surfaces() -> None:
    styles: str = _theme_styles()

    for selector in (
        ".jouzetsu-generation-reasoning {",
        ".jouzetsu-generation-strip {",
    ):
        panel_styles: str = styles.split(selector, maxsplit=1)[1].split(
            "}", maxsplit=1
        )[0]
        assert "var(--jouzetsu-assistant-muted)" in panel_styles
        assert "var(--jouzetsu-app-surface-raised)" in panel_styles
        assert "var(--jouzetsu-user-" not in panel_styles


def test_chat_client_ticks_the_loading_status_elapsed_value() -> None:
    script: str = _chat_client_source()

    assert "const COMPOSER_STATUS_ELAPSED_INTERVAL_MS = 100;" in script
    assert "#updateStatusDuration(" in script
    assert "#syncStatusDuration(" in script
    assert "syncStatusElapsedTimer()" in script
    assert "composerStatusStageElapsedSeconds" in script
    assert "composerStatusOverallElapsedSeconds" in script
    assert "#syncStatusLoadProgress(" in script
    assert "composerStatusLoadProgress" in script
    assert "data-composer-status-progress" in script
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
        chat = await state.new_chat()
        assistant: Message = chat.add_message("assistant", "A complete response")
        assistant.set_reasoning("Reviewed the available context.")
        web: WebApplication = WebApplication(
            state.config, state, on_startup=_noop, on_shutdown=_noop
        )
        try:
            async with httpx.AsyncClient(
                transport=_transport(web, client_ip="127.0.0.1"),
                base_url="http://testserver",
            ) as client:
                page: httpx.Response = await client.get(f"/chats/{state.chats[0].id}")

                assert 'data-testid="message-context-menu"' in page.text
                assert 'data-testid="message-copy-context-action"' in page.text
                assert 'data-testid="message-fork-context-action"' in page.text
                assert 'data-testid="message-details-context-action"' in page.text
                assert 'data-testid="message-details-dialog"' in page.text
                assert 'data-testid="message-details-fork-action"' not in page.text
                assert f'data-message-id="{assistant.id}"' in page.text
                assert "Reviewed the available context." not in page.text

                details: httpx.Response = await client.get(
                    f"/fragments/messages/{assistant.id}/details?chat_id={chat.id}"
                )
                assert details.status_code == 200
                assert 'data-testid="message-details-content"' in details.text
                assert 'data-testid="message-details-model"' in details.text
                assert "Not recorded" in details.text
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


def test_mobile_message_actions_fill_the_footer_before_wrapping() -> None:
    responsive_styles: str = (_THEME_DIRECTORY / "responsive.css").read_text(
        encoding="utf-8"
    )

    footer_styles = responsive_styles.partition(".jouzetsu-message-footer {")[
        2
    ].partition("}")[0]
    action_styles = responsive_styles.partition(".jouzetsu-message-actions {")[
        2
    ].partition("}")[0]
    updated_at_styles = responsive_styles.partition(".jouzetsu-message-updated-at {")[
        2
    ].partition("}")[0]

    assert "grid-template-columns: minmax(0, 1fr);" in footer_styles
    assert '"actions"\n            "updated-at";' in footer_styles
    assert "grid-area: actions;" in action_styles
    assert "width: 100%;" in action_styles
    assert "grid-area: updated-at;" in updated_at_styles


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
                assert started.headers["location"].startswith("/chats/")
                assert state.chats[0].character is not None
                assert state.chats[0].character.id == character_id
                assert state.chats[0].character.revision == character.revision + 1
                assert (
                    "Occupation: Pilot and cartographer" in state.chats[0].system_prompt
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
                assert started.headers["location"].startswith("/chats/")
                assert state.chats[0].title == "Mira & Ren"
                assert state.chats[0].prompt_mode is ChatPromptMode.CUSTOM
                assert tuple(
                    binding.id for binding in state.chats[0].character_cast
                ) == (mira.id, ren.id)
                assert state.chats[0].system_prompt.startswith(
                    "Use concise mission briefings."
                )
                assert "Mira:" in state.chats[0].system_prompt
                assert "Ren:" in state.chats[0].system_prompt

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
                assert state.chats[0].prompt_mode is ChatPromptMode.STORY
                assert (
                    "A tense rescue on an ocean moon." in state.chats[0].system_prompt
                )
                assert "Write in third person" in state.chats[0].system_prompt
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
