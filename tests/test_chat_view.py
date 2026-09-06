"""FastHTML chat-fragment rendering tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import cast

import pytest

from jouzetsu.access import AccessDecision
from jouzetsu.config import AppConfig, MessageActionIconStyle, StarterPrompt
from jouzetsu.models import CharacterChatBinding, Chat, ChatPromptMode, Message
from jouzetsu.runtime import (
    GenerationMetrics,
    ModelDescriptor,
    ModelInstanceDescriptor,
    RuntimePhase,
    RuntimeStatus,
)
from jouzetsu.state import AppState
from jouzetsu.web.views.access import render_locked_access_page
from jouzetsu.web.views.chat import (
    ChatSettingsView,
    ChatView,
    ModelChoice,
    render_chat_fragment,
    render_chat_live_fragment,
    render_composer,
    render_message,
    render_settings_panel,
)
from jouzetsu.web.views.context import PageContext
from jouzetsu.web.views.dialogs import render_loaded_model_card
from jouzetsu.web.views.feedback import render_notice_area


@dataclass(frozen=True, slots=True)
class _SettingsViewState:
    """Minimal state surface used by the active-chat settings renderer."""

    active_chat: Chat
    generating: bool
    config: AppConfig = field(default_factory=AppConfig)
    model_inventory: tuple[ModelDescriptor, ...] = ()
    generating_model_keys: frozenset[str] = frozenset()

    def is_generating(self, chat_id: str | None = None) -> bool:
        _ = chat_id
        return self.generating

    def model_display_name(self, model_key: str, fallback: str = "") -> str:
        return self.config.server.model_aliases.get(model_key, fallback or model_key)

    def is_model_generating(self, model_key: str) -> bool:
        return model_key in self.generating_model_keys


def _render_message(
    messages: list[Message],
    index: int,
    *,
    generating: bool = False,
    muted_actions: bool = False,
) -> str:
    return str(
        render_message(
            _chat_view(
                Chat(id="chat-message-test", messages=messages),
                generating=generating,
                message_action_icon_style="muted_color"
                if muted_actions
                else "monochrome",
            ),
            messages,
            index,
            editing_message_id="",
        )
    )


def _chat_view(
    chat: Chat,
    *,
    generating: bool = False,
    runtime_status: RuntimeStatus | None = None,
    generation_reasoning: str = "",
    empty_state_message: str = "An excellent blank canvas.",
    message_action_icon_style: MessageActionIconStyle = "monochrome",
    starter_prompts: tuple[StarterPrompt, ...] = (),
) -> ChatView:
    """Build the explicit input consumed by the chat renderers."""

    return ChatView(
        chat=chat,
        generating=generating,
        runtime_status=runtime_status or RuntimeStatus(RuntimePhase.READY),
        generation_reasoning=generation_reasoning,
        empty_state_message=empty_state_message,
        message_action_icon_style=message_action_icon_style,
        starter_prompts=starter_prompts,
    )


def test_last_assistant_message_renders_only_valid_actions_in_display_order() -> None:
    messages: list[Message] = [
        Message(id="user-1", role="user", content="Question"),
        Message(id="assistant-1", role="assistant", content="First answer"),
        Message(
            id="assistant-2",
            role="assistant",
            content="Second answer",
            continuity_rewrite="Rewritten second answer",
        ),
    ]

    markup: str = _render_message(messages, 2)
    markers: list[str] = [
        "delete-message-assistant-2",
        "regenerate-message-assistant-2",
        "review-continuity-rewrite-assistant-2",
        "merge-message-assistant-2",
        "edit-message-assistant-2",
        "continue-message-assistant-2",
    ]

    assert [markup.index(f'data-testid="{marker}"') for marker in markers] == sorted(
        markup.index(f'data-testid="{marker}"') for marker in markers
    )
    assert 'action="/messages/assistant-2/regenerate"' in markup
    assert 'href="/chats?continuity_rewrite=assistant-2"' in markup
    assert 'action="/messages/assistant-2/continue"' in markup
    assert 'data-message-action="continue"' in markup
    assert 'data-chat-mutation="true"' in markup
    assert "truncate-message-assistant-2" not in markup
    assert "fork-message-assistant-2" not in markup
    assert 'data-delete-choice-open="assistant-2"' in markup
    assert 'data-delete-choice-has-following="false"' in markup
    assert 'class="jouzetsu-svg-icon"' in markup


def test_last_user_message_renders_resend_in_the_continue_position() -> None:
    messages: list[Message] = [Message(id="user-1", role="user", content="Question")]

    markup: str = _render_message(messages, 0)
    markers: list[str] = [
        "delete-message-user-1",
        "edit-message-user-1",
        "resend-message-user-1",
    ]

    assert [markup.index(f'data-testid="{marker}"') for marker in markers] == sorted(
        markup.index(f'data-testid="{marker}"') for marker in markers
    )
    assert "is-resend is-action-divider" not in markup
    assert "regenerate-message-user-1" not in markup
    assert "fork-message-user-1" not in markup


def test_nonfinal_user_message_hides_resend() -> None:
    messages: list[Message] = [
        Message(id="user-1", role="user", content="Question"),
        Message(id="assistant-1", role="assistant", content="Answer"),
    ]

    assert "resend-message-user-1" not in _render_message(messages, 0)


def test_nonfinal_message_delete_action_marks_following_messages() -> None:
    messages: list[Message] = [
        Message(id="user-1", role="user", content="Question"),
        Message(id="assistant-1", role="assistant", content="Answer"),
    ]

    markup: str = _render_message(messages, 0)

    assert 'data-delete-choice-has-following="true"' in markup


def test_nonfinal_assistant_message_hides_its_continuity_rewrite_action() -> None:
    messages: list[Message] = [
        Message(id="user-1", role="user", content="Question"),
        Message(
            id="assistant-1",
            role="assistant",
            content="Answer",
            continuity_rewrite="Proposed rewrite",
        ),
        Message(id="user-2", role="user", content="Follow-up"),
    ]

    markup: str = _render_message(messages, 1)

    assert "review-continuity-rewrite-assistant-1" not in markup


def test_streaming_message_hides_message_actions() -> None:
    message: Message = Message(id="assistant-stream", role="assistant", content="")

    markup: str = _render_message([message], 0, generating=True)

    assert 'class="jouzetsu-message is-assistant is-streaming"' in markup
    assert 'data-streaming-pending="true"' in markup
    assert 'aria-label="Jouzetsu is responding"' in markup
    for marker in (
        "delete-message-assistant-stream",
        "regenerate-message-assistant-stream",
        "fork-message-assistant-stream",
        "edit-message-assistant-stream",
        "continue-message-assistant-stream",
    ):
        assert marker not in markup


def test_message_renders_its_updated_time_in_the_bubble_footer() -> None:
    markup: str = _render_message(
        [
            Message(
                id="assistant-updated",
                role="assistant",
                content="Answer",
                reasoning="Checked the supplied details.",
                created_at=100.0,
                updated_at=200.0,
            )
        ],
        0,
    )

    assert 'data-testid="message-updated-at-assistant-updated"' in markup
    assert 'data-message-updated-at="200.0"' in markup
    assert "jouzetsu-message-footer" in markup
    assert "jouzetsu-message-updated-at" in markup
    assert "Checked the supplied details." not in markup


def test_completed_message_renders_commonmark() -> None:
    markup: str = _render_message(
        [
            Message(
                id="assistant-markdown",
                role="assistant",
                content="# Heading\n\n**Bold** and `code`\n\n- One\n- Two",
            )
        ],
        0,
    )

    assert "<h1>Heading</h1>" in markup
    assert "<strong>Bold</strong>" in markup
    assert "<code>code</code>" in markup
    assert "<ul>" in markup
    assert "<li>One</li>" in markup


def test_message_markdown_escapes_raw_html_and_blocks_unsafe_links() -> None:
    markup: str = _render_message(
        [
            Message(
                id="assistant-unsafe-markdown",
                role="assistant",
                content='<script>alert("no")</script>\n\n[Unsafe](javascript:alert(1))',
            )
        ],
        0,
    )

    assert "&lt;script&gt;alert(&quot;no&quot;)&lt;/script&gt;" in markup
    assert '<a href="javascript:alert(1)">' not in markup


def test_streaming_message_renders_commonmark_before_completion() -> None:
    markup: str = _render_message(
        [
            Message(
                id="assistant-streaming-markdown",
                role="assistant",
                content="# Heading\n\n**Still streaming**",
            )
        ],
        0,
        generating=True,
    )

    assert "<h1>Heading</h1>" in markup
    assert "<strong>Still streaming</strong>" in markup


def test_live_chat_fragment_contains_only_streaming_values() -> None:
    assistant: Message = Message(
        id="assistant-stream", role="assistant", content="Streamed response"
    )
    view: ChatView = _chat_view(
        Chat(
            id="chat-stream",
            messages=[
                Message(id="user-stream", role="user", content="Question"),
                assistant,
            ],
        ),
        generating=True,
        runtime_status=RuntimeStatus(RuntimePhase.GENERATING, output_tokens=2),
        generation_reasoning="Checking the requested details",
    )

    markup: str = repr(render_chat_live_fragment(view))

    assert 'id="chat-live-fragment"' in markup
    assert 'data-chat-id="chat-stream"' in markup
    assert 'data-live-streaming-message-id="assistant-stream"' in markup
    assert 'data-live-streaming-text="Streamed response"' in markup
    assert "<p>Streamed response</p>" in markup
    assert "Streamed response" in markup
    assert 'data-live-composer-status="true"' in markup
    assert 'data-live-generation-reasoning="true"' in markup
    assert (
        'data-live-generation-reasoning-text="Checking the requested details"' in markup
    )
    assert "jouzetsu-message-surface" not in markup


def test_generating_composer_exposes_a_collapsed_live_reasoning_panel() -> None:
    view: ChatView = _chat_view(
        Chat(id="chat-reasoning"),
        generating=True,
        runtime_status=RuntimeStatus(RuntimePhase.REASONING),
        generation_reasoning="Compare the constraints before answering.",
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))
    panel_opening: str = markup[
        markup.index("<details") : markup.index(">", markup.index("<details"))
    ]

    assert 'data-testid="generation-reasoning"' in markup
    assert "Thinking" in markup
    assert "Compare the constraints before answering." in markup
    assert "hidden" not in panel_opening
    assert "open" not in panel_opening


def test_chat_fragment_exposes_the_active_chat_identity_for_stable_reconciliation() -> (
    None
):
    view: ChatView = _chat_view(
        Chat(id="chat-stable"),
        runtime_status=RuntimeStatus(RuntimePhase.READY),
    )

    markup: str = repr(render_chat_fragment(view))

    assert 'id="chat-fragment"' in markup
    assert 'data-chat-id="chat-stable"' in markup
    assert 'data-testid="empty-chat-message"' in markup
    assert "An excellent blank canvas." in markup
    assert "say hi." not in markup


def test_streaming_message_exposes_its_unmodified_text_for_incremental_animation() -> (
    None
):
    markup: str = _render_message(
        [Message(id="assistant-stream", role="assistant", content="First words")],
        0,
        generating=True,
    )

    assert 'data-streaming-message-id="assistant-stream"' in markup
    assert 'data-streaming-text="First words"' in markup
    assert "data-streaming-pending" not in markup


def test_notice_uses_explicit_status_and_error_semantics_with_a_dismiss_control() -> (
    None
):
    success_markup: str = repr(render_notice_area(notice="Chat saved", error=""))
    error_markup: str = repr(render_notice_area(notice="", error="Could not save chat"))

    assert 'data-notice-kind="success"' in success_markup
    assert 'role="status"' in success_markup
    assert 'data-notice-dismiss="true"' in success_markup
    assert 'data-notice-kind="error"' in error_markup
    assert 'role="alert"' in error_markup


def test_muted_action_style_is_reflected_in_message_markup() -> None:
    markup: str = _render_message(
        [
            Message(
                id="assistant-muted", role="assistant", content="A compact action bar"
            )
        ],
        0,
        muted_actions=True,
    )

    assert "jouzetsu-message-actions jouzetsu-action-icons-muted" in markup
    assert "is-regenerate" in markup
    assert 'aria-label="Regenerate the last assistant message"' in markup


def test_composer_places_the_message_input_before_actions_for_keyboard_navigation() -> (
    None
):
    view: ChatView = _chat_view(
        Chat(id="chat-composer"),
        runtime_status=RuntimeStatus(RuntimePhase.READY),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert markup.index('data-testid="message-input"') < markup.index(
        'data-testid="send-message-button"'
    )
    assert 'rows="1"' in markup
    assert "jouzetsu-composer-input-row" in markup
    assert "jouzetsu-composer-primary" in markup
    assert 'aria-keyshortcuts="Shift+Enter Meta+Enter"' in markup
    assert 'title="Send (Shift+Enter / Cmd+Enter)"' in markup
    assert 'aria-label="Send"' in markup
    assert "jouzetsu-composer-primary is-icon" in markup
    assert "M22 2 11 13" in markup
    assert 'data-chat-mutation="true"' in markup


def test_editing_composer_marks_the_auto_sized_edit_mode() -> None:
    editing_message = Message(id="user-edit", role="user", content="Edit this message")
    view: ChatView = _chat_view(
        Chat(id="chat-edit", messages=[editing_message]),
        runtime_status=RuntimeStatus(RuntimePhase.READY),
    )

    markup: str = str(
        render_composer(
            view, edit_message_id=editing_message.id, editing_message=editing_message
        )
    )

    assert 'data-composer-mode="edit"' in markup
    assert 'data-testid="save-edit-button"' in markup
    assert 'title="Save edit (Shift+Enter / Cmd+Enter)"' in markup
    assert 'data-composer-input="true"' in markup
    assert 'aria-errormessage="composer-edit-validation"' in markup
    assert 'data-composer-validation="true"' in markup
    assert "composer-resize-handle" in markup


def test_editing_composer_disables_an_empty_edit_and_exposes_inline_validation() -> (
    None
):
    editing_message = Message(id="user-empty-edit", role="user", content="")
    view: ChatView = _chat_view(
        Chat(id="chat-empty-edit", messages=[editing_message]),
        runtime_status=RuntimeStatus(RuntimePhase.READY),
    )

    markup: str = str(
        render_composer(
            view, edit_message_id=editing_message.id, editing_message=editing_message
        )
    )
    input_test_id: int = markup.index('data-testid="message-input"')
    input_opening: str = markup[
        markup.rfind("<textarea", 0, input_test_id) : markup.index(">", input_test_id)
    ]
    save_button_start: int = markup.index('data-testid="save-edit-button"')
    save_button_opening: str = markup[
        markup.rfind("<button", 0, save_button_start) : markup.index(
            ">", save_button_start
        )
    ]

    assert "required" in input_opening
    assert "disabled" in save_button_opening
    assert 'data-composer-validation="true"' in markup
    assert "hidden" in markup


def test_generating_composer_keeps_the_draft_input_available_and_uses_the_primary_stop_control() -> (
    None
):
    view: ChatView = _chat_view(
        Chat(id="chat-generating", draft="Next question"),
        generating=True,
        runtime_status=RuntimeStatus(RuntimePhase.GENERATING, output_tokens=12),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))
    input_end: int = markup.index("</textarea>")

    assert 'data-composer-generating="true"' in markup
    assert (
        "disabled"
        not in markup[markup.index('data-testid="message-input"') : input_end]
    )
    assert "required" not in markup[markup.rfind("<textarea", 0, input_end) : input_end]
    assert "jouzetsu-generation-strip" in markup
    assert 'data-testid="stop-generation-button"' in markup
    assert 'action="/chat/stop"' in markup
    assert 'data-testid="send-message-button"' not in markup
    stop_button_start: int = markup.index('data-testid="stop-generation-button"')
    assert (
        "disabled"
        not in markup[
            markup.rfind("<button", 0, stop_button_start) : markup.index(
                ">", stop_button_start
            )
        ]
    )


def test_generating_chat_disables_every_generation_sensitive_setting() -> None:
    view: ChatSettingsView = ChatSettingsView(
        chat=Chat(id="chat-generating-settings"),
        generating=True,
        generation_defaults=AppConfig().generation,
        model_choices=(ModelChoice(key="demo-model", label="Demo Model"),),
    )

    markup: str = repr(render_settings_panel(view))
    for marker in (
        "active-chat-model",
        "active-chat-temperature",
        "active-chat-top-p",
        "active-chat-max-tokens",
        "save-chat-sampling-button",
        "active-chat-prompt",
        "save-chat-prompt-button",
        "active-chat-british-spellings",
        "active-chat-save-reasoning",
    ):
        marker_index: int = markup.index(f'data-testid="{marker}"')
        opening_tag: str = markup[
            markup.rfind("<", 0, marker_index) : markup.index(">", marker_index)
        ]
        assert "disabled" in opening_tag


def test_settings_panel_lists_every_character_in_a_cast() -> None:
    chat = Chat(
        id="chat-cast-settings",
        character_cast=(
            CharacterChatBinding(id="mira123", name="Mira", revision=2),
            CharacterChatBinding(id="ren123", name="Ren", revision=4),
        ),
        prompt_mode=ChatPromptMode.STORY,
    )
    view: ChatSettingsView = ChatSettingsView(
        chat=chat,
        generating=False,
        generation_defaults=AppConfig().generation,
        model_choices=(),
    )

    markup: str = repr(render_settings_panel(view))

    assert "Cast" in markup
    assert "Started as: Story" in markup
    assert "Mira (revision 2)" in markup
    assert "Ren (revision 4)" in markup
    assert 'href="/characters/mira123"' in markup
    assert 'href="/characters/ren123"' in markup


def test_generating_model_cannot_be_unloaded_from_the_models_dialog() -> None:
    model: ModelDescriptor = ModelDescriptor(
        key="demo-model", display_name="Demo Model"
    )
    state: AppState = cast(
        AppState,
        cast(
            object,
            _SettingsViewState(
                active_chat=Chat(id="chat-generating-model"),
                generating=False,
                generating_model_keys=frozenset({model.key}),
            ),
        ),
    )

    markup: str = repr(
        render_loaded_model_card(
            state, model, ModelInstanceDescriptor(id="demo-instance")
        )
    )
    marker_index: int = markup.index('data-testid="unload-model-demo-instance"')
    opening_tag: str = markup[
        markup.rfind("<", 0, marker_index) : markup.index(">", marker_index)
    ]

    assert "disabled" in opening_tag
    assert "Cannot unload a model while it is generating a response" in markup


def test_composer_status_distinguishes_model_loading_from_prompt_processing() -> None:
    overall_started_at: float = time.monotonic() - 5.0
    view: ChatView = _chat_view(
        Chat(id="chat-loading"),
        generating=True,
        runtime_status=RuntimeStatus(
            RuntimePhase.LOADING_MODEL,
            model_name="Cold Model",
            started_at=overall_started_at,
            phase_started_at=time.monotonic() - 1.0,
        ),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert (
        'data-composer-status-state="true" class="jouzetsu-composer-status-state">Loading'
        in markup
    )
    assert (
        'data-composer-status-detail="true" class="jouzetsu-composer-status-detail">Cold Model'
        in markup
    )
    assert "in progress" not in markup
    assert "Processing prompt" not in markup
    assert 'data-composer-status-progress="true"' in markup
    assert "data-composer-status-load-progress" not in markup
    progress_marker_index: int = markup.index('data-composer-status-progress="true"')
    progress_tag: str = markup[
        markup.rfind("<", 0, progress_marker_index) : markup.index(
            ">", progress_marker_index
        )
    ]
    assert "hidden" in progress_tag
    assert 'data-composer-status-timers="true"' in markup
    assert 'data-composer-status-stage-elapsed="true"' in markup
    assert 'data-composer-status-overall-elapsed="true"' in markup
    assert 'data-composer-status-stage-elapsed-seconds="' in markup
    assert 'data-composer-status-overall-elapsed-seconds="' in markup
    assert "1." in markup
    assert "5." in markup

    view = _chat_view(
        Chat(id="chat-processing"),
        generating=True,
        runtime_status=RuntimeStatus(
            RuntimePhase.PROCESSING_PROMPT,
            model_name="Cold Model",
            started_at=overall_started_at,
            phase_started_at=time.monotonic() - 1.0,
        ),
    )

    markup = str(render_composer(view, edit_message_id="", editing_message=None))

    assert ">Processing<" in markup
    assert "Prompt" in markup
    assert "in progress" not in markup
    assert "Cold Model" not in markup
    assert ">Loading<" not in markup
    assert "data-composer-status-load-progress" not in markup

    view = _chat_view(
        Chat(id="chat-processing-progress"),
        generating=True,
        runtime_status=RuntimeStatus(
            RuntimePhase.PROCESSING_PROMPT,
            model_name="Cold Model",
            progress=0.5,
            started_at=overall_started_at,
            phase_started_at=time.monotonic() - 1.0,
        ),
    )

    markup = str(render_composer(view, edit_message_id="", editing_message=None))

    assert ">Processing<" in markup
    assert "Prompt · 50%" in markup
    assert "data-composer-status-load-progress" not in markup


def test_loading_composer_status_renders_numeric_progress_bar() -> None:
    view: ChatView = _chat_view(
        Chat(id="chat-loading-progress"),
        generating=True,
        runtime_status=RuntimeStatus(
            RuntimePhase.LOADING_MODEL,
            model_name="Cold Model",
            progress=0.5,
        ),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert "Cold Model · 50%" in markup
    assert 'data-composer-status-load-progress="50.000"' in markup
    assert 'aria-label="Model loading progress"' in markup
    assert 'aria-valuenow="50.000"' in markup
    progress_marker_index = markup.index('data-composer-status-progress="true"')
    progress_tag = markup[
        markup.rfind("<", 0, progress_marker_index) : markup.index(
            ">", progress_marker_index
        )
    ]
    assert "hidden" not in progress_tag
    assert markup.count('aria-live="off"') == 2
    assert "--jouzetsu-composer-status-load-progress: 50.000%" in markup

    live_markup: str = repr(render_chat_live_fragment(view))

    assert 'data-live-composer-status="true"' in live_markup
    assert 'data-composer-status-load-progress="50.000"' in live_markup
    assert 'aria-valuenow="50.000"' in live_markup


def test_composer_moves_multiple_shortcuts_to_an_action_tray() -> None:
    view: ChatView = _chat_view(
        Chat(id="chat-shortcuts"),
        starter_prompts=(
            StarterPrompt(label="Explain", content="Explain this"),
            StarterPrompt(label="Continue", content="Continue this"),
        ),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert "jouzetsu-composer-footer-has-multiple-actions" in markup
    assert 'data-testid="composer-suggestion-0"' in markup
    assert 'data-testid="composer-suggestion-1"' in markup


def test_ready_composer_status_omits_model_name_but_keeps_completion_metrics() -> None:
    view: ChatView = _chat_view(
        Chat(id="chat-ready"),
        runtime_status=RuntimeStatus(
            RuntimePhase.READY,
            model_name="Ready Model",
            metrics=GenerationMetrics(
                output_tokens=42,
                tokens_per_second=12.3,
                time_to_first_token_seconds=0.45,
            ),
        ),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert ">Ready<" in markup
    assert "Ready Model" not in markup
    assert "12.3 tok/s · 42 tokens · TTFT 0.45s" in markup


def test_composer_rejects_an_unknown_runtime_phase() -> None:
    unknown_phase: RuntimePhase = cast(RuntimePhase, object())
    view: ChatView = _chat_view(
        Chat(id="chat-unknown-phase"),
        runtime_status=RuntimeStatus(unknown_phase),
    )

    with pytest.raises(ValueError, match="unsupported runtime phase"):
        _ = render_composer(view, edit_message_id="", editing_message=None)


def test_unloaded_model_status_omits_redundant_not_loaded_detail() -> None:
    view: ChatView = _chat_view(
        Chat(id="chat-unloaded"),
        runtime_status=RuntimeStatus(
            RuntimePhase.MODEL_UNLOADED, model_name="PaintedFantasy"
        ),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert ">Unloaded<" in markup
    assert (
        'data-composer-status-detail="true" class="jouzetsu-composer-status-detail">PaintedFantasy'
        in markup
    )
    assert "Not loaded" not in markup


def test_composer_status_places_reasoning_and_output_in_the_state_column() -> None:
    for phase, expected_state in (
        (RuntimePhase.REASONING, "Reasoning"),
        (RuntimePhase.GENERATING, "Generating"),
    ):
        view: ChatView = _chat_view(
            Chat(id=f"chat-{phase.value}"),
            generating=True,
            runtime_status=RuntimeStatus(
                phase,
                output_tokens=12,
                started_at=time.monotonic() - 2.0,
                phase_started_at=time.monotonic() - 1.0,
            ),
        )

        markup: str = str(
            render_composer(view, edit_message_id="", editing_message=None)
        )

        assert (
            f'data-composer-status-state="true" class="jouzetsu-composer-status-state">{expected_state}'
            in markup
        )
        assert (
            'data-composer-status-detail="true" class="jouzetsu-composer-status-detail">12 tokens'
            in markup
        )
        assert 'data-composer-status-timers="true"' in markup


def test_reviewing_composer_status_labels_the_review_stage() -> None:
    view: ChatView = _chat_view(
        Chat(id="chat-reviewing"),
        generating=True,
        runtime_status=RuntimeStatus(
            RuntimePhase.REVIEWING,
            model_name="Demo Model",
            started_at=time.monotonic() - 2.0,
            phase_started_at=time.monotonic() - 0.5,
        ),
    )

    markup: str = str(render_composer(view, edit_message_id="", editing_message=None))

    assert ">Reviewing<" in markup
    assert "Response continuity" in markup


def test_composer_status_labels_preparing_and_stopping_stages() -> None:
    for phase, expected_state, expected_detail in (
        (RuntimePhase.STARTING, "Preparing", "Demo Model"),
        (RuntimePhase.STOPPING, "Stopping", ""),
    ):
        view: ChatView = _chat_view(
            Chat(id=f"chat-{phase.value}"),
            generating=True,
            runtime_status=RuntimeStatus(phase, model_name="Demo Model"),
        )

        markup: str = str(
            render_composer(view, edit_message_id="", editing_message=None)
        )

        assert (
            f'data-composer-status-state="true" class="jouzetsu-composer-status-state">{expected_state}'
            in markup
        )
        if expected_detail:
            assert expected_detail in markup


def test_pending_access_page_exposes_the_request_csrf_token() -> None:
    context: PageContext = PageContext(
        decision=AccessDecision(
            access_allowed=False,
            can_use_global_settings=False,
            can_manage_access=False,
            is_localhost=False,
            device_id="dvc_pending-device-0001",
            reason="pending_device",
        ),
        client_label="Remote browser",
        csrf_token="csrf-token-for-pending-browser",
    )

    markup: str = str(
        render_locked_access_page(
            context, bootstrap_device=False, approval_phrase_enabled=True
        )
    )

    assert 'data-csrf-token="csrf-token-for-pending-browser"' in markup
    assert 'action="/access/current-device/pending"' in markup
    assert 'data-testid="access-approval-phrase-input"' in markup
