"""Focused browser coverage for client-side state that unit tests cannot observe."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Final, Literal, TypedDict, cast

import pytest
from browser_support import (
    BrowserServer,
    BrowserTestBackend,
    create_browser_server,
)
from image_support import image_bytes
from playwright.sync_api import (
    Browser,
    CDPSession,
    Error,
    Page,
    Route,
    expect,
    sync_playwright,
)

from jouzetsu.backend_models import RegisteredModel
from jouzetsu.config import (
    CONTINUE_PROMPT_CONTENT,
    ConfigStore,
    DeviceAccessSettings,
    MessageAction,
    MessageActionStyle,
    StarterPrompt,
)
from jouzetsu.models import Chat
from jouzetsu.runtime import ModelCapabilities, ModelDescriptor

_MESSAGE_LIST_SELECTOR: Final[str] = '[data-testid="message-list"]'
_MESSAGE_SELECTOR: Final[str] = "article[data-message-id]"
_SCROLL_RESTORE_TOLERANCE_PX: Final[float] = 2.0
_MANUAL_SCROLL_OFFSET_PX: Final[float] = 17.0
_BROWSER_ACTION_TIMEOUT_MS: Final[float] = 5_000
_LONG_MESSAGE: Final[str] = " ".join(
    "This deliberately long browser-test message keeps the transcript scrollable."
    for _ in range(20)
)
_STREAMING_RESPONSE: Final[str] = "\n\n".join(
    (_LONG_MESSAGE, _LONG_MESSAGE, " ".join("extra" for _ in range(70)))
)
pytestmark = pytest.mark.browser


class ScrollMetrics(TypedDict):
    top: float
    maximum: float


class MessagePositionMetrics(TypedDict):
    """Scroll and viewport position of one transcript message."""

    gap: float
    message_top: float
    top: float


class MessageActionLayoutMetrics(TypedDict):
    """Narrow-screen dimensions of a message's action area."""

    actions_width: float
    footer_width: float
    row_count: int


@pytest.fixture()
def chromium_browser() -> Iterator[Browser]:
    """Run Chromium for one test so Playwright releases its event loop afterward."""

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Error as error:
            pytest.skip(
                "Playwright Chromium is unavailable; run "
                "`uv run --group dev playwright install chromium` "
                f"({error})"
            )
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture()
def browser_server(tmp_path: Path) -> Iterator[BrowserServer]:
    """Provide one isolated app process and state store per browser test."""

    server = create_browser_server(tmp_path)
    try:
        yield server
    finally:
        server.close()


@pytest.fixture()
def browser_page(
    chromium_browser: Browser, browser_server: BrowserServer
) -> Iterator[Page]:
    """Create a desktop page whose same-origin requests target the test server."""

    context = chromium_browser.new_context(
        base_url=browser_server.base_url,
        viewport={"width": 960, "height": 720},
        reduced_motion="reduce",
    )
    page = context.new_page()
    page.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
    try:
        yield page
    finally:
        context.close()


@pytest.fixture()
def touch_browser_page(
    chromium_browser: Browser, browser_server: BrowserServer
) -> Iterator[Page]:
    with chromium_browser.new_context(
        base_url=browser_server.base_url,
        viewport={"width": 390, "height": 844},
        is_mobile=True,
        has_touch=True,
        reduced_motion="reduce",
    ) as context:
        page = context.new_page()
        page.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
        yield page


def test_character_onboarding_randomizes_each_name_part(
    browser_page: Page,
) -> None:
    """Onboarding keeps both independent randomisers live before a character exists."""

    browser_page.add_init_script("Math.random = () => 0;")
    browser_page.goto("/characters")
    with browser_page.expect_navigation(wait_until="load"):
        browser_page.get_by_role("button", name="New character").click()

    expect(browser_page.get_by_test_id("character-editor-form")).to_be_visible()
    expect(browser_page.get_by_role("heading", name="Create character")).to_be_visible()

    given_name = browser_page.get_by_test_id("character-given-name-input")
    family_name = browser_page.get_by_test_id("character-family-name-input")
    browser_page.get_by_test_id("character-given-name-randomize").click()
    first_given_name = given_name.input_value()
    browser_page.get_by_test_id("character-given-name-randomize").click()
    second_given_name = given_name.input_value()
    browser_page.get_by_test_id("character-family-name-randomize").click()
    first_family_name = family_name.input_value()
    browser_page.get_by_test_id("character-family-name-randomize").click()
    second_family_name = family_name.input_value()

    assert first_given_name
    assert second_given_name
    assert first_given_name != second_given_name
    assert first_family_name
    assert second_family_name
    assert first_family_name != second_family_name
    assert given_name.input_value() == second_given_name


def test_message_scroll_stays_pinned_or_restores_its_position(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """A full fragment update follows a pinned reader but leaves a browsing reader put."""

    message_id = _seed_scrollable_transcript(browser_server)
    _open_chat_page(browser_page)
    initial = _scroll_metrics(browser_page)
    assert initial["maximum"] > 300

    _set_message_scroll_top(browser_page, initial["maximum"])
    _edit_message_in_background(
        browser_page,
        message_id,
        f"pinned-fragment-update\n\n{_LONG_MESSAGE}\n\n{_LONG_MESSAGE}",
    )
    _wait_for_message_text(browser_page, message_id, "pinned-fragment-update")
    pinned = _scroll_metrics(browser_page)

    assert pinned["maximum"] > initial["maximum"]
    assert pinned["maximum"] - pinned["top"] <= _SCROLL_RESTORE_TOLERANCE_PX

    expected_scroll_top = pinned["maximum"] * 0.35
    _set_message_scroll_top(browser_page, expected_scroll_top)
    _edit_message_in_background(
        browser_page,
        message_id,
        f"restored-fragment-update\n\n{_LONG_MESSAGE}\n\n{_LONG_MESSAGE}\n\n{_LONG_MESSAGE}",
    )
    _wait_for_message_text(browser_page, message_id, "restored-fragment-update")
    restored = _scroll_metrics(browser_page)

    assert abs(restored["top"] - expected_scroll_top) <= _SCROLL_RESTORE_TOLERANCE_PX
    assert restored["maximum"] - restored["top"] > 100


def test_narrow_message_actions_use_the_full_footer_width_before_wrapping(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """Five mobile controls stay on one row in a 280px-wide message list."""

    browser_page.set_viewport_size({"width": 640, "height": 720})
    chat = browser_server.state.chats[0]
    _ = chat.add_message("user", "Prompt")
    _ = chat.add_message("assistant", "Earlier reply")
    message = chat.add_message("assistant", '"Certainly."')
    _open_chat_page(browser_page)
    _constrain_message_list_width(browser_page, 280)

    selector = _message_selector(message.id)
    assert browser_page.locator(f"{selector} .jouzetsu-message-action").count() == 5
    layout = _message_action_layout_metrics(browser_page, selector)

    assert layout["footer_width"] <= 250
    assert (
        abs(layout["actions_width"] - layout["footer_width"])
        <= _SCROLL_RESTORE_TOLERANCE_PX
    )
    assert layout["row_count"] == 1


def test_new_messages_snap_to_bottom_then_stream_to_top_on_narrow_viewport(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """A mobile turn settles at the tail before its streamed reply follows to the top."""

    browser_page.set_viewport_size({"width": 390, "height": 720})
    _seed_transcript(browser_server, 6, _LONG_MESSAGE)
    completion_gate = _configure_paused_streaming_response(browser_server)
    try:
        _open_chat_page(browser_page)
        browser_page.get_by_test_id("message-input").fill("Generate a long reply")
        browser_page.get_by_test_id("send-message-button").click()
        browser_page.wait_for_selector("article.is-streaming[data-message-id]")
        _wait_for_animation_frames(browser_page)

        browser_page.wait_for_function(
            "() => document.querySelector('article.is-streaming[data-message-id]')"
            "?.innerText.includes('deliberately long')",
        )
        _wait_for_animation_frames(browser_page)
        assistant_turn = _message_position_metrics(
            browser_page, "article.is-streaming[data-message-id]"
        )
        assert abs(assistant_turn["message_top"]) <= _SCROLL_RESTORE_TOLERANCE_PX
        assert assistant_turn["gap"] > _MANUAL_SCROLL_OFFSET_PX

        manual_scroll_top = assistant_turn["top"] + _MANUAL_SCROLL_OFFSET_PX
        _set_message_scroll_top(browser_page, manual_scroll_top)
        manual_position = _message_position_metrics(
            browser_page, "article.is-streaming[data-message-id]"
        )
        assert (
            abs(manual_position["top"] - manual_scroll_top)
            <= _SCROLL_RESTORE_TOLERANCE_PX
        )
        completion_gate.set()
        browser_page.wait_for_function(
            "() => !document.querySelector('article.is-streaming[data-message-id]')"
        )
        _wait_for_animation_frames(browser_page)
        completed = _message_position_metrics(
            browser_page, "article.is-assistant[data-message-id]:last-child"
        )

        assert (
            abs(completed["message_top"] - manual_position["message_top"])
            <= _SCROLL_RESTORE_TOLERANCE_PX
        )
    finally:
        completion_gate.set()


@pytest.mark.parametrize("continuation", (False, True))
def test_regenerate_button_replaces_its_reply_using_only_the_previous_history(
    browser_page: Page,
    browser_server: BrowserServer,
    continuation: bool,
) -> None:
    state = browser_server.state
    chat: Chat = state.chats[0]
    _ = chat.add_message("user", "First question")
    earlier_reply = chat.add_message("assistant", "Keep this earlier answer")
    _ = chat.add_message("user", "Latest question")
    if continuation:
        _ = chat.add_message("assistant", "Keep this response before continuing")
    history: list[tuple[str, str]] = [
        (message.role, message.content) for message in chat.messages
    ]
    previous_ids: list[str] = [message.id for message in chat.messages]
    target = chat.add_message("assistant", "Replace this reply")
    browser_server.client.response_content = "Regenerated answer"
    _open_chat_page(browser_page)
    expect(
        browser_page.get_by_test_id(f"regenerate-message-{earlier_reply.id}")
    ).to_have_count(0)
    browser_page.get_by_test_id("message-input").fill("Keep this unsent draft")
    browser_page.get_by_test_id(f"regenerate-message-{target.id}").click()

    expect(browser_page.get_by_text("Regenerated answer", exact=True)).to_be_visible()
    expect(browser_page.locator(_message_selector(target.id))).to_have_count(1)
    expect(browser_page.locator(_MESSAGE_SELECTOR)).to_have_count(len(previous_ids) + 1)
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "Keep this unsent draft"
    )
    requests = browser_server.client.requests
    assert len(requests) == 1
    expected_history: list[tuple[str, str]] = [
        ("system", state.config.generation.system_prompt),
        *history,
    ]
    if continuation:
        expected_history.append(("user", CONTINUE_PROMPT_CONTENT))
    assert [
        (message.role, message.content) for message in requests[0].messages
    ] == expected_history
    assert [message.id for message in chat.messages[:-1]] == previous_ids
    assert chat.messages[-1].id == target.id
    assert chat.messages[-1].content == "Regenerated answer"
    assert chat.draft == "Keep this unsent draft"


def test_continue_streams_a_new_message_and_preserves_the_original_on_reload(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    chat = browser_server.state.chats[0]
    _ = chat.add_message("user", "Tell me a story")
    original = chat.add_message("assistant", "An unfinished story")
    original_record = original.to_dict()
    gate = Event()
    browser_server.client.completion_gate = gate
    browser_server.client.response_content = "The next part"
    try:
        _open_chat_page(browser_page)
        browser_page.get_by_test_id(f"continue-message-{original.id}").click()
        expect(browser_page.locator("article.is-streaming")).to_contain_text(
            "The next part"
        )
        expect(browser_page.locator(_MESSAGE_SELECTOR)).to_have_count(3)
        expect(browser_page.locator(_message_selector(original.id))).to_contain_text(
            "An unfinished story"
        )
        assert original.to_dict() == original_record
        assert len(chat.messages) == 2
        gate.set()
        browser_page.wait_for_function(
            "() => !document.querySelector('article.is-streaming')"
        )
        assert len(chat.messages) == 3
        assert chat.messages[-2] is original
        assert chat.messages[-1].id != original.id
        assert chat.messages[-1].content == "The next part"
        assert original.to_dict() == original_record
        browser_page.reload()
        expect(browser_page.locator(_MESSAGE_SELECTOR)).to_have_count(3)
        expect(
            browser_page.get_by_text("An unfinished story", exact=True)
        ).to_be_visible()
        expect(browser_page.get_by_text("The next part", exact=True)).to_be_visible()
    finally:
        gate.set()


def _use_vision_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def inventory(
        _client: BrowserTestBackend,
    ) -> tuple[ModelDescriptor, ...]:
        return (
            ModelDescriptor(
                key="browser-test-model",
                display_name="Vision model",
                capabilities=ModelCapabilities(vision=True),
            ),
            ModelDescriptor(key="text-model", display_name="Text model"),
        )

    _ = (monkeypatch, inventory)


def _open_vision_chat(page: Page) -> None:
    page.goto("/chats")


def test_uploads_keep_send_disabled_through_fragment_refreshes(
    browser_page: Page,
    browser_server: BrowserServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_vision_inventory(monkeypatch)
    _open_vision_chat(browser_page)
    browser_page.get_by_test_id("message-input").fill("Describe the image")
    browser_page.evaluate(
        """() => {
            const fetch = window.fetch.bind(window);
            window.fetch = async (url, options) => {
                if (new URL(url, location.href).pathname.endsWith('/images')) {
                    await new Promise((resolve) => { window.releaseImageUpload = resolve; });
                }
                return fetch(url, options);
            };
        }"""
    )
    browser_page.get_by_test_id("image-upload-input").set_input_files(
        {"name": "red.png", "mimeType": "image/png", "buffer": image_bytes()}
    )
    expect(browser_page.get_by_test_id("send-message-button")).to_be_disabled()
    expect(browser_page.locator("[data-image-upload-pending]")).to_be_visible()
    browser_page.get_by_test_id("open-settings-button").click()
    with browser_page.expect_response("**/fragments/chat*"):
        browser_page.locator("#settings-panel [data-model-picker-value]").evaluate(
            "control => control.dispatchEvent(new Event('change', { bubbles: true }))"
        )
    browser_page.get_by_role("button", name="Close settings").click()
    expect(browser_page.get_by_test_id("send-message-button")).to_be_disabled()
    expect(browser_page.locator("[data-image-upload-pending]")).to_be_visible()
    browser_page.evaluate("window.releaseImageUpload()")
    expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
    expect(browser_page.get_by_test_id("send-message-button")).to_be_enabled()
    expect(browser_page.locator("[data-image-upload-pending]")).to_be_hidden()
    browser_page.get_by_test_id("send-message-button").click()
    expect(browser_page.get_by_text("Browser test reply", exact=True)).to_be_visible()
    request = browser_server.client.requests[0]
    assert request.messages[-1].content == "Describe the image"
    assert len(request.messages[-1].images) == 1


@pytest.mark.parametrize("first_upload_succeeds", (True, False))
def test_image_queue_waits_for_added_uploads_and_retains_batch_failures(
    browser_page: Page, first_upload_succeeds: bool
) -> None:
    _open_chat_page(browser_page)
    result = browser_page.evaluate(
        """async (firstUploadSucceeds) => {
            const { ImageController } = await import('/static/app/images.js');
            const releases = [];
            const waiters = [];
            const nextRelease = () => releases.length ? Promise.resolve(releases.shift()) : new Promise((resolve) => waiters.push(resolve));
            const originalFetch = window.fetch.bind(window);
            window.fetch = (url, options) => new URL(url, location.href).pathname.endsWith('/images')
                ? new Promise((resolve) => { const waiter = waiters.shift(); if (waiter) waiter(resolve); else releases.push(resolve); })
                : originalFetch(url, options);
            const images = new ImageController({ announce: () => {}, syncPrimary: () => {}, drafts: {flush: async () => true, serialize: async (operation) => operation(), savedRevision: () => {}} });
            const form = document.createElement('form');
            form.dataset.imageUploadForm = 'true';
            form.action = document.querySelector('[data-image-upload-form]').action;
            images.handleSubmit(form, null);
            let settled = false;
            const flushed = images.flush().then((success) => { settled = true; return success; });
            images.handleSubmit(form, null);
            (await nextRelease())(new Response(JSON.stringify({draft_revision: 1}), { status: firstUploadSucceeds ? 200 : 400 }));
            const secondRelease = await nextRelease();
            const waitedForBoth = !settled && images.isUploading;
            const latestFlush = images.flush();
            secondRelease(new Response(JSON.stringify({draft_revision: 2})));
            const succeeded = await flushed;
            const batchSucceeded = await latestFlush;
            images.handleSubmit(form, null);
            (await nextRelease())(new Response(JSON.stringify({draft_revision: 2})));
            return { waitedForBoth, succeeded, batchSucceeded, recovered: await images.flush() };
        }""",
        first_upload_succeeds,
    )
    assert result == {
        "waitedForBoth": True,
        "succeeded": first_upload_succeeds,
        "batchSucceeded": first_upload_succeeds,
        "recovered": True,
    }


def test_cancel_edit_returns_to_the_chat_and_preserves_the_unsent_draft(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    chat = browser_server.state.chats[0]
    message = chat.add_message("user", "Original message")
    _ = chat.add_message("assistant", "Original reply")
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("message-input").fill("Unsent draft")
    browser_page.get_by_test_id(f"edit-message-{message.id}").click()
    expect(
        browser_page.get_by_role("textbox", name="Edit message", exact=True)
    ).to_have_value("Original message")
    browser_page.get_by_test_id("message-input").fill("Discard this edit")
    browser_page.get_by_test_id("cancel-edit-button").click()
    expect(browser_page).to_have_url(f"{browser_server.base_url}/chats/{chat.id}")
    expect(browser_page.get_by_test_id("message-input")).to_have_value("Unsent draft")
    browser_page.reload()
    expect(browser_page.get_by_test_id("message-input")).to_have_value("Unsent draft")
    assert message.content == "Original message"


@pytest.mark.parametrize("viewport_width", (960, 390))
def test_spelling_settings_collapse_without_losing_values_or_hiding_errors(
    browser_page: Page, browser_server: BrowserServer, viewport_width: int
) -> None:
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    original_replacements = list(
        browser_server.state.config.generation.british_spelling_replacements
    )
    browser_page.goto("/chats?dialog=global")
    details = browser_page.get_by_test_id("spelling-replacements-details")
    expect(details).to_have_js_property("open", False)
    save = browser_page.get_by_test_id("save-global-generation-button")
    expect(save).to_be_visible()
    bounds = save.bounding_box()
    assert bounds is not None and bounds["y"] + bounds["height"] <= 844
    with browser_page.expect_navigation(wait_until="load"):
        save.click()
    assert (
        browser_server.state.config.generation.british_spelling_replacements
        == original_replacements
    )
    details.locator("summary").click()
    source = details.locator("[data-spelling-source]").first
    source.fill("two words")
    details.locator("summary").click()
    save.click()
    expect(details).to_have_js_property("open", True)
    expect(source).to_be_focused()
    expect(details.locator("[data-spelling-validation]")).to_contain_text("single-word")


def test_message_context_menu_copies_and_forks_at_the_selected_message(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """The shared context menu applies actions to its selected transcript message."""

    _ = browser_server.state.chats[0].add_message("user", "Fork source")
    selected = browser_server.state.chats[0].add_message(
        "assistant", "Copy this selected reply"
    )
    _ = browser_server.state.chats[0].add_message("user", "Excluded after fork")
    browser_page.add_init_script(
        """
        Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: {
                writeText: async (text) => { window.__browserTestClipboard = text; },
            },
        });
        """
    )
    _open_chat_page(browser_page)

    selected_message = browser_page.locator(_message_selector(selected.id))
    selected_message.click(button="right")
    menu = browser_page.get_by_test_id("message-context-menu")
    expect(menu).to_be_visible()
    expect(
        browser_page.get_by_test_id("message-fork-context-action")
    ).to_have_attribute("action", f"/messages/{selected.id}/fork")

    browser_page.get_by_test_id("message-copy-context-action").click()
    browser_page.wait_for_function(
        "() => window.__browserTestClipboard === 'Copy this selected reply'"
    )
    expect(menu).not_to_be_visible()

    previous_chat_id = browser_page.locator("#chat-fragment").get_attribute(
        "data-chat-id"
    )
    assert previous_chat_id is not None
    selected_message.click(button="right")
    fork_action = browser_page.get_by_test_id("message-fork-context-action")
    expect(menu).to_be_visible()
    expect(fork_action).to_be_visible()
    fork_action.get_by_role("menuitem", name="Fork chat").click()
    browser_page.wait_for_function(
        """(oldChatId) => document.querySelector('#chat-fragment')?.dataset.chatId !== oldChatId""",
        arg=previous_chat_id,
    )

    expect(browser_page.locator(_MESSAGE_SELECTOR)).to_have_count(2)
    expect(browser_page.get_by_text("Fork source", exact=True)).to_be_visible()
    expect(
        browser_page.get_by_text("Copy this selected reply", exact=True)
    ).to_be_visible()
    expect(browser_page.get_by_text("Excluded after fork")).to_have_count(0)


@pytest.mark.parametrize("viewport_width", [390, 960])
@pytest.mark.parametrize("resized", [False, True])
def test_thinking_disclosure_stays_open_through_reasoning_and_draft_refreshes(
    browser_page: Page,
    browser_server: BrowserServer,
    viewport_width: int,
    resized: bool,
) -> None:
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    gate = Event()
    initial_reasoning = "\n".join(f"Reasoning step {index}" for index in range(20))
    browser_server.client.response_reasoning = initial_reasoning
    browser_server.client.response_content = ""
    browser_server.client.completion_gate = gate
    try:
        _open_chat_page(browser_page)
        browser_page.get_by_test_id("message-input").fill("Think about this")
        browser_page.get_by_test_id("send-message-button").click()
        reasoning = browser_page.get_by_test_id("generation-reasoning")
        expect(reasoning).to_be_visible()
        if resized:
            browser_page.get_by_test_id("composer-resize-handle").press("ArrowUp")
        reasoning.locator("summary").click()
        expect(reasoning).to_have_js_property("open", True)
        expect(reasoning.locator("pre")).to_have_text(initial_reasoning)
        expect(reasoning.locator("pre")).to_be_visible()
        _wait_for_animation_frames(browser_page)
        reasoning_bounds = reasoning.bounding_box()
        input_bounds = browser_page.get_by_test_id("message-input").bounding_box()
        assert reasoning_bounds and input_bounds
        assert (
            reasoning_bounds["y"] + reasoning_bounds["height"] <= input_bounds["y"] + 1
        )
        _ = reasoning.evaluate("element => { window.reasoningElement = element; }")
        browser_page.get_by_test_id("message-input").fill("My next message")
        browser_page.wait_for_function(
            "() => document.querySelector('.jouzetsu-composer-form input[name=draft_revision]').value === '2'"
        )
        expect(reasoning).to_have_js_property("open", True)
        assert browser_page.evaluate(
            "window.reasoningElement === document.getElementById('generation-reasoning')"
        )
        chat_id = browser_server.state.chats[0].id
        browser_server.state.session(chat_id).live_reasoning = "A later reasoning step"
        _create_chat_in_background(browser_page)
        expect(reasoning.locator("pre")).to_have_text("A later reasoning step")
        expect(reasoning).to_have_js_property("open", True)
        assert browser_page.evaluate(
            "window.reasoningElement === document.getElementById('generation-reasoning')"
        )
        reasoning.locator("summary").click()
        expect(reasoning).to_have_js_property("open", False)
    finally:
        gate.set()


def _dispatch_touch(
    session: CDPSession,
    phase: Literal["touchStart", "touchMove", "touchEnd", "touchCancel"],
    x: float = 0,
    y: float = 0,
) -> None:
    send = cast(Callable[[str, dict[str, object]], object], session.send)
    _ = send(
        "Input.dispatchTouchEvent",
        {
            "type": phase,
            "touchPoints": [{"x": x, "y": y}]
            if phase in {"touchStart", "touchMove"}
            else [],
        },
    )


def test_mobile_long_press_opens_message_menu_and_details(
    touch_browser_page: Page, browser_server: BrowserServer
) -> None:
    message = browser_server.state.chats[0].add_message("assistant", "Touch this reply")
    message.set_reasoning("Saved reasoning")
    page = touch_browser_page
    _open_chat_page(page)
    page.clock.install()
    session = page.context.new_cdp_session(page)
    bounds = page.locator(
        _message_selector(message.id) + " .jouzetsu-message-text"
    ).bounding_box()
    assert bounds
    _dispatch_touch(session, "touchStart", bounds["x"] + 20, bounds["y"] + 10)
    page.clock.run_for(600)
    _dispatch_touch(session, "touchEnd")
    menu = page.get_by_test_id("message-context-menu")
    expect(menu).to_be_visible()
    page.get_by_role("menuitem", name="Message details", exact=True).tap()
    expect(page.get_by_test_id("message-details-reasoning")).to_have_text(
        "Saved reasoning"
    )


@pytest.mark.parametrize("gesture", ["tap", "move", "cancel"])
def test_mobile_message_long_press_cancels_without_blocking_the_next_hold(
    touch_browser_page: Page,
    browser_server: BrowserServer,
    gesture: Literal["tap", "move", "cancel"],
) -> None:
    message = browser_server.state.chats[0].add_message("user", "Touch this prompt")
    page = touch_browser_page
    _open_chat_page(page)
    page.clock.install()
    session = page.context.new_cdp_session(page)
    bounds = page.locator(
        _message_selector(message.id) + " .jouzetsu-message-text"
    ).bounding_box()
    assert bounds
    x, y = bounds["x"] + 20, bounds["y"] + 10
    _dispatch_touch(session, "touchStart", x, y)
    if gesture == "move":
        _dispatch_touch(session, "touchMove", x, y + 30)
        page.clock.run_for(600)
    _dispatch_touch(session, "touchCancel" if gesture == "cancel" else "touchEnd")
    page.clock.run_for(600)
    menu = page.get_by_test_id("message-context-menu")
    expect(menu).not_to_be_visible()
    _dispatch_touch(session, "touchStart", x, y)
    page.clock.run_for(600)
    _dispatch_touch(session, "touchEnd")
    expect(menu).to_be_visible()
    assert not browser_server.client.requests


def test_fragment_replacement_reconciles_unchanged_messages(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """A full update keeps the transcript container and unchanged message nodes intact."""

    edited = browser_server.state.chats[0].add_message("user", "Original message")
    stable = browser_server.state.chats[0].add_message("assistant", "Stable reply")
    _open_chat_page(browser_page)
    retained = browser_page.evaluate(
        """(stableId) => {
            const messageList = document.getElementById('message-list');
            const stableMessage = Array.from(
                document.querySelectorAll('article[data-message-id]'),
            ).find((message) => message.dataset.messageId === stableId);
            window.__browserTestMessageList = messageList;
            window.__browserTestStableMessage = stableMessage;
            return Boolean(messageList && stableMessage);
        }""",
        stable.id,
    )
    assert retained is True

    _edit_message_in_background(browser_page, edited.id, "Edited through a fragment")
    _wait_for_message_text(browser_page, edited.id, "Edited through a fragment")

    identities_retained: object = browser_page.evaluate(
        """(stableId) => {
            const stableMessage = Array.from(
                document.querySelectorAll('article[data-message-id]'),
            ).find((message) => message.dataset.messageId === stableId);
            return {
                messageList: document.getElementById('message-list') === window.__browserTestMessageList,
                stableMessage: stableMessage === window.__browserTestStableMessage,
            };
        }""",
        stable.id,
    )
    assert identities_retained == {"messageList": True, "stableMessage": True}


def _open_chat_page(page: Page) -> None:
    """Open the browser workspace after its live refresh connection is established."""

    with page.expect_response(
        lambda response: response.url.split("?", maxsplit=1)[0].endswith(
            "/fragments/chat"
        )
    ) as initial_fragment:
        page.goto("/chats")
    assert initial_fragment.value.ok
    _ = initial_fragment.value.body()
    expect(page.get_by_test_id("message-list")).to_be_visible()
    _wait_for_animation_frames(page)


def _send_only(page: Page) -> None:
    page.get_by_test_id("composer-menu-toggle").click()
    page.get_by_test_id("send-only-button").click()


@pytest.mark.parametrize("viewport_width", [960, 390])
def test_colourway_editor_updates_owned_ui_and_light_mode(
    browser_page: Page,
    browser_server: BrowserServer,
    tmp_path: Path,
    viewport_width: int,
) -> None:
    """Saving four main colours updates real CSS without coupling role accents."""

    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    _ = browser_server.state.chats[0].add_message("user", "My message")
    _ = browser_server.state.chats[0].add_message("assistant", "Assistant reply")
    _open_chat_page(browser_page)
    browser_page.locator('[data-panel-open="navigation"]').click()
    browser_page.get_by_test_id("global-settings-button").click()
    browser_page.get_by_test_id("global-settings-dialog-appearance-tab").click()

    for owner, value in (
        ("app", "#897aab"),
        ("user", "#ff6600"),
        ("assistant", "#00aa88"),
        ("system", "#4488ee"),
    ):
        browser_page.get_by_test_id(f"theme-{owner}-accent-color").fill(value)

    def save() -> None:
        with browser_page.expect_navigation(wait_until="load"):
            browser_page.get_by_test_id("save-global-appearance-button").click()
        browser_page.mouse.move(0, 0)
        expect(browser_page.get_by_test_id("theme-app-colourway")).to_be_visible()

    save()
    expected = {
        "app": "rgb(137, 122, 171)",
        "user": "rgb(255, 102, 0)",
        "assistant": "rgb(0, 170, 136)",
        "system": "rgb(68, 136, 238)",
    }
    assert _rendered_colourways(browser_page) == expected
    expect(browser_page.locator(".jouzetsu-composer-primary")).to_have_css(
        "background-color", expected["user"]
    )

    browser_page.get_by_test_id("theme-user-accent-color").fill("#22bbdd")
    save()
    expected["user"] = "rgb(34, 187, 221)"
    assert _rendered_colourways(browser_page) == expected
    expect(browser_page.locator(".jouzetsu-composer-primary")).to_have_css(
        "background-color", expected["user"]
    )

    browser_page.get_by_test_id("appearance-dark-mode").uncheck()
    save()
    assert _rendered_colourways(browser_page) == expected
    expect(browser_page.locator("body")).to_have_css(
        "background-color", "rgb(255, 255, 255)"
    )
    expect(browser_page.locator("html")).to_have_css("color-scheme", "light")
    _ = browser_page.screenshot(path=str(tmp_path / "appearance-light.png"))


def test_reset_keeps_its_action_after_a_draft_flush(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """A reset submitter remains the target after preserving a pending composer draft."""

    expected_prompt = browser_server.state.config.generation.system_prompt
    browser_server.state.config.generation.system_prompt = "Custom global prompt"

    def block_draft_save(route: Route) -> None:
        route.abort()

    browser_page.route("**/chat/draft", block_draft_save)
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("message-input").fill("Draft that cannot save")
    browser_page.locator('[data-panel-open="navigation"]').click()
    browser_page.get_by_test_id("global-settings-button").click()
    browser_page.once("dialog", lambda dialog: dialog.accept())

    with browser_page.expect_navigation(wait_until="load"):
        browser_page.get_by_test_id("reset-global-prompt-button").click()

    assert browser_server.state.config.generation.system_prompt == expected_prompt


def _rendered_colourways(page: Page) -> dict[str, str]:
    """Sample actual component colours, including the shared-control scope."""

    targets = (
        ("app", '[data-testid="save-global-appearance-button"]', "background-color"),
        (
            "user",
            ".jouzetsu-message.is-user .jouzetsu-message-surface",
            "border-right-color",
        ),
        (
            "assistant",
            ".jouzetsu-message.is-assistant .jouzetsu-message-surface",
            "border-left-color",
        ),
        ("system", ".jouzetsu-dialog-feedback", "border-left-color"),
    )
    # Sample connected elements together; an SSE refresh can replace the dialog between calls.
    return cast(
        dict[str, str],
        page.evaluate(
            """targets => Object.fromEntries(targets.map(([owner, selector, property]) => {
            const element = document.querySelector(selector);
            if (!(element instanceof HTMLElement)) throw new Error(`Missing colour target: ${selector}`);
            return [owner, getComputedStyle(element).getPropertyValue(property)];
        }))""",
            targets,
        ),
    )


def _seed_scrollable_transcript(server: BrowserServer) -> str:
    """Create a static transcript large enough to exercise the list scroll state."""

    return _seed_transcript(server, 12, _LONG_MESSAGE)


def _seed_transcript(server: BrowserServer, turn_count: int, content: str) -> str:
    """Add a predictable sequence of user and assistant messages to the active chat."""

    chat = server.state.chats[0]
    last_message_id = ""
    for turn in range(turn_count):
        _ = chat.add_message("user", f"User turn {turn}: {content}")
        response = chat.add_message("assistant", f"Assistant turn {turn}: {content}")
        last_message_id = response.id
    return last_message_id


def _configure_paused_streaming_response(server: BrowserServer) -> Event:
    """Configure the browser runtime to pause after rendering a long response."""

    completion_gate = Event()
    server.client.response_content = _STREAMING_RESPONSE
    server.client.first_token_delay_seconds = 0.25
    server.client.completion_gate = completion_gate
    return completion_gate


def _edit_message_in_background(page: Page, message_id: str, content: str) -> None:
    """Cause an SSE-driven full refresh without using the foreground chat controller."""

    status: object = page.evaluate(
        """async ({ messageId, content }) => {
            const token = document.querySelector('[data-csrf-token]')?.getAttribute('data-csrf-token') || '';
            const body = new URLSearchParams({
                chat_id: document.getElementById("chat-fragment").dataset.chatId,
                revision: document.getElementById("chat-fragment").dataset.revision,
                message_id: messageId,
                content,
                _jouzetsu_csrf: token,
            });
            const response = await fetch('/messages/edit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body,
            });
            return response.status;
        }""",
        {"messageId": message_id, "content": content},
    )
    assert status == 200


def _wait_for_message_text(page: Page, message_id: str, marker: str) -> None:
    """Wait for the live browser to apply the full fragment containing one marker."""

    page.wait_for_function(
        """({ messageId, marker }) => Array.from(
            document.querySelectorAll('article[data-message-id]'),
        ).some((message) => message.dataset.messageId === messageId && message.innerText.includes(marker))""",
        arg={"messageId": message_id, "marker": marker},
    )


def _wait_for_animation_frames(page: Page) -> None:
    """Let DOM layout and resize observers settle before taking a measurement."""

    page.evaluate(
        """() => new Promise((resolve) => requestAnimationFrame(
            () => requestAnimationFrame(resolve),
        ))"""
    )


def _scroll_metrics(page: Page) -> ScrollMetrics:
    """Read the current and maximum message-list offsets with explicit validation."""

    result: object = page.locator(_MESSAGE_LIST_SELECTOR).evaluate(
        """(element) => ({
            top: element.scrollTop,
            maximum: Math.max(0, element.scrollHeight - element.clientHeight),
        })"""
    )
    if not isinstance(result, dict):
        raise TypeError("message list did not return scroll metrics")
    values: dict[object, object] = cast(dict[object, object], result)
    top = values.get("top")
    maximum = values.get("maximum")
    if (
        isinstance(top, bool)
        or not isinstance(top, int | float)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int | float)
    ):
        raise TypeError("message list returned invalid scroll metrics")
    return {"top": float(top), "maximum": float(maximum)}


def _message_position_metrics(page: Page, selector: str) -> MessagePositionMetrics:
    """Read one message's location relative to the transcript viewport."""

    result: object = page.locator(selector).evaluate(
        """(message) => {
            const messages = document.getElementById('message-list');
            if (!(messages instanceof HTMLElement)) return null;
            return {
                gap: messages.scrollHeight - messages.clientHeight - messages.scrollTop,
                message_top: message.getBoundingClientRect().top - messages.getBoundingClientRect().top,
                top: messages.scrollTop,
            };
        }"""
    )
    if not isinstance(result, dict):
        raise TypeError("message position metrics were unavailable")
    values: dict[object, object] = cast(dict[object, object], result)
    parsed: dict[str, float] = {}
    for key in ("gap", "message_top", "top"):
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"message position metric {key} was invalid")
        parsed[key] = float(value)
    return {
        "gap": parsed["gap"],
        "message_top": parsed["message_top"],
        "top": parsed["top"],
    }


def _message_action_layout_metrics(
    page: Page, selector: str
) -> MessageActionLayoutMetrics:
    """Read the available width and wrapped rows of a message action group."""

    result: object = page.locator(selector).evaluate(
        """(message) => {
            const footer = message.querySelector('.jouzetsu-message-footer');
            const actions = message.querySelector('.jouzetsu-message-actions');
            if (!(footer instanceof HTMLElement) || !(actions instanceof HTMLElement)) return null;
            const rows = new Set(Array.from(
                actions.querySelectorAll('.jouzetsu-message-action'),
                (action) => Math.round(action.getBoundingClientRect().top),
            ));
            return {
                actions_width: actions.getBoundingClientRect().width,
                footer_width: footer.getBoundingClientRect().width,
                row_count: rows.size,
            };
        }"""
    )
    if not isinstance(result, dict):
        raise TypeError("message action layout metrics were unavailable")
    values: dict[object, object] = cast(dict[object, object], result)
    actions_width = values.get("actions_width")
    footer_width = values.get("footer_width")
    row_count = values.get("row_count")
    if (
        isinstance(actions_width, bool)
        or not isinstance(actions_width, int | float)
        or isinstance(footer_width, bool)
        or not isinstance(footer_width, int | float)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
    ):
        raise TypeError("message action layout metrics were invalid")
    return {
        "actions_width": float(actions_width),
        "footer_width": float(footer_width),
        "row_count": row_count,
    }


def _insert_test_style_rules(page: Page, rules: tuple[str, ...]) -> None:
    """Append temporary test styles through a same-origin sheet permitted by CSP."""

    page.evaluate(
        """(rules) => {
            for (const sheet of document.styleSheets) {
                try {
                    rules.forEach((rule) => sheet.insertRule(rule));
                    return;
                } catch {
                    // Try the next same-origin stylesheet.
                }
            }
            throw new Error('no writable stylesheet for browser test');
        }""",
        list(rules),
    )


def _constrain_message_list_width(page: Page, width: int) -> None:
    """Give a browser test a narrow transcript without depending on window sizing."""

    _insert_test_style_rules(
        page, (f".jouzetsu-messages {{ width: {width}px !important; }}",)
    )
    _wait_for_animation_frames(page)


def _set_message_scroll_top(page: Page, scroll_top: float) -> None:
    """Set and publish a message-list scroll position as a user scroll would."""

    page.locator(_MESSAGE_LIST_SELECTOR).evaluate(
        """(element, nextScrollTop) => {
            element.scrollTop = nextScrollTop;
            element.dispatchEvent(new Event('scroll'));
        }""",
        scroll_top,
    )


def _message_selector(message_id: str) -> str:
    """Return the transcript-only selector for one server-generated message id."""

    return f'{_MESSAGE_SELECTOR}[data-message-id="{message_id}"]'


def test_two_browsers_keep_independent_urls_and_history(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    try:
        _open_chat_page(browser_page)
        first_url = browser_page.url
        other.goto(first_url)
        browser_page.get_by_test_id("message-input").fill("Draft in first chat")
        browser_page.get_by_test_id("open-nav-button").click()
        browser_page.get_by_test_id("new-chat-button").click()
        expect(browser_page).not_to_have_url(first_url)
        second_url = browser_page.url.split("?", 1)[0]
        expect(other).to_have_url(first_url)
        expect(other.get_by_test_id("message-input")).to_have_value(
            "Draft in first chat"
        )
        browser_page.go_back()
        expect(browser_page).to_have_url(first_url)
        expect(browser_page.get_by_test_id("message-input")).to_have_value(
            "Draft in first chat"
        )
        browser_page.go_forward()
        expect(browser_page).to_have_url(second_url)
        browser_page.reload()
        expect(browser_page).to_have_url(second_url)
        expect(other).to_have_url(first_url)
    finally:
        context.close()


def test_shared_draft_updates_idle_browser_and_preserves_rejected_local_text(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    try:
        _open_chat_page(browser_page)
        other.goto(browser_page.url)
        browser_page.route(
            "**/chat/draft",
            lambda route: route.fulfill(status=503, body="Controlled draft failure"),
        )
        browser_page.get_by_test_id("message-input").fill("Keep my unsaved text")
        expect(browser_page.locator("#notice-area")).to_contain_text("Could not save")
        other.get_by_test_id("message-input").fill("Saved by other browser")
        other.wait_for_function(
            "() => Number(document.querySelector('.jouzetsu-composer-form input[name=draft_revision]').value) > 0"
        )
        browser_page.wait_for_function(
            "() => Number(document.getElementById('chat-fragment').dataset.draftRevision) > 0"
        )
        expect(browser_page.get_by_test_id("message-input")).to_have_value(
            "Keep my unsaved text"
        )
        browser_page.unroute("**/chat/draft")
        browser_page.get_by_test_id("message-input").fill("Keep my local correction")
        expect(browser_page.locator("#notice-area")).to_contain_text(
            "changed elsewhere"
        )
        expect(browser_page.get_by_test_id("message-input")).to_have_value(
            "Keep my local correction"
        )
        browser_page.reload()
        expect(browser_page.get_by_test_id("message-input")).to_have_value(
            "Saved by other browser"
        )
    finally:
        context.close()


def test_incoming_edit_preserves_focused_editor_and_stale_save_is_visible(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    chat = browser_server.state.chats[0]
    message = chat.add_message("user", "Original")
    _open_chat_page(browser_page)
    browser_page.get_by_test_id(f"edit-message-{message.id}").click()
    expect(browser_page.get_by_test_id("save-edit-button")).to_be_visible()
    browser_page.get_by_test_id("message-input").fill("My unsaved edit")
    _edit_message_in_background(browser_page, message.id, "Other committed edit")
    _wait_for_message_text(browser_page, message.id, "Other committed edit")
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "My unsaved edit"
    )
    browser_page.get_by_test_id("save-edit-button").click()
    expect(browser_page.locator("#notice-area")).to_contain_text("revision changed")
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "My unsaved edit"
    )
    assert message.content == "Other committed edit"


def test_incoming_deletion_preserves_the_local_message_editor(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    chat = browser_server.state.chats[0]
    message = chat.add_message("user", "Original to be deleted")
    _open_chat_page(browser_page)
    browser_page.get_by_test_id(f"edit-message-{message.id}").click()
    expect(browser_page.get_by_test_id("save-edit-button")).to_be_visible()
    browser_page.get_by_test_id("message-input").fill("Keep my local edit")
    assert (
        browser_page.evaluate(
            """async (messageId) => {
            const token = document.querySelector('[data-csrf-token]').dataset.csrfToken;
            const shell = document.getElementById('chat-fragment');
            const response = await fetch(`/messages/${messageId}/delete`, {method: 'POST',
                headers: {'X-Jouzetsu-CSRF': token, 'HX-Request': 'true'},
                body: new URLSearchParams({chat_id: shell.dataset.chatId, revision: shell.dataset.revision})});
            return response.status;
        }""",
            message.id,
        )
        == 200
    )
    expect(browser_page.locator("#notice-area")).to_contain_text("message was deleted")
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "Keep my local edit"
    )
    browser_page.get_by_test_id("save-edit-button").click()
    expect(browser_page.locator("#notice-area")).to_contain_text("revision changed")
    assert not chat.messages
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "Keep my local edit"
    )
    browser_page.get_by_test_id("cancel-edit-button").click()
    expect(browser_page.get_by_test_id("message-input")).to_have_value("")


@pytest.mark.parametrize("viewport_width", [960, 390])
def test_composer_menu_supports_keyboard_images_and_send_only(
    browser_page: Page,
    browser_server: BrowserServer,
    tmp_path: Path,
    viewport_width: int,
) -> None:
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    _open_chat_page(browser_page)
    toggle = browser_page.get_by_test_id("composer-menu-toggle")
    menu = browser_page.locator("[data-composer-menu]")
    send_only = browser_page.get_by_test_id("send-only-button")
    expect(send_only).not_to_be_visible()
    toggle.press("Enter")
    expect(send_only).to_be_visible()
    expect(send_only).to_be_disabled()
    browser_page.keyboard.press("Tab")
    expect(browser_page.get_by_test_id("add-images-button")).to_be_focused()
    browser_page.keyboard.press("Escape")
    expect(send_only).not_to_be_visible()
    expect(toggle).to_be_focused()
    toggle.click()
    browser_page.get_by_test_id("message-input").click()
    expect(send_only).not_to_be_visible()
    toggle.click()
    browser_page.evaluate(
        "window.openComposerMenu = document.querySelector('[data-composer-menu]')"
    )
    browser_page.get_by_test_id("message-input").fill("Save this with an image")
    browser_page.wait_for_function(
        "() => document.getElementById('chat-fragment').dataset.draftRevision === '1'"
    )
    expect(menu).to_have_js_property("open", True)
    assert browser_page.evaluate(
        "openComposerMenu === document.querySelector('[data-composer-menu]')"
    )
    expect(send_only).to_be_enabled()
    assert browser_page.locator(".jouzetsu-composer-menu-options").evaluate("""menu => {
        const bounds = menu.getBoundingClientRect();
        return bounds.left >= 0 && bounds.right <= innerWidth && bounds.top >= 0 && bounds.bottom <= innerHeight;
    }""")
    _ = browser_page.screenshot(path=str(tmp_path / "composer-menu.png"))
    with browser_page.expect_file_chooser() as picker:
        browser_page.get_by_test_id("add-images-button").click()
    picker.value.set_files(
        {"name": "menu.png", "mimeType": "image/png", "buffer": image_bytes()}
    )
    expect(menu).to_have_js_property("open", False)
    expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "Save this with an image"
    )
    _send_only(browser_page)
    expect(browser_page.get_by_test_id("message-input")).to_have_value("")
    expect(menu).to_have_js_property("open", False)
    expect(browser_page.locator(_MESSAGE_SELECTOR)).to_have_count(1)
    message = browser_server.state.chats[0].messages[0]
    assert message.content == "Save this with an image"
    assert len(message.images) == 1
    assert not browser_server.client.requests


@pytest.mark.parametrize("viewport_width", [390, 960])
@pytest.mark.parametrize("existing", ["", "Existing draft", "Existing draft\n"])
def test_quick_templates_insert_from_the_message_actions_menu(
    browser_page: Page,
    browser_server: BrowserServer,
    viewport_width: int,
    existing: str,
) -> None:
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    _open_chat_page(browser_page)
    prompt = browser_server.state.config.ui.starter_prompts[0]
    input_field = browser_page.get_by_test_id("message-input")
    input_field.fill(existing)
    browser_page.get_by_test_id("composer-menu-toggle").press("Enter")
    templates = browser_page.get_by_role("group", name="Quick templates", exact=True)
    expect(templates).to_be_visible()
    expect(
        browser_page.locator(".jouzetsu-composer-footer [data-suggestion]")
    ).to_have_count(0)
    templates.get_by_role("button", name=prompt.label, exact=True).press("Enter")
    separator = "\n" if existing and not existing.endswith("\n") else ""
    expect(input_field).to_have_value(existing + separator + prompt.content)
    expect(input_field).to_be_focused()
    expect(browser_page.locator("[data-composer-menu]")).to_have_js_property(
        "open", False
    )
    assert not browser_server.state.chats[0].messages
    assert not browser_server.client.requests


def test_quick_template_menu_scrolls_within_a_short_mobile_viewport(
    touch_browser_page: Page, browser_server: BrowserServer
) -> None:
    browser_server.state.config.ui.starter_prompts = [
        StarterPrompt(f"Template {index}", f"Content {index}") for index in range(20)
    ]
    page = touch_browser_page
    page.set_viewport_size({"width": 390, "height": 360})
    _open_chat_page(page)
    page.get_by_test_id("composer-menu-toggle").tap()
    _wait_for_animation_frames(page)
    options = page.locator(".jouzetsu-composer-menu-options")
    bounds = options.bounding_box()
    assert bounds
    assert bounds["y"] >= 0
    assert bounds["y"] + bounds["height"] <= 360
    assert options.evaluate("element => element.scrollHeight > element.clientHeight")
    last = page.get_by_test_id("composer-suggestion-19")
    last.scroll_into_view_if_needed()
    last.tap()
    expect(page.get_by_test_id("message-input")).to_have_value("Content 19")


@pytest.mark.parametrize("viewport_width", [960, 390])
def test_saved_approval_phrase_can_be_revealed_edited_and_cleared(
    browser_page: Page,
    browser_server: BrowserServer,
    tmp_path: Path,
    viewport_width: int,
) -> None:
    original = 'Test phrase with & "quotes"'
    browser_server.state.config.access.approval_phrase = original
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("open-nav-button").click()
    browser_page.get_by_test_id("access-settings-button").click()
    phrase = browser_page.get_by_label("Approval phrase", exact=True)
    expect(phrase).to_have_value(original)
    expect(phrase).to_have_attribute("type", "password")
    show = browser_page.get_by_role("button", name="Show approval phrase", exact=True)
    assert show.evaluate("""button => {
        const input = document.getElementById(button.getAttribute('aria-controls'));
        const field = input.getBoundingClientRect();
        const toggle = button.getBoundingClientRect();
        return toggle.left >= field.left && toggle.right <= field.right
            && toggle.top >= field.top && toggle.bottom <= field.bottom;
    }""")
    _ = browser_page.screenshot(path=str(tmp_path / "approval-phrase-control.png"))
    show.click()
    expect(phrase).to_have_attribute("type", "text")
    expect(phrase).to_have_value(original)
    replacement = "Changed approval phrase"
    phrase.fill(replacement)
    browser_page.keyboard.press("Tab")
    hide = browser_page.get_by_role("button", name="Hide approval phrase", exact=True)
    expect(hide).to_be_focused()
    hide.press("Enter")
    expect(phrase).to_have_attribute("type", "password")
    expect(phrase).to_have_value(replacement)
    assert browser_server.state.config.access.approval_phrase == original
    with browser_page.expect_navigation(wait_until="load"):
        browser_page.get_by_role("button", name="Save phrase", exact=True).click()
    expect(phrase).to_have_value(replacement)
    expect(phrase).to_have_attribute("type", "password")
    store = ConfigStore.for_paths(browser_server.state.config.paths)
    assert store.load().access.approval_phrase == replacement
    with browser_page.expect_navigation(wait_until="load"):
        browser_page.get_by_role("button", name="Save phrase", exact=True).click()
    expect(phrase).to_have_value(replacement)
    assert browser_server.state.config.access.approval_phrase == replacement
    phrase.fill("")
    with browser_page.expect_navigation(wait_until="load"):
        browser_page.get_by_role("button", name="Save phrase", exact=True).click()
    expect(phrase).to_have_value("")
    assert store.load().access.approval_phrase == ""


def test_composer_menu_keeps_image_upload_available_during_generation(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    gate = _configure_paused_streaming_response(browser_server)
    try:
        _open_chat_page(browser_page)
        browser_page.get_by_test_id("message-input").fill("Generate a reply")
        browser_page.get_by_test_id("send-message-button").click()
        expect(browser_page.locator("article.is-streaming")).to_be_visible()
        browser_page.get_by_test_id("composer-menu-toggle").click()
        expect(browser_page.get_by_test_id("send-only-button")).not_to_be_visible()
        with browser_page.expect_file_chooser() as picker:
            browser_page.get_by_test_id("add-images-button").click()
        picker.value.set_files(
            {"name": "next.png", "mimeType": "image/png", "buffer": image_bytes()}
        )
        expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
        expect(browser_page.get_by_test_id("stop-generation-button")).to_be_enabled()
        assert len(browser_server.client.requests) == 1
        assert not browser_server.client.requests[0].messages[-1].images
    finally:
        gate.set()


def test_image_picker_survives_generation_finishing(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    gate = _configure_paused_streaming_response(browser_server)
    try:
        _open_chat_page(browser_page)
        browser_page.get_by_test_id("message-input").fill("Generate a reply")
        browser_page.get_by_test_id("send-message-button").click()
        expect(browser_page.locator("article.is-streaming")).to_be_visible()
        browser_page.get_by_test_id("composer-menu-toggle").click()
        with browser_page.expect_file_chooser() as picker:
            browser_page.get_by_test_id("add-images-button").click()
        gate.set()
        expect(browser_page.get_by_test_id("send-message-button")).to_be_visible()
        picker.value.set_files(
            {"name": "next.png", "mimeType": "image/png", "buffer": image_bytes()}
        )
        expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
        assert len(browser_server.client.requests) == 1
        assert not browser_server.client.requests[0].messages[-1].images
    finally:
        gate.set()


def _expect_message_action_colour(
    page: Page, marker: str, action: MessageAction, style: MessageActionStyle
) -> None:
    field = (
        action.color_field.replace("_", "-")
        if style is MessageActionStyle.SEMANTIC
        else "message-actions"
    )
    expected = cast(
        str,
        page.evaluate(
            """variable => {
            const reference = document.createElement('span');
            reference.style.color = `var(${variable})`;
            document.body.append(reference);
            const colour = getComputedStyle(reference).color;
            reference.remove();
            return colour;
        }""",
            f"--jouzetsu-app-{field}",
        ),
    )
    button = page.get_by_test_id(marker)
    expect(button).to_have_css("color", expected)
    icons = button.locator("svg").all()
    assert icons
    for icon in icons:
        expect(icon).to_have_css("fill", expected)
    button.hover()
    for icon in icons:
        expect(icon).to_have_css("fill", expected)
    button.focus()
    for icon in icons:
        expect(icon).to_have_css("fill", expected)


def _expect_icons_painted(page: Page) -> None:
    """Wait for actual external SVG artwork, rather than just its empty container."""

    _ = page.wait_for_function("""() => {
        const icons = [...document.querySelectorAll('.jouzetsu-svg-icon')]
            .filter(icon => icon.getClientRects().length);
        return icons.length > 0 && icons.every(icon => {
            const bounds = icon.getBBox();
            return bounds.width > 0 && bounds.height > 0;
        });
    }""")


def _set_browser_models(
    server: BrowserServer, models: tuple[RegisteredModel, ...]
) -> None:
    configuration = server.state.config.backends
    backend = configuration.backends[0]
    server.state.config.backends = replace(
        configuration, backends=(replace(backend, models=models),)
    )
    server.client.descriptors = tuple(model.descriptor for model in models)


def test_reasoning_control_updates_chat_profile(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    from jouzetsu.backend_models import BackendKind, ModelManagementSettings

    state = browser_server.state
    configuration = state.config.backends
    backend = configuration.backends[0]
    model = backend.models[0]
    model = replace(
        model,
        descriptor=replace(
            model.descriptor,
            capabilities=ModelCapabilities(
                reasoning=True, reasoning_options=("off", "on")
            ),
        ),
    )
    configured = replace(
        backend,
        kind=BackendKind.LM_STUDIO,
        model_management=ModelManagementSettings(),
        models=(model, *backend.models[1:]),
    )
    state.config.backends = replace(configuration, backends=(configured,))
    _open_chat_page(browser_page)
    toggle = browser_page.get_by_test_id("reasoning-level-toggle")
    expect(toggle).to_have_attribute("data-reasoning-state", "automatic")
    expect(toggle).to_have_attribute(
        "aria-label", "Reasoning level: Model default (Automatic)"
    )
    reasoning_menu = browser_page.locator("[data-reasoning-menu]")
    actions_menu = browser_page.locator("[data-composer-menu]")
    toggle.click()
    expect(reasoning_menu).to_have_js_property("open", True)
    browser_page.evaluate(
        "window.reasoningMenu = document.querySelector('[data-reasoning-menu]')"
    )
    browser_page.get_by_test_id("message-input").fill(
        "Draft updated while the menu is open"
    )
    browser_page.wait_for_function(
        "() => document.getElementById('chat-fragment').dataset.draftRevision === '1'"
    )
    expect(reasoning_menu).to_have_js_property("open", True)
    assert browser_page.evaluate(
        "window.reasoningMenu === document.querySelector('[data-reasoning-menu]')"
    )
    browser_page.get_by_test_id("message-input").click()
    expect(reasoning_menu).to_have_js_property("open", False)
    toggle.click()
    browser_page.get_by_test_id("composer-menu-toggle").click()
    expect(reasoning_menu).to_have_js_property("open", False)
    expect(actions_menu).to_have_js_property("open", True)
    toggle.click()
    expect(actions_menu).to_have_js_property("open", False)
    expect(reasoning_menu).to_have_js_property("open", True)
    browser_page.keyboard.press("Escape")
    expect(reasoning_menu).to_have_js_property("open", False)
    expect(toggle).to_be_focused()
    toggle.click()
    model_default = browser_page.get_by_role("button", name="Model default", exact=True)
    expect(model_default).to_have_attribute("aria-pressed", "true")
    with browser_page.expect_response("**/chat/reasoning-level") as response:
        browser_page.get_by_role("button", name="Off", exact=True).click()
    assert response.value.ok
    expect(toggle).to_have_attribute("data-reasoning-state", "off")
    expect(toggle).to_have_attribute("aria-label", "Reasoning level: Off")
    assert (
        toggle.evaluate("element => getComputedStyle(element, '::after').width")
        == "22px"
    )
    expect(browser_page.locator("#chat-fragment")).to_have_attribute(
        "data-revision", str(state.chats[0].revision)
    )
    profile = state.chats[0].inference.selected_profile
    assert profile is not None and profile.reasoning == "off"
    browser_page.reload()
    browser_page.get_by_test_id("reasoning-level-toggle").click()
    off = browser_page.get_by_role("button", name="Off", exact=True)
    expect(off).to_have_attribute("aria-pressed", "true")
    expect(
        browser_page.get_by_role("button", name="Model default", exact=True)
    ).to_have_attribute("aria-pressed", "false")
    with browser_page.expect_response("**/chat/reasoning-level") as on_response:
        browser_page.get_by_role("button", name="On", exact=True).click()
    assert on_response.value.ok
    expect(toggle).to_have_attribute("data-reasoning-state", "on")
    expect(toggle).to_have_attribute("aria-label", "Reasoning level: On")
    browser_page.get_by_test_id("open-settings-button").click()
    browser_page.get_by_test_id("active-chat-model").click()
    with browser_page.expect_response("**/chat/model") as changed:
        browser_page.locator(
            '#active-chat-model-options [data-model-value="text-model"]'
        ).click()
    assert changed.value.ok
    profile = state.chats[0].inference.selected_profile
    assert profile is not None and profile.reasoning == "model_default"
    expect(browser_page.get_by_test_id("reasoning-level-toggle")).to_have_count(0)
    browser_page.goto("/chats?dialog=global&tab=backends")
    browser_page.locator(f"#backend-{configured.id} > summary").click()
    model_form = browser_page.locator(
        f'#backend-{configured.id} form[action$="/alias"]'
    ).filter(
        has=browser_page.locator('input[name="model_id"][value="browser-test-model"]')
    )
    model_form.locator('select[name="reasoning_default"]').select_option("on")
    with browser_page.expect_response(
        f"**/settings/backends/{configured.id}/alias"
    ) as saved:
        model_form.get_by_role("button", name="Save model").click()
    assert saved.value.ok
    assert (
        state.config.backends.backend(configured.id)
        .model("browser-test-model")
        .reasoning_default
        == "on"
    )


def test_unknown_reasoning_capabilities_do_not_erase_model_default(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    from jouzetsu.backend_models import BackendKind, ModelManagementSettings

    state = browser_server.state
    configuration = state.config.backends
    backend = configuration.backends[0]
    model = replace(backend.models[0], reasoning_default="on")
    configured = replace(
        backend,
        kind=BackendKind.LM_STUDIO,
        model_management=ModelManagementSettings(),
        models=(model, *backend.models[1:]),
    )
    state.config.backends = replace(configuration, backends=(configured,))
    browser_page.goto("/chats?dialog=global&tab=backends")
    browser_page.locator(f"#backend-{backend.id} > summary").click()
    form = browser_page.locator(f'#backend-{backend.id} form[action$="/alias"]').filter(
        has=browser_page.locator('input[name="model_id"][value="browser-test-model"]')
    )
    expect(form.locator('select[name="reasoning_default"]')).to_have_value("on")
    expect(form).to_contain_text("previous reasoning default is unavailable")
    form.locator('input[name="alias"]').fill("New alias")
    with browser_page.expect_response(
        f"**/settings/backends/{backend.id}/alias"
    ) as response:
        form.get_by_role("button", name="Save model").click()
    assert response.value.ok
    assert (
        state.config.backends.backend(backend.id)
        .model("browser-test-model")
        .reasoning_default
        == "on"
    )


@pytest.mark.parametrize("viewport_width", [390, 1280])
@pytest.mark.parametrize("dark_mode", [False, True])
def test_capability_badges_render_and_follow_the_selected_model(
    browser_page: Page,
    browser_server: BrowserServer,
    tmp_path: Path,
    viewport_width: int,
    dark_mode: bool,
) -> None:
    state = browser_server.state
    state.config.ui.dark_mode = dark_mode
    backend = state.config.backends.backends[0]
    supported = ModelCapabilities(True, True, True)
    models = tuple(
        replace(
            model,
            descriptor=replace(
                model.descriptor,
                capabilities=supported
                if index == 0
                else ModelCapabilities(False, False, None),
            ),
        )
        for index, model in enumerate(backend.models)
    )
    _set_browser_models(browser_server, models)
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("open-settings-button").click()
    picker = browser_page.get_by_test_id("active-chat-model")
    capabilities = picker.locator(".jouzetsu-model-capabilities")
    for label in ("Vision supported", "Tool use supported", "Reasoning supported"):
        expect(capabilities.get_by_role("img", name=label, exact=True)).to_be_visible()
    _expect_icons_painted(browser_page)
    browser_page.wait_for_function("""() => {
        const panel = document.getElementById('settings-panel');
        return Math.abs(panel.getBoundingClientRect().right - innerWidth) < 0.1;
    }""")
    panel_content = browser_page.locator("#settings-panel .jouzetsu-panel-content")
    scroll_top: object = panel_content.evaluate("element => element.scrollTop")
    picker.click()
    options = browser_page.locator("#active-chat-model-options")
    supported_option = options.locator('[data-model-value="browser-test-model"]')
    expect(supported_option.locator(".jouzetsu-capability-badge")).to_have_count(3)
    text_option = options.locator('[data-model-value="text-model"]')
    expect(text_option.locator(".jouzetsu-capability-badge")).to_have_count(0)
    assert "Unknown" not in options.inner_text()
    assert "Vision" not in options.inner_text()
    assert panel_content.evaluate("element => element.scrollTop") == scroll_top
    _ = browser_page.screenshot(
        path=str(tmp_path / "capability-badges.png"), animations="disabled"
    )
    with browser_page.expect_response("**/chat/model") as saved:
        text_option.click()
    assert saved.value.ok
    expect(capabilities.locator(".jouzetsu-capability-badge")).to_have_count(0)
    expect(picker).to_have_text("Text model")
    profile = browser_server.state.chats[0].inference.selected_profile
    assert profile is not None and profile.model_id == "text-model"
    browser_page.evaluate("""async () => {
        const { createIcon } = await import('/static/app/icons.js');
        document.querySelector('#settings-panel [data-model-choice-content]')
            .append(createIcon('fim'));
    }""")
    _expect_icons_painted(browser_page)
    browser_page.goto("/chats?dialog=global")
    browser_page.get_by_test_id("global-settings-dialog-backends-tab").click()
    backend_editor = browser_page.locator(f"#backend-{backend.id}")
    backend_editor.locator(":scope > summary").click()
    default_picker = browser_page.get_by_test_id(f"backend-model-{backend.id}")
    default_picker.click()
    default_options = browser_page.locator(f"#backend-model-{backend.id}-options")
    expect(
        default_options.locator(
            '[data-model-value="browser-test-model"] .jouzetsu-capability-badge'
        )
    ).to_have_count(3)
    default_picker.press("Escape")
    for label in ("Vision supported", "Tool use supported", "Reasoning supported"):
        expect(
            backend_editor.get_by_role("img", name=label, exact=True).first
        ).to_be_visible()
    _expect_icons_painted(browser_page)
    default_picker.click()
    default_options.locator('[data-model-value="text-model"]').click()
    expect(default_picker).to_have_text("Text model")
    expect(backend_editor.locator("[data-model-picker-value]")).to_have_value(
        "text-model"
    )
    assert state.config.backends.backends[0].defaults.model_id == "browser-test-model"
    with browser_page.expect_response(
        f"**/settings/backends/{backend.id}/save"
    ) as saved_default:
        backend_editor.get_by_role("button", name="Save backend", exact=True).click()
    assert saved_default.value.ok
    assert state.config.backends.backends[0].defaults.model_id == "text-model"


def test_model_picker_keyboard_selection_and_live_refresh(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    chat = browser_server.state.chats[0]
    message = chat.add_message("user", "A message to edit in the background")
    backend = browser_server.state.config.backends.backends[0]
    _set_browser_models(
        browser_server,
        (
            *(
                RegisteredModel(ModelDescriptor(f"extra-{index}", f"Extra {index:02}"))
                for index in range(30)
            ),
            *backend.models,
        ),
    )
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("open-settings-button").click()
    picker = browser_page.get_by_test_id("active-chat-model")
    options = browser_page.locator("#active-chat-model-options")
    text_option = options.locator('[data-model-value="text-model"]')
    picker.focus()
    picker.press("ArrowDown")
    expect(options).to_be_visible()
    picker.press("End")
    expect(picker).to_have_attribute(
        "aria-activedescendant", text_option.get_attribute("id") or ""
    )
    picker.press("Escape")
    expect(options).to_be_hidden()
    expect(browser_page.locator("#settings-panel")).to_have_class(
        re.compile(r"is-open")
    )
    profile = chat.inference.selected_profile
    assert profile is not None and profile.model_id == "browser-test-model"
    picker.press("t")
    expect(picker).to_have_attribute(
        "aria-activedescendant", text_option.get_attribute("id") or ""
    )
    with browser_page.expect_response("**/chat/model") as saved:
        picker.press("Enter")
    assert saved.value.ok
    expect(browser_page.locator("#chat-fragment")).to_have_attribute(
        "data-revision", str(chat.revision)
    )
    expect(picker).to_be_focused()
    expect(picker).to_have_text("Text model")
    picker.press("ArrowDown")
    scroll_top = options.evaluate(
        "element => { element.scrollTop = 400; return element.scrollTop; }"
    )
    browser_page.evaluate(
        "window.previousModelPicker = document.getElementById('active-chat-model')"
    )
    _edit_message_in_background(
        browser_page, message.id, "Updated while the picker is open"
    )
    browser_page.wait_for_function(
        "() => window.previousModelPicker !== document.getElementById('active-chat-model')"
    )
    expect(picker).to_have_attribute("aria-expanded", "true")
    expect(options).to_be_visible()
    expect(picker).to_be_focused()
    expect(picker).to_have_attribute(
        "aria-activedescendant", text_option.get_attribute("id") or ""
    )
    _wait_for_message_text(browser_page, message.id, "Updated while the picker is open")
    expect(
        browser_page.locator(
            '#settings-panel form[action="/chat/model"] input[name="revision"]'
        )
    ).to_have_value(str(chat.revision))
    assert options.evaluate("element => element.scrollTop") == scroll_top
    picker.press("Home")
    with browser_page.expect_response("**/chat/model") as cleared:
        picker.press("Enter")
    assert cleared.value.ok
    profile = chat.inference.selected_profile
    assert profile is not None and profile.model_id is None
    expect(picker).to_have_text("No model selected")
    picker.press("ArrowDown")
    picker.press("Tab")
    expect(options).to_be_hidden()
    expect(browser_page.locator("#settings-panel")).to_have_class(
        re.compile(r"is-open")
    )


def test_model_picker_touch_selection_and_outside_dismissal(
    touch_browser_page: Page, browser_server: BrowserServer
) -> None:
    page = touch_browser_page
    _open_chat_page(page)
    page.get_by_test_id("open-settings-button").tap()
    picker = page.get_by_test_id("active-chat-model")
    options = page.locator("#active-chat-model-options")
    picker.tap()
    expect(options).to_be_visible()
    with page.expect_response("**/chat/model") as saved:
        options.locator('[data-model-value="text-model"]').tap()
    assert saved.value.ok
    expect(picker).to_have_text("Text model")
    expect(options).to_be_hidden()
    profile = browser_server.state.chats[0].inference.selected_profile
    assert profile is not None and profile.model_id == "text-model"
    picker.tap()
    expect(options).to_be_visible()
    page.get_by_test_id("rename-chat-input").tap()
    expect(options).to_be_hidden()


@pytest.mark.parametrize("unsaved", [False, True])
def test_backend_model_picker_survives_configuration_refresh(
    browser_page: Page, browser_server: BrowserServer, unsaved: bool
) -> None:
    """A dialog refresh retains focus, the popup, and any unsaved default model."""

    backend = browser_server.state.config.backends.backends[0]
    browser_page.goto("/chats?dialog=global&tab=backends")
    browser_page.locator(f"#backend-{backend.id} > summary").click()
    picker = browser_page.get_by_test_id(f"backend-model-{backend.id}")
    options = browser_page.locator(f"#backend-model-{backend.id}-options")
    picker.click()
    if unsaved:
        options.locator('[data-model-value="text-model"]').click()
        picker.click()
    assert (
        browser_page.evaluate("""async () => {
        const response = await fetch('/settings/global/prompt', {
            method: 'POST',
            headers: {
                'X-Jouzetsu-CSRF': document.querySelector('[data-csrf-token]').dataset.csrfToken,
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({system_prompt: 'Changed in the background'}),
        });
        return response.status;
    }""")
        == 200
    )
    browser_page.wait_for_function("""() => {
        const dialog = document.getElementById('global-settings-dialog');
        return dialog.open && dialog.querySelector('textarea[name="system_prompt"]')
            .value === 'Changed in the background';
    }""")
    expect(picker).to_be_focused()
    expect(picker).to_have_attribute("aria-expanded", "true")
    expect(options).to_be_visible()
    expect(picker).to_have_text("Text model" if unsaved else "Browser test model")


def test_model_picker_touch_scroll_does_not_select_or_dismiss(
    touch_browser_page: Page, browser_server: BrowserServer
) -> None:
    page = touch_browser_page
    backend = browser_server.state.config.backends.backends[0]
    _set_browser_models(
        browser_server,
        (
            *backend.models,
            *(
                RegisteredModel(ModelDescriptor(f"extra-{index}", f"Extra {index:02}"))
                for index in range(30)
            ),
        ),
    )
    _open_chat_page(page)
    page.get_by_test_id("open-settings-button").tap()
    picker = page.get_by_test_id("active-chat-model")
    picker.tap()
    options = page.locator("#active-chat-model-options")
    bounds = options.bounding_box()
    assert bounds
    session = page.context.new_cdp_session(page)
    x, y = bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] - 20
    _dispatch_touch(session, "touchStart", x, y)
    for distance in range(30, 181, 30):
        _dispatch_touch(session, "touchMove", x, y - distance)
    _dispatch_touch(session, "touchEnd")
    page.wait_for_function(
        "() => document.getElementById('active-chat-model-options').scrollTop > 0"
    )
    expect(picker).to_have_attribute("aria-expanded", "true")
    profile = browser_server.state.chats[0].inference.selected_profile
    assert profile is not None and profile.model_id == "browser-test-model"


@pytest.mark.parametrize("location", ["chat", "backend"])
def test_model_picker_fits_its_scroll_container(
    browser_page: Page,
    browser_server: BrowserServer,
    location: Literal["chat", "backend"],
) -> None:
    """A long list remains reachable when the control is near the viewport bottom."""

    browser_page.set_viewport_size({"width": 390, "height": 500})
    backend = browser_server.state.config.backends.backends[0]
    models = tuple(
        replace(
            backend.models[0],
            descriptor=ModelDescriptor(f"model-{index}", f"Model {index:02}"),
        )
        for index in range(30)
    )
    _set_browser_models(browser_server, (*backend.models, *models))
    if location == "chat":
        _open_chat_page(browser_page)
        browser_page.get_by_test_id("open-settings-button").click()
        picker = browser_page.get_by_test_id("active-chat-model")
    else:
        browser_page.goto("/chats?dialog=global&tab=backends")
        browser_page.locator(f"#backend-{backend.id} > summary").click()
        picker = browser_page.get_by_test_id(f"backend-model-{backend.id}")
    picker.evaluate("element => element.scrollIntoView({block: 'end'})")
    picker.click()
    options = browser_page.locator(f"#{picker.get_attribute('id')}-options")
    expect(options).to_be_visible()
    assert options.evaluate("""element => {
        const bounds = element.getBoundingClientRect();
        const container = element.closest('.jouzetsu-panel-content, dialog');
        const clip = container.getBoundingClientRect();
        return bounds.top >= Math.max(0, clip.top)
            && bounds.bottom <= Math.min(innerHeight, clip.bottom);
    }""")
    picker.press("End")
    last = options.locator('[data-model-value="model-29"]')
    expect(last).to_be_in_viewport()
    last.click()
    expect(picker).to_have_text("Model 29")


def test_model_picker_typeahead_starts_after_the_current_selection(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    backend = browser_server.state.config.backends.backends[0]
    first, second = backend.models
    second = replace(
        second,
        descriptor=replace(second.descriptor, display_name="Browser alternative"),
    )
    _set_browser_models(browser_server, (first, second))
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("open-settings-button").click()
    picker = browser_page.get_by_test_id("active-chat-model")
    picker.focus()
    picker.press("b")
    alternative = browser_page.locator(
        '#active-chat-model-options [data-model-value="text-model"]'
    )
    expect(picker).to_have_attribute(
        "aria-activedescendant", alternative.get_attribute("id") or ""
    )
    picker.press("b")
    original = browser_page.locator(
        '#active-chat-model-options [data-model-value="browser-test-model"]'
    )
    expect(picker).to_have_attribute(
        "aria-activedescendant", original.get_attribute("id") or ""
    )


@pytest.mark.parametrize("viewport_width", [390, 1280])
def test_phosphor_icons_render_and_continue_keeps_its_armed_variant(
    browser_page: Page, browser_server: BrowserServer, viewport_width: int
) -> None:
    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    chat = browser_server.state.chats[0]
    _ = chat.add_message("user", "Hello")
    reply = chat.add_message("assistant", "An example reply.")
    _open_chat_page(browser_page)
    sprite_url = browser_page.locator('meta[name="jouzetsu-icons"]').get_attribute(
        "content"
    )
    assert sprite_url and sprite_url.startswith("/static/icons/sprite.svg?v=")
    sprite = browser_page.request.get(sprite_url)
    assert sprite.status == 200
    assert sprite.headers["content-type"] == "image/svg+xml"
    _expect_icons_painted(browser_page)
    control = browser_page.get_by_test_id(f"continue-message-{reply.id}")
    regular = control.locator(".jouzetsu-icon-regular")
    armed = control.locator(".jouzetsu-icon-armed")
    expect(regular).to_have_css("display", "inline")
    expect(armed).to_have_css("display", "none")
    _ = control.evaluate("control => control.classList.add('is-swipe-armed')")
    expect(regular).to_have_css("display", "none")
    expect(armed).to_have_css("display", "inline")
    _expect_icons_painted(browser_page)
    _ = control.evaluate("control => control.classList.remove('is-swipe-armed')")
    browser_page.get_by_test_id("composer-menu-toggle").click()
    _expect_icons_painted(browser_page)
    _ = browser_page.screenshot(path=f"/tmp/jouzetsu-phosphor-{viewport_width}.png")


def test_dynamically_added_controls_use_the_same_phosphor_sprite(
    browser_page: Page,
) -> None:
    browser_page.goto("/chats?dialog=global")
    details = browser_page.get_by_test_id("spelling-replacements-details")
    details.locator("summary").click()
    original = details.locator(".jouzetsu-spelling-arrow use").first.get_attribute(
        "href"
    )
    assert original and original.endswith("#arrow-right")
    details.get_by_role("button", name="Add replacement", exact=True).click()
    expect(details.locator(".jouzetsu-spelling-arrow use").last).to_have_attribute(
        "href", original
    )
    _expect_icons_painted(browser_page)
    browser_page.goto("/characters")
    with browser_page.expect_navigation(wait_until="load"):
        browser_page.get_by_role("button", name="New character", exact=True).click()
    browser_page.get_by_role("button", name="Add custom field", exact=True).click()
    field_icon = browser_page.locator("[data-character-field-menu-toggle] svg use").last
    expect(field_icon).to_have_attribute(
        "href", original.replace("#arrow-right", "#more")
    )
    _expect_icons_painted(browser_page)


def test_phosphor_access_and_error_icons_preserve_their_behaviour(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    browser_server.state.config.access.devices["dvc_phosphor-test-0001"] = (
        DeviceAccessSettings(label="Example phone")
    )
    browser_page.goto("/access/denied")
    expect(browser_page.locator(".jouzetsu-access-icon use")).to_have_attribute(
        "href", re.compile(r"#lock$")
    )
    _expect_icons_painted(browser_page)
    browser_page.goto("/chats?dialog=access&error=Example%20error")
    _expect_icons_painted(browser_page)
    expect(
        browser_page.locator(".jouzetsu-dialog-feedback .jouzetsu-error-icon")
    ).to_be_visible()
    card = browser_page.locator(".jouzetsu-device-card").first
    expect(card.locator(".jouzetsu-device-expand-icon")).to_be_visible()
    expect(card.locator(".jouzetsu-device-collapse-icon")).not_to_be_visible()
    card.locator("summary").click()
    expect(card.locator(".jouzetsu-device-expand-icon")).not_to_be_visible()
    expect(card.locator(".jouzetsu-device-collapse-icon")).to_be_visible()
    _expect_icons_painted(browser_page)
    _ = browser_page.evaluate("""async () => {
        const { NoticeController } = await import('/static/app/notices.js');
        new NoticeController().announce('Client error', true);
    }""")
    expect(browser_page.locator("#notice-area .jouzetsu-error-icon")).to_be_visible()
    _expect_icons_painted(browser_page)


@pytest.mark.parametrize("dark_mode", [False, True])
@pytest.mark.parametrize("style", list(MessageActionStyle))
def test_every_message_action_icon_uses_its_configured_colour(
    browser_page: Page,
    browser_server: BrowserServer,
    tmp_path: Path,
    dark_mode: bool,
    style: MessageActionStyle,
) -> None:
    state = browser_server.state
    state.config.ui.dark_mode = dark_mode
    state.config.ui.message_action_style = style
    chat = state.chats[0]
    first = chat.add_message("user", "First prompt")
    _ = chat.add_message("user", "Additional prompt")
    _ = chat.add_message("assistant", "Initial reply")
    last = chat.add_message("assistant", "Continued reply")
    _open_chat_page(browser_page)
    for marker, action in (
        (f"delete-message-{first.id}", MessageAction.DELETE),
        (f"truncate-{first.id}", MessageAction.DELETE),
        (f"edit-message-{first.id}", MessageAction.EDIT),
        (f"regenerate-message-{last.id}", MessageAction.REGENERATE),
        (f"merge-message-{last.id}", MessageAction.MERGE),
        (f"continue-message-{last.id}", MessageAction.CONTINUE),
    ):
        _expect_message_action_colour(browser_page, marker, action, style)
    for message in browser_page.locator(_MESSAGE_SELECTOR).all():
        expect(message.locator(".jouzetsu-message-action-divider")).to_be_visible()
    _ = browser_page.screenshot(path=str(tmp_path / "message-action-colours.png"))
    browser_page.get_by_test_id("message-input").fill("Next prompt")
    _send_only(browser_page)
    expect(browser_page.get_by_text("Next prompt", exact=True)).to_be_visible()
    _expect_message_action_colour(
        browser_page,
        f"resend-message-{chat.messages[-1].id}",
        MessageAction.CONTINUE,
        style,
    )
    assert (
        browser_page.evaluate("""async () => {
        const response = await fetch('/settings/global/appearance/reset', {
            method: 'POST',
            headers: {'HX-Request': 'true', 'X-Jouzetsu-CSRF': document.querySelector('[data-csrf-token]').dataset.csrfToken},
        });
        return response.status;
    }""")
        == 200
    )
    assert state.config.ui.message_action_style is MessageActionStyle.SEMANTIC
    browser_page.get_by_test_id("open-nav-button").click()
    browser_page.get_by_test_id("global-settings-button").click()
    browser_page.get_by_test_id("global-settings-dialog-appearance-tab").click()
    expect(browser_page.get_by_test_id("message-action-style")).to_have_value(
        "semantic"
    )


def test_send_only_and_shift_click_save_without_generating(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("message-input").fill("Send only")
    _send_only(browser_page)
    expect(
        browser_page.get_by_test_id("message-list").get_by_text("Send only", exact=True)
    ).to_be_visible()
    expect(browser_page.get_by_test_id("message-input")).to_have_value("")
    assert not browser_server.client.requests
    browser_page.get_by_test_id("message-input").fill("Shift send")
    browser_page.get_by_test_id("send-message-button").click(modifiers=["Shift"])
    expect(
        browser_page.get_by_test_id("message-list").get_by_text(
            "Shift send", exact=True
        )
    ).to_be_visible()
    assert not browser_server.client.requests


def test_rejected_shift_click_does_not_change_the_next_keyboard_send(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    browser_server.state.chats[0].draft = "Keyboard submission"
    _open_chat_page(browser_page)
    browser_page.route(
        "**/chat/send",
        lambda route: route.fulfill(status=409, body="Controlled rejected send"),
    )
    browser_page.get_by_test_id("send-message-button").click(modifiers=["Shift"])
    expect(browser_page.locator("#notice-area")).to_contain_text(
        "Controlled rejected send"
    )
    browser_page.unroute("**/chat/send")
    browser_page.get_by_test_id("message-input").press("Shift+Enter")
    expect(browser_page.get_by_text("Browser test reply", exact=True)).to_be_visible()
    assert len(browser_server.client.requests) == 1


def test_global_backend_controls_are_usable_on_mobile(browser_page: Page) -> None:
    browser_page.set_viewport_size({"width": 390, "height": 844})
    browser_page.goto("/chats?dialog=global&tab=backends")
    form = browser_page.get_by_test_id("add-backend-form")
    expect(form).to_be_visible()
    form.locator('[name="name"]').fill("Second local")
    form.locator('[name="base_url"]').fill("http://localhost:9000/v1")
    with browser_page.expect_navigation():
        form.get_by_role("button", name="Add backend").click()
    expect(browser_page.locator("#global-settings-dialog")).to_be_visible()
    expect(
        browser_page.locator("#global-settings-dialog summary", has_text="Second local")
    ).to_be_visible()


def test_typing_during_send_keeps_the_next_draft_ordered(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _open_chat_page(browser_page)
    browser_page.evaluate("""() => {
        const original = fetch.bind(window);
        window.fetch = async (url, options) => {
            const response = await original(url, options);
            if (new URL(url, location.href).pathname === '/chat/send') await new Promise((resolve) => { window.releaseSend = resolve; });
            return response;
        };
    }""")
    browser_page.get_by_test_id("message-input").fill("First submission")
    _send_only(browser_page)
    browser_page.wait_for_function("() => typeof window.releaseSend === 'function'")
    browser_page.get_by_test_id("message-input").fill("The next unsent draft")
    browser_page.evaluate("window.releaseSend()")
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "The next unsent draft"
    )
    browser_page.wait_for_function(
        "() => document.querySelector('.jouzetsu-composer-form input[name=draft_revision]').value === '3'"
    )
    assert browser_server.state.chats[0].draft == "The next unsent draft"
    assert [message.content for message in browser_server.state.chats[0].messages] == [
        "First submission"
    ]


def test_rejected_attachment_stays_available_for_explicit_retry(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _open_chat_page(browser_page)
    browser_page.route(
        "**/chats/*/images",
        lambda route: route.fulfill(status=409, body="Shared draft changed"),
    )
    browser_page.get_by_test_id("image-upload-input").set_input_files(
        {"name": "keep-this.png", "mimeType": "image/png", "buffer": image_bytes()}
    )
    retained = browser_page.locator('[data-failed-attachments="true"]')
    expect(retained.get_by_role("link", name="keep-this.png")).to_be_visible()
    assert not browser_server.state.chats[0].draft_images
    browser_page.unroute("**/chats/*/images")
    retained.get_by_role("button", name="Retry attachments", exact=True).click()
    expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
    expect(retained).to_have_count(0)
    assert len(browser_server.state.chats[0].draft_images) == 1


def test_incoming_settings_preserve_focused_unsaved_form(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    browser_page.goto("/chats?dialog=global")
    prompt = browser_page.locator(
        '#global-settings-dialog textarea[name="system_prompt"]'
    )
    prompt.fill("Keep this unsaved prompt")
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    try:
        other.goto(browser_page.url)
        status = other.evaluate("""async () => {
            const token = document.querySelector('[data-csrf-token]').dataset.csrfToken;
            const response = await fetch('/settings/global/prompt', {method: 'POST', headers: {'X-Jouzetsu-CSRF': token, 'Content-Type': 'application/x-www-form-urlencoded'}, body: new URLSearchParams({system_prompt: 'Other saved prompt'})});
            return response.status;
        }""")
        assert status == 200
        browser_page.wait_for_function(
            "() => document.querySelector('#global-settings-dialog').open"
        )
        expect(prompt).to_have_value("Keep this unsaved prompt")
        expect(prompt).to_be_focused()
        assert (
            browser_server.state.config.generation.system_prompt == "Other saved prompt"
        )
    finally:
        context.close()


def test_missing_original_renders_unavailable_placeholder(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    image = browser_server.state.images.store(image_bytes(), "missing-original.png")
    message = browser_server.state.chats[0].add_message("user", "Retained reference")
    message.images = (image,)
    browser_server.state.images.path_for(image).unlink()
    _open_chat_page(browser_page)
    expect(
        browser_page.get_by_text("Image unavailable: missing-original.png", exact=True)
    ).to_be_visible()
    assert message.images == (image,)


def test_send_uses_secure_fallback_when_random_uuid_is_unavailable(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    browser_page.add_init_script(
        "Object.defineProperty(Crypto.prototype, 'randomUUID', {value: undefined});"
    )
    _open_chat_page(browser_page)
    browser_page.get_by_test_id("message-input").fill("HTTP LAN submission")
    browser_page.get_by_test_id("send-message-button").click()
    expect(browser_page.get_by_text("Browser test reply", exact=True)).to_be_visible()
    assert len(browser_server.client.requests) == 1
    receipt = browser_server.state.chats[0].submissions[0]
    assert len(receipt.id) == 32
    assert int(receipt.id, 16) != 0


def test_continuous_typing_publishes_shared_drafts_without_a_pause(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _open_chat_page(browser_page)
    browser_page.clock.install()
    browser_page.clock.pause_at("2030-01-01T00:00:00Z")
    for index in range(8):
        browser_page.get_by_test_id("message-input").fill(f"Continuous input {index}")
        browser_page.clock.run_for(100)
    browser_page.wait_for_function(
        "() => Number(document.querySelector('.jouzetsu-composer-form input[name=draft_revision]').value) > 0"
    )
    assert browser_server.state.chats[0].draft.startswith("Continuous input")


def test_slow_draft_requests_coalesce_continued_typing(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _open_chat_page(browser_page)
    browser_page.clock.install()
    browser_page.clock.pause_at("2030-01-01T00:00:00Z")
    browser_page.evaluate("""() => {
        window.draftRequests = 0;
        const original = fetch.bind(window);
        window.fetch = async (url, options) => {
            if (new URL(url, location.href).pathname === '/chat/draft') {
                window.draftRequests += 1;
                if (window.draftRequests === 1) await new Promise((resolve) => { window.releaseDraft = resolve; });
            }
            return original(url, options);
        };
    }""")
    browser_page.get_by_test_id("message-input").fill("First pending draft")
    browser_page.clock.run_for(250)
    browser_page.wait_for_function("() => typeof window.releaseDraft === 'function'")
    for index in range(6):
        browser_page.get_by_test_id("message-input").fill(f"Latest draft {index}")
        browser_page.clock.run_for(250)
    browser_page.evaluate("window.releaseDraft()")
    browser_page.clock.resume()
    browser_page.wait_for_function(
        "() => document.querySelector('.jouzetsu-composer-form input[name=draft_revision]').value === '2'"
    )
    assert browser_server.state.chats[0].draft == "Latest draft 5"
    assert browser_page.evaluate("window.draftRequests") == 2


@pytest.mark.parametrize("resized", [False, True])
@pytest.mark.parametrize("interaction", ["echo", "typing", "composition", "backspace"])
def test_delayed_draft_echo_keeps_mobile_keyboard_input_connected(
    chromium_browser: Browser,
    browser_server: BrowserServer,
    interaction: Literal["echo", "typing", "composition", "backspace"],
    resized: bool,
) -> None:
    with chromium_browser.new_context(
        base_url=browser_server.base_url,
        viewport={"width": 412, "height": 915},
        is_mobile=True,
        has_touch=True,
    ) as context:
        page = context.new_page()
        page.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
        _open_chat_page(page)
        if resized:
            page.get_by_test_id("composer-resize-handle").press("ArrowUp")
        page.evaluate("""() => {
            const original = fetch.bind(window);
            window.fetch = async (url, options) => {
                const response = await original(url, options);
                if (new URL(url, location.href).pathname === '/fragments/chat/composer' && !window.heldComposer) {
                    window.heldComposer = true;
                    const markup = await response.text();
                    await new Promise(resolve => { window.releaseComposer = resolve; });
                    return new Response(markup, {status: response.status});
                }
                return response;
            };
        }""")
        input_field = page.get_by_test_id("message-input")
        input_field.fill("Mobile keyboard input")
        page.wait_for_function("() => typeof window.releaseComposer === 'function'")
        page.clock.install()
        page.clock.pause_at("2030-01-01T00:00:00Z")
        input_field.focus()
        if interaction == "composition":
            input_field.evaluate("""input => {
                input.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true, data: ''}));
                input.setRangeText(' composing', input.value.length, input.value.length, 'end');
                input.dispatchEvent(new InputEvent('input', {bubbles: true, data: ' composing', isComposing: true, inputType: 'insertCompositionText'}));
            }""")
        elif interaction == "backspace":
            page.keyboard.down("Backspace")
        elif interaction == "typing":
            page.keyboard.insert_text(" still typing")
        value = input_field.input_value()
        page.evaluate("""() => {
            window.keyboardInput = document.querySelector('[data-testid=message-input]');
            window.keyboardSelection = [keyboardInput.selectionStart, keyboardInput.selectionEnd];
            window.keyboardBlurred = false;
            window.inputValueWrites = 0;
            window.selectionWrites = 0;
            keyboardInput.addEventListener('blur', () => { window.keyboardBlurred = true; });
            const value = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            Object.defineProperty(keyboardInput, 'value', {
                get() { return value.get.call(this); },
                set(text) { window.inputValueWrites += 1; value.set.call(this, text); },
            });
            const setSelection = keyboardInput.setSelectionRange.bind(keyboardInput);
            keyboardInput.setSelectionRange = (...args) => { window.selectionWrites += 1; setSelection(...args); };
            window.releaseComposer();
        }""")
        page.wait_for_function(
            "() => document.getElementById('chat-fragment').dataset.draftRevision === '1'"
        )
        assert page.evaluate("""() => (
            keyboardInput === document.querySelector('[data-testid=message-input]')
            && document.activeElement === keyboardInput && !keyboardBlurred
            && inputValueWrites === 0 && selectionWrites === 0
            && keyboardInput.selectionStart === keyboardSelection[0]
            && keyboardInput.selectionEnd === keyboardSelection[1]
        )""")
        expect(input_field).to_have_value(value)
        if interaction == "composition":
            input_field.evaluate(
                "input => input.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true, data: ' composing'}))"
            )
            page.keyboard.insert_text(" finished")
            value += " finished"
        elif interaction == "backspace":
            for _ in range(6):
                page.keyboard.down("Backspace")
            page.keyboard.up("Backspace")
            value = value[:-6]
        else:
            page.keyboard.insert_text(" finished")
            value += " finished"
        expect(input_field).to_have_value(value)
        page.clock.resume()
        page.wait_for_function(
            "() => document.querySelector('.jouzetsu-composer-form input[name=draft_revision]').value === '2'"
        )
        assert browser_server.state.chats[0].draft == value


def _create_chat_in_background(page: Page) -> None:
    assert (
        page.evaluate("""async () => {
        const token = document.querySelector('[data-csrf-token]').dataset.csrfToken;
        const response = await fetch('/chats/new', {method: 'POST',
            headers: {'X-Jouzetsu-CSRF': token, 'HX-Request': 'true'}});
        return response.status;
    }""")
        == 200
    )


def test_navigation_cannot_retain_another_chats_dirty_settings(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    first = browser_server.state.chats[0]
    _open_chat_page(browser_page)
    _create_chat_in_background(browser_page)
    second = browser_server.state.chats[0]
    assert first is not second
    browser_page.get_by_test_id("open-settings-button").click()
    browser_page.get_by_test_id("active-chat-prompt").fill("Unsaved first-chat prompt")
    browser_page.keyboard.press("Escape")
    browser_page.get_by_test_id("open-nav-button").click()
    browser_page.locator(f'a[href="/chats/{second.id}"]').click()
    expect(browser_page).to_have_url(f"{browser_server.base_url}/chats/{second.id}")
    browser_page.wait_for_function(
        "id => document.getElementById('settings-panel').dataset.chatId === id",
        arg=second.id,
    )
    browser_page.keyboard.press("Escape")
    browser_page.get_by_test_id("open-settings-button").click()
    expect(browser_page.get_by_test_id("active-chat-prompt")).to_have_value("")
    assert not first.system_prompt and not second.system_prompt


def test_panel_refresh_repeats_when_a_newer_change_arrives_during_fetch(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    chat = browser_server.state.chats[0]
    message = chat.add_message("user", "Original")
    _open_chat_page(browser_page)
    browser_page.evaluate("""() => {
        const original = fetch.bind(window);
        window.fetch = async (url, options) => {
            const response = await original(url, options);
            if (new URL(url, location.href).pathname === '/fragments/settings' && !window.heldSettings) {
                window.heldSettings = true;
                await new Promise((resolve) => { window.releaseSettings = resolve; });
            }
            return response;
        };
    }""")
    _edit_message_in_background(browser_page, message.id, "First committed change")
    browser_page.wait_for_function("() => typeof window.releaseSettings === 'function'")
    _wait_for_message_text(browser_page, message.id, "First committed change")
    _edit_message_in_background(browser_page, message.id, "Second committed change")
    _wait_for_message_text(browser_page, message.id, "Second committed change")
    browser_page.evaluate("window.releaseSettings()")
    browser_page.wait_for_function(
        "revision => document.querySelector('#settings-panel input[name=revision]').value === revision",
        arg=str(chat.revision),
    )


def test_delayed_send_acknowledgement_does_not_reverse_chat_navigation(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    first = browser_server.state.chats[0]
    _open_chat_page(browser_page)
    _create_chat_in_background(browser_page)
    second = browser_server.state.chats[0]
    browser_page.evaluate("""() => {
        const original = fetch.bind(window);
        window.fetch = async (url, options) => {
            const response = await original(url, options);
            if (new URL(url, location.href).pathname === '/chat/send') {
                await new Promise((resolve) => { window.releaseSend = resolve; });
                window.sendAcknowledged = true;
            }
            if (new URL(url, location.href).pathname === '/fragments/chat' && window.sendAcknowledged) window.acknowledgementRefreshed = true;
            return response;
        };
    }""")
    browser_page.get_by_test_id("message-input").fill("Save in first chat")
    _send_only(browser_page)
    browser_page.wait_for_function("() => typeof window.releaseSend === 'function'")
    browser_page.get_by_test_id("open-nav-button").click()
    browser_page.locator(f'a[href="/chats/{second.id}"]').click()
    browser_page.wait_for_function(
        "id => document.getElementById('chat-fragment').dataset.chatId === id",
        arg=second.id,
    )
    browser_page.evaluate("window.releaseSend()")
    browser_page.wait_for_function("() => window.acknowledgementRefreshed === true")
    expect(browser_page).to_have_url(f"{browser_server.base_url}/chats/{second.id}")
    assert [message.content for message in first.messages] == ["Save in first chat"]
    assert not second.messages


def test_refreshed_dialog_retains_backdrop_dismissal(browser_page: Page) -> None:
    browser_page.goto("/chats?dialog=global")
    browser_page.wait_for_function(
        "() => document.getElementById('global-settings-dialog').open"
    )
    assert (
        browser_page.evaluate("""async () => {
        window.originalDialog = document.getElementById('global-settings-dialog');
        const token = document.querySelector('[data-csrf-token]').dataset.csrfToken;
        const response = await fetch('/settings/global/prompt', {method: 'POST',
            headers: {'X-Jouzetsu-CSRF': token, 'Content-Type': 'application/x-www-form-urlencoded'},
            body: new URLSearchParams({system_prompt: 'Refresh the dialog'})});
        return response.status;
    }""")
        == 200
    )
    browser_page.wait_for_function(
        "() => document.getElementById('global-settings-dialog') !== window.originalDialog"
    )
    expect(
        browser_page.locator('#global-settings-dialog textarea[name="system_prompt"]')
    ).to_have_value("Refresh the dialog")
    browser_page.mouse.click(1, 1)
    expect(browser_page.locator("#global-settings-dialog")).not_to_be_visible()


def test_storage_warning_survives_transient_notice_updates(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    storage = browser_server.state.storage
    (storage.directory / "damaged.json").write_text("{broken", encoding="utf-8")
    storage.load_all()
    _open_chat_page(browser_page)
    warning = browser_page.locator('[data-storage-warning="true"]')
    expect(warning).to_contain_text("1 chat files")
    browser_page.get_by_test_id("message-input").fill("Save with warning visible")
    browser_page.get_by_test_id("send-message-button").click()
    expect(browser_page.get_by_text("Browser test reply", exact=True)).to_be_visible()
    # Dismissing ordinary feedback must not erase the recovery warning.
    browser_page.evaluate("""async () => {
        const { NoticeController } = await import('/static/app/notices.js');
        const notices = new NoticeController();
        notices.announce('Temporary feedback');
        notices.dismiss();
    }""")
    expect(warning).to_be_visible()


@pytest.mark.parametrize("with_image", (False, True))
def test_lost_send_response_can_retry_the_same_submission(
    browser_page: Page, browser_server: BrowserServer, with_image: bool
) -> None:
    _open_chat_page(browser_page)
    if with_image:
        browser_page.get_by_test_id("image-upload-input").set_input_files(
            {"name": "retry.png", "mimeType": "image/png", "buffer": image_bytes()}
        )
        expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
    browser_page.evaluate("""() => {
        const original = fetch.bind(window);
        window.fetch = async (url, options) => {
            const response = await original(url, options);
            if (new URL(url, location.href).pathname === '/chat/send' && !window.lostSend) {
                window.lostSend = true;
                throw new TypeError('Controlled lost acknowledgement');
            }
            return response;
        };
    }""")
    browser_page.get_by_test_id("message-input").fill("Retry this accepted submission")
    _send_only(browser_page)
    expect(browser_page.locator("#notice-area")).to_contain_text(
        "Could not save this change"
    )
    browser_page.wait_for_function(
        "() => document.getElementById('chat-fragment').dataset.revision === '1'"
    )
    expect(browser_page.get_by_test_id("message-input")).to_have_value(
        "Retry this accepted submission"
    )
    expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(
        1 if with_image else 0
    )
    _send_only(browser_page)
    expect(browser_page.get_by_test_id("message-input")).to_have_value("")
    chat = browser_server.state.chats[0]
    assert len(chat.messages) == len(chat.submissions) == 1
    assert len(chat.messages[0].images) == int(with_image)
    assert not browser_server.client.requests


def test_chat_history_and_reload_restore_browser_scroll_positions(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _seed_scrollable_transcript(browser_server)
    _open_chat_page(browser_page)
    first_url = browser_page.url
    _create_chat_in_background(browser_page)
    second = browser_server.state.chats[0]
    expected_top = _scroll_metrics(browser_page)["maximum"] * 0.3
    _set_message_scroll_top(browser_page, expected_top)
    browser_page.get_by_test_id("open-nav-button").click()
    browser_page.locator(f'a[href="/chats/{second.id}"]').click()
    browser_page.wait_for_function(
        "id => document.getElementById('chat-fragment').dataset.chatId === id",
        arg=second.id,
    )
    browser_page.go_back()
    expect(browser_page).to_have_url(first_url)
    browser_page.wait_for_function(
        "top => Math.abs(document.getElementById('message-list').scrollTop - top) <= 2",
        arg=expected_top,
    )
    browser_page.reload()
    browser_page.wait_for_function(
        "top => Math.abs(document.getElementById('message-list').scrollTop - top) <= 2",
        arg=expected_top,
    )


def test_appearance_changes_refresh_another_browsers_styles(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    _open_chat_page(browser_page)
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    other.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
    try:
        other.goto("/chats?dialog=global&tab=appearance")
        other.get_by_test_id("theme-user-accent-color").fill("#22bbdd")
        other.get_by_test_id("appearance-dark-mode").uncheck()
        with other.expect_navigation():
            other.get_by_test_id("save-global-appearance-button").click()
        expect(browser_page.locator("body")).to_have_css(
            "background-color", "rgb(255, 255, 255)"
        )
        expect(browser_page.locator(".jouzetsu-composer-primary")).to_have_css(
            "background-color", "rgb(34, 187, 221)"
        )
    finally:
        context.close()


def test_two_browsers_reject_stale_backend_settings_without_losing_edits(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    configured = browser_server.state.config.backends.backends[0]
    path = f"/settings/backends/{configured.id}/save"
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    other.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
    try:
        for page in (browser_page, other):
            page.goto("/chats?dialog=global&tab=backends")
            page.locator(f"#backend-{configured.id} > summary").click()
        form_a = browser_page.locator(f'form[action="{path}"]')
        form_b = other.locator(f'form[action="{path}"]')
        token = form_b.locator('[name="backend_fingerprint"]').input_value()
        form_a.locator('[name="name"]').fill("First browser saved")
        form_b.locator('[name="name"]').fill("Second browser unsaved")
        with browser_page.expect_response(
            lambda response: response.url.endswith(path)
        ) as accepted:
            form_a.get_by_role("button", name="Save backend", exact=True).click()
        assert accepted.value.status == 200
        expect(other.locator(f"#backend-{configured.id} > summary")).to_contain_text(
            "First browser saved"
        )
        expect(form_b.locator('[name="backend_fingerprint"]')).to_have_value(token)
        with other.expect_response(
            lambda response: response.url.endswith(path)
        ) as rejected:
            form_b.get_by_role("button", name="Save backend", exact=True).click()
        assert rejected.value.status == 409
        expect(other.locator("#notice-area")).to_contain_text("changed elsewhere")
        expect(form_b.locator('[name="name"]')).to_have_value("Second browser unsaved")
        assert (
            browser_server.state.config.backends.backend(configured.id).name
            == "First browser saved"
        )
        loaded = ConfigStore.for_paths(browser_server.state.config.paths).load()
        assert loaded.backends.backend(configured.id).name == "First browser saved"
        expect(form_a.locator('[name="backend_fingerprint"]')).not_to_have_value(token)
        form_a.locator('[name="name"]').fill("Subsequent valid edit")
        with browser_page.expect_response(
            lambda response: response.url.endswith(path)
        ) as valid:
            form_a.get_by_role("button", name="Save backend", exact=True).click()
        assert valid.value.status == 200
        expect(form_b.locator('[name="name"]')).to_have_value("Second browser unsaved")
        assert (
            browser_server.state.config.backends.backend(configured.id).name
            == "Subsequent valid edit"
        )
    finally:
        context.close()


def test_chats_workspace_restores_independent_browser_selections(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    first = browser_server.state.chats[0]
    message = first.add_message("user", "First chat")
    _open_chat_page(browser_page)
    first_url = browser_page.url
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    other.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
    try:
        _open_chat_page(other)
        _create_chat_in_background(browser_page)
        second = browser_server.state.chats[0]
        browser_page.get_by_test_id("open-nav-button").click()
        browser_page.locator(f'a[href="/chats/{second.id}"]').click()
        browser_page.wait_for_function(
            "id => document.getElementById('chat-fragment').dataset.chatId === id",
            arg=second.id,
        )
        second_url = browser_page.url
        _edit_message_in_background(
            other, message.id, "Make the first chat globally newest"
        )
        assert browser_server.state.chats[0] is first
        for page, expected in ((browser_page, second_url), (other, first_url)):
            page.goto("/characters")
            page.goto("/")
            page.get_by_test_id("launch-chats-link").click()
            expect(page).to_have_url(expected)
            page.reload()
            expect(page).to_have_url(expected)
            page.goto("/chats")
            expect(page).to_have_url(expected)
        assert (
            browser_page.evaluate("""async () => {
            const token = document.querySelector('[data-csrf-token]').dataset.csrfToken;
            const shell = document.getElementById('chat-fragment');
            const response = await fetch(`/chats/${shell.dataset.chatId}/delete`, {method: 'POST',
                headers: {'X-Jouzetsu-CSRF': token, 'HX-Request': 'true'},
                body: new URLSearchParams({chat_id: shell.dataset.chatId, revision: shell.dataset.revision})});
            return response.status;
        }""")
            == 200
        )
        browser_page.goto("/chats")
        expect(browser_page).to_have_url(first_url)
        expect(other).to_have_url(first_url)
    finally:
        context.close()


def test_long_transcript_draft_updates_only_refresh_composers(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    _seed_transcript(browser_server, 80, _LONG_MESSAGE)
    _open_chat_page(browser_page)
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    other.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
    try:
        _open_chat_page(other)
        for page in (browser_page, other):
            page.evaluate("""() => {
                performance.clearResourceTimings();
                window.transcriptMutations = 0;
                new MutationObserver((changes) => { window.transcriptMutations += changes.length; })
                    .observe(document.getElementById('message-list'), {subtree: true, childList: true, attributes: true, characterData: true});
            }""")
        for index in range(3):
            value = f"Composer-only draft {index}"
            browser_page.get_by_test_id("message-input").fill(value)
            expect(other.get_by_test_id("message-input")).to_have_value(value)
        expect(browser_page.get_by_test_id("message-input")).to_be_focused()
        other.get_by_test_id("image-upload-input").set_input_files(
            {"name": "shared.png", "mimeType": "image/png", "buffer": image_bytes()}
        )
        for page in (browser_page, other):
            expect(page.get_by_test_id("remove-draft-image")).to_have_count(1)
        browser_page.route("**/chat/draft", lambda route: route.fulfill(status=503))
        browser_page.get_by_test_id("message-input").fill("Keep my unsaved local text")
        expect(browser_page.locator("#notice-area")).to_contain_text("Could not save")
        other.get_by_test_id("message-input").fill("Newer shared text")
        browser_page.wait_for_function(
            "() => document.getElementById('chat-fragment').dataset.draftRevision === '5'"
        )
        expect(browser_page.get_by_test_id("message-input")).to_have_value(
            "Keep my unsaved local text"
        )
        browser_page.unroute("**/chat/draft")
        browser_page.get_by_test_id("message-input").fill("Keep my rejected correction")
        expect(browser_page.locator("#notice-area")).to_contain_text(
            "changed elsewhere"
        )
        expect(browser_page.get_by_test_id("remove-draft-image")).to_have_count(1)
        for page in (browser_page, other):
            assert page.evaluate("window.transcriptMutations") == 0
            assert (
                page.evaluate(
                    "performance.getEntriesByType('resource').filter(e => new URL(e.name).pathname === '/fragments/chat').length"
                )
                == 0
            )
            assert (
                page.evaluate(
                    "performance.getEntriesByType('resource').filter(e => new URL(e.name).pathname === '/fragments/chat/composer').length"
                )
                > 0
            )
        assert len(browser_server.state.chats[0].messages) == 160
    finally:
        context.close()


def test_focused_draft_refresh_preserves_running_generation_status(
    browser_page: Page, browser_server: BrowserServer, chromium_browser: Browser
) -> None:
    gate = _configure_paused_streaming_response(browser_server)
    context = chromium_browser.new_context(base_url=browser_server.base_url)
    other = context.new_page()
    other.set_default_timeout(_BROWSER_ACTION_TIMEOUT_MS)
    try:
        _open_chat_page(browser_page)
        browser_page.get_by_test_id("message-input").fill("Generate while drafting")
        browser_page.get_by_test_id("send-message-button").click()
        expect(browser_page.locator("article.is-streaming")).to_be_visible()
        _open_chat_page(other)
        other.get_by_test_id("message-input").fill("Draft during held generation")
        expect(browser_page.get_by_test_id("message-input")).to_have_value(
            "Draft during held generation"
        )
        for page in (browser_page, other):
            expect(page.get_by_test_id("stop-generation-button")).to_be_enabled()
            expect(page.locator("article.is-streaming")).to_contain_text(
                "deliberately long"
            )
        gate.set()
        for page in (browser_page, other):
            expect(page.get_by_test_id("send-message-button")).to_be_visible()
            expect(page.get_by_test_id("message-input")).to_have_value(
                "Draft during held generation"
            )
        assert len(browser_server.client.requests) == 1
    finally:
        gate.set()
        context.close()


def test_delayed_composer_response_cannot_replace_a_newly_selected_chat(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    _open_chat_page(browser_page)
    _create_chat_in_background(browser_page)
    second = browser_server.state.chats[0]
    browser_page.evaluate("""() => {
        const original = fetch.bind(window);
        window.fetch = async (url, options) => {
            const response = await original(url, options);
            if (new URL(url, location.href).pathname === '/fragments/chat/composer' && !window.heldComposer) {
                window.heldComposer = true;
                await new Promise((resolve) => { window.releaseComposer = resolve; });
                window.composerReleased = true;
            }
            return response;
        };
    }""")
    browser_page.get_by_test_id("message-input").fill("First-chat draft")
    browser_page.wait_for_function("() => typeof window.releaseComposer === 'function'")
    browser_page.get_by_test_id("open-nav-button").click()
    browser_page.locator(f'a[href="/chats/{second.id}"]').click()
    browser_page.wait_for_function(
        "id => document.getElementById('chat-fragment').dataset.chatId === id",
        arg=second.id,
    )
    browser_page.evaluate("window.releaseComposer()")
    browser_page.wait_for_function("() => window.composerReleased === true")
    expect(browser_page.get_by_test_id("message-input")).to_have_value("")
    expect(browser_page).to_have_url(f"{browser_server.base_url}/chats/{second.id}")
