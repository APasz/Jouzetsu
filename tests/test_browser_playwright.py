"""Focused browser coverage for client-side state that unit tests cannot observe."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event
from typing import Final, TypedDict, cast

import pytest
from browser_support import BrowserServer, create_browser_server
from playwright.sync_api import Browser, Error, Page, Route, expect, sync_playwright

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
    chat = browser_server.state.active_chat
    _ = chat.add_message("user", "Prompt")
    _ = chat.add_message("assistant", "Earlier reply")
    message = chat.add_message("assistant", '"Certainly."')
    _open_chat_page(browser_page)
    _constrain_message_list_width(browser_page, 280)

    selector = _message_selector(message.id)
    assert browser_page.locator(f"{selector} .jouzetsu-message-action").count() == 5
    layout = _message_action_layout_metrics(browser_page, selector)

    assert layout["footer_width"] <= 250
    assert abs(layout["actions_width"] - layout["footer_width"]) <= _SCROLL_RESTORE_TOLERANCE_PX
    assert layout["row_count"] == 1


def test_last_message_hides_the_delete_following_choice(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """The tail-delete action is unavailable for the final transcript message."""

    message = browser_server.state.active_chat.add_message("assistant", "Last reply")
    _open_chat_page(browser_page)

    browser_page.get_by_test_id(f"delete-message-{message.id}").click()
    dialog = browser_page.get_by_test_id("message-delete-dialog")

    expect(dialog).to_be_visible()
    expect(dialog.locator('[data-delete-choice-form="single"]')).to_be_visible()
    expect(dialog.locator('[data-delete-choice-form="following"]')).to_be_hidden()


@pytest.mark.parametrize(
    "manual_scroll_to_bottom",
    [False, True],
    ids=("automatic", "manual"),
)
def test_stream_completion_preserves_a_top_locked_response_position(
    browser_page: Page,
    browser_server: BrowserServer,
    manual_scroll_to_bottom: bool,
) -> None:
    """Completing a top-locked response must not reinterpret it as bottom-pinned."""

    completion_gate = _configure_paused_streaming_response(browser_server)
    try:
        _open_chat_page(browser_page)
        browser_page.evaluate("() => document.fonts.ready")
        browser_page.get_by_test_id("message-input").fill(
            "Generate a nearly long reply"
        )
        browser_page.get_by_test_id("send-message-button").click()
        browser_page.wait_for_function(
            "() => document.querySelector('article.is-streaming[data-message-id]')"
            "?.innerText.includes('deliberately long')",
        )
        expected = _message_position_metrics(
            browser_page, "article.is-streaming[data-message-id]"
        )
        assert abs(expected["message_top"]) <= _SCROLL_RESTORE_TOLERANCE_PX
        assert 0 < expected["gap"] <= 24

        if manual_scroll_to_bottom:
            _set_message_scroll_top(browser_page, expected["top"] + expected["gap"])
            expected = _message_position_metrics(
                browser_page, "article.is-streaming[data-message-id]"
            )
            assert expected["gap"] <= _SCROLL_RESTORE_TOLERANCE_PX

        completion_gate.set()
        browser_page.wait_for_function(
            "() => !document.querySelector('article.is-streaming[data-message-id]')"
        )
        _wait_for_animation_frames(browser_page)
        completed = _message_position_metrics(
            browser_page, "article.is-assistant[data-message-id]:last-child"
        )

        assert abs(completed["top"] - expected["top"]) <= _SCROLL_RESTORE_TOLERANCE_PX
        assert (
            abs(completed["message_top"] - expected["message_top"])
            <= _SCROLL_RESTORE_TOLERANCE_PX
        )
    finally:
        completion_gate.set()


@pytest.mark.parametrize(
    "manual_scroll",
    [False, True],
    ids=("automatic", "manual"),
)
def test_stream_completion_preserves_a_top_locked_response_anchor_with_history(
    browser_page: Page,
    browser_server: BrowserServer,
    manual_scroll: bool,
) -> None:
    """Earlier messages expanding at completion cannot move a top-locked response."""

    browser_page.set_viewport_size({"width": 640, "height": 720})
    _seed_transcript(browser_server, 6, "Earlier transcript message")
    completion_gate = _configure_paused_streaming_response(browser_server)
    try:
        _open_chat_page(browser_page)
        _constrain_message_list_width(browser_page, 320)
        browser_page.get_by_test_id("message-input").fill("Generate a nearly long reply")
        browser_page.get_by_test_id("send-message-button").click()
        browser_page.wait_for_selector("article.is-streaming[data-message-id]")
        _set_message_scroll_top(browser_page, _scroll_metrics(browser_page)["maximum"])
        browser_page.wait_for_function(
            "() => document.querySelector('article.is-streaming[data-message-id]')"
            "?.innerText.includes('deliberately long')",
        )
        expected = _message_position_metrics(
            browser_page, "article.is-streaming[data-message-id]"
        )
        assert abs(expected["message_top"]) <= _SCROLL_RESTORE_TOLERANCE_PX

        if manual_scroll:
            _set_message_scroll_top(
                browser_page, expected["top"] + _MANUAL_SCROLL_OFFSET_PX
            )
            expected = _message_position_metrics(
                browser_page, "article.is-streaming[data-message-id]"
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
            abs(completed["message_top"] - expected["message_top"])
            <= _SCROLL_RESTORE_TOLERANCE_PX
        )
    finally:
        completion_gate.set()


def test_message_context_menu_copies_and_forks_at_the_selected_message(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """The shared context menu applies actions to its selected transcript message."""

    _ = browser_server.state.active_chat.add_message("user", "Fork source")
    selected = browser_server.state.active_chat.add_message(
        "assistant", "Copy this selected reply"
    )
    _ = browser_server.state.active_chat.add_message("user", "Excluded after fork")
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


def test_fragment_replacement_reconciles_unchanged_messages(
    browser_page: Page, browser_server: BrowserServer
) -> None:
    """A full update keeps the transcript container and unchanged message nodes intact."""

    edited = browser_server.state.active_chat.add_message("user", "Original message")
    stable = browser_server.state.active_chat.add_message("assistant", "Stable reply")
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
    initial_fragment.value.finished()
    expect(page.get_by_test_id("message-list")).to_be_visible()
    _wait_for_animation_frames(page)


@pytest.mark.parametrize("viewport_width", [960, 390])
def test_colourway_editor_updates_owned_ui_and_light_mode(
    browser_page: Page,
    browser_server: BrowserServer,
    tmp_path: Path,
    viewport_width: int,
) -> None:
    """Saving four main colours updates real CSS without coupling role accents."""

    browser_page.set_viewport_size({"width": viewport_width, "height": 844})
    _ = browser_server.state.active_chat.add_message("user", "My message")
    _ = browser_server.state.active_chat.add_message("assistant", "Assistant reply")
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
    return {
        owner: cast(
            str,
            page.locator(selector).evaluate(
                "(element, property) => getComputedStyle(element).getPropertyValue(property)",
                property,
            ),
        )
        for owner, selector, property in targets
    }


def _seed_scrollable_transcript(server: BrowserServer) -> str:
    """Create a static transcript large enough to exercise the list scroll state."""

    return _seed_transcript(server, 12, _LONG_MESSAGE)


def _seed_transcript(server: BrowserServer, turn_count: int, content: str) -> str:
    """Add a predictable sequence of user and assistant messages to the active chat."""

    chat = server.state.active_chat
    last_message_id = ""
    for turn in range(turn_count):
        _ = chat.add_message("user", f"User turn {turn}: {content}")
        response = chat.add_message(
            "assistant", f"Assistant turn {turn}: {content}"
        )
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
