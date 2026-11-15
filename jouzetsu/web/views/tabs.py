"""Reusable accessible tab controls for dialog content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..html import HTML, Button, Div, HtmlChild, Small, Span

type TabIndicator = Literal["success", "error", "warning", "neutral"]


@dataclass(frozen=True, slots=True)
class DialogTab:
    """One named, labelled tab in a dialog."""

    name: str
    label: str
    detail: str = ""
    indicator: TabIndicator | None = None
    indicator_subdued: bool = False


def render_dialog_tab_list(
    dialog_id: str,
    tabs: tuple[DialogTab, ...],
    active_tab: str,
    *,
    label: str,
    pinned_tab: str = "",
) -> HTML:
    """Render the controls for a server-selected dialog tab."""

    return Div(
        *(
            Button(
                tab.label,
                Span(
                    cls="jouzetsu-tab-indicator is-subdued" if tab.indicator_subdued else "jouzetsu-tab-indicator",
                    data_tab_indicator=tab.indicator,
                    aria_hidden="true",
                )
                if tab.indicator is not None
                else Small(tab.detail, cls="jouzetsu-tab-detail")
                if tab.detail
                else None,
                type="button",
                id=_tab_id(dialog_id, tab),
                role="tab",
                aria_controls=_panel_id(dialog_id, tab),
                aria_selected="true" if tab.name == active_tab else "false",
                aria_label=f"{tab.label}: {tab.detail}" if tab.indicator else None,
                title=tab.detail if tab.indicator else None,
                tabindex=0 if tab.name == active_tab else -1,
                cls="jouzetsu-tab",
                data_tab_target=_panel_id(dialog_id, tab),
                data_tab_name=tab.name,
                data_tab_pinned="true" if tab.name == pinned_tab else None,
                data_testid=f"{dialog_id}-{tab.name}-tab",
            )
            for tab in tabs
        ),
        id=f"{dialog_id}-tabs",
        role="tablist",
        aria_label=label,
        cls="jouzetsu-tab-list",
        data_tab_list="true",
    )


def render_dialog_tab_panel(
    dialog_id: str,
    tab: DialogTab,
    active_tab: str,
    *contents: HtmlChild,
) -> HTML:
    """Render one dialog tab panel, hidden unless it is active."""

    panel_id: str = _panel_id(dialog_id, tab)
    return Div(
        *contents,
        id=panel_id,
        role="tabpanel",
        aria_labelledby=_tab_id(dialog_id, tab),
        hidden=tab.name != active_tab,
        cls="jouzetsu-tab-panel",
        data_tab_panel=panel_id,
        data_testid=panel_id,
    )


def _tab_id(dialog_id: str, tab: DialogTab) -> str:
    return f"{dialog_id}-{tab.name}-tab"


def _panel_id(dialog_id: str, tab: DialogTab) -> str:
    return f"{dialog_id}-{tab.name}-panel"
