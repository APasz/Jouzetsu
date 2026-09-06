"""Reusable accessible tab controls for dialog content."""

from __future__ import annotations

from dataclasses import dataclass

from ..html import HTML, Button, Div, HtmlChild


@dataclass(frozen=True, slots=True)
class DialogTab:
    """One named, labelled tab in a dialog."""

    name: str
    label: str


def render_dialog_tab_list(
    dialog_id: str,
    tabs: tuple[DialogTab, ...],
    active_tab: str,
    *,
    label: str,
) -> HTML:
    """Render the controls for a server-selected dialog tab."""

    return Div(
        *(
            Button(
                tab.label,
                type="button",
                id=_tab_id(dialog_id, tab),
                role="tab",
                aria_controls=_panel_id(dialog_id, tab),
                aria_selected="true" if tab.name == active_tab else "false",
                tabindex=0 if tab.name == active_tab else -1,
                cls="jouzetsu-tab",
                data_tab_target=_panel_id(dialog_id, tab),
                data_testid=f"{dialog_id}-{tab.name}-tab",
            )
            for tab in tabs
        ),
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
