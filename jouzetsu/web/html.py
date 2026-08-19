# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Typed boundary around FastHTML's dynamically generated element factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]
from fasthtml.common import (  # pyright: ignore[reportMissingTypeStubs]
    H1 as _H1,
    H2 as _H2,
    H3 as _H3,
    A as _A,
    Article as _Article,
    Aside as _Aside,
    Button as _Button,
    Details as _Details,
    Dialog as _Dialog,
    Div as _Div,
    Footer as _Footer,
    Form as _Form,
    Header as _Header,
    Img as _Img,
    Input as _Input,
    Label as _Label,
    Link as _Link,
    Main as _Main,
    Meta as _Meta,
    Option as _Option,
    P as _P,
    Pre as _Pre,
    Section as _Section,
    Select as _Select,
    Script as _Script,
    Small as _Small,
    Span as _Span,
    Summary as _Summary,
    Textarea as _Textarea,
)


type HTML = FT
type HtmlChild = HTML | str | int | float | bool | None


class HtmlTag(Protocol):
    """One FastHTML element factory with deliberately broad HTML attributes."""

    def __call__(self, *children: HtmlChild, **attributes: object) -> HTML: ...


type _RawTag = Callable[..., object]


def _tag(factory: object) -> HtmlTag:
    """Cast one dynamically declared FastHTML tag at the integration boundary."""

    raw_factory: _RawTag = cast(_RawTag, factory)

    def render(*children: HtmlChild, **attributes: object) -> HTML:
        return cast(HTML, raw_factory(*children, **attributes))

    return render


A: HtmlTag = _tag(_A)
Article: HtmlTag = _tag(_Article)
Aside: HtmlTag = _tag(_Aside)
Button: HtmlTag = _tag(_Button)
Details: HtmlTag = _tag(_Details)
Dialog: HtmlTag = _tag(_Dialog)
Div: HtmlTag = _tag(_Div)
Footer: HtmlTag = _tag(_Footer)
Form: HtmlTag = _tag(_Form)
H1: HtmlTag = _tag(_H1)
H2: HtmlTag = _tag(_H2)
H3: HtmlTag = _tag(_H3)
Header: HtmlTag = _tag(_Header)
Img: HtmlTag = _tag(_Img)
Input: HtmlTag = _tag(_Input)
Label: HtmlTag = _tag(_Label)
Link: HtmlTag = _tag(_Link)
Main: HtmlTag = _tag(_Main)
Meta: HtmlTag = _tag(_Meta)
Option: HtmlTag = _tag(_Option)
P: HtmlTag = _tag(_P)
Pre: HtmlTag = _tag(_Pre)
Section: HtmlTag = _tag(_Section)
Select: HtmlTag = _tag(_Select)
Script: HtmlTag = _tag(_Script)
Small: HtmlTag = _tag(_Small)
Span: HtmlTag = _tag(_Span)
Summary: HtmlTag = _tag(_Summary)
Textarea: HtmlTag = _tag(_Textarea)
