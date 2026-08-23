from __future__ import annotations

from jouzetsu.config import SpellingReplacement
from jouzetsu.text_processing import apply_case_preserving_word_replacements


def test_case_preserving_word_replacements_keep_common_case_shapes() -> None:
    replacements = [SpellingReplacement("mom", "mum")]

    result = apply_case_preserving_word_replacements("mom Mom MOM MoM", replacements)

    assert result == "mum Mum MUM MuM"


def test_word_replacements_do_not_replace_inside_identifiers() -> None:
    replacements = [SpellingReplacement("color", "colour")]

    result = apply_case_preserving_word_replacements(
        "color my_color color2 color-coded", replacements
    )

    assert result == "colour my_color color2 colour-coded"
