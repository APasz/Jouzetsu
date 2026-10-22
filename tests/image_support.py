"""Small real encoded images shared by vision and browser tests."""

from __future__ import annotations

from io import BytesIO
from typing import Literal

from PIL import Image


def image_bytes(
    *,
    format: Literal["PNG", "JPEG", "WEBP", "GIF"] = "PNG",
    color: str = "red",
    size: tuple[int, int] = (48, 32),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=format)
    return output.getvalue()
