"""Validated original images and small previews stored by content digest."""

from __future__ import annotations

import base64
import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

from PIL import Image, ImageOps, UnidentifiedImageError

from .atomic_write import atomic_write_bytes
from .models import ImageAttachment, ImageMediaType

_IMAGE_FORMATS: Final[dict[str, ImageMediaType]] = {
    "JPEG": ImageMediaType.JPEG,
    "PNG": ImageMediaType.PNG,
    "WEBP": ImageMediaType.WEBP,
}
_PREVIEW_SIZE: Final[tuple[int, int]] = (320, 240)


def _write_if_changed(path: Path, data: bytes) -> None:
    """Repair damaged files without rewriting identical shared originals."""

    try:
        if path.read_bytes() == data:
            return
    except FileNotFoundError:
        pass
    atomic_write_bytes(path, data)


@dataclass(frozen=True, slots=True)
class ImageLimits:
    """Central upload policy, ready to expose through configuration later."""

    max_image_bytes: int = 64 * 1024 * 1024
    max_batch_bytes: int = 512 * 1024 * 1024
    max_image_pixels: int = 100_000_000
    max_images_per_message: int = 32

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.max_image_bytes,
                self.max_batch_bytes,
                self.max_image_pixels,
                self.max_images_per_message,
            )
        ):
            raise ValueError("image limits must be positive integers")

    def validate_image_count(self, count: int) -> None:
        """Apply the same per-message count policy to uploads and merged turns."""

        if count > self.max_images_per_message:
            raise ValueError(
                f"a message can contain at most {self.max_images_per_message} images"
            )


class ImageStorage:
    """Keep originals shared safely by drafts, messages, forks, undo, and trash."""

    def __init__(self, directory: Path, limits: ImageLimits | None = None) -> None:
        self.directory: Path = directory
        self.limits: ImageLimits = limits or ImageLimits()

    def store(self, data: bytes, name: str) -> ImageAttachment:
        """Decode and validate an image before publishing its original and preview."""

        if not data or len(data) > self.limits.max_image_bytes:
            raise ValueError(
                f"each image must be between 1 and {self.limits.max_image_bytes:,} bytes"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    media_type: ImageMediaType | None = _IMAGE_FORMATS.get(
                        source.format or ""
                    )
                    if media_type is None:
                        raise ValueError("images must be JPEG, PNG, or WebP")
                    width, height = source.size
                    if width * height > self.limits.max_image_pixels:
                        raise ValueError("image dimensions exceed the pixel limit")
                    if getattr(source, "is_animated", False):
                        raise ValueError("animated images are not supported")
                    source.verify()
                with Image.open(BytesIO(data)) as source:
                    source.load()
                    ImageOps.exif_transpose(source, in_place=True)
                    source.thumbnail(_PREVIEW_SIZE)
                    preview_data = BytesIO()
                    source.convert("RGB").save(preview_data, format="JPEG", quality=85)
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as exc:
            raise ValueError(
                "image is invalid, damaged, or too large to decode"
            ) from exc

        clean_name: str = Path(name.replace("\\", "/")).name
        clean_name = (
            "".join(
                character for character in clean_name if character.isprintable()
            ).strip()
            or "image"
        )
        image = ImageAttachment(
            id=hashlib.sha256(data).hexdigest(),
            name=clean_name,
            media_type=media_type,
            width=width,
            height=height,
            size_bytes=len(data),
        )
        original_path: Path = self.path_for(image)
        preview_path: Path = self.path_for(image, thumbnail=True)
        _write_if_changed(original_path, data)
        _write_if_changed(preview_path, preview_data.getvalue())
        return image

    def path_for(self, image: ImageAttachment, *, thumbnail: bool = False) -> Path:
        """Resolve only validated content identifiers, never uploaded path names."""

        suffix: str = (
            "preview.jpg" if thumbnail else image.media_type.value.split("/")[1]
        )
        return self.directory / f"{image.id}.{suffix}"

    def validate_available(self, images: Sequence[ImageAttachment]) -> None:
        """Reject missing originals before mutating a transcript for generation."""

        for image in {image.id: image for image in images}.values():
            _ = self._read_original(image)

    def _read_original(self, image: ImageAttachment) -> bytes:
        try:
            data: bytes = self.path_for(image).read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(
                f"original image is missing or damaged: {image.name}"
            ) from exc
        if (
            len(data) != image.size_bytes
            or hashlib.sha256(data).hexdigest() != image.id
        ):
            raise ValueError(f"original image is missing or damaged: {image.name}")
        return data

    def data_url(self, image: ImageAttachment) -> str:
        """Encode full-resolution pixels for LM Studio's image content part.

        LM Studio 0.4.26 rejects WebP data URLs on the compatibility endpoint.
        PNG transport preserves decoded WebP pixels and transparency losslessly;
        the uploaded original remains untouched.
        """

        data: bytes = self._read_original(image)
        media_type: ImageMediaType = image.media_type
        if media_type is ImageMediaType.WEBP:
            with Image.open(BytesIO(data)) as source:
                output = BytesIO()
                ImageOps.exif_transpose(source, in_place=True)
                source.save(output, format="PNG")
                data = output.getvalue()
            media_type = ImageMediaType.PNG
        encoded: str = base64.b64encode(data).decode("ascii")
        return f"data:{media_type.value};base64,{encoded}"
