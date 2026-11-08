"""Image persistence, history, model compatibility, and protected upload coverage."""

from __future__ import annotations

import base64
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from image_support import image_bytes
from PIL import Image

from jouzetsu.image_storage import ImageLimits, ImageStorage
from jouzetsu.models import CHAT_SCHEMA_VERSION, Chat, ImageMediaType, Message


@pytest.fixture()
def image_storage(tmp_path: Path) -> ImageStorage:
    return ImageStorage(tmp_path / "images")


@pytest.mark.parametrize(
    "format,media_type",
    (
        ("PNG", ImageMediaType.PNG),
        ("JPEG", ImageMediaType.JPEG),
        ("WEBP", ImageMediaType.WEBP),
    ),
)
def test_image_storage_preserves_originals_and_decodes_actual_format(
    tmp_path: Path,
    format: Literal["PNG", "JPEG", "WEBP"],
    media_type: ImageMediaType,
) -> None:
    data = image_bytes(format=format)
    storage = ImageStorage(tmp_path)
    image = storage.store(data, "../../picture.wrong-extension")

    assert image.media_type is media_type
    assert image.name == "picture.wrong-extension"
    assert storage.path_for(image).read_bytes() == data
    transport_type = (
        ImageMediaType.PNG if media_type is ImageMediaType.WEBP else media_type
    )
    data_url = storage.data_url(image)
    assert data_url.startswith(f"data:{transport_type.value};base64,")
    with (
        Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as transported,
        Image.open(BytesIO(data)) as original,
    ):
        assert transported.size == original.size
        assert (
            transported.convert("RGBA").tobytes() == original.convert("RGBA").tobytes()
        )
    with Image.open(storage.path_for(image, thumbnail=True)) as preview:
        assert preview.width <= 320 and preview.height <= 240
    assert storage.store(data, image.name) == image
    assert len(list(tmp_path.iterdir())) == 2


def test_webp_transport_preserves_transparency(image_storage: ImageStorage) -> None:
    data = BytesIO()
    Image.new("RGBA", (37, 29), (20, 40, 60, 128)).save(
        data, format="WEBP", lossless=True
    )
    storage = image_storage
    image = storage.store(data.getvalue(), "transparent.webp")
    data_url = storage.data_url(image)
    assert data_url.startswith("data:image/png;base64,")
    with Image.open(
        BytesIO(base64.b64decode(data_url.split(",", 1)[1]))
    ) as transported:
        assert transported.size == (37, 29)
        assert transported.convert("RGBA").getpixel((0, 0)) == (20, 40, 60, 128)
    assert storage.path_for(image).read_bytes() == data.getvalue()


@pytest.mark.parametrize(
    "data", (b"not an image", b"<svg></svg>", image_bytes(format="GIF"))
)
def test_invalid_and_unsupported_images_never_reach_storage(
    tmp_path: Path, data: bytes
) -> None:
    with pytest.raises(ValueError, match="image|JPEG"):
        _ = ImageStorage(tmp_path).store(data, "image.png")
    assert list(tmp_path.iterdir()) == []


def test_image_limits_and_missing_originals_fail_explicitly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _ = ImageLimits(max_images_per_message=0)
    with pytest.raises(ValueError, match="each image"):
        _ = ImageStorage(tmp_path, ImageLimits(max_image_bytes=2)).store(
            image_bytes(), "large.png"
        )
    with pytest.raises(ValueError, match="pixel limit"):
        _ = ImageStorage(tmp_path, ImageLimits(max_image_pixels=2)).store(
            image_bytes(), "large.png"
        )
    storage = ImageStorage(tmp_path)
    image = storage.store(image_bytes(), "original.png")
    _ = storage.path_for(image).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="damaged"):
        _ = storage.data_url(image)
    with pytest.raises(ValueError, match="missing or damaged"):
        storage.validate_available((image,))


def test_reupload_repairs_damaged_originals_and_previews(tmp_path: Path) -> None:
    storage = ImageStorage(tmp_path)
    data = image_bytes()
    image = storage.store(data, "original.png")
    preview_path = storage.path_for(image, thumbnail=True)
    preview_data = preview_path.read_bytes()
    _ = storage.path_for(image).write_bytes(b"x" * len(data))
    _ = preview_path.write_bytes(b"damaged preview")

    assert storage.store(data, image.name) == image
    assert storage.path_for(image).read_bytes() == data
    assert preview_path.read_bytes() == preview_data
    storage.validate_available((image,))


def test_images_round_trip_and_old_chats_load_without_attachments(
    image_storage: ImageStorage,
) -> None:
    image = image_storage.store(image_bytes(), "saved.png")
    chat = Chat(draft_images=(image,), messages=[Message(role="user", images=(image,))])
    assert Chat.from_dict(chat.to_dict()) == chat
    assert chat.to_dict()["schema_version"] == CHAT_SCHEMA_VERSION
    old: dict[str, object] = {
        "schema_version": 1,
        "messages": [{"role": "user", "content": "Old message"}],
    }
    loaded = Chat.from_dict(old)
    assert loaded.draft_images == () and loaded.messages[0].images == ()
    with pytest.raises(ValueError, match="only user"):
        _ = Message(role="assistant", images=(image,))
    with pytest.raises(ValueError, match="SHA-256"):
        _ = replace(image, id="../escape")
