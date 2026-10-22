"""Image previews shared by saved messages and the composer draft."""

from __future__ import annotations

from typing import Final

from ...models import Chat, ImageAttachment, ImageMediaType
from ..html import HTML, A, Button, Div, Form, Img, Input, Label, Noscript, Small
from .icons import render_icon

_IMAGE_UPLOAD_FORM_ID: Final[str] = "composer-image-upload"


def render_image_gallery(
    images: tuple[ImageAttachment, ...], *, chat_id: str = ""
) -> HTML:
    """Show protected previews with original-image links and optional draft removal."""

    return Div(
        *(
            Div(
                A(
                    Img(
                        src=f"/images/{image.id}?thumbnail=1",
                        alt=image.name,
                        width=image.width,
                        height=image.height,
                        loading="lazy",
                    ),
                    href=f"/images/{image.id}",
                    target="_blank",
                    rel="noopener",
                    title=f"{image.name} · {image.width:,} × {image.height:,}",
                ),
                Small(image.name, cls="jouzetsu-image-name"),
                Button(
                    "Remove",
                    type="submit",
                    formaction=f"/chats/{chat_id}/images/{image.id}/remove",
                    cls="jouzetsu-button jouzetsu-image-remove",
                    aria_label=f"Remove {image.name}",
                    data_testid="remove-draft-image",
                )
                if chat_id
                else None,
                cls="jouzetsu-image-preview",
                data_image_id=image.id,
            )
            for image in images
        ),
        cls="jouzetsu-image-gallery",
        data_testid="image-gallery",
    )


def render_image_upload(chat: Chat) -> HTML:
    """Keep uploaded draft attachments on the server before sending a message."""

    return Form(
        render_image_gallery(chat.draft_images, chat_id=chat.id),
        Small(
            "Saving images…",
            hidden=True,
            role="status",
            cls="jouzetsu-form-help",
            data_image_upload_pending="true",
        ),
        Noscript(
            Button("Attach selected images", type="submit", cls="jouzetsu-button")
        ),
        action=f"/chats/{chat.id}/images",
        id=_IMAGE_UPLOAD_FORM_ID,
        method="post",
        enctype="multipart/form-data",
        cls="jouzetsu-image-upload-form",
        data_image_upload_form="true",
        data_chat_id=chat.id,
    )


def render_image_upload_control() -> HTML:
    """Attach an inline file picker to the separate multipart upload form."""

    return Label(
        render_icon("image"),
        Input(
            type="file",
            name="images",
            multiple=True,
            accept=",".join(media.value for media in ImageMediaType),
            form=_IMAGE_UPLOAD_FORM_ID,
            aria_label="Add images",
            cls="jouzetsu-image-file-input",
            data_testid="image-upload-input",
        ),
        title="Add images (or paste/drop)",
        cls="jouzetsu-button jouzetsu-composer-attachment",
        data_testid="add-images-button",
    )
