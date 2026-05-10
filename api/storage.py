"""결과 이미지를 admin-web이 정적 서빙하는 디렉터리에 저장."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Tuple

from PIL import Image

from api.config import settings


def save_result_image(image: Image.Image, ext: str = "jpg") -> Tuple[Path, str]:
    """저장된 파일의 (절대경로, public URL) 반환.

    public URL은 admin-web의 /uploads/fitting-results/ 라우트와 일치.
    """
    settings.ensure_dirs()
    filename = f"{uuid.uuid4()}.{ext}"
    abs_path = settings.uploads_dir / filename
    save_kwargs = {"quality": 92} if ext.lower() in {"jpg", "jpeg"} else {}
    fmt = "JPEG" if ext.lower() in {"jpg", "jpeg"} else ext.upper()
    image.convert("RGB").save(abs_path, format=fmt, **save_kwargs)
    public_url = f"{settings.public_url_prefix.rstrip('/')}/{filename}"
    return abs_path, public_url


def image_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
