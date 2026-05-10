"""CAPVirtualTryOn 싱글턴.

uvicorn workers=1 가정. GPU 메모리를 많이 쓰므로 멀티프로세스 로드를 피한다.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from api.config import settings

log = logging.getLogger("capvton.deps")

_lock = threading.Lock()
_instance = None


def get_vton():
    """CAPVirtualTryOn 인스턴스 반환. CAPVTON_DISABLE_MODEL=1이면 None."""
    global _instance
    if settings.disable_model:
        return None
    if _instance is None:
        with _lock:
            if _instance is None:
                from vton_script import CAPVirtualTryOn  # 지연 임포트 (CUDA init 비용)

                log.info("[Model] CAPVirtualTryOn 로드 시작 (ckpt_dir=%s)", settings.ckpt_dir)
                _instance = CAPVirtualTryOn(ckpt_dir=settings.ckpt_dir)
                log.info("[Model] CAPVirtualTryOn 로드 완료")
    return _instance
