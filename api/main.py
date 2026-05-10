"""FastAPI 진입점.

실행:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

환경변수는 api/config.py 참고.
fit_light_app/web/server.js 가 /tryon → http://localhost:8000 으로 프록시.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.config import settings
from api.deps import get_vton
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("capvton.main")

app = FastAPI(title="capvton-api", version="0.1.0")

# fit_light_app/web 프록시 뒤에서 호출되므로 보통 CORS 불필요하지만,
# 직접 호출(개발용)도 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Record-Id", "X-Result-Path", "X-Coin-Balance"],
)

app.include_router(router)


@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    resp = await call_next(request)
    dur = (time.perf_counter() - start) * 1000
    log.info("%s %s → %s (%.0fms)", request.method, request.url.path, resp.status_code, dur)
    return resp


@app.on_event("startup")
async def _startup():
    settings.ensure_dirs()
    db.init_pool()  # 실패해도 서버는 살아있음 (require_db=True면 예외)
    if not settings.disable_model:
        # 모델은 첫 요청 시 로드. 사전로드 원하면 아래 주석 해제.
        # get_vton()
        pass
    log.info(
        "[startup] uploads_dir=%s public_url_prefix=%s db_ready=%s",
        settings.uploads_dir,
        settings.public_url_prefix,
        db.is_ready(),
    )
