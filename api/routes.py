"""/tryon — 가상 피팅 추론 + (선택) Oracle DB 저장."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from api import db
from api.config import settings
from api.deps import get_vton
from api.storage import image_to_jpeg_bytes, save_result_image

log = logging.getLogger("capvton.api")

router = APIRouter()

ALLOWED_GARMENTS = {"upper_body", "lower_body", "dresses", "short_sleeve", "shorts"}


def _save_upload(tmp_dir: Path, upload: UploadFile, name: str) -> Path:
    if upload is None:
        raise HTTPException(status_code=400, detail=f"{name} 파일이 필요합니다.")
    suffix = Path(upload.filename or f"{name}.jpg").suffix or ".jpg"
    dst = tmp_dir / f"{name}{suffix}"
    with dst.open("wb") as f:
        f.write(upload.file.read())
    return dst


def _parse_optional_json(raw: Optional[str], field_name: str) -> Optional[dict]:
    if raw is None or raw.strip() == "":
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} JSON 파싱 실패: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"{field_name}는 객체(dict)여야 합니다.")
    return data


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": (not settings.disable_model),
        "db_ready": db.is_ready(),
    }


@router.post("/tryon")
async def tryon(
    src_image: UploadFile = File(...),
    ref_image: UploadFile = File(...),
    garment_type: str = Form("upper_body"),
    step: int = Form(20),
    seed: int = Form(42),
    vt_model_type: str = Form("viton_hd"),
    vt_repaint: bool = Form(False),
    # ── (선택) Oracle DB 저장용 ──
    user_id: Optional[str] = Form(None),
    product_id: Optional[str] = Form(None),
    coin_used: Optional[int] = Form(None),
    # ── (선택) Fit 추론 ──
    user_measurements: Optional[str] = Form(None),     # JSON string
    garment_measurements: Optional[str] = Form(None),  # JSON string
    preferred_fit: Optional[str] = Form(None),
    # ── 응답 형식 ──
    response_format: str = Form("image"),  # "image" | "json"
):
    """가상 피팅 1회 실행.

    Form:
        src_image, ref_image (필수): 사용자/의류 이미지
        garment_type: upper_body | lower_body | dresses | short_sleeve | shorts
        step, seed, vt_model_type, vt_repaint: 모델 파라미터
        user_id, product_id, coin_used: 모두 주어지면 fitting_records INSERT
        user_measurements / garment_measurements: JSON string. 있으면 핏 판정 수행
        preferred_fit: slim | regular | loose | oversized
        response_format: image=JPEG bytes, json={imageBase64, recordId, ...}

    Response (image, default):
        Content-Type: image/jpeg
        Headers (DB 저장 시):
          X-Record-Id, X-Result-Path, X-Coin-Balance
    """
    if garment_type not in ALLOWED_GARMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"garment_type은 {sorted(ALLOWED_GARMENTS)} 중 하나여야 합니다.",
        )

    coin_used_value = (
        coin_used if coin_used is not None and coin_used > 0 else settings.default_coin_used
    )
    persist_to_db = bool(user_id) and bool(product_id)

    user_meas_dict = _parse_optional_json(user_measurements, "user_measurements")
    garment_meas_dict = _parse_optional_json(garment_measurements, "garment_measurements")

    # ── 치수 수신 현황 로깅 ──────────────────────────────────────────────
    if user_meas_dict:
        log.info("[/tryon] ✅ 사용자 신체 치수 수신: %s", list(user_meas_dict.keys()))
    else:
        log.warning(
            "[/tryon] ❌ 사용자 신체 치수 누락 — raw=%s (클라이언트가 user_measurements를 전송하지 않음)",
            repr(user_measurements)[:120],
        )

    if garment_meas_dict:
        log.info("[/tryon] ✅ 의류 치수 수신: %s", list(garment_meas_dict.keys()))
    else:
        log.warning(
            "[/tryon] ❌ 의류 치수 누락 — raw=%s (클라이언트가 garment_measurements를 전송하지 않음)",
            repr(garment_measurements)[:120],
        )

    if not user_meas_dict and not garment_meas_dict:
        log.warning("[/tryon] ⚠️ 신체·의류 치수 모두 누락 → 핏 추론 건너뜀")
    elif not user_meas_dict or not garment_meas_dict:
        missing = "신체 치수" if not user_meas_dict else "의류 치수"
        log.warning("[/tryon] ⚠️ %s 누락 → 핏 추론 건너뜀 (둘 다 있어야 핏 추론 실행됨)", missing)

    vton = get_vton()
    if vton is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다 (DISABLE_MODEL).")

    # 업로드 파일을 임시 디렉터리에 저장 → CAPVirtualTryOn은 path 기반
    with tempfile.TemporaryDirectory(prefix="capvton_") as tmp:
        tmp_dir = Path(tmp)
        src_path = _save_upload(tmp_dir, src_image, "src")
        ref_path = _save_upload(tmp_dir, ref_image, "ref")

        loop = asyncio.get_running_loop()

        def _run():
            return vton.capvton_predict(
                src_image_path=str(src_path),
                ref_image_path=str(ref_path),
                control_type="virtual_tryon",
                vt_model_type=vt_model_type,
                vt_garment_type=garment_type,
                vt_repaint=vt_repaint,
                step=step,
                seed=seed,
                user_measurements=user_meas_dict,
                garment_measurements=garment_meas_dict,
                preferred_fit=preferred_fit,
            )

        try:
            result = await loop.run_in_executor(None, _run)
        except Exception as exc:  # noqa: BLE001
            log.exception("[/tryon] 추론 실패")
            raise HTTPException(status_code=500, detail=f"추론 실패: {exc}") from exc

    # capvton_predict는 fit_outputs 유무에 따라 tuple 또는 dict를 반환
    if isinstance(result, tuple):
        gen_image: Image.Image = result[0]
    elif isinstance(result, dict):
        gen_image = result["generated_image"]
    else:
        raise HTTPException(status_code=500, detail="알 수 없는 추론 결과 형식.")

    record_id: Optional[int] = None
    public_url: Optional[str] = None
    coin_balance_after: Optional[int] = None

    if persist_to_db:
        try:
            _abs_path, public_url = save_result_image(gen_image, ext="jpg")
            db_row = db.insert_fitting_record(
                user_id=user_id,
                product_id=product_id,
                result_image_path=public_url,
                coin_used=coin_used_value,
            )
            record_id = db_row["record_id"]
            coin_balance_after = db_row["coin_balance"]
        except db.DBUnavailable as exc:
            log.warning("[/tryon] DB 미가용으로 저장 건너뜀: %s", exc)
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("[/tryon] DB 저장 실패")
            raise HTTPException(status_code=500, detail=f"DB 저장 실패: {exc}") from exc

    if response_format == "json":
        fit_report_data = None
        if isinstance(result, dict) and result.get("fit_report") is not None:
            try:
                fit_report_data = result["fit_report"].to_dict()
                log.info(
                    "[/tryon] fitReport 생성 완료 — 종합점수=%.0f%% 사이즈추천=%s 부위=%s",
                    fit_report_data.get("overallScore", 0) * 100,
                    fit_report_data.get("sizeRecommendation", "-"),
                    list(fit_report_data.get("parts", {}).keys()),
                )
            except Exception as exc:
                log.warning("[/tryon] fitReport 직렬화 실패: %s", exc)
        else:
            log.info("[/tryon] fitReport 없음 (신체·의류 치수 미전송 또는 fit 추론 미실행)")
        body = {
            "imageBase64": base64.b64encode(image_to_jpeg_bytes(gen_image)).decode("ascii"),
            "recordId": record_id,
            "resultPath": public_url,
            "coinBalance": coin_balance_after,
            "coinUsed": coin_used_value if persist_to_db else 0,
            "fitReport": fit_report_data,
        }
        return JSONResponse(body)

    headers: dict[str, str] = {}
    if record_id is not None:
        headers["X-Record-Id"] = str(record_id)
    if public_url is not None:
        headers["X-Result-Path"] = public_url
    if coin_balance_after is not None:
        headers["X-Coin-Balance"] = str(coin_balance_after)

    return Response(
        content=image_to_jpeg_bytes(gen_image),
        media_type="image/jpeg",
        headers=headers,
    )
