"""환경 변수 기반 설정.

admin-web의 .env 또는 OS 환경변수와 동일한 키를 사용한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


@dataclass
class Settings:
    # ── 서버 ──
    host: str = field(default_factory=lambda: os.environ.get("CAPVTON_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("CAPVTON_PORT", 8000))

    # ── 모델 체크포인트 ──
    ckpt_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CAPVTON_CKPT_DIR",
            str(Path(__file__).resolve().parent.parent / "ckpts"),
        )
    )

    # ── 결과 이미지 저장 디렉터리 ──
    # admin-web과 동일한 경로를 마운트해두면 그대로 서빙 가능.
    # 기본값: D:/fit_light/admin-web/uploads/fitting-results 가정.
    uploads_dir: Path = field(
        default_factory=lambda: _env_path(
            "CAPVTON_UPLOADS_DIR",
            Path("/mnt/d/fit_light/admin-web/uploads/fitting-results"),
        )
    )
    # DB에 저장될 public URL prefix (admin-web 정적 라우트와 일치)
    public_url_prefix: str = field(
        default_factory=lambda: os.environ.get(
            "CAPVTON_PUBLIC_URL_PREFIX", "/uploads/fitting-results"
        )
    )

    # ── Oracle DB ──
    db_user: str = field(default_factory=lambda: os.environ.get("ORACLE_DB_USER", "ADMIN"))
    db_password: str = field(
        default_factory=lambda: os.environ.get("ORACLE_DB_PASSWORD", "Fitmate1234!")
    )
    db_dsn: str = field(
        default_factory=lambda: os.environ.get("ORACLE_DB_CONNECT_STRING", "fitlight_low")
    )
    db_config_dir: Path = field(
        default_factory=lambda: _env_path(
            "ORACLE_DB_CONFIG_DIR",
            Path("/mnt/d/fit_light/admin-web/Wallet_fitlight"),
        )
    )
    db_wallet_location: Path = field(
        default_factory=lambda: _env_path(
            "ORACLE_DB_WALLET_LOCATION",
            Path("/mnt/d/fit_light/admin-web/Wallet_fitlight"),
        )
    )
    db_wallet_password: str = field(
        default_factory=lambda: os.environ.get("ORACLE_DB_WALLET_PASSWORD", "Fitmate1234!")
    )
    db_pool_min: int = field(default_factory=lambda: _env_int("ORACLE_DB_POOL_MIN", 1))
    db_pool_max: int = field(default_factory=lambda: _env_int("ORACLE_DB_POOL_MAX", 4))

    # ── 동작 옵션 ──
    # DB 미가용 시(워크스테이션에서 모델만 띄우는 경우) 서버는 계속 살아있음.
    require_db: bool = field(
        default_factory=lambda: os.environ.get("CAPVTON_REQUIRE_DB", "0") == "1"
    )
    # 모델 로드 비활성 — 개발/테스트 시 헬스체크만 켜고 싶을 때 사용.
    disable_model: bool = field(
        default_factory=lambda: os.environ.get("CAPVTON_DISABLE_MODEL", "0") == "1"
    )
    default_coin_used: int = field(
        default_factory=lambda: _env_int("CAPVTON_DEFAULT_COIN_USED", 10)
    )

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
