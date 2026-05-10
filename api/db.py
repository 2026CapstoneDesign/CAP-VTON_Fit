"""Oracle DB 연결/INSERT 헬퍼.

admin-web (server.js)의 fitting_records 스키마와 동작을 그대로 따른다:
  - INSERT INTO fitting_records (...) RETURNING id
  - UPDATE app_users SET coin_balance, fitting_count
  - 트랜잭션은 호출자가 commit/rollback
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from api.config import settings

log = logging.getLogger("capvton.db")

try:
    import oracledb  # type: ignore
except ImportError:  # pragma: no cover
    oracledb = None  # type: ignore

_pool = None
_pool_lock = threading.Lock()


class DBUnavailable(RuntimeError):
    """oracledb 미설치 또는 연결 실패."""


def _build_pool():
    if oracledb is None:
        raise DBUnavailable("oracledb 패키지가 설치되어 있지 않습니다 (pip install oracledb).")

    return oracledb.create_pool(
        user=settings.db_user,
        password=settings.db_password,
        dsn=settings.db_dsn,
        config_dir=str(settings.db_config_dir),
        wallet_location=str(settings.db_wallet_location),
        wallet_password=settings.db_wallet_password,
        min=settings.db_pool_min,
        max=settings.db_pool_max,
        increment=1,
    )


def init_pool() -> bool:
    """Eager 초기화. 실패해도 서버는 계속 동작 (require_db=True면 예외)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            return True
        try:
            _pool = _build_pool()
            log.info("[DB] Oracle pool ready (%s @ %s)", settings.db_user, settings.db_dsn)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("[DB] 풀 초기화 실패: %s", exc)
            if settings.require_db:
                raise
            return False


def is_ready() -> bool:
    return _pool is not None


@contextmanager
def get_connection() -> Iterator["oracledb.Connection"]:
    if _pool is None:
        if not init_pool():
            raise DBUnavailable("DB 연결 풀을 사용할 수 없습니다.")
    conn = _pool.acquire()
    try:
        yield conn
    finally:
        try:
            _pool.release(conn)
        except Exception:  # noqa: BLE001
            pass


def insert_fitting_record(
    user_id: str,
    product_id: str,
    result_image_path: str,
    coin_used: int,
) -> dict:
    """fitting_records INSERT + 코인 차감 + fitting_count 증가.

    admin-web의 POST /api/app/users/:id/fittings 와 동일한 트랜잭션 로직.
    """
    if oracledb is None:
        raise DBUnavailable("oracledb 미설치")

    with get_connection() as conn:
        cur = conn.cursor()
        try:
            # 1) 사용자 검증
            cur.execute(
                "SELECT id, coin_balance, active FROM app_users WHERE id = :user_id",
                {"user_id": user_id},
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"user_id={user_id} 를 찾을 수 없습니다.")
            _, coin_balance, active = row
            if int(active or 0) != 1:
                raise PermissionError("비활성화된 계정입니다.")
            if int(coin_balance or 0) < coin_used:
                raise ValueError("코인이 부족합니다.")

            # 2) 상품 검증
            cur.execute(
                "SELECT id, active FROM products WHERE id = :product_id",
                {"product_id": product_id},
            )
            prod = cur.fetchone()
            if prod is None or int(prod[1] or 0) != 1:
                raise ValueError(f"product_id={product_id} (활성 상품) 을 찾을 수 없습니다.")

            # 3) INSERT … RETURNING id
            record_id_var = cur.var(oracledb.DB_TYPE_NUMBER)
            cur.execute(
                """
                INSERT INTO fitting_records (
                    user_id, product_id, result_image_path, coin_used, created_at
                ) VALUES (
                    :user_id, :product_id, :result_image_path, :coin_used, SYSTIMESTAMP
                )
                RETURNING id INTO :record_id
                """,
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "result_image_path": result_image_path,
                    "coin_used": coin_used,
                    "record_id": record_id_var,
                },
            )
            new_id = int(record_id_var.getvalue()[0])

            # 4) 코인 차감 + fitting_count 증가
            cur.execute(
                """
                UPDATE app_users
                SET coin_balance = coin_balance - :coin_delta,
                    fitting_count = fitting_count + 1
                WHERE id = :user_id
                """,
                {"coin_delta": coin_used, "user_id": user_id},
            )

            conn.commit()
            new_balance = int(coin_balance) - int(coin_used)
            return {
                "record_id": new_id,
                "user_id": user_id,
                "product_id": product_id,
                "result_image_path": result_image_path,
                "coin_used": int(coin_used),
                "coin_balance": new_balance,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
