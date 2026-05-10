#!/usr/bin/env bash
# capvton-api 실행 스크립트
# fit_light_app/web/server.js 가 /tryon → http://localhost:8000 으로 프록시.

set -euo pipefail

cd "$(dirname "$0")/.."

# ── 환경변수 (필요 시 .env로 export) ──
export CAPVTON_HOST="${CAPVTON_HOST:-0.0.0.0}"
export CAPVTON_PORT="${CAPVTON_PORT:-8000}"
export CAPVTON_CKPT_DIR="${CAPVTON_CKPT_DIR:-$(pwd)/ckpts}"

# admin-web과 동일한 정적 디렉터리에 결과 이미지를 저장하면
# /uploads/fitting-results/<file>.jpg 가 그대로 서빙됨.
export CAPVTON_UPLOADS_DIR="${CAPVTON_UPLOADS_DIR:-/mnt/d/fit_light/admin-web/uploads/fitting-results}"
export CAPVTON_PUBLIC_URL_PREFIX="${CAPVTON_PUBLIC_URL_PREFIX:-/uploads/fitting-results}"

# Oracle Wallet 위치 (admin-web 와 공유)
export ORACLE_DB_USER="${ORACLE_DB_USER:-ADMIN}"
export ORACLE_DB_PASSWORD="${ORACLE_DB_PASSWORD:-Fitmate1234!}"
export ORACLE_DB_CONNECT_STRING="${ORACLE_DB_CONNECT_STRING:-fitlight_low}"
export ORACLE_DB_CONFIG_DIR="${ORACLE_DB_CONFIG_DIR:-/mnt/d/fit_light/admin-web/Wallet_fitlight}"
export ORACLE_DB_WALLET_LOCATION="${ORACLE_DB_WALLET_LOCATION:-$ORACLE_DB_CONFIG_DIR}"
export ORACLE_DB_WALLET_PASSWORD="${ORACLE_DB_WALLET_PASSWORD:-Fitmate1234!}"

exec uvicorn api.main:app \
    --host "$CAPVTON_HOST" \
    --port "$CAPVTON_PORT" \
    --workers 1 \
    --timeout-keep-alive 75
