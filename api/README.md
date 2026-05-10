# capvton-api

`CAPVirtualTryOn`을 HTTP로 노출하는 FastAPI 서버.

```
fit_light_app/web/server.js
        │ /tryon   (proxy)
        ▼
http://localhost:8000      ← 이 서버
        │
        ├── 모델: vton_script.CAPVirtualTryOn (GPU)
        └── DB:   ORACLE fitting_records  (admin-web과 동일 wallet 공유)
```

## 설치

```bash
pip install -r requirements-api.txt
```

## 실행

```bash
# 1) 환경변수만 export 하고 직접 실행
bash scripts/run_capvton_api.sh

# 2) 또는 uvicorn 직접
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CAPVTON_PORT` | `8000` | 서버 포트 (web/server.js 프록시 타깃) |
| `CAPVTON_CKPT_DIR` | `<repo>/ckpts` | 모델 체크포인트 디렉터리 |
| `CAPVTON_UPLOADS_DIR` | `/mnt/d/fit_light/admin-web/uploads/fitting-results` | 결과 이미지 저장 경로 (admin-web과 공유) |
| `CAPVTON_PUBLIC_URL_PREFIX` | `/uploads/fitting-results` | DB에 저장될 URL prefix |
| `CAPVTON_DEFAULT_COIN_USED` | `10` | 코인 미지정 시 차감량 |
| `CAPVTON_REQUIRE_DB` | `0` | `1`이면 DB 미연결 시 기동 실패 |
| `CAPVTON_DISABLE_MODEL` | `0` | `1`이면 모델 로드 생략 (헬스체크 전용) |
| `ORACLE_DB_*` | admin-web과 동일 | wallet 위치 등 |

## 엔드포인트

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "db_ready": true}
```

### `POST /tryon`

**Request — `multipart/form-data`**

| 필드 | 타입 | 필수 | 비고 |
|------|------|------|------|
| `src_image` | File | ✅ | 사용자 사진 |
| `ref_image` | File | ✅ | 의류 이미지 |
| `garment_type` | str | ✅ | `upper_body` / `lower_body` / `dresses` / `short_sleeve` / `shorts` |
| `step` | int | | 기본 20 |
| `seed` | int | | 기본 42 |
| `vt_model_type` | str | | `viton_hd` (기본) / `viton_dc` |
| `vt_repaint` | bool | | 기본 false |
| `user_id` | str | | **있으면 DB 저장** |
| `product_id` | str | | **`user_id`와 함께 필수** |
| `coin_used` | int | | 기본 `CAPVTON_DEFAULT_COIN_USED` |
| `user_measurements` | str (JSON) | | `{"height_cm":175, "chest_circumference_cm":95, ...}` |
| `garment_measurements` | str (JSON) | | `{"category":"tshirt", "chest_circumference_cm":104, ...}` |
| `preferred_fit` | str | | `slim` / `regular` / `loose` / `oversized` |
| `response_format` | str | | `image` (기본) / `json` |

**Response (image, default)**

- `Content-Type: image/jpeg`
- 본문: 결과 이미지 (JPEG bytes)
- 헤더 (DB 저장 시):
  - `X-Record-Id`: fitting_records.id
  - `X-Result-Path`: `/uploads/fitting-results/<uuid>.jpg`
  - `X-Coin-Balance`: 차감 후 잔액

**Response (`response_format=json`)**

```json
{
  "imageBase64": "...",
  "recordId": 27,
  "resultImagePath": "/uploads/fitting-results/abc.jpg",
  "coinBalance": 90,
  "coinUsed": 10
}
```

## 프론트엔드 연동 패턴

현재 [fit_light_app/web/js/api.js](../../../fit_light/fit_light_app/web/js/api.js)는 다음 3-stage 흐름을 사용한다:

```
1) /tryon                                  → blob 반환
2) /api/app/fittings/upload-result         → blob 업로드, public URL 반환
3) /api/app/users/:id/fittings (POST)      → fitting_records INSERT, 코인 차감
```

`capvton-api`는 `/tryon` 단계에서 2+3을 통째로 대신할 수 있다.
프론트가 폼에 `user_id`/`product_id`를 함께 실어 보내면 단일 호출로 끝난다:

```js
// js/api.js 의 runTryOn 옵션 확장 예
runTryOn: async (srcFile, refBlob, garmentType, step=20, opts={}) => {
  const fd = new FormData();
  fd.append('src_image', srcFile, 'src_image.jpg');
  fd.append('ref_image', refBlob, 'ref_image.jpg');
  fd.append('garment_type', garmentType);
  fd.append('step', String(step));
  if (opts.userId)    fd.append('user_id',    opts.userId);
  if (opts.productId) fd.append('product_id', opts.productId);
  if (opts.coinUsed)  fd.append('coin_used',  String(opts.coinUsed));
  const r = await fetch(`${TRYON_BASE_URL}tryon`, { method:'POST', body: fd });
  if (!r.ok) throw new Error(`TryOn HTTP ${r.status}`);
  return {
    blob: await r.blob(),
    recordId:    r.headers.get('X-Record-Id'),
    resultPath:  r.headers.get('X-Result-Path'),
    coinBalance: r.headers.get('X-Coin-Balance'),
  };
},
```

`user_id`를 보내지 않으면 기존 흐름(2+3 별도 호출)이 그대로 유효하므로, 점진적 마이그레이션 가능.

## DB 저장 동작

`fitting_records` (admin-web 스키마와 동일):

| 컬럼 | 값 |
|------|-----|
| `id` | IDENTITY |
| `user_id` | 폼의 `user_id` |
| `product_id` | 폼의 `product_id` |
| `result_image_path` | `/uploads/fitting-results/<uuid>.jpg` |
| `coin_used` | 폼의 `coin_used` (기본 10) |
| `created_at` | `SYSTIMESTAMP` |

INSERT와 함께 `app_users.coin_balance -= coin_used`, `fitting_count += 1`.
검증 실패(사용자/상품 미존재, 비활성, 코인 부족) 시 400 반환.
DB 미가용이면 이미지만 응답하고 헤더는 비움 (서버는 계속 동작).
