# api_server.py
# FastAPI 서버: 6개의 파라미터를 받아 예측 + FITS 선택 + 3D 볼륨 생성 후
# .npy 바이너리로 반환합니다.
from __future__ import annotations
import io, traceback, sys
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

import NASAdata as nd  # 같은 폴더의 모듈

app = FastAPI(title="Galaxy Volume API", version="1.0.0")

# CORS: 프론트가 file:// 로 열려도 편히 테스트 가능
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predict_volume.npy")
def predict_volume_npy(
    sersic_n: float = Query(2.0, description="Sersic index"),
    ba_ratio: float = Query(0.7, description="b/a axis ratio"),
    sigma: float    = Query(120.0, description="central velocity dispersion"),
    sfr: float      = Query(0.03, description="star formation rate"),
    redshift: float = Query(0.05, description="redshift"),
    sb_1re: float   = Query(0.4, description="surface brightness at 1Re"),
    depth: int      = Query(64, ge=8, le=256, description="Z slices"),
    thickness: float= Query(0.35, ge=0.05, le=1.0, description="disk thickness"),
    out_xy: int     = Query(256, ge=32, le=512, description="XY resolution"),
):
    try:
        _, _, _, vol = nd.predict_and_build_volume(
            sersic_n, ba_ratio, sigma, sfr, redshift, sb_1re,
            depth=depth, thickness=thickness, out_xy=out_xy
        )
        buf = io.BytesIO()
        np.save(buf, vol.astype(np.float32))  # .npy 직렬화
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="volume.npy"'}
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)  # ★ 콘솔에 스택 출력
        return JSONResponse({"error": str(e)}, status_code=500)


# uvicorn api_server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
