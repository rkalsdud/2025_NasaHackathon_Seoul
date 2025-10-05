# NASAdata.py
# -----------------
# 6개 물리량을 입력받아 저장된 scikit-learn 모델로 은하 "형태/크기"를 예측하고,
# 가장 가까운 샘플 FITS를 골라 3D numpy 볼륨(HxWxD, float32 0..1)로 생성합니다.

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from astropy.io import fits

# ------------------------------------------------------------
# 안전한 경로 유틸
HERE = os.path.dirname(os.path.abspath(__file__))

def _abs_path(*parts) -> str:
    return os.path.join(HERE, *parts)

# ------------------------------------------------------------
# 모델 로딩 (경로는 스크립트 기준)
TYPE_MODEL_PATH = _abs_path("galaxy_type_classifier.joblib")
SIZE_MODEL_PATH = _abs_path("galaxy_size_regressor.joblib")

print(">>> 1단계: 저장된 머신러닝 모델을 불러옵니다...")
try:
    model_type = joblib.load(TYPE_MODEL_PATH)
    model_size = joblib.load(SIZE_MODEL_PATH)
    # print("✅ 모델 로딩 성공!")
except Exception as e:
    print("🚨 [오류] 모델 로드 실패:", repr(e))
    print("   경로를 확인하세요:", TYPE_MODEL_PATH, SIZE_MODEL_PATH)
    # 모델이 꼭 필요하므로 예외를 그대로 올려 종료
    raise

# ------------------------------------------------------------
# 샘플 메타 및 예측
def get_sample_galaxy_data() -> pd.DataFrame:
    """샘플 은하 정보 DataFrame 반환 (형태, 은하명, Re, FITS 파일명)"""
    sample_data = [
        ["Late-type (Spiral)", "M51", 47, "sample_m51.fits"],
        ["Late-type (Spiral)", "M63", 30, "sample_m63.fits"],
        ["Late-type (Spiral)", "NGC 5866", 26, "sample_ngc 5866.fits"],
        ["Late-type (Spiral)", "NGC 1300", 18, "sample_ngc 1300.fits"],
        ["Late-type (Spiral)", "NGC 7479", 13, "sample_ngc 7479.fits"],
        ["Early-type (Elliptical)", "M87", 72, "sample_m87.fits"],
        ["Early-type (Elliptical)", "M49", 50, "sample_m49.fits"],
        ["Early-type (Elliptical)", "M104", 38, "sample_m104.fits"],
        ["Early-type (Elliptical)", "M59", 21, "sample_m59.fits"],
        ["Irregular", "NGC 6822", 105, "sample_ngc 6822.fits"],
        ["Irregular", "M82", 35, "sample_m82.fits"],
        ["Irregular", "NGC 5204", 13, "sample_ngc 5204.fits"],
    ]
    df = pd.DataFrame(sample_data, columns=["형태", "은하", "Re", "FITS"])
    return df

def normalize_pred_type(t: str) -> str:
    """모델 출력 문자열을 샘플 표의 3가지 레이블 중 하나로 보정"""
    if not isinstance(t, str):
        return "Late-type (Spiral)"
    s = t.strip().lower()
    # 흔한 변형들 인식
    if "irr" in s or "irreg" in s:
        return "Irregular"
    if "early" in s or "ellip" in s or "e-type" in s or "ell" in s:
        return "Early-type (Elliptical)"
    if "late" in s or "spiral" in s or "s-type" in s or "disk" in s:
        return "Late-type (Spiral)"
    # 못 알아들으면 기본값
    return "Late-type (Spiral)"


def predict_galaxy_all(
    sersic_n: float,
    ba_ratio: float,
    sigma: float,
    sfr: float,
    redshift: float,
    sb_1re: float,
):
    """모양(분류) + 크기(회귀) 통합 예측"""
    x = pd.DataFrame({
        "NSA_SERSIC_N": [sersic_n],
        "NSA_ELPETRO_BA": [ba_ratio],
        "STELLAR_SIGMA_1RE": [sigma],
        "SFR_TOT": [sfr],
        "Z": [redshift],
        "SB_1RE": [sb_1re],
    })
    y_type = model_type.predict(x)[0]
    y_size = float(model_size.predict(x)[0])
    return y_type, y_size

def find_closest_fits(input_type: str, input_re: float, df: pd.DataFrame) -> str | None:
    """예측된 형태/크기에 가장 가까운 FITS 파일명"""
    sub = df[df["형태"] == input_type].copy()
    if sub.empty:
        return None
    sub["diff"] = (sub["Re"] - input_re).abs()
    return sub.loc[sub["diff"].idxmin(), "FITS"]

# ------------------------------------------------------------
# FITS -> 2D -> 3D 볼륨 생성
def _load_fits_2d(fits_path: str) -> np.ndarray:
    """FITS에서 2D 이미지 추출(float32, NaN/inf 제거)"""
    with fits.open(fits_path) as hdul:
        data = None
        if len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data
        elif hdul[0].data is not None:
            data = hdul[0].data
        else:
            raise RuntimeError("No valid image data in FITS")
    img = np.asarray(data, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    # contrast stretch (1~99.5%)
    vmin, vmax = np.percentile(img, [1.0, 99.5])
    img = np.clip(img, vmin, vmax)
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img[:] = 0.0
    return img

def _resize_bilinear(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """src(H0,W0) -> (out_h,out_w) bilinear, numpy-only (scikit-image 없이 동작)"""
    H0, W0 = src.shape
    H, W = int(out_h), int(out_w)
    out = np.empty((H, W), dtype=np.float32)

    # target pixel centers mapped to source space
    ys = (np.linspace(0.5, H - 0.5, H) / H) * H0 - 0.5
    xs = (np.linspace(0.5, W - 0.5, W) / W) * W0 - 0.5

    y0 = np.clip(np.floor(ys).astype(np.int32), 0, H0 - 2)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, W0 - 2)
    dy = ys - y0
    dx = xs - x0

    for i in range(H):
        yi0 = y0[i]; yi1 = yi0 + 1
        wy0 = (1.0 - dy[i]); wy1 = dy[i]

        # 열 방향 벡터화 보간 (브로드캐스팅 안전)
        v00 = src[yi0, x0]      # (W,)
        v10 = src[yi0, x0 + 1]
        v01 = src[yi1, x0]
        v11 = src[yi1, x0 + 1]

        out[i, :] = ( (1.0 - dx) * (wy0 * v00) +
                      dx         * (wy0 * v10) +
                      (1.0 - dx) * (wy1 * v01) +
                      dx         * (wy1 * v11) )

    # 0..1 정규화
    mn, mx = float(out.min()), float(out.max())
    if mx > mn:
        out = (out - mn) / (mx - mn)
    else:
        out[:] = 0.0
    return out.astype(np.float32)


def build_volume_from_fits(
    fits_path: str,
    depth: int = 64,
    thickness: float = 0.35,
    out_xy: int = 256,
) -> np.ndarray:
    """
    중심이 밝은 은하 가정으로 2D 이미지를 Z로 '두께' 있게 확장한 3D 볼륨을 만듭니다.
    반환 shape: (H, W, D), dtype float32, [0..1]
    """
    img2d = _load_fits_2d(fits_path)
    # 안전한 bilinear 리사이즈로 교체
    H = int(out_xy); W = int(out_xy); D = int(depth)
    img_res = _resize_bilinear(img2d, H, W)

    # Z 프로파일(벌지+디스크)
    z = np.linspace(-1, 1, D).astype(np.float32)
    z_prof = np.exp(-(z ** 2) / (2 * (max(thickness, 1e-3) ** 2))).astype(np.float32)
    z_prof /= (z_prof.max() + 1e-8)

    # 중심 강조(반경 가중)
    yy2, xx2 = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rr = np.sqrt(((xx2 - cx) / (0.5 * W)) ** 2 + ((yy2 - cy) / (0.5 * H)) ** 2).astype(np.float32)
    rr = np.clip(rr, 0, 1)
    disk = (1 - rr) ** 0.6  # 중심 밝게

    vol = np.empty((H, W, D), dtype=np.float32)
    for k in range(D):
        vol[:, :, k] = img_res * (0.7 + 0.3 * z_prof[k]) * disk

    # 최종 정규화
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol[:] = 0.0
    return vol.astype(np.float32)
def build_synthetic_volume(out_xy=256, depth=64, sersic_n=2.0, ba=0.7, thickness=0.35):
    H = W = int(out_xy); D = int(depth)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W-1)/2.0, (H-1)/2.0
    xn = (xx - cx) / (0.45 * W)
    yn = (yy - cy) / (0.45 * H) / max(ba, 1e-3)  # b/a 적용
    r = np.sqrt(xn**2 + yn**2)

    b = np.exp(-(r ** (1.0 / max(sersic_n, 1e-3))))
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)

    z = np.linspace(-1, 1, D).astype(np.float32)
    z_prof = np.exp(-(z**2) / (2 * max(thickness, 1e-3)**2)).astype(np.float32)
    z_prof /= (z_prof.max() + 1e-8)

    vol = np.empty((H, W, D), dtype=np.float32)
    for k in range(D):
        vol[:, :, k] = b * (0.7 + 0.3 * z_prof[k])
    return vol

# ------------------------------------------------------------
# 통합 파이프라인
def predict_and_build_volume(
    sersic_n: float,
    ba_ratio: float,
    sigma: float,
    sfr: float,
    redshift: float,
    sb_1re: float,
    depth: int = 64,
    thickness: float = 0.35,
    out_xy: int = 256,
):
    """
    6개 입력 → (형태, 크기) 예측 → 가장 가까운 FITS 선택 → 3D 볼륨 생성
    실패 시 합성 볼륨으로 폴백.
    """
    df = get_sample_galaxy_data()

    # 1) 모델 예측
    raw_type, pred_size = predict_galaxy_all(sersic_n, ba_ratio, sigma, sfr, redshift, sb_1re)
    pred_type = normalize_pred_type(raw_type)

    # 2) 샘플 FITS 매칭
    fits_name = find_closest_fits(pred_type, pred_size, df)
    vol = None
    fits_path = None

    # 3) FITS 경로가 있으면 우선 시도
    if fits_name:
        fits_path = _abs_path("data", fits_name)
        if os.path.exists(fits_path):
            try:
                vol = build_volume_from_fits(fits_path, depth=depth, thickness=thickness, out_xy=out_xy)
            except Exception:
                vol = None  # 실패 시 폴백으로 넘어감

    # 4) 폴백: 합성 볼륨
    if vol is None:
        # 합성 볼륨으로 폴백
        vol = build_synthetic_volume(
            out_xy=out_xy, depth=depth,
            sersic_n=sersic_n, ba=ba_ratio, thickness=thickness
        )
        fits_name = fits_name or "synthetic"

    return pred_type, float(pred_size), fits_name, vol

# (직접 실행용 간단 테스트)
if __name__ == "__main__":
    t, s, f, v = predict_and_build_volume(2.3, 0.6, 90.0, 0.03, 0.05, 0.4, depth=48, thickness=0.35, out_xy=128)
    print("pred:", t, s, f, "| vol:", v.shape, v.dtype, v.min(), v.max())
