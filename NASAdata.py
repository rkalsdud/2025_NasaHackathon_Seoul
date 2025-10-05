# predict_galaxy.py
# -----------------
# 이 스크립트는 사전에 학습되고 저장된 머신러닝 모델을 불러와
# 사용자가 입력한 6가지 물리량으로 은하의 형태와 크기를 예측하고,
# 가장 유사한 샘플 은하의 이미지 파일(FITS)을 찾아줍니다.

# 필수 라이브러리
import pandas as pd
import numpy as np
import joblib
from astropy.io import fits # FITS 파일 처리를 위해 필요

# 저장된 모델 불러오기
print(">>> 1단계: 저장된 머신러닝 모델을 불러옵니다...")
try:
    # joblib.load() 함수로 바이너리 파일을 다시 모델 객체로 복원
    model_type = joblib.load('galaxy_type_classifier.joblib')
    model_size = joblib.load('galaxy_size_regressor.joblib')
    print("✅ 모델 로딩 성공!")
except FileNotFoundError:
    print("🚨 [오류] 모델 파일을 찾을 수 없습니다.")
    print("이 스크립트와 같은 위치에 'galaxy_type_classifier.joblib'와 'galaxy_size_regressor.joblib' 파일이 있는지 확인해주세요.")
    exit() # 프로그램 종료

# 예측 및 파일 검색을 위한 함수 정의

def get_sample_galaxy_data():
    """샘플 은하 정보가 담긴 DataFrame을 생성하고 반환하는 함수"""
    sample_data = [
        ["Late-type (Spiral)", "M51", 47, "sample_m51.fits"],
        ["Late-type (Spiral)", "M63", 30, "sample_m63.fits"],
        ["Late-type (Spiral)", "NGC 5866", 26, "sample_ngc5866.fits"],
        ["Late-type (Spiral)", "NGC 1300", 18, "sample_ngc1300.fits"],
        ["Late-type (Spiral)", "NGC 7479", 13, "sample_ngc7479.fits"],
        ["Early-type (Elliptical)", "M87", 72, "sample_m87.fits"],
        ["Early-type (Elliptical)", "M49", 50, "sample_m49.fits"],
        ["Early-type (Elliptical)", "M104", 38, "sample_m104.fits"],
        ["Early-type (Elliptical)", "M59", 21, "sample_m59.fits"],
        ["Irregular", "NGC 6822", 105, "sample_ngc6822.fits"],
        ["Irregular", "M82", 35, "sample_m82.fits"],
        ["Irregular", "NGC 5204", 13, "sample_ngc5204.fits"]
    ]
    df_samples = pd.DataFrame(sample_data, columns=["형태", "은하", "Re", "FITS"])
    return df_samples

def predict_galaxy_all():
    """사용자로부터 직접 6개 값을 입력받아 은하 모양과 크기를 모두 예측하는 함수"""
    print("\n>>> 2단계: 은하의 특징을 나타내는 6가지 값을 입력해주세요.")
    
    # 사용자 입력을 받아 변수에 저장
    sersic_n = float(input("  1. 세르식 지수 (타원은하 ~4, 나선은하 ~1): "))
    ba_ratio = float(input("  2. 장축 대 단축 비율 (둥글수록 1, 납작할수록 0): "))
    sigma = float(input("  3. 중심 속도 분산 (타원은하 ~200, 나선은하 ~70): "))
    sfr = float(input("  4. 총 별 형성률 (타원은하 <0.01, 나선/불규칙 >0.01): "))
    redshift = float(input("  5. 적색편이 (거리가 멀수록 큼, 예: 0.1): "))
    sb_1re = float(input("  6. 표면 밝기 (SB_1RE) (밝을수록 작음, 예: 0.4): "))

    # 입력을 DataFrame 형식으로 변환
    input_data = pd.DataFrame({
        'NSA_SERSIC_N': [sersic_n], 
        'NSA_ELPETRO_BA': [ba_ratio],
        'STELLAR_SIGMA_1RE': [sigma], 
        'SFR_TOT': [sfr],
        'Z': [redshift], 
        'SB_1RE': [sb_1re]
    })

    # 모델 1: 모양 예측 (분류)
    type_prediction = model_type.predict(input_data)[0]
    type_probabilities = model_type.predict_proba(input_data)[0]

    # 모델 2: 크기 예측 (회귀)
    size_prediction = model_size.predict(input_data)[0]

    print("\n--- 💡 통합 예측 결과 ---")
    print(f"➡️  은하 모양: '{type_prediction}'일 가능성이 높습니다.")
    print(f"➡️  은하 크기: 유효반경 약 {size_prediction:.2f} arcsec로 예측됩니다.")
    print("\n[상세] 모양별 확률:")
    for i, class_name in enumerate(model_type.classes_):
        print(f"  - {class_name}: {type_probabilities[i]*100:.2f}%")

    return type_prediction, size_prediction

def find_closest_fits(input_type, input_re, df_samples):
    """예측된 결과와 가장 유사한 샘플 은하의 FITS 파일명을 찾는 함수"""
    # 예측된 형태와 일치하는 은하들만 필터링
    subset = df_samples[df_samples['형태'] == input_type].copy()
    if subset.empty:
        print(f"\n[알림] '{input_type}' 형태와 일치하는 샘플 은하가 목록에 없습니다.")
        return None

    # 예측된 크기(Re)와의 차이를 계산하여 가장 차이가 적은 은하를 찾음
    subset['diff'] = np.abs(subset['Re'] - input_re)
    closest = subset.loc[subset['diff'].idxmin()]
    return closest['FITS']


#  메인 코드 실행 ---
if __name__ == "__main__":
    # 1. 샘플 은하 데이터 준비
    df_galaxy_samples = get_sample_galaxy_data()
    
    # 2. 사용자 입력 및 은하 특성 예측
    predicted_type, predicted_size = predict_galaxy_all()

    # 3. 예측 결과와 가장 유사한 샘플 FITS 파일 찾기
    print("\n>>> 3단계: 예측 결과와 가장 유사한 샘플 은하를 찾습니다...")
    closest_file = find_closest_fits(predicted_type, predicted_size, df_galaxy_samples)

    if closest_file:
        print(f"✅ 찾은 FITS 파일: {closest_file}")
        try:
            # hdu = fits.open(closest_file)
            pass
        except FileNotFoundError:
            print(f"'{closest_file}' 파일을 찾을 수 없습니다. 샘플 FITS 파일들이 있는지 확인하세요.")