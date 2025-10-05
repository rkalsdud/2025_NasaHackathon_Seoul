# Flask 및 관련 라이브러리 설치


from flask import Flask, request, jsonify
import numpy as np
from astropy.io import fits
from skimage.transform import resize
import os

def load_raw_fits_image(file_path):
    """FITS 파일을 열어 첫 번째 또는 두 번째 HDU에서 원본 데이터를 찾아 반환합니다."""
    try:
        with fits.open(file_path) as hdul:
            image_data = None
            # Extension HDU (주로 SCI 데이터)를 먼저 시도
            if len(hdul) > 1 and hdul[1].data is not None:
                image_data = hdul[1].data
            # 없다면 Primary HDU 시도
            elif hdul[0].data is not None:
                image_data = hdul[0].data
            else:
                print(f"Error: No valid image data found in any HDU for {file_path}.")
                return None
            
            image_data = image_data.astype(np.float32)
            image_data = np.nan_to_num(image_data)
            return image_data
            
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None

def create_3d_voxel_grid(image_2d, depth=32, threshold=0.01):
    """재조정된 2D 이미지를 받아 3D 복셀 그리드를 생성합니다."""
    height, width = image_2d.shape
    voxel_grid = np.zeros((height, width, depth), dtype=np.float32)
    print(f"Creating {height}x{width}x{depth} voxel grid...")

    # NumPy 벡터화 연산으로 성능 향상
    y, x = np.where(image_2d > threshold)
    brightness = image_2d[y, x]
    
    fill_depth = np.round(brightness * depth).astype(int)
    fill_depth[fill_depth == 0] = 1
    
    start_z = np.round((depth / 2) - (fill_depth / 2)).astype(int)
    end_z = np.round((depth / 2) + (fill_depth / 2)).astype(int)
    
    for i in range(len(x)):
        sz, ez = start_z[i], end_z[i]
        if sz >= ez: ez = sz + 1
        voxel_grid[y[i], x[i], sz:ez] = brightness[i]
            
    print("Voxel grid creation complete.")
    return voxel_grid

def convert_fits_to_3d_array(fits_path, output_xy_size=256, output_depth=32):
    """
    하나의 FITS 파일을 최종 3D NumPy 배열로 변환하는 메인 함수.
    """
    print(f"\n--- Processing file: {fits_path} ---")
    
    # 1. 원본 FITS 로드
    raw_image_2d = load_raw_fits_image(fits_path)
    if raw_image_2d is None:
        return None

    # 2. 고해상도 2D 이미지 재조정 (Contrast Stretching)
    print("Scaling 2D image...")
    vmin, vmax = np.percentile(raw_image_2d, [1, 99.5])
    clipped_image = np.clip(raw_image_2d, vmin, vmax)
    scaled_image_2d = (clipped_image - vmin) / (vmax - vmin)
    print("Image scaling complete.")
    
    # 3. '작은' 2D 이미지를 먼저 만듭니다.
    resized_2d_for_3d = resize(scaled_image_2d, (output_xy_size, output_xy_size), anti_aliasing=True)

    # 4. '작은' 2D 이미지로 '작은' 3D 복셀 그리드를 생성합니다.
    voxel_3d = create_3d_voxel_grid(resized_2d_for_3d, depth=output_depth)
    
    return voxel_3d
# --- Flask API 서버 설정 ---
app = Flask(__name__)

# 미리 정해진 FITS 파일들의 경로를 딕셔너리로 관리
AVAILABLE_FITS = {
    "m51": "data/m51.fits",
    "m101": "data/m101.fits",
    "ngc1300": "data/ngc1300.fits"
}

@app.route('/generate_3d', methods=['GET'])
def generate_3d():
    # 1. 클라이언트가 요청한 파일 키를 가져옵니다 (예: 'm51')
    file_key = request.args.get('file')

    # 2. 요청이 유효한지 확인
    if not file_key or file_key not in AVAILABLE_FITS:
        return jsonify({"error": "Invalid or missing file key"}), 400

    fits_path = AVAILABLE_FITS[file_key]
    if not os.path.exists(fits_path):
        return jsonify({"error": f"FITS file not found at path: {fits_path}"}), 404

    # 3. 실시간으로 3D 배열 생성
    voxel_3d_array = convert_fits_to_3d_array(fits_path, output_xy_size=128, output_depth=32)

    if voxel_3d_array is None:
        return jsonify({"error": "Failed to process FITS file"}), 500

    # 4. NumPy 배열을 JSON으로 보낼 수 있도록 리스트로 변환하여 전송
    return jsonify({
        "file_key": file_key,
        "shape": voxel_3d_array.shape,
        "voxel_data": voxel_3d_array.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)