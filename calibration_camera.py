##### 카메라 캘리브레이션을 통해

import yaml
import cv2
import numpy as np
import os
import glob

# config 설정
with open("/home/soda/Documents/Devleopment/real_camera_lidar_calibration/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    print(config)

    pattern_size = (config['cols'], config['rows'])
    square_size = config['s_height']
    capture_mode = config['capture_mode']  # True면 캡처 모드
    capture_max_count = config['capture_max_count']

# 카메라 설정
window_name = "calibration"
cap = cv2.VideoCapture(1)
cap_width = 960
cap_height = 540
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, cap_width, cap_height)

# 저장할 이미지의 인덱스
img_index = 0
img_end_index = 30

# 이미지를 저장할 폴더 생성 (폴더가 없는 경우)
save_folder = "saved"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

while capture_mode:
    if img_index >= img_end_index:
        break

    ret, frame = cap.read()
    if not ret:
        print("카메라를 사용할 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # 체커보드의 코너를 화면에 표시
        _frame = frame.copy()
        cv2.drawChessboardCorners(_frame, pattern_size, corners, ret)

        # 인식된 체커보드 결과를 보여줌
        cv2.imshow(window_name, _frame)

        # 여기에서 사용자 입력을 대기
        key = cv2.waitKey(0) & 0xFF  # 실시간 캡처를 멈추고 사용자의 입력을 기다림

        if key == ord(' '):
            # 스페이스바를 누르면 이미지 저장
            img_name = f"calibration_{img_index}.jpg"
            save_path = os.path.join(save_folder, img_name)
            cv2.imwrite(save_path, frame)

            print(f"{save_path} 저장됨")
            img_index += 1

        elif key == 27:  # ESC 키
            continue
        elif key == ord('q'):
            break
    else:
        # 체커보드가 인식되지 않으면 실시간으로 다음 프레임을 계속 보여줌
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 저장된 이미지로 내부 행을 얻음

# 미세조정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 세계 좌표점 초기화
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 2D 점과 3D 점을 저장할 리스트
objpoints = []  # 3D 세계 좌표 점
imgpoints = []  # 2D 이미지 좌표 점

# 저장된 이미지에서 캘리브레이션 데이터 추출
images = glob.glob(os.path.join(save_folder, '*.jpg'))
for i, image_path in enumerate(images):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # cv2.imshow(f'{i} img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")  # 내부 카메라 행렬
print(mtx)

print("dist : \n")  # 렌즈 왜곡 계수(Lens distortion coefficients)
print(dist)


# 결과를 저장할 폴더 생성 (폴더가 없는 경우)
result_folder = "result"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 카메라 캘리브레이션 결과를 저장할 YAML 파일 경로
result_file_path = os.path.join(result_folder, "calibration_result.yaml")

# 카메라 캘리브레이션 결과 데이터
calibration_data = {
    "camera_matrix": mtx.tolist(),  # numpy 배열을 리스트로 변환
    "distortion_coefficients": dist.tolist()
}

# YAML 파일에 결과 데이터 저장
with open(result_file_path, 'w', encoding='utf-8') as f:
    yaml.dump(calibration_data, f)

print(f"카메라 캘리브레이션 결과가 {result_file_path}에 저장되었습니다.")
