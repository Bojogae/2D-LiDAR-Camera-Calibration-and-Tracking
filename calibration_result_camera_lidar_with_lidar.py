import cv2
import numpy as np
from scipy.optimize import least_squares

import configparser
from crplidar import LiDAR
from sklearn.cluster import DBSCAN
from ExtendedKalmanFilter import ExtendedKalmanFilter
import time
from cluster import Cluster
from tracker import Tracking



from aprilgrid import Detector
import os

def load_extrinsic_parameters_from_config(filename='extrinsic_parameters.ini'):
    """
    설정 파일에서 회전 행렬과 이동 벡터를 불러옵니다.

    Parameters:
    - filename: 설정 파일의 이름

    Returns:
    - rotation_matrix: 회전 행렬 (Numpy 배열)
    - tvec: 이동 벡터 (Numpy 배열)
    """
    config = configparser.ConfigParser()
    config.read(filename)

    # 회전 행렬 불러오기
    rotation_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            rotation_matrix[i, j] = float(config[f'RotationMatrix_Row{i}'][f'col{j}'])

    # 이동 벡터 불러오기
    tvec = np.array([float(config['TranslationVector'][f'tvec_{i}']) for i in range(3)]).reshape((3, 1))

    return rotation_matrix, tvec


# 카메라 내부 매개변수와 왜곡 계수
# 카메라 내부 매개변수와 왜곡 계수
camera_matrix = np.array([[718.0185818253857, 0, 323.4567247857622],
                          [0, 718.7782113904716, 235.99239839273147],
                          [0, 0, 1]])
dist_coeffs = np.array([0.09119377823619247, 0.0626265233941177, -0.007768343835168918, -0.004451209268144528, 0.48630799030494654])

rvec, tvec = load_extrinsic_parameters_from_config()


device = 1
cap_width = 300
cap_height = 300

cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


resolution = 7000

output_size = (cap_width, cap_height)



def scale_point(x, y, input_scale, output_size):
    """
    x, y: 입력 좌표 (LiDAR 좌표)
    input_scale: LiDAR의 최대 측정 거리
    output_size: 출력 이미지의 크기 (width, height)
    """
    # LiDAR 좌표를 이미지의 중앙을 기준으로 변환
    x_adjusted = x + input_scale / 2
    y_adjusted = y + input_scale / 2
    
    # 변환된 좌표를 이미지 해상도에 맞게 스케일링
    x_scaled = int(x_adjusted / input_scale * output_size[0])
    y_scaled = int(y_adjusted / input_scale * output_size[1])
    return x_scaled, y_scaled


def divide_points(p1, p2, num_divisions):
    """
    두 점 사이를 주어진 숫자만큼 나누고, 각각의 좌표를 반환합니다.

    Parameters:
    - p1, p2: 나누고자 하는 두 점의 좌표 (x, y)
    - num_divisions: 나누고자 하는 구간의 수

    Returns:
    - points: 나눈 점들의 좌표 리스트
    """
    # 시작점과 끝점 사이의 벡터를 계산
    vector = np.array(p2) - np.array(p1)
    
    # 구간 당 벡터의 증분을 계산
    increment = vector / num_divisions
    
    # 각 구간의 좌표를 계산하여 저장
    points = [tuple(np.array(p1) + increment * i) for i in range(num_divisions + 1)]
    
    return points



def project_lidar_to_camera(image, lidar_points, rvec, tvec, camera_matrix, dist_coeffs, distances):
    """
    LiDAR 포인트를 카메라 이미지 상에 투영합니다.

    Parameters:
    - image: 투영할 대상인 카메라 이미지
    - lidar_points: LiDAR로부터 얻은 3D 좌표 목록
    - rvec: 회전 벡터
    - tvec: 평행이동 벡터
    - camera_matrix: 카메라의 내부 행렬
    - dist_coeffs: 카메라의 왜곡 계수
    - distances: 각 LiDAR 포인트까지의 거리
    """


    # LiDAR 포인트를 이미지 상의 포인트로 변환
    img_points, _ = cv2.projectPoints(lidar_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 변환된 포인트를 이미지에 표시
    for i, p in enumerate(img_points):
        # 유효한 좌표 범위 내에 있는지 확인
        x, y = int(p[0][0]), int(p[0][1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # 거리에 따라 색상을 결정
            color = color_by_distance(distances[i], min_distance=0, max_distance=resolution)
            cv2.circle(image, (x, y), 3, color, -1)  # 색상을 거리에 따라 변경
            # cv2.putText(image, str(distances[i]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



# 거리에 따라 색상을 결정하는 함수
def color_by_distance(distance, min_distance, max_distance):
    # 거리 값을 0에서 120 사이의 Hue 값으로 변환 (녹색에서 빨간색으로)
    hue = int((1 - (distance - min_distance) / (max_distance - min_distance)) * 120)
    # HSV 색상을 RGB로 변환
    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
    return color.tolist()


def main():

    global tvec, rvec

    lidar = LiDAR()
    lidar.startMotor()

    rvec = np.array(rvec)
    tvec = np.array(tvec)

    # print(rvec)
    # print(tvec)

    while True:
        # CAMERA
        ret, frame = cap.read()
        if not ret:
            break

        coords = lidar.getXY_with_distance()  # LiDAR로부터 XY 좌표 얻기
        image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        # print(len(coords))

        scaled_coords = []
        distances = []

        for x, y, distance in coords:
            scaled_coords.append(scale_point(x, y, resolution, output_size))
            distances.append(distance)

        
        # numpy array로 변환
        points_array = np.array(scaled_coords)

        # 새로운 차원 추가
        new_dimension = np.zeros((points_array.shape[0], 1))  # 모든 원소에 대해 0 값을 가지는 새로운 차원 생성
        new_points_array = np.hstack((points_array, new_dimension))  # 기존 배열에 새로운 차원을 추가

        # # LiDAR 포인트를 카메라 이미지 상에 투영
        project_lidar_to_camera(frame, new_points_array, rvec, tvec, camera_matrix, dist_coeffs, distances)
        cv2.imshow("Projected LiDAR on Camera", frame)

        for x, y in scaled_coords:
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
            
        cv2.imshow('LiDAR Scan', image)

        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
            break


    lidar.stopMotor()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()