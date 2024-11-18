import cv2
import numpy as np
from scipy.optimize import least_squares

from crplidar import LiDAR
from sklearn.cluster import DBSCAN
from ExtendedKalmanFilter import ExtendedKalmanFilter
import time
from cluster import Cluster
from tracker import Tracking



from aprilgrid import Detector
import os



# 카메라 내부 매개변수와 왜곡 계수
# 카메라 내부 매개변수와 왜곡 계수
camera_matrix = np.array([[718.0185818253857, 0, 323.4567247857622],
                          [0, 718.7782113904716, 235.99239839273147],
                          [0, 0, 1]])
dist_coeffs = np.array([0.09119377823619247, 0.0626265233941177, -0.007768343835168918, -0.004451209268144528, 0.48630799030494654])



rvec = [[ 0.98146676, -0.19072319, -0.01864553],
 [-0.00126777, -0.10375852,  0.99460171],
 [-0.19162824, -0.97614488, -0.10207733]]

tvec = [[-347.79113757],
 [  21.96899218],
 [ 325.4898965 ]]


device = 1

cap_width = 800
cap_height = 480

length_magnification = 3
cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


resolution = 20000
epsilon = 12
min_samples = 5

output_size = (cap_width, cap_height)
current_clusters = {}
image = None

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



def draw_coordinates_with_clustering(points, scaled_point_array, input_scale=resolution):
    """
    coords: LiDAR로부터 얻은 좌표 리스트
    input_scale: LiDAR의 최대 측정 거리 (mm 단위가 일반적)
    output_size: 출력 이미지의 크기 (width, height)
    """

    global image, current_clusters

    # DBSCAN 클러스터링 수행
    if len(points) > 0:
        # 스케일링된 좌표 준비
                
        # DBSCAN 알고리즘 적용
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(scaled_point_array)
        
        # 클러스터링 결과
        labels = dbscan.labels_ # 라벨 
        unique_labels = set(labels) - {-1}  # 이상치 추적 X 

        # HSV 색상 공간에서 라벨별로 색상 생성
        hsv_colors = [(int(label * 255 / len(unique_labels)), 255, 255) for label in unique_labels]
        rgb_colors = [cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0] for hsv_color in hsv_colors]
        color_map = {label: rgb_color for label, rgb_color in zip(unique_labels, rgb_colors)}

        # 클러스터링 결과 시각화
        for (x, y), label in zip(scaled_point_array, labels):
            if label != -1:  # 라벨이 -1이 아닌 경우에만 색상을 적용
                color = color_map[label]
                cv2.circle(image, (x, y), 1, color.tolist(), -1)
            else:  # 잡음(이상치)은 회색으로 표시
                cv2.circle(image, (x, y), 1, (125, 125, 125), -1)


        # 클러스터링 중심 시각화 \
        for label in unique_labels:
            cluster = points[labels == label]
            centroid = np.mean(cluster, axis=0)
            x, y = scale_point(centroid[0], centroid[1], input_scale, output_size)

            cluster_instance = Cluster(label, (x, y))
            current_clusters[cluster_instance.label] = cluster_instance

            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(image, f'{cluster_instance.label}', (cluster_instance.position.x, cluster_instance.position.y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          



def main():

    global image, tvec, rvec

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

        points = []
        distances = []

        for x, y, distance in coords:
            points.append((x, y))
            distances.append(distance)

        scaled_coords = [scale_point(x, y, resolution, output_size) for x, y in points]

        # numpy array로 변환
        scaled_point_array = np.array(scaled_coords)
        points = np.array(points)
        draw_coordinates_with_clustering(points, scaled_point_array, resolution)

        # 새로운 차원 추가
        new_dimension = np.zeros((scaled_point_array.shape[0], 1))  # 모든 원소에 대해 0 값을 가지는 새로운 차원 생성
        new_points_array = np.hstack((scaled_point_array, new_dimension))  # 기존 배열에 새로운 차원을 추가

        # LiDAR 포인트를 카메라 이미지 상에 투영
        project_lidar_to_camera(frame, new_points_array, rvec, tvec, camera_matrix, dist_coeffs, distances)
        cv2.imshow("Projected LiDAR on Camera", frame)

        cv2.imshow('LiDAR Scan', image)

        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
            break


    lidar.stopMotor()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()