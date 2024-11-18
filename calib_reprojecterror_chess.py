import cv2
import numpy as np
from scipy.optimize import least_squares

from crplidar import LiDAR
from sklearn.cluster import DBSCAN
from ExtendedKalmanFilter import ExtendedKalmanFilter
import time
from cluster import Cluster
from tracker import Tracking

import json


from aprilgrid import Detector
import os



# 카메라 내부 매개변수와 왜곡 계수
# 카메라 내부 매개변수와 왜곡 계수
camera_matrix = np.array([[718.0185818253857, 0, 323.4567247857622],
                          [0, 718.7782113904716, 235.99239839273147],
                          [0, 0, 1]])
dist_coeffs = np.array([0.09119377823619247, 0.0626265233941177, -0.007768343835168918, -0.004451209268144528, 0.48630799030494654])


# AprilTag 보드의 물리적 크기 설정 (80x80cm, 각 태그는 8.7cm)
board_width = 0.8  # meter
board_height = 0.8  # meter
tag_size = 0.087  # meter

lidar_2d_positions = []     # [(x, y), (x, y)], 2d lidar에서 x, y 좌표 
chess_2d_positions = []     # [(id, (x1, y1), (x2, y2)), (id, (x1, y1), (x2, y2))] x1, y1은 좌하단, x2, y2는 우하단


# 아래 리스트는 실제로 캘리브레이션에 사용될 좌표들임
calibration_for_lidar_points = []   # [(x, y, 0), (x, y, 0), (x, y, 0)] lidar 좌표들 모음
calibration_for_chess_points = []   # [(x, y), (x, y), (x, y)] april 태그의 좌표들 모음

chessboard_size = (9, 6)  # 체스보드의 코너 수 (가로, 세로)
square_size = 0.025

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

device = 1
cap_width = 900
cap_height = 480

length_magnification = 3
cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


resolution = 15000
epsilon = 12
min_samples = 5

ekf_next_id = 0  # 새로 추가할 확장칼만필터 추적 id 
current_tracking = {}  # 추적 객체

current_clusters = {} # 현재 클러스터의 중심점을 저장

output_size = (cap_width, cap_height)

previous_time = time.time()
cancel_tracking_threshold = 5

image = None

success = None
rvec = None
tvec = None
extrinsic = None


TARGET_DESCRIPTION = "3_BaF"

calibration_count = 5

def save_extrinsic_to_json(rotation_matrix, translation_vector, extrinsic_matrix, reprojection_error, description, filename="chess_extrinsic_data.json"):
    # NumPy array의 데이터를 Python 기본 타입으로 변환
    data = {
        "rotation_matrix": rotation_matrix.astype(float).tolist(),  # NumPy float32를 float로 변환
        "translation_vector": translation_vector.astype(float).tolist(),
        "extrinsic_matrix": extrinsic_matrix.astype(float).tolist(),
        "reprojection_error": float(reprojection_error),  # NumPy float32를 float로 명시적 변환
        "description": description
    }

    # 파일이 존재하는지 확인하고 데이터 로드
    if os.path.exists(filename):
        try:
            with open(filename, "r") as file:
                existing_data = json.load(file)
                new_id = existing_data["last_id"] + 1
        except json.JSONDecodeError:
            existing_data = {"data": [], "last_id": -1}
            new_id = 0
    else:
        existing_data = {"data": [], "last_id": -1}
        new_id = 0

    data["id"] = new_id
    existing_data["data"].append(data)
    existing_data["last_id"] = new_id

    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)



def find_chessboard_corners(frame):
    """체스보드 코너 검출과 그 결과를 이미지에 표시하는 함수"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        # 코너를 더 정밀하게 찾기 위해
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
    return ret, corners

def extract_specific_row_corners(corners, row_number):
    """특정 행의 코너 좌표를 추출"""
    if corners is not None:
        # 특정 행의 코너들을 선택
        row_start = row_number * chessboard_size[0]
        row_end = row_start + chessboard_size[0]
        specific_row_corners = corners[row_start:row_end]
        return specific_row_corners


def calculate_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    재투영 오차를 계산하는 함수입니다.

    Parameters:
    - object_points: 3D 월드 좌표
    - image_points: 2D 이미지 좌표
    - rvec: 회전 벡터
    - tvec: 평행이동 벡터
    - camera_matrix: 카메라 내부 매개변수
    - dist_coeffs: 카메라 왜곡 계수
    """
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    
    # projected_points의 형태 확인
    print("Projected Points Shape:", projected_points.shape)
    print("Image Points Shape:", image_points.shape)
    
    # 오차 계산
    errors = np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1)
    mean_error = np.mean(errors)
    mean_error = np.sqrt(mean_error)

    return mean_error



def find_nearest_cluster(click_position):
    min_distance = float('inf')
    nearest_cluster = None

    # 현재 클러스터 아이템을 순회 하면서 가장 가까운 클러스터를 찾음
    for cluster in current_clusters.values():
        # 클릭 좌표와 클러스터의 거리를 구함
        distance = np.linalg.norm(np.array(click_position) - np.array(cluster.position.tuple()))

        if distance < min_distance:
            min_distance = distance
            nearest_cluster = cluster   # 최소 거리에 해당하면 클러스터 저장 

    return nearest_cluster, min_distance

def find_nearest_ekf(click_position):
    global current_tracking
    min_distance = float('inf')
    nearest_ekf_key = None
    for key, tracking_obj in current_tracking.items():
        # Tracking 객체의 위치를 얻음
        tracking_position = np.array(tracking_obj.getPosition())
        # 클릭된 위치와 Tracking 객체 위치와의 유클리드 거리를 계산
        distance = np.linalg.norm(click_position - tracking_position)
        if distance < min_distance:
            min_distance = distance
            nearest_ekf_key = key
    return nearest_ekf_key, min_distance




def mouse_click(event, x, y, flags, param):
    global ekf_next_id, current_tracking
    click_position = np.array([x, y])

    # 컨트롤 키와 함께 마우스 왼쪽 버튼 클릭 이벤트 처리
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            nearest_ekf_key, min_distance = find_nearest_ekf(click_position)
            # 거리 임곗값을 확인하여 가까운 객체에 대해서만 추적을 취소
            if nearest_ekf_key is not None and min_distance < cancel_tracking_threshold:
                del current_tracking[nearest_ekf_key]
                # print(f"Tracking {nearest_ekf_key} cancelled.")

        else:
            # 단순 좌클릭: 가장 가까운 클러스터 찾기 및 추적 시작
            if current_clusters:
                nearest_cluster, nearest_distance = find_nearest_cluster(click_position)

            # print("클릭된 가장 가까운 클러스터: ", nearest_cluster)
            if nearest_cluster is not None:
                current_tracking.clear()

                ekf = ExtendedKalmanFilter()    # 확장칼만필터 생성
                ekf_id = ekf_next_id
                ekf_next_id = 0 # 무조건 추적 하나 이상 하지 말것

                tracking = Tracking(ekf_id, ekf, nearest_distance)    # 추적 인스턴스 생성
                tracking.ekf.correct(nearest_cluster.position.tuple())
                tracking.ekf.predict()
                current_tracking[tracking.id] = tracking
        
        # print("current_clusters: ", current_clusters)
        # print("current_tracking: ", current_tracking)


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


def draw_coordinates_with_clustering(coords, dt, input_scale=resolution):
    """
    coords: LiDAR로부터 얻은 좌표 리스트
    input_scale: LiDAR의 최대 측정 거리 (mm 단위가 일반적)
    output_size: 출력 이미지의 크기 (width, height)
    """

    global next_label, image, current_clusters, current_tracking, lidar_2d_positions


    # DBSCAN 클러스터링 수행
    if len(coords) > 0:
        # 스케일링된 좌표 준비
        scaled_coords = [scale_point(x, y, input_scale, output_size) for x, y in coords]

        X = np.array(scaled_coords)
        
        # DBSCAN 알고리즘 적용
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        
        # 클러스터링 결과
        labels = dbscan.labels_ # 라벨 
        unique_labels = set(labels) - {-1}  # 이상치 추적 X 

        for label in unique_labels:
            cluster = coords[labels == label]
            centroid = np.mean(cluster, axis=0)
            x, y = scale_point(centroid[0], centroid[1], input_scale, output_size)

            cluster_instance = Cluster(label, (x, y))
            current_clusters[cluster_instance.label] = cluster_instance

            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(image, f'{cluster_instance.label}', (cluster_instance.position.x, cluster_instance.position.y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          

        aprilboard_cluster = None

        # 추적 결과 시각화
        for tracking in current_tracking.values():
            tracking.ekf.predict(dt)
            tracking_x, tracking_y = map(int, (tracking.ekf.state[0], tracking.ekf.state[1]))
            nearset_cluster, _ = find_nearest_cluster((tracking_x, tracking_y))
            tracking.ekf.correct(nearset_cluster.position.tuple())
            
            # April 태그 보드가 해당하는 추적 id는 무조건 0으로 가정
            if tracking.id == 0:
                aprilboard_cluster = nearset_cluster

            cv2.circle(image, (tracking_x, tracking_y), 3, (0, 0, 255), -1)
            cv2.putText(image, f'{tracking.id}', (tracking_x, tracking_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

         
        # 클러스터링 결과 시각화
        for (x, y), label in zip(scaled_coords, labels):
            color = (255, 255, 255) if label != -1 else (125, 125, 125)  # 잡음은 회색, 아니면 흰색

            if aprilboard_cluster != None and label == aprilboard_cluster.label:
                color = (255, 0, 0)
                lidar_2d_positions.append((x, y, 0))

            cv2.circle(image, (x, y), 1, color, -1)



# AprilTag 검출 및 3D-2D 점 쌍 생성 함수
def detect_apriltag(detector, frame):
    global chess_2d_positions

    # 이미지 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    # 검출된 태그 순회
    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners

        corner_01 = (int(corners[0][0][0]), int(corners[0][0][1]))
        corner_02 = (int(corners[1][0][0]), int(corners[1][0][1]))
        corner_03 = (int(corners[2][0][0]), int(corners[2][0][1]))
        corner_04 = (int(corners[3][0][0]), int(corners[3][0][1]))

        center_x = int((corner_01[0] + corner_02[0] + corner_03[0] + corner_04[0]) / 4)
        center_y = int((corner_01[1] + corner_02[1] + corner_03[1] + corner_04[1]) / 4)
        tag_center = (center_x, center_y)

        cv2.line(frame, corner_01, corner_02, (255, 0, 0), 1)
        cv2.line(frame, corner_02, corner_03, (255, 102, 0), 1)
        cv2.line(frame, corner_03, corner_04, (0, 255, 102), 1)
        cv2.line(frame, corner_04, corner_01, (0, 255, 0), 1)

        cv2.putText(frame, str(tag_id), (center_x - 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.circle(frame, tag_center, 3, (0, 0, 255), 1)

        if tag_id in [6, 7, 8, 9, 10, 11]:
            chess_2d_positions.append((tag_id, corner_01, corner_02))   # 태그의 좌, 우 하단의 모서리 좌표 추가


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

def calibrate_camera_lidar(april_points, lidar_points, camera_matrix, dist_coeffs):
    """
    SolvePnP를 사용하여 카메라와 LiDAR 간의 외부 행렬을 계산합니다.

    Parameters:
    - april_points: 카메라 이미지 상의 AprilTag 위치 좌표 목록 (2D 이미지 포인트)
    - lidar_points: LiDAR로부터 얻은 3D 좌표 목록
    - camera_matrix: 카메라의 내부 행렬
    - dist_coeffs: 카메라의 왜곡 계수

    Returns:
    - success: solvePnP 호출 성공 여부
    - rvec: 회전 벡터
    - tvec: 평행이동 벡터
    """
    # AprilTag 포인트는 2D, LiDAR 포인트는 3D이므로 solvePnP에 적합
    object_points = np.array(lidar_points, dtype=np.float32)
    image_points = np.array(april_points, dtype=np.float32)

    # solvePnP 호출
    # success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)



    # using solvePnP to calculate R, t
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    rvec, _ = cv2.Rodrigues(rvec)
    tvec = tvec

    extrinsic = np.append(rvec, tvec, axis=1)
    extrinsic = np.append(extrinsic, [[0, 0, 0, 1]], axis=0)

    return success, rvec, tvec, extrinsic


def project_lidar_to_camera(image, lidar_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    LiDAR 포인트를 카메라 이미지 상에 투영합니다.

    Parameters:
    - image: 투영할 대상인 카메라 이미지
    - lidar_points: LiDAR로부터 얻은 3D 좌표 목록
    - rvec: 회전 벡터
    - tvec: 평행이동 벡터
    - camera_matrix: 카메라의 내부 행렬q
    - dist_coeffs: 카메라의 왜곡 계수
    """


    # LiDAR 포인트를 이미지 상의 포인트로 변환
    img_points, _ = cv2.projectPoints(lidar_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 변환된 포인트를 이미지에 표시
    for i, p in enumerate(img_points):
        # 유효한 좌표 범위 내에 있는지 확인
        color = None
        x, y = int(p[0][0]), int(p[0][1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # if 0 < i < 180:
            #     color = (255, 0, 0)
            # elif 180 <= i < 360:
            #     color = (0, 255, 0)
            # elif 360 <= i < 540:
            #     color = (0, 0, 0)
            # else:
            #     color = (0, 0, 255)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)


def detect_chessboard(frame):
    found, corners = find_chessboard_corners(frame)
    if found:
        # 두 번째 행의 코너 추출 (0-indexed에서 두 번째 행은 인덱스 1)
        second_row_corners = extract_specific_row_corners(corners, 1)
        if second_row_corners is not None:
            # 추출된 코너 좌표에 대한 처리
            for corner in second_row_corners:
                x, y = int(corner.ravel()[0]), int(corner.ravel()[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 추출된 코너 시각화
                chess_2d_positions.append((x, y))

def main():
    global previous_time, image, lidar_2d_positions, chess_2d_positions, success, rvec, tvec, extrinsic

    cv2.namedWindow("LiDAR Scan with Clustering")
    cv2.setMouseCallback("LiDAR Scan with Clustering", mouse_click)

    lidar = LiDAR()
    lidar.startMotor()


    detector = Detector("t36h11")

    current_count = 0
    
    while True:
        
        current_time = time.time()  # 현재 시간
        dt = current_time - previous_time

        # CAMERA
        ret, frame = cap.read()
        if not ret:
            break
        
        #detect_apriltag(detector, frame)
        

        # 2D LIDAR 
        image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        coords = lidar.getXY()  # LiDAR로부터 XY 좌표 얻기

        #print(len(coords))

        draw_coordinates_with_clustering(coords, dt, resolution)  # 클러스터링 결과와 함께 좌표를 이미지 상에 표시
        cv2.imshow('LiDAR Scan with Clustering', image)



        key = cv2.waitKey(1)

        if key == ord('q'):  # 'q' 키를 누르면 루프 종료
            lidar.stopMotor()
            cv2.destroyAllWindows()
            break
        if key == ord('a'):
            detect_chessboard(frame)

            
        if key == ord('y'):
            detect_chessboard(frame)
            lidar_2d_positions = list(set(lidar_2d_positions))
            chess_2d_positions = list(set(chess_2d_positions))

            # 2D LiDAR에서 April tag를 추적하고 있고, 카메라로 April tag를 찾았을 때 캘리브레이션 진행
            if len(lidar_2d_positions) > 0 and len(chess_2d_positions) == chessboard_size[0]:
                # 좌표들 오름차순 정렬
                lidar_2d_positions.sort(key=lambda x: (x[0], x[1]))
                chess_2d_positions.sort(key=lambda x: (x[0]))

                # 맨 처음 좌표와 맨 끝 좌표를 기준으로 좌표를 나눔 
                divide_num = len(chess_2d_positions)
                divide_lidar_points = divide_points(lidar_2d_positions[0], lidar_2d_positions[-1], divide_num-1)

                # 라이다, april tag를 모음
                calibration_for_lidar_points.extend(divide_lidar_points)
                calibration_for_chess_points.extend(chess_2d_positions)


                # print(len(calibration_for_lidar_points))
                # print(len(calibration_for_chess_points))
            
                # print("현재까지 모은 lidar 좌표들")
                # print(calibration_for_lidar_points)
                # print("현재까지 모은 chess 좌표들")
                # print(calibration_for_chess_points)
                print()

                current_count += 1
                if current_count >= calibration_count:
                    success, rvec, tvec, extrinsic = calibrate_camera_lidar(calibration_for_chess_points, calibration_for_lidar_points, camera_matrix, dist_coeffs)
                    break

        cv2.imshow("april_result", frame)

        lidar_2d_positions.clear()
        chess_2d_positions.clear()

        previous_time = current_time

    # 기존의 창들 닫음
    cv2.destroyAllWindows()

    print(rvec)
    print(tvec)
    print(extrinsic)

    time.sleep(2)

    reprojection_error = None
    if success:
        # print(calibration_for_lidar_points)
        # print(calibration_for_chess_points)

        # 객체 점과 이미지 점을 numpy 배열로 변환
        lidar_points = np.array(calibration_for_lidar_points, dtype=np.float32)
        image_points = np.array(calibration_for_chess_points, dtype=np.float32)  # 직접 변환

        # 재투영 오차 계산
        reprojection_error = calculate_reprojection_error(
            lidar_points, 
            image_points, 
            rvec, 
            tvec, 
            camera_matrix, 
            dist_coeffs
        )
    print(f"재투영 오차: {reprojection_error}")
    save_extrinsic_to_json(rvec, tvec, extrinsic, reprojection_error, TARGET_DESCRIPTION)

    while success:
        # CAMERA
        ret, frame = cap.read()
        if not ret:
            break

        coords = lidar.getXY()  # LiDAR로부터 XY 좌표 얻기
        # print(len(coords))
        scaled_coords = [scale_point(x, y, resolution, output_size) for x, y in coords]

        # numpy array로 변환
        points_array = np.array(scaled_coords)

        # 새로운 차원 추가
        new_dimension = np.zeros((points_array.shape[0], 1))  # 모든 원소에 대해 0 값을 가지는 새로운 차원 생성
        new_points_array = np.hstack((points_array, new_dimension))  # 기존 배열에 새로운 차원을 추가

        # LiDAR 포인트를 카메라 이미지 상에 투영
        project_lidar_to_camera(frame, new_points_array, rvec, tvec, camera_matrix, dist_coeffs)
        cv2.imshow("Projected LiDAR on Camera", frame)

        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
            break


    lidar.stopMotor()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()