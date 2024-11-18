import cv2
import numpy as np
from scipy.optimize import least_squares

from crplidar import LiDAR
import time



# 카메라 내부 매개변수와 왜곡 계수
# 카메라 내부 매개변수와 왜곡 계수
camera_matrix = np.array([[718.0185818253857, 0, 323.4567247857622],
                          [0, 718.7782113904716, 235.99239839273147],
                          [0, 0, 1]])
dist_coeffs = np.array([0.09119377823619247, 0.0626265233941177, -0.007768343835168918, -0.004451209268144528, 0.48630799030494654])


# 아래 리스트는 실제로 캘리브레이션에 사용될 좌표들임
calibration_for_lidar_points = []   # [(x, y, 0), (x, y, 0), (x, y, 0)] lidar 좌표들 모음
calibration_for_camera_points = []   # [(x, y), (x, y), (x, y)] april 태그의 좌표들 모음

device = 1
cap_width = 900
cap_height = 480

cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

resolution = 15000
output_size = (cap_width, cap_height)

success = None
rvec = None
tvec = None
extrinsic = None

camera_view_name = "Camera View"
lidar_view_name = "LiDAR View"

camera_frame = None
lidar_map = None


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

    return mean_error



# 마우스 클릭 리스너
def mouse_click(event, x, y, flags, param):
    global calibration_for_lidar_points, calibration_for_camera_points
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = np.array([x, y])
        if param == camera_view_name:
            calibration_for_camera_points.append((x, y))
            cv2.circle(camera_frame, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(param, camera_frame)
            print(f"Clicked at {click_position[0]}, {click_position[1]} on {param} view, camera size: {len(calibration_for_camera_points)}")

        else:
            calibration_for_lidar_points.append((x, y, 0))
            cv2.circle(lidar_map, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(param, lidar_map)
            print(f"Clicked at {click_position[0]}, {click_position[1]} on {param} view, lidar size: {len(calibration_for_lidar_points)}")


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
    for p in img_points:
        # 유효한 좌표 범위 내에 있는지 확인
        x, y = int(p[0][0]), int(p[0][1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)



def main():
    global calibration_for_camera_points, calibration_for_lidar_points, success, rvec, tvec, extrinsic
    global camera_frame, lidar_map

    cv2.namedWindow(camera_view_name)
    cv2.namedWindow(lidar_view_name)
    cv2.setMouseCallback(camera_view_name, mouse_click, camera_view_name)
    cv2.setMouseCallback(lidar_view_name, mouse_click, lidar_view_name)

    lidar = LiDAR()
    lidar.startMotor()

    while True:
        # CAMERA
        ret, camera_frame = cap.read()
        if not ret:
            break
        
        # 2D LIDAR 
        lidar_map = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        coords = lidar.getXY()  # LiDAR로부터 XY 좌표 얻기
        scaled_coords = [scale_point(x, y, resolution, output_size) for x, y in coords]
        for coord in scaled_coords:
            cv2.circle(lidar_map, coord, 1, (255, 255, 255), -1)
        
        # 라이다 화면
        cv2.imshow(lidar_view_name, lidar_map)
            
        # 카메라 화면
        cv2.imshow(camera_view_name, camera_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):  # 'q' 키를 누르면 루프 종료
            success, rvec, tvec, extrinsic = calibrate_camera_lidar(calibration_for_camera_points, calibration_for_lidar_points, camera_matrix, dist_coeffs)
            cv2.destroyAllWindows()
            break


    # 기존의 창들 닫음
    cv2.destroyAllWindows()

    print(rvec)
    print(tvec)
    print(extrinsic)

    time.sleep(2)

    reprojection_error = None
    if success:
        print(calibration_for_lidar_points)
        print(calibration_for_camera_points)

        # 객체 점과 이미지 점을 numpy 배열로 변환
        lidar_points = np.array(calibration_for_lidar_points, dtype=np.float32)
        image_points = np.array(calibration_for_camera_points, dtype=np.float32)  

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