import cv2
import numpy as np
from crplidar import LiDAR
from sklearn.cluster import DBSCAN
from ExtendedKalmanFilter import ExtendedKalmanFilter
import time
from cluster import Cluster
from tracker import Tracking


resolution = 20000
epsilon = 12
min_samples = 5

ekf_next_id = 0  # 새로 추가할 확장칼만필터 추적 id 
current_tracking = {}  # 추적 객체

current_clusters = {} # 현재 클러스터의 중심점을 저장

output_size = (800, 600)

previous_time = time.time()
cancel_tracking_threshold = 3

image = None

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
                print(f"Tracking {nearest_ekf_key} cancelled.")

        else:
            # 단순 좌클릭: 가장 가까운 클러스터 찾기 및 추적 시작
            if current_clusters:
                nearest_cluster, nearest_distance = find_nearest_cluster(click_position)

            # print("클릭된 가장 가까운 클러스터: ", nearest_cluster)
            if nearest_cluster is not None:
                ekf = ExtendedKalmanFilter()    # 확장칼만필터 생성
                ekf_id = ekf_next_id
                ekf_next_id += 1

                tracking = Tracking(ekf_id, ekf, nearest_distance)    # 추적 인스턴스 생성
                tracking.ekf.correct(nearest_cluster.position.tuple())
                tracking.ekf.predict()
                current_tracking[tracking.id] = tracking
        
        # print("current_clusters: ", current_clusters)


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

    global next_label, image, current_clusters, current_tracking

    

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
          

        # 클러스터링 결과 시각화
        for (x, y), label in zip(scaled_coords, labels):
            color = (255, 255, 255) if label != -1 else (125, 125, 125)  # 잡음은 회색, 아니면 흰색
            cv2.circle(image, (x, y), 1, color, -1)

        # 추적 결과 시각화
        for tracking in current_tracking.values():
            tracking.ekf.predict(dt) # 예측 단계

            # # 가장 가까운 클러스터를 찾음
            # tracking_x, tracking_y = map(int, (tracking.ekf.state[0], tracking.ekf.state[1]))
            # nearset_cluster, nearst_distance = find_nearest_cluster((tracking_x, tracking_y))

            # # 좌표 근처에 클러스터가 없을 경우 무언가 가로막힌 상황
            # if tracking.last_nearst_distance > nearst_distance * 2:
            #     print(nearst_distance, "nearst_distance: 예측 단계")
            #     print(tracking.last_nearst_distance, "last_nearst_distanc: 예측 단계")

            # else:
            #     tracking.last_nearst_distance = nearst_distance
            #     tracking.ekf.correct(nearset_cluster.position.tuple())
            #     print(nearst_distance, "nearst_distance: 보정 단계")

            tracking_x, tracking_y = map(int, (tracking.ekf.state[0], tracking.ekf.state[1]))
            nearset_cluster, nearst_distance = find_nearest_cluster((tracking_x, tracking_y))
            tracking.ekf.correct(nearset_cluster.position.tuple())
            
            # nearset_cluster, _ = find_nearest_cluster((tracking_x, tracking_y))
            # tracking.ekf.correct(nearset_cluster.position.tuple())

            cv2.circle(image, (tracking_x, tracking_y), 3, (0, 0, 255), -1)
            cv2.putText(image, f'{tracking.id}', (tracking_x, tracking_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def main():
    global previous_time, image

    cv2.namedWindow("LiDAR Scan with Clustering")
    cv2.setMouseCallback("LiDAR Scan with Clustering", mouse_click)


    lidar = LiDAR()
    lidar.startMotor()
    
    while True:
        current_time = time.time()  # 현재 시간
        dt = current_time - previous_time

        image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

        coords = lidar.getXY()  # LiDAR로부터 XY 좌표 얻기
        
        draw_coordinates_with_clustering(coords, dt, resolution)  # 클러스터링 결과와 함께 좌표를 이미지 상에 표시

        cv2.imshow('LiDAR Scan with Clustering', image)

        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

        previous_time = current_time

    lidar.stopMotor()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()