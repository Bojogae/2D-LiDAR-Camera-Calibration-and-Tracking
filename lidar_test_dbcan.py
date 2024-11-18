import cv2
import numpy as np
from crplidar import LiDAR
from sklearn.cluster import DBSCAN
from cluster import Cluster
import matplotlib.pyplot as plt

resolution = 20000
epsilon = 12
min_samples = 5

current_clusters = {} # 현재 클러스터의 중심점을 저장

output_size = (800, 600)

cancel_tracking_threshold = 3

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


def draw_coordinates_with_clustering(coords, input_scale=resolution):
    """
    coords: LiDAR로부터 얻은 좌표 리스트
    input_scale: LiDAR의 최대 측정 거리 (mm 단위가 일반적)
    output_size: 출력 이미지의 크기 (width, height)
    """

    global next_label, image, current_clusters

    

    # DBSCAN 클러스터링 수행
    if len(coords) > 0:
        print(coords)
        print(type(coords))
        # 스케일링된 좌표 준비
        scaled_coords = [scale_point(x, y, input_scale, output_size) for x, y in coords]
        X = np.array(scaled_coords)
        
        # DBSCAN 알고리즘 적용
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        
        # 클러스터링 결과
        labels = dbscan.labels_ # 라벨 
        unique_labels = set(labels) - {-1}  # 이상치 추적 X 

        # HSV 색상 공간에서 라벨별로 색상 생성
        hsv_colors = [(int(label * 255 / len(unique_labels)), 255, 255) for label in unique_labels]
        rgb_colors = [cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0] for hsv_color in hsv_colors]
        color_map = {label: rgb_color for label, rgb_color in zip(unique_labels, rgb_colors)}

        # 클러스터링 결과 시각화
        for (x, y), label in zip(scaled_coords, labels):
            if label != -1:  # 라벨이 -1이 아닌 경우에만 색상을 적용
                color = color_map[label]
                cv2.circle(image, (x, y), 1, color.tolist(), -1)
            else:  # 잡음(이상치)은 회색으로 표시
                cv2.circle(image, (x, y), 1, (125, 125, 125), -1)


        # 클러스터링 중심 시각화 \
        for label in unique_labels:
            cluster = coords[labels == label]
            centroid = np.mean(cluster, axis=0)
            x, y = scale_point(centroid[0], centroid[1], input_scale, output_size)

            cluster_instance = Cluster(label, (x, y))
            current_clusters[cluster_instance.label] = cluster_instance

            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(image, f'{cluster_instance.label}', (cluster_instance.position.x, cluster_instance.position.y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          

         


def main():
    global previous_time, image

    lidar = LiDAR()
    lidar.startMotor()
    
    while True:
        image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

        coords = lidar.getXY()  # LiDAR로부터 XY 좌표 얻기
        
        draw_coordinates_with_clustering(coords, resolution)  # 클러스터링 결과와 함께 좌표를 이미지 상에 표시

        cv2.imshow('LiDAR Scan with Clustering', image)

        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
            break


    lidar.stopMotor()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()