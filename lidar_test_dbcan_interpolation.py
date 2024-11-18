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



image = None


interpolation_density = 2


def interpolate_points(p1, p2, num_points):
    """ 주어진 두 점 p1, p2 사이에 num_points 만큼의 점을 선형 보간하여 생성 """
    return [(p1[0] + i / (num_points + 1) * (p2[0] - p1[0]), 
             p1[1] + i / (num_points + 1) * (p2[1] - p1[1])) for i in range(1, num_points + 1)]



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
    global image, current_clusters

    
    if len(coords) > 0:
        scaled_coords = [scale_point(x, y, input_scale, output_size) for x, y in coords]
        X = np.array(scaled_coords)
        
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        labels = dbscan.labels_
        unique_labels = set(labels) - {-1}  # 이상치를 제외한 유니크 라벨

        # HSV 색상 공간에서 라벨별로 색상 생성
        hsv_colors = [(int(label * 255 / len(unique_labels)), 255, 255) for label in unique_labels]
        rgb_colors = [cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0] for hsv_color in hsv_colors]
        color_map = {label: tuple(color.tolist()) for label, color in zip(unique_labels, rgb_colors)}

        # 이상치(-1)에 대한 회색 색상 추가
        color_map[-1] = (125, 125, 125)

        # 모든 점을 저장할 임시 리스트
        points_to_draw = []

        for label in set(labels):  # 이상치를 포함한 모든 라벨 처리
            cluster_points = coords[labels == label]
            if label != -1 and len(cluster_points) > 1:
                # 정렬
                sorted_indices = np.lexsort((cluster_points[:, 1], cluster_points[:, 0]))
                cluster_points = cluster_points[sorted_indices]

                # 보간
                interpolated_points = []
                for i in range(len(cluster_points) - 1):
                    p1, p2 = cluster_points[i], cluster_points[i+1]
                    interpolated_points.extend(interpolate_points(p1, p2, interpolation_density))

                # 전체 점들 추가
                all_points = np.vstack([cluster_points, interpolated_points])
            else:
                all_points = cluster_points  # 이상치인 경우 보간 없이 직접 추가

            for point in all_points:
                x, y = scale_point(point[0], point[1], input_scale, output_size)
                points_to_draw.append((x, y, color_map[label]))

            if label != -1:
                # 중심 계산 및 저장
                centroid = np.mean(cluster_points, axis=0)
                x, y = scale_point(centroid[0], centroid[1], input_scale, output_size)
                cluster_instance = Cluster(label, (x, y))
                current_clusters[cluster_instance.label] = cluster_instance
                points_to_draw.append((x, y, (0, 255, 0)))  # 중심점
                points_to_draw.append(("text", x, y - 20, f'{label}', 0.5, (0, 255, 0)))

        # 모든 점 그리기
        for item in points_to_draw:
            if item[0] == "text":
                cv2.putText(image, item[3], (item[1], item[2]), cv2.FONT_HERSHEY_SIMPLEX, item[4], item[5], 2)
            else:
                cv2.circle(image, (item[0], item[1]), 1, item[2], -1)


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