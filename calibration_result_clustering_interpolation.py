import cv2
import numpy as np
from crplidar import LiDAR
from sklearn.cluster import DBSCAN
from cluster import Cluster

# 카메라 설정
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

# LiDAR 및 이미지 설정
resolution = 20000
output_size = (800, 600)
epsilon = 12
min_samples = 5
interpolation_density = 2
device = 1
cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_size[1])

def scale_point(x, y, input_scale, output_size):
    x_adjusted = x + input_scale / 2
    y_adjusted = y + input_scale / 2
    x_scaled = int(x_adjusted / input_scale * output_size[0])
    y_scaled = int(y_adjusted / input_scale * output_size[1])
    return x_scaled, y_scaled

def interpolate_points(p1, p2, num_points):
    """ 주어진 두 점 p1, p2 사이에 num_points 만큼의 점을 선형 보간하여 생성 """
    return [(p1[0] + i / (num_points + 1) * (p2[0] - p1[0]), 
             p1[1] + i / (num_points + 1) * (p2[1] - p1[1])) for i in range(1, num_points + 1)]


def project_points_to_image(points, rvec, tvec, camera_matrix, dist_coeffs):
    lidar_points = np.array([[x, y, z] for x, y, z in points])
    img_points, _ = cv2.projectPoints(lidar_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2)  # (N, 2) 형태로 변경

    # 유효하지 않은 좌표 제거 (NaN 또는 무한대)
    valid_points = []
    for point in img_points:
        x, y = point
        if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
            valid_points.append((int(x), int(y)))

    return valid_points

def main():
    global rvec, tvec

    lidar = LiDAR()
    lidar.startMotor()

    rvec = np.array(rvec)
    tvec = np.array(tvec)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        coords_with_distances = lidar.getXY_with_distance()

        points = []
        for x, y, distance in coords_with_distances:
            tmp = []
            tmp.extend(scale_point(x, y, resolution, output_size))
            tmp.append(distance)
            points.append(tuple(tmp))


        X = np.array([(x, y) for x, y, _ in points])
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        labels = dbscan.labels_

        result_coords = []
        for label in set(labels):
            cluster_points = [points[i] for i in range(len(points)) if labels[i] == label]
            interpolated_points = []
            for i in range(len(cluster_points) - 1):
                interpolated_points += interpolate_points(cluster_points[i], cluster_points[i+1], interpolation_density)
            
            result_coords.extend(interpolated_points)
            
            img_points = project_points_to_image(interpolated_points, rvec, tvec, camera_matrix, dist_coeffs)
            for x, y in img_points:
                if 0 <= x < output_size[0] and 0 <= y < output_size[1]:  # 화면 범위 내에서만 그림
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)



        image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        for x, y, _ in result_coords:
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)
        cv2.imshow('LiDAR Scan', image)

        cv2.imshow('Projected LiDAR on Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    lidar.stopMotor()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    