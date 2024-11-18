import cv2
import numpy as np
from crplidar import LiDAR


resolution = 20000
output_size = (800, 600)


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


def main():
    global previous_time
    
    lidar = LiDAR()
    lidar.startMotor()
    
    while True:
        image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

        coords = lidar.getXY()  # LiDAR로부터 XY 좌표 얻기
        scaled_coords = [scale_point(x, y, resolution, output_size) for x, y in coords]

        for x, y in scaled_coords:
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)     
        
        cv2.imshow('LiDAR Scan', image)

        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

    lidar.stopMotor()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()