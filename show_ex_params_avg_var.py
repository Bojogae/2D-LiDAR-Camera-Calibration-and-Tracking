import json
import numpy as np
import matplotlib.pyplot as plt

TARGET_DESCRIPTION = "1_SbS"

# JSON 파일에서 데이터 읽기
def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# 각 행렬과 평균 행렬 사이의 유클리드 거리 계산
def calculate_distances(average_extrinsic, extrinsics):
    distances = [np.linalg.norm(extrinsic - average_extrinsic) for extrinsic in extrinsics]
    return distances

# 외부 행렬의 평균 계산
def calculate_average_extrinsics(data):
    extrinsics = [np.array(item['extrinsic_matrix']) for item in data if item['description'] == TARGET_DESCRIPTION]
    if not extrinsics:
        return None
    average_extrinsic = np.mean(extrinsics, axis=0)
    return average_extrinsic, extrinsics

# 결과 시각화
def display_results(average_extrinsic, distances):
    print(average_extrinsic)

    # 평균 행렬 플로팅
    plt.axhline(y=np.mean(distances), color='gray', linestyle='-', label='Average Distance')

    # 각 행렬과 평균 사이의 거리 플로팅
    
    plt.plot(distances, marker='o', label='Distances from Average')

    plt.xlabel('Extrinsic Matrix ID')
    plt.ylabel('Distance from Average')
    plt.title('Distances of Extrinsic Matrices from Average')
    plt.legend()
    plt.show()

# 메인 함수
def main():
    filename = '/home/soda/Documents/Devleopment/chess_extrinsic_data.json'  # JSON 파일 경로 지정
    data = read_json(filename)
    data = data['data']  # 데이터 항목만 추출

    average_extrinsic, extrinsics = calculate_average_extrinsics(data)
    if average_extrinsic is not None:
        distances = calculate_distances(average_extrinsic, extrinsics)
        display_results(average_extrinsic, distances)
    else:
        print("No data found with description '1_SbS'.")

if __name__ == '__main__':
    main()
