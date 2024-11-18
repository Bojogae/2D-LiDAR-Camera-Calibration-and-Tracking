import json
import numpy as np
import matplotlib.pyplot as plt

# JSON 파일에서 데이터 읽기
def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['data']

# 모든 외부 행렬의 평균과 description 별 평균 계산
def calculate_averages(data):
    all_extrinsics = [np.array(item['extrinsic_matrix']) for item in data]
    overall_average = np.mean(all_extrinsics, axis=0)

    description_averages = {}
    for item in data:
        desc = item['description']
        if desc not in description_averages:
            description_averages[desc] = []
        description_averages[desc].append(np.array(item['extrinsic_matrix']))

    for key in description_averages:
        description_averages[key] = np.mean(np.array(description_averages[key], dtype=float), axis=0)

    return overall_average, description_averages

# 차이 계산
def calculate_differences(overall_average, description_averages):
    differences = {}
    for desc, average in description_averages.items():
        differences[desc] = np.linalg.norm(average - overall_average)
    return differences

# 결과 시각화 및 가장 큰/작은 차이 출력
def display_results(differences):
    descriptions = list(differences.keys())
    distances = list(differences.values())

    plt.figure(figsize=(10, 5))
    plt.bar(descriptions, distances, color='b')
    plt.xlabel('Description')
    plt.ylabel('Distance from Overall Average')
    plt.title('Difference of Description Averages from Overall Average')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    max_diff = max(distances)
    min_diff = min(distances)
    max_desc = descriptions[distances.index(max_diff)]
    min_desc = descriptions[distances.index(min_diff)]

    print(f"Largest Difference: {max_diff} (Description: {max_desc})")
    print(f"Smallest Difference: {min_diff} (Description: {min_desc})")

# 메인 함수
def main():
    filename = '/home/soda/Documents/Devleopment/extrinsic_data.json'
    data = read_json(filename)
    overall_average, description_averages = calculate_averages(data)
    differences = calculate_differences(overall_average, description_averages)
    display_results(differences)

if __name__ == '__main__':
    main()
