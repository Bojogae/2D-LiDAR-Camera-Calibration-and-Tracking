import numpy as np 

class ExtendedKalmanFilter:
    def __init__(self):
        # 초기 상태 벡터: [x, y, vx, vy, ax, ay] 위치, 속도, 가속도
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # 초기 공분산 행렬, 6x6으로 확장, 초기 상태 추정의 불확실성을 나타냅니다. 이 매트릭스는 대각선이 큰 값을 가질수록 초기 상태 추정에 대한 불확실성이 크다는 것을 의미
        self.P = np.eye(6, dtype=np.float32) * 1000
        # 프로세스 노이즈, 6x6으로 확장,  시스템 동작 중 발생할 수 있는 예측되지 않은 변화를 모델링
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        # 측정 노이즈, 위치 x, y만 측정하므로 2x2, 측정 과정에서 발생하는 오차를 나타냅니다. 이는 센서의 정확도와 직접 관련이 있으며, 일반적으로 센서의 사양을 기반으로 더 높은 값은 더 높은 불확실성을 의미합니다.
        self.R = np.eye(2, dtype=np.float32) * 5

    def predict(self, dt=10):
        # 상태 전이 모델을 확장하여 가속도 고려
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],  # x = x + vx*dt + 0.5*ax*dt^2
            [0, 1, 0, dt, 0, 0.5*dt**2],  # y = y + vy*dt + 0.5*ay*dt^2
            [0, 0, 1, 0, dt, 0],          # vx = vx + ax*dt
            [0, 0, 0, 1, 0, dt],          # vy = vy + ay*dt
            [0, 0, 0, 0, 1, 0],           # ax = ax
            [0, 0, 0, 0, 0, 1]            # ay = ay
        ], dtype=np.float32)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def correct(self, measurement):
        # 측정 업데이트, 여기서는 위치 x, y만 측정
        H = np.array([
            [1, 0, 0, 0, 0, 0],  # x 측정
            [0, 1, 0, 0, 0, 0]   # y 측정
        ], dtype=np.float32)
        z = np.array(measurement, dtype=np.float32)
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


"""

import numpy as np

class ExtendedKalmanFilter:
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 10
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 5

    def predict(self, dt=10):
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def correct(self, measurement):
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        z = np.array(measurement, dtype=np.float32)
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.state += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
"""



