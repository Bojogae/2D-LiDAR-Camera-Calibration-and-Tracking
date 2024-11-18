import numpy as np

class KalmanFilter:
    def __init__(self):
        # 초기 상태 벡터: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        # 초기 공분산 행렬
        self.P = np.eye(4, dtype=np.float32) * 500
        # 프로세스 노이즈
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        # 측정 노이즈
        self.R = np.eye(2, dtype=np.float32) * 4
        # 측정 행렬
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    def predict(self):
        # 상태 전이 모델
        dt = 1.0  # 시간 간격
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def correct(self, measurement):
        z = np.array(measurement, dtype=np.float32)
        y = z - np.dot(self.H, self.state)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)