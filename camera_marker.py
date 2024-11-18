from aprilgrid import Detector
import time
import os

import cv2
import numpy as np

K = (718.0185818253857, 718.7782113904716, 323.4567247857622, 235.99239839273147)
dist = (0.09119377823619247, 0.0626265233941177, -0.007768343835168918, -0.004451209268144528, 0.48630799030494654)
mm = 87

device = 1
cap_width = 760
cap_height = 540

length_magnification = 3
cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    detector = Detector("t36h11")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        tag_id = detection.tag_id
        corners = detection.corners

        corner_01 = (int(corners[0][0][0]), int(corners[0][0][1]))
        corner_02 = (int(corners[1][0][0]), int(corners[1][0][1]))
        corner_03 = (int(corners[2][0][0]), int(corners[2][0][1]))
        corner_04 = (int(corners[3][0][0]), int(corners[3][0][1]))

        center_x = int((corner_01[0] + corner_02[0] + corner_03[0] + corner_04[0]) / 4)
        center_y = int((corner_01[1] + corner_02[1] + corner_03[1] + corner_04[1]) / 4)

        cv2.line(frame, corner_01, corner_02, (255, 0, 0), 2)
        cv2.line(frame, corner_02, corner_03, (255, 102, 100), 2)
        cv2.line(frame, corner_03, corner_04, (0, 255, 102), 2)
        cv2.line(frame, corner_04, corner_01, (100, 255, 0), 2)

        cv2.putText(frame, str(tag_id), (center_x - 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), 1)

        if tag_id == 0:
            # 여기 카메라 포즈를 추정하여 x,y,z 축 그리기
            pass

    cv2.imshow("result", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
