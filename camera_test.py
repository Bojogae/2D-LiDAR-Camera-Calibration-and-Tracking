import numpy as np

points_2D = np.array([
            (155, 330),
            (210, 280),
            (320, 280),
            (415, 280),
        ], dtype="double")


import cv2


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for coord in points_2D:
        cv2.circle(frame, tuple(map(int, coord)), 2, (0, 255, 0), -1)


    cv2.imshow("Projected LiDAR on Camera", frame)
    
    if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
        break
    
    
cv2.destroyAllWindows()
cap.release()