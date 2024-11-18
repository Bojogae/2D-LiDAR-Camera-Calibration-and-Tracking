# 2D LiDAR-Camera Calibration

## 프로젝트 소개
DBSCAN 클러스터링과 확장 칼만 필터를 활용하여 2D LiDAR와 카메라 센서를 정밀하게 캘리브레이션하는 프로젝트입니다. AprilTag를 추적하여 두 센서 간의 외부 파라미터를 계산하고, 이를 기반으로 환경 정보를 융합합니다.

## 주요 기능
- **DBSCAN 클러스터링**: 2D LiDAR 포인트 클라우드에서 밀집 영역 추출 및 노이즈 제거.
- **확장 칼만 필터**: AprilTag를 안정적으로 추적하여 특징점 확보.
- **다항식 보간법**: 부족한 대응점 보완을 통한 정밀 캘리브레이션.
- **PnP 알고리즘**: 카메라와 LiDAR 간의 외부 파라미터 계산.

## 기술 스택
- OpenCV
- DBSCAN 알고리즘
- 확장 칼만 필터
- PnP 알고리즘

## 설치 및 실행
1. OpenCV와 필요한 라이브러리 설치.
2. AprilTag 코드를 프로젝트에 통합.
3. LiDAR와 카메라 데이터를 캡처하여 캘리브레이션 수행.

## 안내
시스템 개발 도중에 백업용으로 만든 코드들입니다.
현재는 ROS2 Foxy 기반의 시스템을 다시 개발하고 있습니다.

아래 경로에서 업데이트된 상황을 확인할 수 있습니다.
!(https://github.com/wndudwkd003/2D-LiDAR-Camera-Calibration)


## 사진
![캘리브레이션 결과](https://github.com/Bojogae/Materials/blob/main/2D-LiDAR-Camera-Calibration/1.png)
![캘리브레이션 결과](https://github.com/Bojogae/Materials/blob/main/2D-LiDAR-Camera-Calibration/2.png)
![캘리브레이션 결과](https://github.com/Bojogae/Materials/blob/main/2D-LiDAR-Camera-Calibration/3.png)
![캘리브레이션 결과](https://github.com/Bojogae/Materials/blob/main/2D-LiDAR-Camera-Calibration/4.png)
![캘리브레이션 결과](https://github.com/Bojogae/Materials/blob/main/2D-LiDAR-Camera-Calibration/5.png)
