from pyrplidar import PyRPlidar
import time
import traceback  
import numpy as np 
import math 

class LiDAR:
    def __init__(self, RPM=880):
        self.RPM = RPM

        try:
            self.lidar = PyRPlidar()
            self.lidar.connect(port="/dev/ttyUSB0", baudrate=256000, timeout=3)
            print(f"LiDAR Success connect!")

        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.
            

    def startMotor(self):
        try:
            self.lidar.set_motor_pwm(self.RPM)
            self.scan_generator = self.lidar.start_scan_express(4)

            print(f"Current RPM is {self.RPM}")
            time.sleep(2)

        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.
            self.lidar.stop()
            self.lidar.set_motor_pwm(0)
            self.lidar.disconnect()

    def stopMotor(self):
        try:
            self.lidar.stop()
            self.lidar.set_motor_pwm(0)
            self.lidar.disconnect()
        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.


    def disconnection(self):
        try:
            self.lidar.disconnect()
        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.

             
    def getScan(self):
        scans = []

        try:
            start_angle = None
            has_started = False

            for scan in self.scan_generator():
                scans.append(scan)
                if scan.start_flag == True:
                    if has_started == True:
                        break
                    has_started = True


            # for scan in self.scan_generator():
            #     scans.append(scan)
            #     if scan.start_flag == True:
            #         start_angle = scan.angle
            #         has_started = True
            #     else:
            #         if has_started and abs(scan.angle - start_angle) <= 0.05:
            #             break  
                    
        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.
            self.lidar.stop()
            self.lidar.set_motor_pwm(0)
            self.lidar.disconnect()

        return scans

    def getXY(self):
        coords = []

        try:
            scans = self.getScan()
            angles = np.radians([scan.angle for scan in scans]) - np.pi/2
            # angles = np.radians([scan.angle for scan in scans]) 
            distances = np.array([scan.distance for scan in scans])

            x = distances * np.cos(angles)  # X 좌표 계산
            y = distances * np.sin(angles)  # Y 좌표 계산

            coords = np.vstack((x, y)).T


        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.
            self.lidar.stop()
            self.lidar.set_motor_pwm(0)
            self.lidar.disconnect()
        
        return coords


    def getXY_with_distance(self):
        coords = []

        try:
            scans = self.getScan()
            angles = np.radians([scan.angle for scan in scans]) - np.pi/2
            # angles = np.radians([scan.angle for scan in scans]) 
            distances = np.array([scan.distance for scan in scans])

            x = distances * np.cos(angles)  # X 좌표 계산
            y = distances * np.sin(angles)  # Y 좌표 계산

            coords = np.vstack((x, y, distances)).T


        except Exception as e:  # 연결 관련 예외를 캐치합니다.
            print('Error during connection or setup:', str(e))
            traceback.print_exc()  # 예외의 상세한 정보를 출력합니다.
            self.lidar.stop()
            self.lidar.set_motor_pwm(0)
            self.lidar.disconnect()
        
        return coords


    def test(self):
        self.lidar.connect(port="/dev/ttyUSB0", baudrate=256000, timeout=3)
        info = self.lidar.get_info()
        print("info :", info)

        health = self.lidar.get_health()
        print("health :", health)

        samplerate = self.lidar.get_samplerate()
        print("samplerate :", samplerate)


        scan_modes = self.lidar.get_scan_modes()
        print("scan modes :")
        for scan_mode in scan_modes:
            print(scan_mode)


        self.lidar.disconnect()
