from math import pi
from queue import PriorityQueue
from collections import deque
import numpy as np
import threading
import time

from pop import Pilot

import utils.loadSensor as loadSensor

rb = 10
max_speed = 50.0
min_speed = 30.0
safe_thres = 2.0
danger_thres = 1.0
cut_thres = 2.0
max_gap = 300
min_gap = 50
window_size = 20

def preprocess_lidar(ranges):
    proc_ranges = []
    
    for i in range(0, len(ranges), window_size):
        cur_mean = round(sum(ranges[i:i + window_size]) / window_size, 5)
        for _ in range(window_size):
            proc_ranges.append(cur_mean)
    proc_ranges = np.array(proc_ranges)
    return proc_ranges

def refine_danger_range(start_i, end_i, ranges):
    p = start_i
    while p < end_i:
        if ranges[p] <= danger_thres:
            ranges[max(0, p - rb):p+rb] = 0
            p += rb
        else:
            p += 1
    return ranges

def find_best_point(start_i, end_i, ranges):
    safe_p_left = start_i
    safe_p_right = end_i
    p = start_i

    safe_range = PriorityQueue()

    while p < end_i:
        if ranges[p] >= safe_thres:
            safe_p_left = p
            p += 1

            while p < end_i and ranges[p] >= safe_thres and p - safe_p_left <= max_gap and ranges[p] - ranges[max(0, p-1)] < cut_thres:
                p += 1

            safe_p_right = p - 1

            if safe_p_right != safe_p_left:
                safe_range.put((-(np.max(ranges[safe_p_left:safe_p_right])), (safe_p_left,safe_p_right)))
        else:
            p += 1
    if safe_range.empty():
        print('no safe range')
        return np.argmax(ranges)
    else:
        while not safe_range.empty():
            safe_p_left, safe_p_right = safe_range.get()[1]
            target = (safe_p_left + safe_p_right) // 2

            if 179 <= target <= 900 and safe_p_right - safe_p_left > min_gap:
                # print(f'left: {safe_p_left}, right: {safe_range}')
                return target
        return target

def rescale_steering_angle(steering_angle, min_angle, max_angle):
    return 2 * (steering_angle - min_angle) / (max_angle - min_angle) - 1



def lidar_callback(car, lidar, stop_event):
    car.forward()
    while not stop_event.is_set():
        scans = lidar.getRanges()

        angles = scans[:, 0]
        ranges = scans[:, 1]
        ranges = ranges * 0.001

    

        angle_differences = np.diff(angles)
        angle_increment = np.mean(angle_differences)


        angle_min = np.min(angles)

        # print(f"angle_increment: {angle_increment}")

        # print(scans)

        proc_ranges = preprocess_lidar(ranges)
        proc_ranges = refine_danger_range(start_i=0, end_i=len(proc_ranges), ranges=proc_ranges)

        farmost_p_idx = find_best_point(start_i=0, end_i=len(proc_ranges), ranges=proc_ranges)
        steering_angle = angle_min + farmost_p_idx * angle_increment
        steering_angle = rescale_steering_angle(steering_angle, -np.pi, np.pi)
        car.steering = steering_angle



        # print(f"farmost_p_idx: {farmost_p_idx}")
        # print(f"farmost_p_range: {proc_ranges[farmost_p_idx]}")
        # print(f"steering_angle: {steering_angle * 180 / pi}")

        velocity = max_speed

        print(f"steering_angle: {steering_angle}")

        if abs(steering_angle) >= 0.3:
            velocity = min_speed
        
        # print(steering_angle)
        car.steering = steering_angle
        car.setSpeed(velocity)








def main():
    car = Pilot.AutoCar()
    car.setSpeed(1)

    lidar = loadSensor.getLiDAR()
    stop_event = threading.Event()

    lidar_thead = threading.Thread(target=lidar_callback, args=(car, lidar, stop_event))
    lidar_thead.start()

    try:
        while True:
            result = input()
            if result == 'q':
                break
    finally:
        stop_event.set()
        lidar_thead.join()
        lidar.stopMotor()
        car.steering = 0
        car.stop()



if __name__ == "__main__":
    main()