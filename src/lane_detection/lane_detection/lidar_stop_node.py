import math
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# 파라미터 설정
MAX_STEER = 0.42
MAX_SPEED = 1.00
MIN_SPEED = 0.25

Kp = 1.20
Kd = 0.45

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

class WhiteLineControl(Node):
    def __init__(self):
        super().__init__('white_line_control_node')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/white/image_raw', 10)

        # 이미지 및 라이다 구독
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.img_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.prev_error = 0.0
        self.prev_lane_center = None
        self.prev_speed = 0.0
        self.last_lane_width = 240
        
        # 라이다 상태 변수
        self.front_obs = False
        self.left_obs_dist = 9.9
        self.right_obs_dist = 9.9

        self.get_logger().info('white line control node (Avoidance Mode) start')

    def scan_callback(self, msg):
        self.front_obs = False
        self.left_obs_dist = 9.9
        self.right_obs_dist = 9.9
        
        front_hit = 0
        for i, dist in enumerate(msg.ranges):
            if dist <= 0.0 or math.isinf(dist) or math.isnan(dist): continue
            
            angle = math.degrees(msg.angle_min + i * msg.angle_increment)
            while angle > 180: angle -= 360
            while angle < -180: angle += 360
            
            # 정면 (-45 ~ 45도): 비상 정지 구역
            if -20.0 <= angle <= 20.0 and dist < 0.4:
                front_hit += 1
            # 측면 (회피 구역)
            elif 20.0 < angle <= 90.0 and dist < 0.6:
                if dist < self.left_obs_dist: self.left_obs_dist = dist
            elif -90.0 <= angle < -20.0 and dist < 0.6:
                if dist < self.right_obs_dist: self.right_obs_dist = dist
                    
        self.front_obs = (front_hit >= 5)

    def detect_white(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_lower = np.array([0, 0, 170])
        white_upper = np.array([179, 90, 255])
        mask = cv2.inRange(hsv, white_lower, white_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def img_warp(self, img):
        img_x, img_y = img.shape[1], img.shape[0]
        src_center_offset = [200, 315]
        src = np.float32([[0, 479], [src_center_offset[0], src_center_offset[1]], [640 - src_center_offset[0], src_center_offset[1]], [639, 479]])
        dst_offset = [round(img_x * 0.125), 0]
        dst = np.float32([[dst_offset[0], img_y], [dst_offset[0], 0], [img_x - dst_offset[0], 0], [img_x - dst_offset[0], img_y]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (img_x, img_y))

    def get_filtered_line_center(self, mask, x_offset=0, image_center=320):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300: continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx = int(M['m10'] / M['m00']) + x_offset
            centers.append(cx)
        if len(centers) == 0: return None
        centers.sort()
        filtered = []
        min_gap = 60
        for cx in centers:
            if len(filtered) == 0: filtered.append(cx)
            else:
                if abs(cx - filtered[-1]) < min_gap:
                    if abs(cx - image_center) < abs(filtered[-1] - image_center): filtered[-1] = cx
                else: filtered.append(cx)
        return min(filtered, key=lambda x: abs(x - image_center))

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            warp_img = self.img_warp(img)
            mask = self.detect_white(warp_img)

            height, width = mask.shape
            center_x = width // 2
            roi = mask[int(height * 0.45):height, :]
            roi_h, roi_w = roi.shape
            mid_x = roi_w // 2

            look_y1, look_y2 = int(roi_h * 0.15), int(roi_h * 0.30)
            look_band = roi[look_y1:look_y2, :]
            
            left_cx = self.get_filtered_line_center(look_band[:, :mid_x], 0, center_x)
            right_cx = self.get_filtered_line_center(look_band[:, mid_x:], mid_x, center_x)

            lane_center = None
            if left_cx is not None and right_cx is not None:
                lane_center = (left_cx + right_cx) // 2
                new_width = right_cx - left_cx
                if 80 < new_width < width: self.last_lane_width = int(0.8 * self.last_lane_width + 0.2 * new_width)
            elif left_cx is not None: lane_center = left_cx + self.last_lane_width // 2
            elif right_cx is not None: lane_center = right_cx - self.last_lane_width // 2

            cmd = Twist()
            status_text = "NORMAL"

            if lane_center is not None:
                smooth_center = int(0.65 * (self.prev_lane_center or lane_center) + 0.35 * lane_center)
                self.prev_lane_center = smooth_center

                # PID 조향 계산
                error = center_x - smooth_center
                norm_error = error / center_x
                derivative = norm_error - self.prev_error
                self.prev_error = norm_error
                base_steer = Kp * norm_error + Kd * derivative
                
                # 라이다 회피 가중치 적용
                avoid_offset = 0.0
                if self.left_obs_dist < 0.6: 
                    avoid_offset = -0.5 * (0.6 - self.left_obs_dist)
                    status_text = "AVOID RIGHT"
                elif self.right_obs_dist < 0.6: 
                    avoid_offset = 0.5 * (0.6 - self.right_obs_dist)
                    status_text = "AVOID LEFT"
                
                steer = clamp(base_steer + avoid_offset, -MAX_STEER, MAX_STEER)
                
                # 속도 제어
                target_speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * abs(steer / MAX_STEER)
                target_speed = clamp(target_speed, MIN_SPEED, MAX_SPEED)
                
                # 비상 정지 우선권
                if self.front_obs:
                    speed = 0.0
                    steer = 0.0
                    status_text = "FRONT STOP"
                else:
                    speed = min(self.prev_speed + 0.015, target_speed) if target_speed > self.prev_speed else max(self.prev_speed - 0.08, target_speed)

                self.prev_speed = speed
                cmd.linear.x = speed
                cmd.angular.z = steer
            else:
                self.prev_speed = 0.0
                status_text = "LOST LINE"

            self.cmd_pub.publish(cmd)

            # 시각화
            show = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            if lane_center: cv2.circle(show, (lane_center, (look_y1 + look_y2) // 2), 9, (0, 0, 255), -1)
            cv2.putText(show, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('white_roi_control', show)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhiteLineControl()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()