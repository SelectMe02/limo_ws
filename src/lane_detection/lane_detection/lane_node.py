import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


MAX_STEER = 0.42
MAX_SPEED = 1.00
MIN_SPEED = 0.25

Kp = 1.20
Kd = 0.45

TUNNEL_BRIGHTNESS = 70
MIN_WHITE_PIXELS = 500
TUNNEL_SPEED = 0.35


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class WhiteLineControl(Node):
    def __init__(self):
        super().__init__('white_line_control_node')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.debug_pub = self.create_publisher(
            CompressedImage,
            '/white/compressed',
            10
        )

        self.sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.img_callback,
            qos_profile_sensor_data
        )

        self.prev_error = 0.0
        self.prev_lane_center = None
        self.prev_speed = 0.0
        self.last_lane_width = 240

        self.get_logger().info('white line control node start')

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
        img_x = img.shape[1]
        img_y = img.shape[0]

        src_center_offset = [200, 315]

        src = np.float32([
            [0, 479],
            [src_center_offset[0], src_center_offset[1]],
            [640 - src_center_offset[0], src_center_offset[1]],
            [639, 479],
        ])

        dst_offset = [round(img_x * 0.125), 0]

        dst = np.float32([
            [dst_offset[0], img_y],
            [dst_offset[0], 0],
            [img_x - dst_offset[0], 0],
            [img_x - dst_offset[0], img_y],
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (img_x, img_y))

    def get_filtered_line_center(self, mask, x_offset=0, image_center=320):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 300:
                continue

            M = cv2.moments(cnt)

            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00']) + x_offset
            centers.append(cx)

        if len(centers) == 0:
            return None

        centers.sort()

        filtered = []
        min_gap = 60

        for cx in centers:
            if len(filtered) == 0:
                filtered.append(cx)
            else:
                if abs(cx - filtered[-1]) < min_gap:
                    if abs(cx - image_center) < abs(filtered[-1] - image_center):
                        filtered[-1] = cx
                else:
                    filtered.append(cx)

        return min(filtered, key=lambda x: abs(x - image_center))

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            warp_img = self.img_warp(img)
            mask = self.detect_white(warp_img)

            height, width = mask.shape
            center_x = width // 2

            roi = mask[int(height * 0.45):height, :]
            roi_h, roi_w = roi.shape
            mid_x = roi_w // 2

            look_y1 = int(roi_h * 0.15)
            look_y2 = int(roi_h * 0.30)
            look_band = roi[look_y1:look_y2, :]

            white_pixels = cv2.countNonZero(look_band)
            gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            tunnel_straight_mode = (
                avg_brightness < TUNNEL_BRIGHTNESS and
                white_pixels < MIN_WHITE_PIXELS
            )

            left_mask = look_band[:, :mid_x]
            right_mask = look_band[:, mid_x:]

            left_cx = self.get_filtered_line_center(left_mask, 0, center_x)
            right_cx = self.get_filtered_line_center(right_mask, mid_x, center_x)

            lane_center = None

            if left_cx is not None and right_cx is not None:
                lane_center = (left_cx + right_cx) // 2
                new_width = right_cx - left_cx

                if 80 < new_width < width:
                    self.last_lane_width = int(
                        0.8 * self.last_lane_width + 0.2 * new_width
                    )

            elif left_cx is not None:
                lane_center = left_cx + self.last_lane_width // 2

            elif right_cx is not None:
                lane_center = right_cx - self.last_lane_width // 2

            cmd = Twist()

            if tunnel_straight_mode:
                cmd.linear.x = TUNNEL_SPEED
                cmd.angular.z = 0.0
                self.prev_speed = TUNNEL_SPEED
                self.prev_error = 0.0
                self.prev_lane_center = None
                self.get_logger().warn(
                    f'dark tunnel mode: straight, brightness={avg_brightness:.1f}, white={white_pixels}'
                )

            elif lane_center is not None:
                if self.prev_lane_center is None:
                    smooth_center = lane_center
                else:
                    smooth_center = int(
                        0.65 * self.prev_lane_center + 0.35 * lane_center
                    )

                self.prev_lane_center = smooth_center

                error = center_x - smooth_center
                norm_error = error / center_x

                derivative = norm_error - self.prev_error
                self.prev_error = norm_error

                steer = clamp(
                    Kp * norm_error + Kd * derivative,
                    -MAX_STEER,
                    MAX_STEER
                )

                target_speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * abs(steer / MAX_STEER)
                target_speed = clamp(target_speed, MIN_SPEED, MAX_SPEED)

                max_accel_step = 0.015
                max_decel_step = 0.08

                if target_speed > self.prev_speed:
                    speed = min(self.prev_speed + max_accel_step, target_speed)
                else:
                    speed = max(self.prev_speed - max_decel_step, target_speed)

                self.prev_speed = speed

                cmd.linear.x = speed
                cmd.angular.z = steer

                self.get_logger().info(
                    f'speed={speed:.2f}, target={target_speed:.2f}, '
                    f'steer={steer:.2f}, left={left_cx}, right={right_cx}, '
                    f'center={smooth_center}, width={self.last_lane_width}, '
                    f'brightness={avg_brightness:.1f}, white={white_pixels}'
                )

            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.prev_speed = 0.0
                self.get_logger().warn(
                    f'line not found. stop. brightness={avg_brightness:.1f}, white={white_pixels}'
                )

            self.cmd_pub.publish(cmd)

            debug_img = cv2.bitwise_and(warp_img, warp_img, mask=mask)
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            self.debug_pub.publish(debug_msg)

            show = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            cv2.line(show, (center_x, 0), (center_x, roi_h), (255, 0, 0), 2)
            cv2.rectangle(show, (0, look_y1), (roi_w, look_y2), (0, 255, 255), 2)

            if left_cx is not None:
                cv2.circle(show, (left_cx, (look_y1 + look_y2) // 2), 7, (255, 0, 0), -1)

            if right_cx is not None:
                cv2.circle(show, (right_cx, (look_y1 + look_y2) // 2), 7, (0, 255, 0), -1)

            if lane_center is not None:
                cv2.circle(show, (lane_center, (look_y1 + look_y2) // 2), 9, (0, 0, 255), -1)

            if tunnel_straight_mode:
                cv2.putText(
                    show,
                    'TUNNEL STRAIGHT MODE',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow('white_roi_control', show)
            cv2.imshow('white_mask', mask)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'image callback error: {e}')

    def destroy_node(self):
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WhiteLineControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()