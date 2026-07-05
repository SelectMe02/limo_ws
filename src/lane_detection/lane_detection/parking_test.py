#!/usr/bin/env python3

import math
from enum import Enum

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry


class ParkingState(Enum):
    FOLLOW_LANE = "FOLLOW_LANE"
    ENTRY_OFFSET_FORWARD = "ENTRY_OFFSET_FORWARD"
    HARD_LEFT_ENTER = "HARD_LEFT_ENTER"
    RIGHT_COUNTER_ALIGN = "RIGHT_COUNTER_ALIGN"
    STRAIGHT_IN_SLOT = "STRAIGHT_IN_SLOT"
    FINAL_STOP = "FINAL_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_angle(angle):
    """
    angle을 -pi ~ +pi 범위로 정규화한다.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    """
    geometry_msgs/Quaternion에서 yaw만 추출한다.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ParkingTestNode(Node):
    def __init__(self):
        super().__init__("parking_test_node")

        # =========================
        # Topic parameters
        # =========================
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/cmd_vel")

        self.camera_topic = self.get_parameter("camera_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        # =========================
        # Control parameters
        # =========================
        self.declare_parameter("timer_hz", 20.0)
        self.declare_parameter("max_steer", 0.42)

        # 일반적으로 +angular.z는 좌회전이다.
        # 만약 실제 LIMO에서 반대로 움직이면 left_turn_sign을 -1.0으로 바꾼다.
        self.declare_parameter("left_turn_sign", 1.0)

        self.declare_parameter("follow_speed", 0.10)
        self.declare_parameter("lane_lost_speed", 0.05)
        self.declare_parameter("lane_kp", 0.35)

        self.declare_parameter("entry_offset_dist", 0.10)
        self.declare_parameter("entry_offset_speed", 0.07)
        self.declare_parameter("entry_offset_timeout", 1.5)

        self.declare_parameter("enter_speed", 0.05)
        self.declare_parameter("hard_left_min_time", 0.35)
        self.declare_parameter("hard_left_timeout", 4.0)

        self.declare_parameter("align_speed", 0.04)
        self.declare_parameter("counter_right_w", 0.28)
        self.declare_parameter("align_min_time", 0.40)
        self.declare_parameter("align_timeout", 5.0)

        self.declare_parameter("straight_speed", 0.05)
        self.declare_parameter("straight_timeout", 4.0)

        self.timer_hz = float(self.get_parameter("timer_hz").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.left_turn_sign = float(self.get_parameter("left_turn_sign").value)

        self.follow_speed = float(self.get_parameter("follow_speed").value)
        self.lane_lost_speed = float(self.get_parameter("lane_lost_speed").value)
        self.lane_kp = float(self.get_parameter("lane_kp").value)

        self.entry_offset_dist = float(self.get_parameter("entry_offset_dist").value)
        self.entry_offset_speed = float(self.get_parameter("entry_offset_speed").value)
        self.entry_offset_timeout = float(self.get_parameter("entry_offset_timeout").value)

        self.enter_speed = float(self.get_parameter("enter_speed").value)
        self.hard_left_min_time = float(self.get_parameter("hard_left_min_time").value)
        self.hard_left_timeout = float(self.get_parameter("hard_left_timeout").value)

        self.align_speed = float(self.get_parameter("align_speed").value)
        self.counter_right_w = float(self.get_parameter("counter_right_w").value)
        self.align_min_time = float(self.get_parameter("align_min_time").value)
        self.align_timeout = float(self.get_parameter("align_timeout").value)

        self.straight_speed = float(self.get_parameter("straight_speed").value)
        self.straight_timeout = float(self.get_parameter("straight_timeout").value)

        # =========================
        # Parking detection parameters
        # =========================
        self.declare_parameter("left_open_dist", 0.45)
        self.declare_parameter("open_hold_time", 0.25)
        self.declare_parameter("open_hold_dist", 0.12)

        self.declare_parameter("left_front_wall_min", 0.20)
        self.declare_parameter("left_front_wall_max", 0.90)

        self.declare_parameter("front_safe_dist", 0.35)
        self.declare_parameter("front_stop_dist", 0.25)
        self.declare_parameter("emergency_front_stop_dist", 0.12)

        self.declare_parameter("left_wall_near_dist", 0.22)
        self.declare_parameter("target_yaw_delta_deg", 90.0)
        self.declare_parameter("enter_yaw_delta_deg", 60.0)
        self.declare_parameter("yaw_tolerance_deg", 8.0)

        self.declare_parameter("wall_parallel_tolerance", 0.04)
        self.declare_parameter("wall_parallel_max_dist", 0.80)

        self.left_open_dist = float(self.get_parameter("left_open_dist").value)
        self.open_hold_time = float(self.get_parameter("open_hold_time").value)
        self.open_hold_dist = float(self.get_parameter("open_hold_dist").value)

        self.left_front_wall_min = float(self.get_parameter("left_front_wall_min").value)
        self.left_front_wall_max = float(self.get_parameter("left_front_wall_max").value)

        self.front_safe_dist = float(self.get_parameter("front_safe_dist").value)
        self.front_stop_dist = float(self.get_parameter("front_stop_dist").value)
        self.emergency_front_stop_dist = float(self.get_parameter("emergency_front_stop_dist").value)

        self.left_wall_near_dist = float(self.get_parameter("left_wall_near_dist").value)
        self.target_yaw_delta = math.radians(float(self.get_parameter("target_yaw_delta_deg").value))
        self.enter_yaw_delta = math.radians(float(self.get_parameter("enter_yaw_delta_deg").value))
        self.yaw_tolerance = math.radians(float(self.get_parameter("yaw_tolerance_deg").value))

        self.wall_parallel_tolerance = float(self.get_parameter("wall_parallel_tolerance").value)
        self.wall_parallel_max_dist = float(self.get_parameter("wall_parallel_max_dist").value)

        # =========================
        # Camera lane detection parameters
        # =========================
        self.declare_parameter("roi_y_start_ratio", 0.55)
        self.declare_parameter("white_v_min", 160)
        self.declare_parameter("white_s_max", 90)
        self.declare_parameter("min_lane_area", 300)

        self.roi_y_start_ratio = float(self.get_parameter("roi_y_start_ratio").value)
        self.white_v_min = int(self.get_parameter("white_v_min").value)
        self.white_s_max = int(self.get_parameter("white_s_max").value)
        self.min_lane_area = int(self.get_parameter("min_lane_area").value)

        # =========================
        # Sensor timeout parameters
        # =========================
        self.declare_parameter("scan_timeout", 0.5)
        self.declare_parameter("image_timeout", 0.7)

        self.scan_timeout = float(self.get_parameter("scan_timeout").value)
        self.image_timeout = float(self.get_parameter("image_timeout").value)

        # =========================
        # ROS pub/sub
        # =========================
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(1.0 / self.timer_hz, self.control_loop)

        # =========================
        # Internal states
        # =========================
        self.state = ParkingState.FOLLOW_LANE

        self.last_scan_time = None
        self.last_image_time = None
        self.last_timer_time = self.get_clock().now()
        self.last_log_time = self.get_clock().now()

        self.scan_angles_deg = None
        self.scan_ranges = None
        self.scan_default_max = 10.0

        self.lane_visible = False
        self.lane_error = 0.0
        self.lane_area = 0.0

        self.pose_xy = None
        self.yaw = None
        self.prev_loop_pose = None

        self.state_start_time = self.get_clock().now()
        self.state_start_pose = None
        self.parking_start_yaw = None

        self.slot_open_time = 0.0
        self.slot_open_distance = 0.0

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.get_logger().info("parking_test_node started.")
        self.get_logger().info(f"camera_topic: {self.camera_topic}")
        self.get_logger().info(f"scan_topic  : {self.scan_topic}")
        self.get_logger().info(f"odom_topic  : {self.odom_topic}")
        self.get_logger().info(f"cmd_topic   : {self.cmd_topic}")

    # ============================================================
    # Callback functions
    # ============================================================
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)

        if math.isfinite(msg.range_max) and msg.range_max > 0.0:
            self.scan_default_max = float(msg.range_max)
        else:
            self.scan_default_max = 10.0

        # inf는 라이다가 닿지 않은 먼 거리로 취급한다.
        ranges[np.isinf(ranges)] = self.scan_default_max

        # nan, 0, 범위 밖 값은 invalid 처리한다.
        ranges[~np.isfinite(ranges)] = np.nan
        ranges[ranges < max(msg.range_min, 0.01)] = np.nan
        ranges[ranges > self.scan_default_max] = np.nan

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles_deg = np.rad2deg(angles)
        angles_deg = (angles_deg + 180.0) % 360.0 - 180.0

        self.scan_ranges = ranges
        self.scan_angles_deg = angles_deg
        self.last_scan_time = self.get_clock().now()

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                self.lane_visible = False
                return

            h, w = image.shape[:2]
            roi_y = int(h * self.roi_y_start_ratio)
            roi = image[roi_y:h, :]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 흰색 차선 검출
            lower_white = np.array([0, 0, self.white_v_min])
            upper_white = np.array([180, self.white_s_max, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            moments = cv2.moments(mask)
            area = moments["m00"]

            self.lane_area = area

            if area > self.min_lane_area:
                cx = int(moments["m10"] / area)

                # lane_error > 0이면 차선 중심이 이미지 오른쪽에 있다는 의미
                self.lane_error = (cx - (w / 2.0)) / (w / 2.0)
                self.lane_visible = True
            else:
                self.lane_error = 0.0
                self.lane_visible = False

            self.last_image_time = self.get_clock().now()

        except Exception as e:
            self.lane_visible = False
            self.get_logger().warn(f"image_callback error: {e}")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        self.pose_xy = (x, y)
        self.yaw = yaw

    # ============================================================
    # Utility functions
    # ============================================================
    def publish_cmd(self, v, w):
        w = clamp(w, -self.max_steer, self.max_steer)

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

        self.last_cmd_v = float(v)
        self.last_cmd_w = float(w)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def set_state(self, new_state):
        if self.state == new_state:
            return

        self.get_logger().info(f"[STATE] {self.state.value} -> {new_state.value}")

        self.state = new_state
        self.state_start_time = self.get_clock().now()
        self.state_start_pose = self.pose_xy

    def state_elapsed(self):
        now = self.get_clock().now()
        return (now - self.state_start_time).nanoseconds * 1e-9

    def moved_from_state_start(self):
        """
        현재 상태가 시작된 이후 이동 거리.
        odom이 있으면 odom 기준, 없으면 시간 * 속도 근사값을 사용한다.
        """
        if self.pose_xy is not None and self.state_start_pose is not None:
            dx = self.pose_xy[0] - self.state_start_pose[0]
            dy = self.pose_xy[1] - self.state_start_pose[1]
            return math.hypot(dx, dy)

        return abs(self.last_cmd_v) * self.state_elapsed()

    def incremental_distance(self, dt):
        """
        제어 루프 사이에 이동한 거리.
        주차공간 open distance 누적에 사용한다.
        """
        if self.pose_xy is None:
            return abs(self.last_cmd_v) * dt

        if self.prev_loop_pose is None:
            self.prev_loop_pose = self.pose_xy
            return 0.0

        dx = self.pose_xy[0] - self.prev_loop_pose[0]
        dy = self.pose_xy[1] - self.prev_loop_pose[1]
        dist = math.hypot(dx, dy)

        self.prev_loop_pose = self.pose_xy
        return dist

    def scan_is_ready(self):
        if self.last_scan_time is None:
            return False

        now = self.get_clock().now()
        age = (now - self.last_scan_time).nanoseconds * 1e-9
        return age < self.scan_timeout

    def image_is_recent(self):
        if self.last_image_time is None:
            return False

        now = self.get_clock().now()
        age = (now - self.last_image_time).nanoseconds * 1e-9
        return age < self.image_timeout

    def sector_values(self, deg_min, deg_max):
        """
        라이다 각도 구간의 range 값을 가져온다.
        기준:
            0도      : 전방
            +90도    : 왼쪽
            -90도    : 오른쪽
            ±180도   : 후방
        """
        if self.scan_ranges is None or self.scan_angles_deg is None:
            return np.array([], dtype=np.float32)

        a = self.scan_angles_deg

        if deg_min <= deg_max:
            mask = (a >= deg_min) & (a <= deg_max)
        else:
            # 예: 170도 ~ -170도처럼 -180/180을 가로지르는 구간
            mask = (a >= deg_min) | (a <= deg_max)

        values = self.scan_ranges[mask]
        values = values[np.isfinite(values)]

        return values

    def sector_median(self, deg_min, deg_max):
        values = self.sector_values(deg_min, deg_max)
        if values.size == 0:
            return self.scan_default_max
        return float(np.median(values))

    def sector_percentile(self, deg_min, deg_max, percentile):
        values = self.sector_values(deg_min, deg_max)
        if values.size == 0:
            return self.scan_default_max
        return float(np.percentile(values, percentile))

    def get_scan_features(self):
        """
        주차 FSM에서 사용할 라이다 특징값을 계산한다.
        """
        if not self.scan_is_ready():
            return None

        features = {}

        # 전방 장애물
        features["front"] = self.sector_percentile(-20, 20, 20)

        # 왼쪽 주차공간 감지용
        features["left_front"] = self.sector_median(30, 70)
        features["left_side"] = self.sector_median(70, 110)
        features["left_rear"] = self.sector_median(110, 150)

        # 왼쪽 벽 접근 판단용
        # 사용자가 말한 130~180도 기준도 포함될 수 있도록 넓게 잡음.
        features["left_wall_near"] = self.sector_percentile(70, 170, 20)

        # 벽과 평행한지 판단하기 위한 두 영역
        left_front_wall = self.sector_median(60, 80)
        left_rear_wall = self.sector_median(100, 120)

        features["left_front_wall"] = left_front_wall
        features["left_rear_wall"] = left_rear_wall

        if (
            left_front_wall < self.wall_parallel_max_dist and
            left_rear_wall < self.wall_parallel_max_dist
        ):
            features["wall_parallel_error"] = left_front_wall - left_rear_wall
        else:
            features["wall_parallel_error"] = None

        return features

    def get_signed_parking_yaw_delta(self):
        """
        주차 시작 시점 yaw 기준으로 현재 yaw가 얼마나 왼쪽으로 회전했는지 계산한다.
        left_turn_sign이 1이면 좌회전 yaw 증가를 양수로 본다.
        """
        if self.parking_start_yaw is None or self.yaw is None:
            return None

        delta = normalize_angle(self.yaw - self.parking_start_yaw)

        # 실제 조향 부호가 반대인 경우에도 판단이 가능하도록 sign 반영
        return delta * self.left_turn_sign

    # ============================================================
    # Detection logic
    # ============================================================
    def detect_left_parking_slot(self, features, dt, moved_delta):
        """
        왼쪽 ㄷ자 주차공간 감지 조건.

        1. 왼쪽 측면이 충분히 열려 있음
        2. 열린 상태가 일정 시간 이상 유지됨
        3. 열린 상태로 일정 거리 이상 이동함
        4. 왼쪽 전방 또는 왼쪽 후방 쪽에 ㄷ자 벽 일부가 감지됨
        5. 전방 안전거리가 확보됨
        """
        left_side_open = features["left_side"] > self.left_open_dist

        left_front_wall_seen = (
            self.left_front_wall_min < features["left_front"] < self.left_front_wall_max
        )

        left_rear_wall_seen = (
            self.left_front_wall_min < features["left_rear"] < self.left_front_wall_max
        )

        u_wall_seen = left_front_wall_seen or left_rear_wall_seen
        front_safe = features["front"] > self.front_safe_dist

        candidate = left_side_open and u_wall_seen and front_safe

        if candidate:
            self.slot_open_time += dt

            if moved_delta > 0.0001:
                self.slot_open_distance += moved_delta
            else:
                self.slot_open_distance += abs(self.last_cmd_v) * dt
        else:
            self.slot_open_time = 0.0
            self.slot_open_distance = 0.0

        detected = (
            self.slot_open_time >= self.open_hold_time and
            self.slot_open_distance >= self.open_hold_dist
        )

        return detected

    # ============================================================
    # State handlers
    # ============================================================
    def handle_follow_lane(self, features, dt, moved_delta):
        """
        기존 CV 차선주행을 수행하면서 왼쪽 주차공간을 감지한다.
        """
        slot_detected = self.detect_left_parking_slot(features, dt, moved_delta)

        if slot_detected:
            self.parking_start_yaw = self.yaw
            self.slot_open_time = 0.0
            self.slot_open_distance = 0.0
            self.set_state(ParkingState.ENTRY_OFFSET_FORWARD)
            self.publish_cmd(0.0, 0.0)
            return

        # 카메라 차선 추종
        if self.image_is_recent() and self.lane_visible:
            # lane_error > 0이면 차선이 오른쪽에 있음.
            # 이때 차량은 오른쪽으로 가야 하므로 angular.z는 음수.
            w = -self.lane_kp * self.lane_error
            self.publish_cmd(self.follow_speed, w)
        else:
            # 차선을 잠깐 잃어버리면 천천히 직진
            self.publish_cmd(self.lane_lost_speed, 0.0)

    def handle_entry_offset_forward(self):
        """
        주차공간을 감지한 직후 바로 꺾지 않고,
        입구 기준점을 맞추기 위해 조금 더 직진한다.
        """
        self.publish_cmd(self.entry_offset_speed, 0.0)

        moved = self.moved_from_state_start()
        elapsed = self.state_elapsed()

        if moved >= self.entry_offset_dist or elapsed >= self.entry_offset_timeout:
            self.set_state(ParkingState.HARD_LEFT_ENTER)

    def handle_hard_left_enter(self, features):
        """
        왼쪽 최대 조향으로 ㄷ자 주차공간 안쪽으로 진입한다.
        """
        hard_left_w = self.left_turn_sign * self.max_steer
        self.publish_cmd(self.enter_speed, hard_left_w)

        elapsed = self.state_elapsed()
        yaw_delta = self.get_signed_parking_yaw_delta()

        left_wall_near = features["left_wall_near"] < self.left_wall_near_dist

        yaw_enough = False
        if yaw_delta is not None:
            yaw_enough = yaw_delta >= self.enter_yaw_delta

        timeout = elapsed >= self.hard_left_timeout

        # 너무 빨리 넘어가지 않도록 최소 시간 조건을 둔다.
        if elapsed >= self.hard_left_min_time and (left_wall_near or yaw_enough or timeout):
            self.set_state(ParkingState.RIGHT_COUNTER_ALIGN)

    def handle_right_counter_align(self, features):
        """
        왼쪽으로 들어간 뒤, 오른쪽 조향으로 차체 헤딩을 복원한다.
        목표는 왼쪽 벽과 평행하게 만드는 것이다.
        """
        counter_w = -self.left_turn_sign * self.counter_right_w
        self.publish_cmd(self.align_speed, counter_w)

        elapsed = self.state_elapsed()
        yaw_delta = self.get_signed_parking_yaw_delta()

        yaw_parallel_ok = False
        if yaw_delta is not None:
            yaw_parallel_ok = abs(yaw_delta - self.target_yaw_delta) < self.yaw_tolerance

        wall_parallel_ok = False
        wall_error = features["wall_parallel_error"]
        if wall_error is not None:
            wall_parallel_ok = abs(wall_error) < self.wall_parallel_tolerance

        # 이미 전방 벽이 가까우면 더 이상 자세 보정을 오래 하지 않고 직진/정지 단계로 넘긴다.
        front_almost_close = features["front"] < (self.front_stop_dist + 0.07)

        timeout = elapsed >= self.align_timeout

        if elapsed >= self.align_min_time:
            if yaw_parallel_ok or wall_parallel_ok or front_almost_close or timeout:
                self.set_state(ParkingState.STRAIGHT_IN_SLOT)

    def handle_straight_in_slot(self, features):
        """
        벽과 평행하다고 판단한 뒤 조향 0으로 천천히 전진한다.
        전방 벽이 일정 거리 이내로 들어오면 정지한다.
        """
        if features["front"] <= self.front_stop_dist:
            self.set_state(ParkingState.FINAL_STOP)
            self.stop_robot()
            return

        if self.state_elapsed() >= self.straight_timeout:
            self.set_state(ParkingState.FINAL_STOP)
            self.stop_robot()
            return

        self.publish_cmd(self.straight_speed, 0.0)

    # ============================================================
    # Main control loop
    # ============================================================
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_timer_time).nanoseconds * 1e-9
        self.last_timer_time = now

        if dt <= 0.0:
            dt = 1.0 / self.timer_hz

        moved_delta = self.incremental_distance(dt)

        features = self.get_scan_features()

        if features is None:
            self.stop_robot()
            self.get_logger().warn("No recent /scan data. Robot stopped.")
            return

        # 전방 긴급 정지
        if self.state not in [ParkingState.FINAL_STOP, ParkingState.EMERGENCY_STOP]:
            if features["front"] < self.emergency_front_stop_dist:
                self.get_logger().warn(
                    f"Emergency stop. front={features['front']:.3f} m"
                )
                self.set_state(ParkingState.EMERGENCY_STOP)
                self.stop_robot()
                return

        if self.state == ParkingState.FOLLOW_LANE:
            self.handle_follow_lane(features, dt, moved_delta)

        elif self.state == ParkingState.ENTRY_OFFSET_FORWARD:
            self.handle_entry_offset_forward()

        elif self.state == ParkingState.HARD_LEFT_ENTER:
            self.handle_hard_left_enter(features)

        elif self.state == ParkingState.RIGHT_COUNTER_ALIGN:
            self.handle_right_counter_align(features)

        elif self.state == ParkingState.STRAIGHT_IN_SLOT:
            self.handle_straight_in_slot(features)

        elif self.state == ParkingState.FINAL_STOP:
            self.stop_robot()

        elif self.state == ParkingState.EMERGENCY_STOP:
            self.stop_robot()

        self.print_debug_log(features)

    def print_debug_log(self, features):
        now = self.get_clock().now()
        elapsed = (now - self.last_log_time).nanoseconds * 1e-9

        if elapsed < 0.8:
            return

        self.last_log_time = now

        yaw_delta = self.get_signed_parking_yaw_delta()
        if yaw_delta is None:
            yaw_delta_deg = None
        else:
            yaw_delta_deg = math.degrees(yaw_delta)

        wall_error = features["wall_parallel_error"]

        self.get_logger().info(
            "[DEBUG] "
            f"state={self.state.value}, "
            f"front={features['front']:.2f}, "
            f"L_front={features['left_front']:.2f}, "
            f"L_side={features['left_side']:.2f}, "
            f"L_rear={features['left_rear']:.2f}, "
            f"L_near={features['left_wall_near']:.2f}, "
            f"open_t={self.slot_open_time:.2f}, "
            f"open_d={self.slot_open_distance:.2f}, "
            f"lane_visible={self.lane_visible}, "
            f"lane_error={self.lane_error:.2f}, "
            f"yaw_delta={yaw_delta_deg}, "
            f"wall_err={wall_error}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = ParkingTestNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected. Stopping robot.")

    finally:
        try:
            node.stop_robot()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()