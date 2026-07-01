import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String


# ============================================================
# Utility
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def valid_range(value):
    return value > 0.0 and not math.isinf(value) and not math.isnan(value)


class Mission:
    WAIT_TRAFFIC = 'WAIT_TRAFFIC'
    PEDESTRIAN = 'PEDESTRIAN'
    BOX1 = 'BOX1'
    TUNNEL = 'TUNNEL'
    ROTARY = 'ROTARY'
    CONE = 'CONE'
    BOX2 = 'BOX2'
    PARKING = 'PARKING'
    FINISHED = 'FINISHED'


MISSION_ORDER = [
    Mission.WAIT_TRAFFIC,
    Mission.PEDESTRIAN,
    Mission.BOX1,
    Mission.TUNNEL,
    Mission.ROTARY,
    Mission.CONE,
    Mission.BOX2,
    Mission.PARKING,
    Mission.FINISHED,
]


class FrameDebouncer:
    def __init__(self, on_frames=4, off_frames=4):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.hit_count = 0
        self.clear_count = 0
        self.active = False

    def update(self, detected):
        if detected:
            self.hit_count += 1
            self.clear_count = 0
        else:
            self.hit_count = 0
            self.clear_count += 1

        if self.hit_count >= self.on_frames:
            self.active = True
        if self.clear_count >= self.off_frames:
            self.active = False

        return self.active


@dataclass
class LidarCluster:
    points: list
    cx: float
    cy: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: float
    count: int
    nearest: float


@dataclass
class LaneResult:
    center_x: int
    left_x: int
    right_x: int
    confidence: float
    lane_width_px: int
    tunnel_like: bool
    white_pixels: int
    brightness: float


# ============================================================
# Main mission node
# ============================================================

class LimoMissionFSMNode(Node):
    def __init__(self):
        super().__init__('limo_mission_fsm_node')

        self.bridge = CvBridge()

        # -------------------------
        # Topics
        # -------------------------
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('cone_blocked_topic', '/cone/blocked_lanes')
        self.declare_parameter('debug_topic', '/mission/debug/compressed')
        self.declare_parameter('state_topic', '/mission/state')
        self.declare_parameter('manual_state_topic', '/mission/set_state')

        camera_topic = self.get_parameter('camera_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        cone_topic = self.get_parameter('cone_blocked_topic').value
        debug_topic = self.get_parameter('debug_topic').value
        state_topic = self.get_parameter('state_topic').value
        manual_topic = self.get_parameter('manual_state_topic').value

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)

        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.cone_sub = self.create_subscription(
            String, cone_topic, self.cone_callback, 10
        )
        self.manual_sub = self.create_subscription(
            String, manual_topic, self.manual_state_callback, 10
        )

        # -------------------------
        # Vehicle / control params
        # -------------------------
        self.declare_parameter('max_speed', 0.55)
        self.declare_parameter('min_speed', 0.18)
        self.declare_parameter('curve_min_speed', 0.24)
        self.declare_parameter('tunnel_speed', 0.25)
        self.declare_parameter('avoid_speed', 0.24)
        self.declare_parameter('follow_speed', 0.22)
        self.declare_parameter('max_yaw_rate', 1.20)
        self.declare_parameter('kp', 1.15)
        self.declare_parameter('kd', 0.35)
        self.declare_parameter('yaw_rate_sign', 1.0)
        self.declare_parameter('max_accel_step', 0.020)
        self.declare_parameter('max_decel_step', 0.090)

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.curve_min_speed = float(self.get_parameter('curve_min_speed').value)
        self.tunnel_speed = float(self.get_parameter('tunnel_speed').value)
        self.avoid_speed = float(self.get_parameter('avoid_speed').value)
        self.follow_speed = float(self.get_parameter('follow_speed').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.yaw_rate_sign = float(self.get_parameter('yaw_rate_sign').value)
        self.max_accel_step = float(self.get_parameter('max_accel_step').value)
        self.max_decel_step = float(self.get_parameter('max_decel_step').value)

        # -------------------------
        # Track / fusion params
        # -------------------------
        self.declare_parameter('track_width_m', 0.60)
        self.declare_parameter('robot_width_m', 0.13)
        self.declare_parameter('safety_margin_m', 0.05)
        self.declare_parameter('lane_width_px_init', 240)
        self.declare_parameter('lane_bias_sign', 1.0)

        self.track_width_m = float(self.get_parameter('track_width_m').value)
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.safety_margin_m = float(self.get_parameter('safety_margin_m').value)
        self.drivable_half_width = (
            self.track_width_m / 2.0 - self.robot_width_m / 2.0 - self.safety_margin_m
        )
        self.drivable_half_width = max(0.12, self.drivable_half_width)
        self.last_lane_width_px = int(self.get_parameter('lane_width_px_init').value)
        self.lane_bias_sign = float(self.get_parameter('lane_bias_sign').value)

        # -------------------------
        # Vision params
        # -------------------------
        self.declare_parameter('white_v_min', 180)
        self.declare_parameter('white_s_max', 90)
        self.declare_parameter('tunnel_brightness', 70)
        self.declare_parameter('tunnel_white_pixels', 500)
        self.declare_parameter('max_lost_frames', 12)

        self.white_v_min = int(self.get_parameter('white_v_min').value)
        self.white_s_max = int(self.get_parameter('white_s_max').value)
        self.tunnel_brightness = float(self.get_parameter('tunnel_brightness').value)
        self.tunnel_white_pixels = int(self.get_parameter('tunnel_white_pixels').value)
        self.max_lost_frames = int(self.get_parameter('max_lost_frames').value)

        # -------------------------
        # Mission timing params
        # -------------------------
        self.declare_parameter('box_min_time', 1.0)
        self.declare_parameter('box_clear_time', 0.7)
        self.declare_parameter('ped_clear_time', 0.8)
        self.declare_parameter('tunnel_min_time', 1.0)
        self.declare_parameter('rotary_min_time', 5.0)
        self.declare_parameter('cone_min_time', 3.0)
        self.declare_parameter('cone_hold_time', 2.0)
        self.declare_parameter('box2_to_parking_delay', 1.0)

        self.box_min_time = float(self.get_parameter('box_min_time').value)
        self.box_clear_time = float(self.get_parameter('box_clear_time').value)
        self.ped_clear_time = float(self.get_parameter('ped_clear_time').value)
        self.tunnel_min_time = float(self.get_parameter('tunnel_min_time').value)
        self.rotary_min_time = float(self.get_parameter('rotary_min_time').value)
        self.cone_min_time = float(self.get_parameter('cone_min_time').value)
        self.cone_hold_time = float(self.get_parameter('cone_hold_time').value)
        self.box2_to_parking_delay = float(self.get_parameter('box2_to_parking_delay').value)

        # -------------------------
        # Mission enable / start params
        # -------------------------
        # 신호등 미션을 임시 제외하려면 use_traffic_light:=false 상태로 실행한다.
        # 기본값을 false로 두었기 때문에 현재 코드는 PEDESTRIAN부터 시작한다.
        self.declare_parameter('use_traffic_light', False)
        self.declare_parameter('start_state', '')

        self.use_traffic_light = bool(self.get_parameter('use_traffic_light').value)
        start_state_param = str(self.get_parameter('start_state').value).strip().upper()

        if self.use_traffic_light:
            self.mission_order = MISSION_ORDER.copy()
            default_start_state = Mission.WAIT_TRAFFIC
        else:
            self.mission_order = [
                Mission.PEDESTRIAN,
                Mission.BOX1,
                Mission.TUNNEL,
                Mission.ROTARY,
                Mission.CONE,
                Mission.BOX2,
                Mission.PARKING,
                Mission.FINISHED,
            ]
            default_start_state = Mission.PEDESTRIAN

        if start_state_param in MISSION_ORDER:
            if (not self.use_traffic_light) and start_state_param == Mission.WAIT_TRAFFIC:
                self.get_logger().warn('use_traffic_light is false. start_state WAIT_TRAFFIC is ignored.')
                self.state = default_start_state
            else:
                self.state = start_state_param
        else:
            self.state = default_start_state

        # -------------------------
        # Internal state
        # -------------------------
        self.state_enter_time = time.monotonic()
        self.prev_speed = 0.0
        self.prev_error = 0.0
        self.prev_lane_center = None
        self.prev_lane_result = None
        self.lost_frames = 0
        self.path_center_y_m = 0.0

        self.lidar_points = []
        self.lidar_clusters = []
        self.last_scan_time = 0.0

        self.red_db = FrameDebouncer(4, 4)
        self.green_db = FrameDebouncer(4, 4)
        self.ped_db = FrameDebouncer(3, 5)
        self.box_db = FrameDebouncer(3, 5)
        self.rotary_db = FrameDebouncer(3, 5)
        self.finish_db = FrameDebouncer(4, 5)

        self.pedestrian_was_seen = False
        self.last_ped_clear_time = None

        self.box_avoid_started = False
        self.box_clear_start = None
        self.last_box_side = 0.0

        self.rotary_vehicle_distance = 9.9

        self.blocked_cone_lanes = set()
        self.cone_target_lane = 'center'
        self.last_cone_msg_time = 0.0
        self.cone_shift_until = 0.0

        self.parking_started = False
        self.parking_step_index = 0
        self.parking_step_start = 0.0
        self.parking_sequence = [
            # duration, linear.x, angular.z
            (0.55, 0.18, 0.00),
            (0.95, 0.16, -0.55),
            (0.85, 0.16, 0.00),
            (0.70, 0.14, 0.45),
            (0.30, 0.08, 0.00),
            (999.0, 0.00, 0.00),
        ]

        self.state_timer = self.create_timer(0.2, self.publish_state)

        self.get_logger().info(
            f'limo mission fsm node start. use_traffic_light={self.use_traffic_light}, start_state={self.state}. '
            'manual override: ros2 topic pub --once /mission/set_state std_msgs/msg/String "{data: NEXT}"'
        )

    # ========================================================
    # Mission state helpers
    # ========================================================
    def set_state(self, new_state, reason=''):
        if new_state not in MISSION_ORDER:
            self.get_logger().warn(f'unknown mission state: {new_state}')
            return

        if (not self.use_traffic_light) and new_state == Mission.WAIT_TRAFFIC:
            self.get_logger().warn('WAIT_TRAFFIC is disabled. Use PEDESTRIAN or set use_traffic_light:=true.')
            new_state = Mission.PEDESTRIAN

        if new_state == self.state:
            return

        old = self.state
        self.state = new_state
        self.state_enter_time = time.monotonic()

        # Reset state-local variables
        self.last_ped_clear_time = None
        self.box_clear_start = None
        self.box_avoid_started = False
        self.last_box_side = 0.0
        self.parking_started = False
        self.parking_step_index = 0
        self.parking_step_start = 0.0

        self.get_logger().warn(f'MISSION {old} -> {new_state} {reason}')

    def next_state(self, reason=''):
        try:
            idx = self.mission_order.index(self.state)
            next_idx = min(idx + 1, len(self.mission_order) - 1)
            self.set_state(self.mission_order[next_idx], reason)
        except ValueError:
            self.set_state(self.mission_order[0], reason)

    def state_elapsed(self):
        return time.monotonic() - self.state_enter_time

    def manual_state_callback(self, msg):
        cmd = msg.data.strip().upper()
        aliases = {
            'TRAFFIC': Mission.WAIT_TRAFFIC,
            'WAIT': Mission.WAIT_TRAFFIC,
            'PED': Mission.PEDESTRIAN,
            'PEDESTRIAN': Mission.PEDESTRIAN,
            'BOX': Mission.BOX1,
            'BOX1': Mission.BOX1,
            'TUNNEL': Mission.TUNNEL,
            'ROTARY': Mission.ROTARY,
            'ROUNDABOUT': Mission.ROTARY,
            'CONE': Mission.CONE,
            'BOX2': Mission.BOX2,
            'PARK': Mission.PARKING,
            'PARKING': Mission.PARKING,
            'FINISH': Mission.FINISHED,
            'FINISHED': Mission.FINISHED,
        }

        if cmd == 'NEXT':
            self.next_state('(manual NEXT)')
        elif cmd in aliases:
            self.set_state(aliases[cmd], '(manual set)')
        else:
            self.get_logger().warn(f'unknown manual command: {cmd}')

    def publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    # ========================================================
    # LiDAR processing
    # ========================================================
    def scan_callback(self, msg):
        points = []
        front_scan_points = []

        for i, dist in enumerate(msg.ranges):
            if not valid_range(dist):
                continue
            if dist < 0.05 or dist > 2.0:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)

            # Robot 기준 앞쪽 중심부만 mission 판단에 사용한다.
            if -0.20 <= x <= 1.50 and abs(y) <= 0.80:
                points.append((x, y, dist, angle))
                if x > 0.0:
                    front_scan_points.append((x, y, dist, angle))

        self.lidar_points = points
        self.lidar_clusters = self.make_clusters(front_scan_points)
        self.last_scan_time = time.monotonic()

    def make_clusters(self, scan_points):
        if len(scan_points) == 0:
            return []

        # LaserScan 순서대로 들어온 점을 거리 차이 기준으로 군집화한다.
        clusters_raw = []
        current = []
        prev_xy = None

        for x, y, dist, angle in scan_points:
            if prev_xy is None:
                current = [(x, y, dist, angle)]
                prev_xy = (x, y)
                continue

            px, py = prev_xy
            gap = math.hypot(x - px, y - py)

            if gap < 0.10:
                current.append((x, y, dist, angle))
            else:
                if len(current) >= 2:
                    clusters_raw.append(current)
                current = [(x, y, dist, angle)]

            prev_xy = (x, y)

        if len(current) >= 2:
            clusters_raw.append(current)

        clusters = []
        for c in clusters_raw:
            xs = [p[0] for p in c]
            ys = [p[1] for p in c]
            ds = [p[2] for p in c]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = math.hypot(max_x - min_x, max_y - min_y)
            clusters.append(
                LidarCluster(
                    points=c,
                    cx=float(np.mean(xs)),
                    cy=float(np.mean(ys)),
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    width=width,
                    count=len(c),
                    nearest=min(ds),
                )
            )

        return clusters

    def is_wall_like(self, cluster):
        # 벽은 점이 길게 이어지고 폭이 크다. 차량/박스/사람 후보는 비교적 짧은 덩어리다.
        if cluster.width > 0.65:
            return True
        if (cluster.max_x - cluster.min_x) > 0.75:
            return True
        return False

    def obstacle_in_corridor(self, x_min, x_max, half_width, min_points=3, ignore_wall=True):
        best = None
        path_y = self.path_center_y_m

        for cluster in self.lidar_clusters:
            if cluster.count < min_points:
                continue
            if ignore_wall and self.is_wall_like(cluster):
                continue
            if not (x_min <= cluster.cx <= x_max):
                continue
            if abs(cluster.cy - path_y) > half_width:
                continue

            if best is None or cluster.nearest < best.nearest:
                best = cluster

        return best

    def obstacle_in_box_roi(self):
        return self.obstacle_in_corridor(
            0.20,
            0.95,
            self.drivable_half_width + 0.04,
            min_points=3,
            ignore_wall=True,
        )

    def obstacle_in_pedestrian_roi(self):
        # 보행자는 회피 대상이 아니라 정지 대상이라 y를 박스보다 조금 넓게 본다.
        return self.obstacle_in_corridor(
            0.20,
            1.15,
            min(0.30, self.drivable_half_width + 0.12),
            min_points=3,
            ignore_wall=True,
        )

    def rotary_vehicle_candidate(self):
        best = None

        for cluster in self.lidar_clusters:
            if cluster.count < 3:
                continue
            if self.is_wall_like(cluster):
                continue

            # 회전교차로 진입/추종 위험 영역: 전방과 좌전방을 조금 넓게 본다.
            in_roi = (
                0.15 <= cluster.cx <= 1.25 and
                -0.25 <= cluster.cy <= 0.55
            )
            if not in_roi:
                continue

            # 차량 후보는 너무 작지도, 벽처럼 너무 길지도 않은 군집으로 제한한다.
            if not (0.04 <= cluster.width <= 0.55):
                continue

            if best is None or cluster.nearest < best.nearest:
                best = cluster

        return best

    # ========================================================
    # Vision processing
    # ========================================================
    def img_warp(self, img):
        h, w = img.shape[:2]

        src_top_x = 200
        src_top_y = 315
        src = np.float32([
            [0, h - 1],
            [src_top_x, src_top_y],
            [w - src_top_x, src_top_y],
            [w - 1, h - 1],
        ])

        dst_margin = int(w * 0.125)
        dst = np.float32([
            [dst_margin, h - 1],
            [dst_margin, 0],
            [w - dst_margin, 0],
            [w - 1 - dst_margin, h - 1],
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (w, h))

    def detect_white(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.white_v_min])
        upper = np.array([179, self.white_s_max, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_mid = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mid, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        return mask

    def get_line_center_from_mask(self, mask, x_offset=0, image_center=320):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 250:
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
        min_gap = 55

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

    def detect_lane(self, warp_img, mask):
        h, w = mask.shape
        center_x = w // 2

        gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        look_band_for_tunnel = mask[int(h * 0.45):int(h * 0.65), :]
        white_pixels = int(cv2.countNonZero(look_band_for_tunnel))
        tunnel_like = brightness < self.tunnel_brightness and white_pixels < self.tunnel_white_pixels

        roi = mask[int(h * 0.45):h, :]
        roi_h, roi_w = roi.shape
        mid_x = roi_w // 2

        # 너무 아래만 보면 가까운 노이즈가 많아서 중간 band를 사용한다.
        look_y1 = int(roi_h * 0.15)
        look_y2 = int(roi_h * 0.32)
        look_band = roi[look_y1:look_y2, :]

        left_x = self.get_line_center_from_mask(look_band[:, :mid_x], 0, center_x)
        right_x = self.get_line_center_from_mask(look_band[:, mid_x:], mid_x, center_x)

        lane_center = None
        confidence = 0.0

        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) // 2
            measured_width = right_x - left_x
            if 120 <= measured_width <= 420:
                self.last_lane_width_px = int(0.85 * self.last_lane_width_px + 0.15 * measured_width)
            confidence = 1.0
        elif left_x is not None:
            lane_center = left_x + self.last_lane_width_px // 2
            confidence = 0.65
        elif right_x is not None:
            lane_center = right_x - self.last_lane_width_px // 2
            confidence = 0.65

        if lane_center is None:
            return None, tunnel_like, white_pixels, brightness

        if self.prev_lane_center is None:
            smooth_center = lane_center
        else:
            smooth_center = int(0.65 * self.prev_lane_center + 0.35 * lane_center)

        self.prev_lane_center = smooth_center

        return LaneResult(
            center_x=smooth_center,
            left_x=-1 if left_x is None else left_x,
            right_x=-1 if right_x is None else right_x,
            confidence=confidence,
            lane_width_px=self.last_lane_width_px,
            tunnel_like=tunnel_like,
            white_pixels=white_pixels,
            brightness=brightness,
        ), tunnel_like, white_pixels, brightness

    def detect_traffic_light(self, img):
        h, w = img.shape[:2]

        # 화면 상단 중앙 영역만 사용한다. 카메라 각도에 따라 아래 ROI는 현장에서 조절한다.
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        y1 = int(h * 0.02)
        y2 = int(h * 0.38)
        roi = img[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, np.array([0, 90, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 90, 80]), np.array([179, 255, 255]))
        red = cv2.bitwise_or(red1, red2)
        green = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([90, 255, 255]))

        red_pixels = int(cv2.countNonZero(red))
        green_pixels = int(cv2.countNonZero(green))

        red_seen = red_pixels > 60 and red_pixels > green_pixels * 1.2
        green_seen = green_pixels > 60 and green_pixels > red_pixels * 1.2

        red_active = self.red_db.update(red_seen)
        green_active = self.green_db.update(green_seen)

        return red_active, green_active, red_pixels, green_pixels, (x1, y1, x2, y2)

    def detect_finish_line(self, mask):
        h, w = mask.shape
        band = mask[int(h * 0.70):int(h * 0.92), :]

        # 종료선/주차선은 보통 가로 흰색 영역이 넓게 잡힌다.
        white_pixels = int(cv2.countNonZero(band))
        ratio = white_pixels / float(band.shape[0] * band.shape[1])

        detected = ratio > 0.22
        return self.finish_db.update(detected), ratio

    # ========================================================
    # Cone input
    # ========================================================
    def cone_callback(self, msg):
        raw = msg.data.strip().lower()
        lanes = set()
        for token in raw.replace(';', ',').replace(' ', ',').split(','):
            token = token.strip()
            if token in ('left', 'center', 'right'):
                lanes.add(token)

        self.blocked_cone_lanes = lanes
        self.last_cone_msg_time = time.monotonic()

        if 'center' in lanes and 'left' in lanes:
            self.cone_target_lane = 'right'
        elif 'center' in lanes and 'right' in lanes:
            self.cone_target_lane = 'left'
        elif 'left' in lanes and 'right' not in lanes:
            self.cone_target_lane = 'right'
        elif 'right' in lanes and 'left' not in lanes:
            self.cone_target_lane = 'left'
        else:
            self.cone_target_lane = 'center'

        self.cone_shift_until = time.monotonic() + self.cone_hold_time

    # ========================================================
    # Control helpers
    # ========================================================
    def smooth_speed(self, target_speed):
        if target_speed > self.prev_speed:
            speed = min(self.prev_speed + self.max_accel_step, target_speed)
        else:
            speed = max(self.prev_speed - self.max_decel_step, target_speed)
        self.prev_speed = speed
        return speed

    def make_cmd(self, speed, yaw_rate):
        cmd = Twist()
        cmd.linear.x = float(clamp(speed, 0.0, self.max_speed))
        cmd.angular.z = float(clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))
        return cmd

    def stop_cmd(self):
        self.prev_speed = 0.0
        self.prev_error = 0.0
        return Twist()

    def lane_follow_cmd(self, lane_result, extra_yaw=0.0, speed_limit=None, lane_bias_px=0.0):
        if lane_result is None:
            return self.memory_lane_cmd(tunnel_like=False)

        image_center = 320
        desired_center = image_center + lane_bias_px

        error = desired_center - lane_result.center_x
        norm_error = error / float(image_center)
        derivative = norm_error - self.prev_error
        self.prev_error = norm_error

        yaw_rate = self.yaw_rate_sign * (self.kp * norm_error + self.kd * derivative)
        yaw_rate += extra_yaw
        yaw_rate = clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # camera-lidar fusion용: 현재 차선 중심이 차량 기준 좌우 어디에 있는지 대략 추정한다.
        xm_per_px = self.track_width_m / max(float(self.last_lane_width_px), 1.0)
        self.path_center_y_m = clamp((image_center - lane_result.center_x) * xm_per_px, -0.25, 0.25)

        turn_ratio = min(abs(yaw_rate) / self.max_yaw_rate, 1.0)
        target_speed = self.max_speed - (self.max_speed - self.curve_min_speed) * turn_ratio
        if lane_result.confidence < 0.75:
            target_speed = min(target_speed, 0.35)
        if speed_limit is not None:
            target_speed = min(target_speed, speed_limit)

        speed = self.smooth_speed(clamp(target_speed, self.min_speed, self.max_speed))
        return self.make_cmd(speed, yaw_rate)

    def memory_lane_cmd(self, tunnel_like=False):
        self.lost_frames += 1

        if self.prev_lane_result is not None and self.lost_frames <= self.max_lost_frames:
            target = self.tunnel_speed if tunnel_like else self.min_speed
            speed = self.smooth_speed(target)
            # 터널에서는 직진 구간이라는 전제라 조향을 빠르게 0으로 죽인다.
            yaw = self.yaw_rate_sign * clamp(self.prev_error * 0.45, -0.25, 0.25)
            return self.make_cmd(speed, yaw)

        return self.stop_cmd()

    # ========================================================
    # Mission handlers
    # ========================================================
    def handle_wait_traffic(self, lane_result, traffic):
        red_active, green_active, _, _, _ = traffic

        if green_active:
            self.next_state('(green light)')
            return self.lane_follow_cmd(lane_result, speed_limit=0.30)

        # 빨간색 또는 미검출이면 출발하지 않는다.
        return self.stop_cmd()

    def handle_pedestrian(self, lane_result):
        cluster = self.obstacle_in_pedestrian_roi()
        ped_active = self.ped_db.update(cluster is not None)
        now = time.monotonic()

        if ped_active:
            self.pedestrian_was_seen = True
            self.last_ped_clear_time = None
            return self.stop_cmd(), 'PED_STOP'

        if self.pedestrian_was_seen:
            if self.last_ped_clear_time is None:
                self.last_ped_clear_time = now
            elif now - self.last_ped_clear_time >= self.ped_clear_time:
                self.next_state('(pedestrian clear)')

        return self.lane_follow_cmd(lane_result, speed_limit=0.36), 'PED_CLEAR'

    def handle_box(self, lane_result, next_after_clear=True):
        cluster = self.obstacle_in_box_roi()
        box_active = self.box_db.update(cluster is not None)
        now = time.monotonic()

        extra_yaw = 0.0
        status = 'BOX_SEARCH'

        if box_active and cluster is not None:
            self.box_avoid_started = True
            self.box_clear_start = None

            # 장애물이 차선 중심 기준 왼쪽이면 오른쪽 회피, 오른쪽이면 왼쪽 회피
            side = 1.0 if cluster.cy > self.path_center_y_m else -1.0
            self.last_box_side = side
            extra_yaw = -0.55 * side
            status = 'BOX_AVOID_RIGHT' if side > 0 else 'BOX_AVOID_LEFT'

            cmd = self.lane_follow_cmd(lane_result, extra_yaw=extra_yaw, speed_limit=self.avoid_speed)
            return cmd, status

        if self.box_avoid_started:
            if self.box_clear_start is None:
                self.box_clear_start = now
            clear_long_enough = now - self.box_clear_start >= self.box_clear_time
            min_time_ok = self.state_elapsed() >= self.box_min_time

            if next_after_clear and clear_long_enough and min_time_ok:
                self.next_state('(box clear)')

            # 회피 직후에는 조향을 바로 0으로 만들지 말고 살짝 복귀시킨다.
            recover_yaw = 0.20 * self.last_box_side
            cmd = self.lane_follow_cmd(lane_result, extra_yaw=recover_yaw, speed_limit=0.30)
            return cmd, 'BOX_RECOVER'

        return self.lane_follow_cmd(lane_result, speed_limit=0.36), status

    def handle_tunnel(self, lane_result, tunnel_like):
        if lane_result is None or tunnel_like:
            cmd = self.memory_lane_cmd(tunnel_like=True)
            return cmd, 'TUNNEL_MEMORY'

        self.lost_frames = 0
        cmd = self.lane_follow_cmd(lane_result, speed_limit=0.34)

        # 터널은 짧고 직진이라고 했으므로, 일정 시간 이후 차선이 안정적으로 돌아오면 다음 미션으로 넘긴다.
        if self.state_elapsed() >= self.tunnel_min_time and lane_result.confidence >= 0.65:
            self.next_state('(lane recovered after tunnel)')

        return cmd, 'TUNNEL_LANE'

    def handle_rotary(self, lane_result):
        cluster = self.rotary_vehicle_candidate()
        active = self.rotary_db.update(cluster is not None)
        status = 'ROTARY_CLEAR'

        if active and cluster is not None:
            self.rotary_vehicle_distance = cluster.nearest

            if cluster.nearest < 0.45:
                return self.stop_cmd(), 'ROTARY_STOP'
            elif cluster.nearest < 0.75:
                return self.lane_follow_cmd(lane_result, speed_limit=self.follow_speed), 'ROTARY_FOLLOW'
            else:
                return self.lane_follow_cmd(lane_result, speed_limit=0.30), 'ROTARY_SLOW'

        if self.state_elapsed() >= self.rotary_min_time:
            self.next_state('(rotary time done)')

        return self.lane_follow_cmd(lane_result, speed_limit=0.34), status

    def handle_cone(self, lane_result):
        now = time.monotonic()
        lane_bias_px = 0.0
        status = f'CONE_{self.cone_target_lane.upper()}'

        # YOLO가 center+left를 보면 오른쪽 차선, center+right를 보면 왼쪽 차선으로 이동한다.
        if now <= self.cone_shift_until:
            if self.cone_target_lane == 'left':
                lane_bias_px = -self.lane_bias_sign * self.last_lane_width_px * 0.75
            elif self.cone_target_lane == 'right':
                lane_bias_px = self.lane_bias_sign * self.last_lane_width_px * 0.75

        # 라이다는 D구간에서 의미 분류가 아니라 안전거리 보조만 한다.
        close_cluster = self.obstacle_in_corridor(
            0.12,
            0.55,
            self.drivable_half_width,
            min_points=3,
            ignore_wall=True,
        )
        if close_cluster is not None and close_cluster.nearest < 0.25:
            return self.stop_cmd(), 'CONE_SAFE_STOP'

        if self.state_elapsed() >= self.cone_min_time and now > self.cone_shift_until:
            self.next_state('(cone done)')

        cmd = self.lane_follow_cmd(lane_result, speed_limit=0.30, lane_bias_px=lane_bias_px)
        return cmd, status

    def handle_parking(self, lane_result, finish_detected):
        now = time.monotonic()

        if not self.parking_started:
            if finish_detected:
                self.parking_started = True
                self.parking_step_index = 0
                self.parking_step_start = now
                self.get_logger().warn('parking sequence start')
            else:
                return self.lane_follow_cmd(lane_result, speed_limit=0.25), 'WAIT_FINISH_LINE'

        duration, v, w = self.parking_sequence[self.parking_step_index]
        if now - self.parking_step_start >= duration:
            self.parking_step_index = min(self.parking_step_index + 1, len(self.parking_sequence) - 1)
            self.parking_step_start = now
            duration, v, w = self.parking_sequence[self.parking_step_index]

        if self.parking_step_index == len(self.parking_sequence) - 1:
            # 마지막 step은 정지 유지. 여기서 FINISHED로 넘긴다.
            self.set_state(Mission.FINISHED, '(parking complete)')
            return self.stop_cmd(), 'PARK_DONE'

        return self.make_cmd(v, w), f'PARK_STEP_{self.parking_step_index}'

    # ========================================================
    # Debug
    # ========================================================
    def draw_debug(self, warp_img, mask, lane_result, traffic_roi, status, finish_ratio):
        debug = warp_img.copy()
        h, w = mask.shape
        center_x = w // 2

        colored_mask = np.zeros_like(debug)
        colored_mask[:, :, 1] = mask
        debug = cv2.addWeighted(debug, 0.75, colored_mask, 0.25, 0)

        cv2.line(debug, (center_x, 0), (center_x, h), (255, 0, 0), 2)

        if lane_result is not None:
            cv2.circle(debug, (lane_result.center_x, int(h * 0.70)), 9, (0, 0, 255), -1)
            if lane_result.left_x >= 0:
                cv2.circle(debug, (lane_result.left_x, int(h * 0.70)), 7, (255, 0, 0), -1)
            if lane_result.right_x >= 0:
                cv2.circle(debug, (lane_result.right_x, int(h * 0.70)), 7, (0, 255, 0), -1)

        x1, y1, x2, y2 = traffic_roi
        if x2 > x1 and y2 > y1:
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

        text1 = f'state={self.state} status={status}'
        text2 = f'v={self.prev_speed:.2f} lane_w={self.last_lane_width_px} path_y={self.path_center_y_m:.2f}'
        text3 = f'cones={sorted(list(self.blocked_cone_lanes))} target={self.cone_target_lane} finish={finish_ratio:.2f}'
        cv2.putText(debug, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(debug, dst_format='jpg')
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'debug publish error: {e}')

        try:
            cv2.imshow('mission_debug', debug)
            cv2.imshow('mission_mask', mask)
            cv2.waitKey(1)
        except Exception:
            pass

    # ========================================================
    # Main image callback
    # ========================================================
    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            warp_img = self.img_warp(img)
            mask = self.detect_white(warp_img)

            lane_result, tunnel_like, _, _ = self.detect_lane(warp_img, mask)

            if lane_result is not None:
                self.lost_frames = 0
                self.prev_lane_result = lane_result

            if self.use_traffic_light:
                traffic = self.detect_traffic_light(img)
            else:
                # 신호등 미션 비활성화 상태: debug용 ROI만 더미로 넘기고 색상 판단은 하지 않는다.
                h_img, w_img = img.shape[:2]
                traffic = (False, False, 0, 0, (0, 0, 0, 0))

            finish_detected, finish_ratio = self.detect_finish_line(mask)

            cmd = Twist()
            status = 'IDLE'

            if self.state == Mission.WAIT_TRAFFIC:
                if self.use_traffic_light:
                    cmd = self.handle_wait_traffic(lane_result, traffic)
                    status = 'WAIT_GREEN'
                else:
                    self.set_state(Mission.PEDESTRIAN, '(traffic disabled)')
                    cmd, status = self.handle_pedestrian(lane_result)

            elif self.state == Mission.PEDESTRIAN:
                cmd, status = self.handle_pedestrian(lane_result)

            elif self.state == Mission.BOX1:
                cmd, status = self.handle_box(lane_result, next_after_clear=True)

            elif self.state == Mission.TUNNEL:
                cmd, status = self.handle_tunnel(lane_result, tunnel_like)

            elif self.state == Mission.ROTARY:
                cmd, status = self.handle_rotary(lane_result)

            elif self.state == Mission.CONE:
                cmd, status = self.handle_cone(lane_result)

            elif self.state == Mission.BOX2:
                cmd, status = self.handle_box(lane_result, next_after_clear=True)
                # BOX2가 끝나면 next_state가 PARKING으로 넘어간다.

            elif self.state == Mission.PARKING:
                cmd, status = self.handle_parking(lane_result, finish_detected)

            elif self.state == Mission.FINISHED:
                cmd = self.stop_cmd()
                status = 'FINISHED_STOP'

            else:
                cmd = self.stop_cmd()
                status = 'UNKNOWN_STOP'

            self.cmd_pub.publish(cmd)
            self.draw_debug(warp_img, mask, lane_result, traffic[4], status, finish_ratio)

        except Exception as e:
            self.get_logger().error(f'image callback error: {e}')
            self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LimoMissionFSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()