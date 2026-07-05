import math
import time
from dataclasses import dataclass
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import String


# =========================
# Vehicle / command limits
# =========================
WHEELBASE_M = 0.20
TRACK_M = 0.172
BASE_X_SIZE_M = 0.13
BASE_Y_SIZE_M = 0.12

# Stanley가 계산하는 값은 조향각으로 취급한다.
MAX_STEERING_ANGLE = 0.40  # rad

# /cmd_vel.angular.z로 보낼 yaw rate 제한
MAX_YAW_RATE = 1.50

# 조향 방향이 반대로 들어가면 -1.0으로 바꿔서 테스트
STEER_SIGN = 1.0

# =========================
# Speed params
# =========================
MAX_SPEED = 0.90
STRAIGHT_SPEED = 0.78
CURVE_MIN_SPEED = 0.50
BAD_CONF_SPEED = 0.48
LOST_LANE_SPEED = 0.50
TUNNEL_SPEED = 0.50

CALIBRATION_SPEED_LIMIT = 0.40

MAX_ACCEL_STEP = 0.050
MAX_DECEL_STEP = 0.12

# =========================
# Stanley controller params
# =========================
K_STANLEY = 1.90
K_HEADING = 1.65
K_DAMPING = 0.30
STANLEY_SOFTENING = 0.20

STEER_SMOOTH_ALPHA = 0.85
STEER_RATE_LIMIT = 0.38

# =========================
# Vision / lane params
# =========================
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

INITIAL_LANE_WIDTH_PX = 240

# 곡선/BEV 왜곡 때문에 두 차선 간격이 200px 아래로 잡힐 수 있으므로 완화
MIN_LANE_WIDTH_PX = 160
MAX_LANE_WIDTH_PX = 400

LANE_WIDTH_M = 0.60

AUTO_CALIBRATE_LANE_WIDTH = False
LANE_WIDTH_CALIB_FRAMES = 25
LANE_WIDTH_CALIB_MIN = 200
LANE_WIDTH_CALIB_MAX = 380
LANE_WIDTH_CALIB_STABLE_STD = 35

BEV_FORWARD_M = 0.90

CENTER_BIAS_PX = 0

# =========================
# Pair acceptance params
# =========================
# 두 차선이 모두 보일 때 pair로 인정할 폭 범위
# BEV 화면에서 아래쪽 차선 간격이 크게 보일 수 있으므로 넓게 허용
PAIR_MIN_LANE_WIDTH_PX = 130
PAIR_MAX_LANE_WIDTH_PX = 560

# last_lane_width는 single lane 주행 안정성에 영향을 주므로 너무 큰 값으로 업데이트하지 않음
LANE_WIDTH_UPDATE_MIN_PX = 160
LANE_WIDTH_UPDATE_MAX_PX = 380

# =========================
# Both-lane curve handling
# =========================
# 두 차선이 모두 보이는 곡선 구간에서는 평균 중앙선 대신 바깥쪽 차선을 기준으로 중앙선 생성
USE_OUTER_LANE_IN_BOTH_CURVE = True

# 이 값보다 heading이 크면 곡선 both 모드로 판단
BOTH_CURVE_HEADING_TH = 0.08

# both 차선의 아래쪽 폭과 앞쪽 폭 차이가 이 값보다 크면 평균 중앙선 신뢰도 낮음
BOTH_WIDTH_DIFF_TH_PX = 35

# =========================
# Tight white detection params
# =========================
# 햇빛/반사광을 흰색 선으로 오인하지 않도록 매우 타이트하게 설정
WHITE_S_MAX = 45         # 흰색은 채도가 낮아야 함
WHITE_V_MIN = 215        # 아주 밝은 흰색만 허용
WHITE_GRAY_MIN = 215     # Gray 밝기 기준
WHITE_LAB_L_MIN = 215    # LAB L 채널 밝기 기준
WHITE_BGR_MIN = 190      # BGR 세 채널 모두 일정 이상이어야 함
WHITE_BGR_DIFF_MAX = 45  # B,G,R 차이가 너무 크면 흰색 아님

# =========================
# Sliding window
# =========================
N_WINDOWS = 9
WINDOW_MARGIN = 70
MIN_PIXELS_TO_RECENTER = 35
MIN_LINE_PIXELS = 90
HIST_PEAK_MIN = 15

# 곡선에서 정상 좌우 차선 후보가 같은 군집으로 묶이지 않도록 완화
CLOSE_LINE_REJECT_PX = 140

# 아주 가까운 peak 노이즈 병합용
PEAK_REGION_MERGE_PX = 30

MAX_CANDIDATES_TOTAL = 8
MAX_CANDIDATES_PER_SIDE = 4

LOOKAHEAD_Y_RATIO = 0.58

TUNNEL_BRIGHTNESS = 70
MIN_WHITE_PIXELS = 500
MAX_LOST_FRAMES = 8

SINGLE_SIDE_DEADBAND_PX = 55

# =========================
# Multi-band histogram params
# =========================
# 사람 눈에는 두 차선이 보이지만 아래쪽 히스토그램에는 한쪽만 잡히는 문제 해결용
USE_MULTI_BAND_HISTOGRAM = True

# 기존에는 binary 상단 35%를 버렸는데, 곡선에서 반대쪽 차선이 위쪽에만 보이면 놓칠 수 있음
TOP_IGNORE_RATIO = 0.25

# 후보를 찾을 히스토그램 영역들
# 아래쪽, 중간, 위쪽을 모두 사용
HISTOGRAM_BANDS = [
    (0.55, 1.00),  # 기존 아래쪽 영역
    (0.38, 0.75),  # 중간 영역
    (0.25, 0.55),  # 위쪽 영역
]

SHOW_DEBUG = True


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def poly_x(fit, y):
    return fit[0] * y * y + fit[1] * y + fit[2]


def normalize_deg(angle_deg):
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def angle_in_range(angle_deg, min_deg, max_deg):
    """Return True when angle_deg is inside [min_deg, max_deg]. Supports wrap-around."""
    angle_deg = normalize_deg(angle_deg)
    min_deg = normalize_deg(min_deg)
    max_deg = normalize_deg(max_deg)

    if min_deg <= max_deg:
        return min_deg <= angle_deg <= max_deg

    return angle_deg >= min_deg or angle_deg <= max_deg




# ============================================================
# Mission FSM helpers
# ============================================================

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


MISSION_ORDER_ALL = [
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


def valid_range(value):
    return value > 0.0 and not math.isinf(value) and not math.isnan(value)


class FrameDebouncer:
    def __init__(self, on_frames=4, off_frames=4):
        self.on_frames = int(on_frames)
        self.off_frames = int(off_frames)
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

    def reset(self):
        self.hit_count = 0
        self.clear_count = 0
        self.active = False


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


class StanleyMissionFSMNode(Node):
    def __init__(self):
        super().__init__('stanley_mission_fsm_node')

        self.bridge = CvBridge()

        # -------------------------
        # Topic params
        # -------------------------
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('cone_blocked_topic', '/cone/blocked_lanes')
        self.declare_parameter('manual_state_topic', '/mission/set_state')
        self.declare_parameter('state_topic', '/mission/state')
        self.declare_parameter('debug_topic', '/white/debug/compressed')

        camera_topic = self.get_parameter('camera_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        cone_topic = self.get_parameter('cone_blocked_topic').value
        manual_topic = self.get_parameter('manual_state_topic').value
        state_topic = self.get_parameter('state_topic').value
        debug_topic = self.get_parameter('debug_topic').value

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)

        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.img_callback,
            qos_profile_sensor_data
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.cone_sub = self.create_subscription(
            String,
            cone_topic,
            self.cone_callback,
            10
        )

        self.manual_sub = self.create_subscription(
            String,
            manual_topic,
            self.manual_state_callback,
            10
        )

        # -------------------------
        # Original lane-control internal variables
        # 원래 lane_node.py의 변수 이름을 유지한다.
        # -------------------------
        self.prev_speed = 0.0
        self.prev_steer = 0.0
        self.prev_cte_norm = 0.0

        self.last_lane_width = INITIAL_LANE_WIDTH_PX
        self.prev_center_fit = None
        self.last_single_side = None
        self.lost_frames = 0

        self.lane_width_samples = []
        self.lane_width_calibrated = False

        # Mission에서 차선 목표 중심만 옆 차선으로 밀 때 사용한다.
        # 0이면 원래 lane_node와 동일한 Stanley 제어가 된다.
        self.current_lane_bias_px = 0.0

        # -------------------------
        # Mission params
        # -------------------------
        self.declare_parameter('use_traffic_light', False)
        self.declare_parameter('start_state', '')

        self.declare_parameter('track_width_m', 0.60)
        self.declare_parameter('robot_width_m', 0.13)
        self.declare_parameter('safety_margin_m', 0.05)

        # 상태 전환이 너무 빠르게 넘어가지 않도록 보수적으로 잡은 기본값
        self.declare_parameter('ped_clear_time', 1.2)
        self.declare_parameter('box_min_time', 2.5)
        self.declare_parameter('box_clear_time', 1.2)
        self.declare_parameter('tunnel_min_time', 1.5)
        self.declare_parameter('tunnel_recover_frames', 6)
        self.declare_parameter('rotary_min_time', 4.0)
        self.declare_parameter('rotary_clear_time', 2.0)
        self.declare_parameter('rotary_require_seen', True)
        self.declare_parameter('cone_min_time', 5.0)
        self.declare_parameter('cone_latched_min_time', 2.0)
        self.declare_parameter('cone_target_vote_window', 5)
        self.declare_parameter('cone_lane_bias_ratio', 0.62)
        self.declare_parameter('cone_recover_time', 1.0)
        self.declare_parameter('cone_recover_confidence', 0.55)
        self.declare_parameter('cone_lidar_clear_distance', 0.50)
        self.declare_parameter('cone_lidar_side_angle_min_deg', 20.0)
        self.declare_parameter('cone_lidar_side_angle_max_deg', 120.0)
        self.declare_parameter('cone_lidar_min_hits', 3)
        self.declare_parameter('cone_lidar_clear_frames', 3)
        self.declare_parameter('cone_extra_yaw', 0.28)

        # -------------------------
        # Simple cone force-shift params
        # -------------------------
        # CONE 상태에서 target lane이 left/right로 확정되면
        # 일정 시간 동안 차선추종을 무시하고 강제 조향한 뒤, 다시 정상 차선추종으로 복귀한다.
        self.declare_parameter('cone_simple_force_enabled', True)
        self.declare_parameter('cone_force_duration', 2.00)
        self.declare_parameter('cone_force_speed', 0.40)
        self.declare_parameter('cone_force_yaw', 0.45)
        self.declare_parameter('cone_after_force_speed_limit', 0.45)

        # 일반적으로 /cmd_vel.angular.z 양수가 왼쪽 회전이다.
        # 실제 차량이 반대로 움직이면 False로 바꿔서 테스트한다.
        self.declare_parameter('cone_left_yaw_positive', True)

        # 왼쪽 회피 시 콘은 오른쪽에 남으므로 오른쪽 0~120도 sector를 본다.
        # 오른쪽 회피 시 콘은 왼쪽에 남으므로 왼쪽 -120~0도 sector를 본다.
        self.declare_parameter('cone_side_distance', 0.50)
        self.declare_parameter('cone_side_clear_frames', 3)
        self.declare_parameter('cone_side_min_clusters', 2)
        self.declare_parameter('cone_side_cluster_gap_m', 0.12)
        self.declare_parameter('cone_side_cluster_min_points', 1)

        # Cone latch 안정화:
        # center+right 또는 center+left 두 위치가 확정되면 이후 새 위치는 추가하지 않는다.
        # center만 계속 보이면 left가 잘 안 보이는 상황으로 보고 center+left로 보정한다.
        self.declare_parameter('cone_freeze_after_pair', True)
        self.declare_parameter('cone_center_only_assume_left_time', 999.0)
        self.declare_parameter('cone_center_only_auto_assume_left', False)

        self.declare_parameter('cone_right_angle_min_deg', 0.0)
        self.declare_parameter('cone_right_angle_max_deg', 120.0)
        self.declare_parameter('cone_left_angle_min_deg', -120.0)
        self.declare_parameter('cone_left_angle_max_deg', 0.0)

        # CONE 상태부터 들어오는 고깔 정보만 저장한다.
        # ROTARY에서 너무 일찍 들어온 고깔 정보는 오판단/freeze 원인이 될 수 있으므로 무시한다.
        self.declare_parameter('cone_accept_states', ['CONE'])
        self.declare_parameter('parking_finish_ratio', 0.22)

        # 미션별 속도 제한. 실제 제어 계산은 원래 Stanley 제어가 하고,
        # 여기서는 상태별로 최고속도만 제한한다.
        self.declare_parameter('ped_speed_limit', 0.65)
        self.declare_parameter('box_speed_limit', 0.30)
        self.declare_parameter('avoid_speed_limit', 0.24)
        self.declare_parameter('rotary_speed_limit', 0.70)
        self.declare_parameter('follow_speed_limit', 0.22)
        self.declare_parameter('cone_speed_limit', 0.50)
        self.declare_parameter('parking_approach_speed_limit', 0.24)

        # -------------------------
        # Tight LiDAR sector params
        # -------------------------
        # 보행자: 전방 ±35도, 0.50m 이내만 정지 대상으로 본다.
        self.declare_parameter('ped_front_angle_deg', 40.0)
        self.declare_parameter('ped_front_distance', 0.50)
        self.declare_parameter('ped_front_min_hits', 2)
        self.declare_parameter('ped_front_wall_width_threshold', 0.30)
        self.declare_parameter('ped_line_roi_y_min_ratio', 0.42)
        self.declare_parameter('ped_line_roi_y_max_ratio', 0.78)
        self.declare_parameter('ped_line_fill_ratio', 0.55)
        self.declare_parameter('ped_line_min_width_px', 90)
        self.declare_parameter('ped_line_min_rows', 4)
        self.declare_parameter('ped_line_stop_drop_ratio', 0.35)

        # 박스:
        # 1) 현재 주행 중인 트랙 corridor 안에 있는 LiDAR 클러스터인지 먼저 확인한다.
        # 2) 그 클러스터가 설정한 LiDAR 각도 영역에 걸쳐 있어야 한다.
        # 3) 추가로 10cm 이내의 가까운 점과 L-shape 특징을 만족해야 박스로 인정한다.
        # 이렇게 해야 다른 트랙/벽의 물체가 단순 각도 조건만으로 박스가 되는 것을 줄일 수 있다.
        self.declare_parameter('box_front_angle_deg', 60.0)
        self.declare_parameter('box_front_stop_distance', 0.45)
        self.declare_parameter('box_front_min_hits', 2)
        self.declare_parameter('box_side_angle_min_deg', 20.0)
        self.declare_parameter('box_side_angle_max_deg', 90.0)
        self.declare_parameter('box_side_distance', 0.65)
        self.declare_parameter('box_side_min_hits', 2)
        self.declare_parameter('box1_angle_min_deg', -90.0)
        self.declare_parameter('box1_angle_max_deg', 20.0)
        self.declare_parameter('box1_front_stop_distance', 0.60)
        self.declare_parameter('box_track_x_min', 0.15)
        self.declare_parameter('box_track_x_max', 0.95)
        self.declare_parameter('box_track_half_width', 0.32)
        self.declare_parameter('box_track_min_points', 2)
        self.declare_parameter('box_wall_width_threshold', 0.58)
        self.declare_parameter('box1_close_side_distance', 0.20)
        self.declare_parameter('box1_close_side_min_hits', 2)
        self.declare_parameter('box1_close_wall_width_threshold', 0.38)
        self.declare_parameter('box1_close_wall_x_span', 0.30)

        # 박스 L-shape 판정 파라미터.
        # box_lshape_near_distance는 20cm 이내의 가까운 점까지 L-shape 후보로 본다.
        self.declare_parameter('box_lshape_near_distance', 0.20)
        self.declare_parameter('box_lshape_min_points', 6)
        self.declare_parameter('box_lshape_min_x_span', 0.05)
        self.declare_parameter('box_lshape_min_y_span', 0.05)
        self.declare_parameter('box_lshape_corner_band', 0.035)
        self.declare_parameter('box_lshape_min_leg_points', 2)
        self.declare_parameter('box_avoid_hold_time', 0.5)
        self.declare_parameter('box1_extra_yaw', 0.38)

        # BOX2는 큰 곡률 구간에서 보이므로 BOX1보다 ROI/각도를 조금 넓게 쓴다.
        self.declare_parameter('box2_track_half_width', 0.34)
        self.declare_parameter('box2_front_angle_deg', 70.0)
        self.declare_parameter('box2_front_min_hits', 3)
        self.declare_parameter('box2_side_min_hits', 2)
        self.declare_parameter('box2_avoid_hold_time', 0.9)
        self.declare_parameter('box2_extra_yaw', 0.85)
        self.declare_parameter('box2_left_fallback_enabled', True)

        # 터널:
        # 짧은 터널이므로 카메라 밝기 대신 LiDAR 좌/우 벽 검출로 진입/탈출을 판단한다.
        self.declare_parameter('tunnel_enter_frames', 2)
        self.declare_parameter('tunnel_exit_frames', 2)
        self.declare_parameter('tunnel_enter_min_time', 0.2)
        self.declare_parameter('tunnel_memory_min_time', 0.2)
        self.declare_parameter('tunnel_wall_distance', 0.40)
        self.declare_parameter('tunnel_wall_min_y', 0.08)
        self.declare_parameter('tunnel_wall_x_min', -0.05)
        self.declare_parameter('tunnel_wall_x_max', 0.70)
        self.declare_parameter('tunnel_wall_min_hits', 3)
        self.declare_parameter('tunnel_exit_front_angle_deg', 90.0)
        # 터널 탈출 판단은 전방 전체(-90~90)가 아니라 측면 밴드만 본다.
        # 기본값: -90~-60, 60~90도.
        self.declare_parameter('tunnel_exit_side_angle_min_abs_deg', 60.0)
        self.declare_parameter('tunnel_exit_side_angle_max_abs_deg', 90.0)
        self.declare_parameter('tunnel_center_x_min', 0.05)
        self.declare_parameter('tunnel_center_x_max', 0.60)
        self.declare_parameter('tunnel_wall_center_gain', 3.0)
        self.declare_parameter('tunnel_exit_drive_distance', 1.25)

        # 회전교차로 차량: 사용자가 보낸 TrackVehicleFollowNode의 전방 차량 추종 구조를 반영한다.
        self.declare_parameter('rotary_front_x_min', 0.15)
        self.declare_parameter('rotary_front_x_max', 0.80)
        self.declare_parameter('rotary_front_half_width', 0.36)
        self.declare_parameter('rotary_front_angle_deg', 35.0)
        self.declare_parameter('rotary_cluster_min_points', 3)
        self.declare_parameter('rotary_wall_width_threshold', 0.65)
        self.declare_parameter('rotary_emergency_stop_distance', 0.28)
        self.declare_parameter('rotary_follow_distance', 0.40)
        self.declare_parameter('rotary_slow_distance', 0.50)
        self.declare_parameter('rotary_clear_speed_limit', 0.845)
        self.declare_parameter('rotary_clear_speed_gain', 1.30)
        self.declare_parameter('rotary_slow_speed_limit', 0.416)
        self.declare_parameter('rotary_follow_speed_limit', 0.26)
        self.declare_parameter('rotary_min_moving_speed', 0.156)
        self.declare_parameter('rotary_wait_before_seen_speed', 0.0)

        self.use_traffic_light = bool(self.get_parameter('use_traffic_light').value)
        self.track_width_m = float(self.get_parameter('track_width_m').value)
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.safety_margin_m = float(self.get_parameter('safety_margin_m').value)
        self.drivable_half_width = max(
            0.12,
            self.track_width_m / 2.0 - self.robot_width_m / 2.0 - self.safety_margin_m
        )

        self.ped_clear_time = float(self.get_parameter('ped_clear_time').value)
        self.box_min_time = float(self.get_parameter('box_min_time').value)
        self.box_clear_time = float(self.get_parameter('box_clear_time').value)
        self.tunnel_min_time = float(self.get_parameter('tunnel_min_time').value)
        self.tunnel_recover_frames = int(self.get_parameter('tunnel_recover_frames').value)
        self.rotary_min_time = float(self.get_parameter('rotary_min_time').value)
        self.rotary_clear_time = float(self.get_parameter('rotary_clear_time').value)
        self.rotary_require_seen = bool(self.get_parameter('rotary_require_seen').value)
        self.cone_min_time = float(self.get_parameter('cone_min_time').value)
        self.cone_latched_min_time = float(self.get_parameter('cone_latched_min_time').value)
        self.cone_target_vote_window = int(self.get_parameter('cone_target_vote_window').value)
        self.cone_lane_bias_ratio = float(self.get_parameter('cone_lane_bias_ratio').value)
        self.cone_recover_time = float(self.get_parameter('cone_recover_time').value)
        self.cone_recover_confidence = float(self.get_parameter('cone_recover_confidence').value)
        self.cone_lidar_clear_distance = float(self.get_parameter('cone_lidar_clear_distance').value)
        self.cone_lidar_side_angle_min_deg = float(self.get_parameter('cone_lidar_side_angle_min_deg').value)
        self.cone_lidar_side_angle_max_deg = float(self.get_parameter('cone_lidar_side_angle_max_deg').value)
        self.cone_lidar_min_hits = int(self.get_parameter('cone_lidar_min_hits').value)
        self.cone_lidar_clear_frames = int(self.get_parameter('cone_lidar_clear_frames').value)
        self.cone_extra_yaw = float(self.get_parameter('cone_extra_yaw').value)

        self.cone_simple_force_enabled = bool(self.get_parameter('cone_simple_force_enabled').value)
        self.cone_force_duration = float(self.get_parameter('cone_force_duration').value)
        self.cone_force_speed = float(self.get_parameter('cone_force_speed').value)
        self.cone_force_yaw = float(self.get_parameter('cone_force_yaw').value)
        self.cone_after_force_speed_limit = float(self.get_parameter('cone_after_force_speed_limit').value)
        if self.cone_force_speed <= 0.0:
            self.cone_force_speed = self.cone_after_force_speed_limit
        self.cone_left_yaw_positive = bool(self.get_parameter('cone_left_yaw_positive').value)

        self.cone_side_distance = float(self.get_parameter('cone_side_distance').value)
        self.cone_side_clear_frames = int(self.get_parameter('cone_side_clear_frames').value)
        self.cone_side_min_clusters = int(self.get_parameter('cone_side_min_clusters').value)
        self.cone_side_cluster_gap_m = float(self.get_parameter('cone_side_cluster_gap_m').value)
        self.cone_side_cluster_min_points = int(self.get_parameter('cone_side_cluster_min_points').value)
        self.cone_freeze_after_pair = bool(self.get_parameter('cone_freeze_after_pair').value)
        self.cone_center_only_assume_left_time = float(
            self.get_parameter('cone_center_only_assume_left_time').value
        )
        self.cone_center_only_auto_assume_left = bool(
            self.get_parameter('cone_center_only_auto_assume_left').value
        )
        self.cone_right_angle_min_deg = float(self.get_parameter('cone_right_angle_min_deg').value)
        self.cone_right_angle_max_deg = float(self.get_parameter('cone_right_angle_max_deg').value)
        self.cone_left_angle_min_deg = float(self.get_parameter('cone_left_angle_min_deg').value)
        self.cone_left_angle_max_deg = float(self.get_parameter('cone_left_angle_max_deg').value)

        self.cone_accept_states = set([
            str(x).strip().upper()
            for x in self.get_parameter('cone_accept_states').value
        ])
        self.parking_finish_ratio = float(self.get_parameter('parking_finish_ratio').value)

        self.ped_speed_limit = float(self.get_parameter('ped_speed_limit').value)
        self.box_speed_limit = float(self.get_parameter('box_speed_limit').value)
        self.avoid_speed_limit = float(self.get_parameter('avoid_speed_limit').value)
        self.rotary_speed_limit = float(self.get_parameter('rotary_speed_limit').value)
        self.follow_speed_limit = float(self.get_parameter('follow_speed_limit').value)
        self.cone_speed_limit = float(self.get_parameter('cone_speed_limit').value)
        self.parking_approach_speed_limit = float(self.get_parameter('parking_approach_speed_limit').value)

        self.ped_front_angle_deg = float(self.get_parameter('ped_front_angle_deg').value)
        self.ped_front_distance = float(self.get_parameter('ped_front_distance').value)
        self.ped_front_min_hits = int(self.get_parameter('ped_front_min_hits').value)
        self.ped_front_wall_width_threshold = float(self.get_parameter('ped_front_wall_width_threshold').value)
        self.ped_line_roi_y_min_ratio = float(self.get_parameter('ped_line_roi_y_min_ratio').value)
        self.ped_line_roi_y_max_ratio = float(self.get_parameter('ped_line_roi_y_max_ratio').value)
        self.ped_line_fill_ratio = float(self.get_parameter('ped_line_fill_ratio').value)
        self.ped_line_min_width_px = int(self.get_parameter('ped_line_min_width_px').value)
        self.ped_line_min_rows = int(self.get_parameter('ped_line_min_rows').value)
        self.ped_line_stop_drop_ratio = float(self.get_parameter('ped_line_stop_drop_ratio').value)

        self.box_front_angle_deg = float(self.get_parameter('box_front_angle_deg').value)
        self.box_front_stop_distance = float(self.get_parameter('box_front_stop_distance').value)
        self.box_front_min_hits = int(self.get_parameter('box_front_min_hits').value)
        self.box_side_angle_min_deg = float(self.get_parameter('box_side_angle_min_deg').value)
        self.box_side_angle_max_deg = float(self.get_parameter('box_side_angle_max_deg').value)
        self.box_side_distance = float(self.get_parameter('box_side_distance').value)
        self.box_side_min_hits = int(self.get_parameter('box_side_min_hits').value)
        self.box1_angle_min_deg = float(self.get_parameter('box1_angle_min_deg').value)
        self.box1_angle_max_deg = float(self.get_parameter('box1_angle_max_deg').value)
        self.box1_front_stop_distance = float(self.get_parameter('box1_front_stop_distance').value)
        self.box_track_x_min = float(self.get_parameter('box_track_x_min').value)
        self.box_track_x_max = float(self.get_parameter('box_track_x_max').value)
        self.box_track_half_width = float(self.get_parameter('box_track_half_width').value)
        self.box_track_min_points = int(self.get_parameter('box_track_min_points').value)
        self.box_wall_width_threshold = float(self.get_parameter('box_wall_width_threshold').value)
        self.box1_close_side_distance = float(self.get_parameter('box1_close_side_distance').value)
        self.box1_close_side_min_hits = int(self.get_parameter('box1_close_side_min_hits').value)
        self.box1_close_wall_width_threshold = float(self.get_parameter('box1_close_wall_width_threshold').value)
        self.box1_close_wall_x_span = float(self.get_parameter('box1_close_wall_x_span').value)
        self.box_lshape_near_distance = float(self.get_parameter('box_lshape_near_distance').value)
        self.box_lshape_min_points = int(self.get_parameter('box_lshape_min_points').value)
        self.box_lshape_min_x_span = float(self.get_parameter('box_lshape_min_x_span').value)
        self.box_lshape_min_y_span = float(self.get_parameter('box_lshape_min_y_span').value)
        self.box_lshape_corner_band = float(self.get_parameter('box_lshape_corner_band').value)
        self.box_lshape_min_leg_points = int(self.get_parameter('box_lshape_min_leg_points').value)
        self.box_avoid_hold_time = float(self.get_parameter('box_avoid_hold_time').value)
        self.box1_extra_yaw = float(self.get_parameter('box1_extra_yaw').value)
        self.box2_track_half_width = float(self.get_parameter('box2_track_half_width').value)
        self.box2_front_angle_deg = float(self.get_parameter('box2_front_angle_deg').value)
        self.box2_front_min_hits = int(self.get_parameter('box2_front_min_hits').value)
        self.box2_side_min_hits = int(self.get_parameter('box2_side_min_hits').value)
        self.box2_avoid_hold_time = float(self.get_parameter('box2_avoid_hold_time').value)
        self.box2_extra_yaw = float(self.get_parameter('box2_extra_yaw').value)
        self.box2_left_fallback_enabled = bool(self.get_parameter('box2_left_fallback_enabled').value)

        self.tunnel_enter_frames = int(self.get_parameter('tunnel_enter_frames').value)
        self.tunnel_exit_frames = int(self.get_parameter('tunnel_exit_frames').value)
        self.tunnel_enter_min_time = float(self.get_parameter('tunnel_enter_min_time').value)
        self.tunnel_memory_min_time = float(self.get_parameter('tunnel_memory_min_time').value)
        self.tunnel_wall_distance = float(self.get_parameter('tunnel_wall_distance').value)
        self.tunnel_wall_min_y = float(self.get_parameter('tunnel_wall_min_y').value)
        self.tunnel_wall_x_min = float(self.get_parameter('tunnel_wall_x_min').value)
        self.tunnel_wall_x_max = float(self.get_parameter('tunnel_wall_x_max').value)
        self.tunnel_wall_min_hits = int(self.get_parameter('tunnel_wall_min_hits').value)
        self.tunnel_exit_front_angle_deg = float(self.get_parameter('tunnel_exit_front_angle_deg').value)
        self.tunnel_exit_side_angle_min_abs_deg = float(
            self.get_parameter('tunnel_exit_side_angle_min_abs_deg').value
        )
        self.tunnel_exit_side_angle_max_abs_deg = float(
            self.get_parameter('tunnel_exit_side_angle_max_abs_deg').value
        )
        self.tunnel_center_x_min = float(self.get_parameter('tunnel_center_x_min').value)
        self.tunnel_center_x_max = float(self.get_parameter('tunnel_center_x_max').value)
        self.tunnel_wall_center_gain = float(self.get_parameter('tunnel_wall_center_gain').value)
        self.tunnel_exit_drive_distance = float(self.get_parameter('tunnel_exit_drive_distance').value)

        self.rotary_front_x_min = float(self.get_parameter('rotary_front_x_min').value)
        self.rotary_front_x_max = float(self.get_parameter('rotary_front_x_max').value)
        self.rotary_front_half_width = float(self.get_parameter('rotary_front_half_width').value)
        self.rotary_front_angle_deg = float(self.get_parameter('rotary_front_angle_deg').value)
        self.rotary_cluster_min_points = int(self.get_parameter('rotary_cluster_min_points').value)
        self.rotary_wall_width_threshold = float(self.get_parameter('rotary_wall_width_threshold').value)
        self.rotary_emergency_stop_distance = float(self.get_parameter('rotary_emergency_stop_distance').value)
        self.rotary_follow_distance = float(self.get_parameter('rotary_follow_distance').value)
        self.rotary_slow_distance = float(self.get_parameter('rotary_slow_distance').value)
        self.rotary_clear_speed_limit = float(self.get_parameter('rotary_clear_speed_limit').value)
        self.rotary_clear_speed_gain = float(self.get_parameter('rotary_clear_speed_gain').value)
        self.rotary_slow_speed_limit = float(self.get_parameter('rotary_slow_speed_limit').value)
        self.rotary_follow_speed_limit = float(self.get_parameter('rotary_follow_speed_limit').value)
        self.rotary_min_moving_speed = float(self.get_parameter('rotary_min_moving_speed').value)
        self.rotary_wait_before_seen_speed = float(self.get_parameter('rotary_wait_before_seen_speed').value)

        if self.use_traffic_light:
            self.mission_order = MISSION_ORDER_ALL.copy()
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

        start_state = str(self.get_parameter('start_state').value).strip().upper()
        if start_state in MISSION_ORDER_ALL and (self.use_traffic_light or start_state != Mission.WAIT_TRAFFIC):
            self.state = start_state
        else:
            self.state = default_start_state

        self.state_enter_time = time.monotonic()

        # -------------------------
        # Mission internal state
        # -------------------------
        self.path_center_y_m = 0.0
        self.lidar_clusters = []
        self.lidar_points = []
        self.last_scan_time = 0.0

        # LiDAR sector 상태. 보행자/박스는 넓은 클러스터 ROI 대신 각도 기반으로 좁게 본다.
        self.ped_front_obs = False
        self.ped_front_hit = 0
        self.ped_front_min_dist = 9.9

        self.box_front_obs = False
        self.box_front_hit = 0
        self.box_front_min_dist = 9.9
        self.box_left_obs = False
        self.box_left_hit = 0
        self.box_left_obs_dist = 9.9
        self.box_right_obs = False
        self.box_right_hit = 0
        self.box_right_obs_dist = 9.9
        self.box_track_candidate = None

        self.rotary_follow_status = 'WAIT_VEHICLE'

        self.red_db = FrameDebouncer(4, 6)
        self.green_db = FrameDebouncer(4, 6)
        self.ped_db = FrameDebouncer(2, 6)
        self.ped_line_db = FrameDebouncer(2, 3)
        self.box_db = FrameDebouncer(2, 6)
        self.rotary_db = FrameDebouncer(3, 6)
        self.finish_db = FrameDebouncer(5, 8)

        self.ped_stop_line_was_seen = False
        self.ped_waiting_after_line = False
        self.ped_stop_line_ratio = 0.0
        self.ped_stop_line_peak_ratio = 0.0
        self.pedestrian_was_seen = False
        self.last_ped_clear_time = None

        self.box_avoid_started = False
        self.box_clear_start = None
        self.box_seen_time = None
        self.last_box_side = 0.0

        self.tunnel_was_seen = False
        self.tunnel_recover_count = 0
        self.tunnel_candidate_count = 0
        self.tunnel_seen_time = None
        self.tunnel_exit_start_time = None

        self.rotary_vehicle_seen = False
        self.rotary_clear_start = None
        self.rotary_vehicle_distance = 9.9

        # YOLO 지연 대응용 latch.
        # 한 번이라도 left/center/right가 들어오면 상태가 끝날 때까지 지우지 않는다.
        self.cone_latched = False
        self.cone_latched_lanes = set()
        self.cone_target_lane = 'center'
        self.cone_target_locked = False
        self.cone_target_votes = []
        self.cone_first_latch_time = None
        self.last_cone_msg_time = 0.0
        self.cone_center_only_start_time = None
        self.cone_latch_frozen = False
        self.cone_recover_start_time = None
        self.cone_lidar_clear_count = 0
        self.reset_simple_cone_vars()

        self.parking_started = False
        self.parking_step_index = 0
        self.parking_step_start = 0.0
        self.parking_sequence = [
            (0.55, 0.18, 0.00),
            (0.95, 0.16, -0.55),
            (0.85, 0.16, 0.00),
            (0.70, 0.14, 0.45),
            (0.30, 0.08, 0.00),
            (999.0, 0.00, 0.00),
        ]

        self.state_timer = self.create_timer(0.2, self.publish_state)

        self.get_logger().info(
            f'stanley mission fsm node start. state={self.state}, use_traffic_light={self.use_traffic_light}. '
            'This node keeps the original Stanley lane controller and adds mission gating.'
        )

    # =========================
    # Image preprocessing
    # =========================
    def detect_white(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        h, s, v = cv2.split(hsv)
        l_channel, _, _ = cv2.split(lab)

        b, g, r = cv2.split(img)

        # 1) HSV 기준:
        # 흰색은 채도가 낮고 밝기가 높아야 함
        mask_hsv = cv2.inRange(
            hsv,
            np.array([0, 0, WHITE_V_MIN]),
            np.array([179, WHITE_S_MAX, 255])
        )

        # 2) Gray 밝기 기준:
        # 단순히 밝은 영역만 통과
        mask_gray = cv2.inRange(gray, WHITE_GRAY_MIN, 255)

        # 3) LAB L 채널 기준:
        # 조도 변화가 있어도 밝은 흰색만 통과시키기 위한 보조 조건
        mask_lab = cv2.inRange(l_channel, WHITE_LAB_L_MIN, 255)

        # 4) BGR 균형 기준:
        # 햇빛 반사나 노란빛/파란빛이 강한 영역 제거
        min_bgr = np.minimum(np.minimum(b, g), r)
        max_bgr = np.maximum(np.maximum(b, g), r)
        bgr_diff = max_bgr - min_bgr

        mask_bgr_min = cv2.inRange(min_bgr, WHITE_BGR_MIN, 255)
        mask_bgr_balanced = cv2.inRange(bgr_diff, 0, WHITE_BGR_DIFF_MAX)

        # 핵심:
        # 기존처럼 OR로 합치지 않고, 조건을 모두 만족하는 픽셀만 흰색으로 인정
        mask = cv2.bitwise_and(mask_hsv, mask_gray)
        mask = cv2.bitwise_and(mask, mask_lab)
        mask = cv2.bitwise_and(mask, mask_bgr_min)
        mask = cv2.bitwise_and(mask, mask_bgr_balanced)

        # 노이즈 제거
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_mid = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mid, iterations=1)

        # 너무 많이 부풀리면 햇빛 반사와 선이 붙을 수 있으므로 dilation은 약하게
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        return mask

    def img_warp(self, img):
        h, w = img.shape[:2]

        src_top_x = 175
        src_top_y = 305

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
        warp = cv2.warpPerspective(img, matrix, (w, h))

        return warp

    # =========================
    # Candidate extraction / filtering
    # =========================
    def extract_raw_histogram_candidates(self, histogram):
        candidates = []
        start = None

        for i, value in enumerate(histogram):
            if value > HIST_PEAK_MIN:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i
                    region = histogram[start:end]

                    if len(region) > 0:
                        xs = np.arange(start, end)
                        weight_sum = float(np.sum(region))

                        if weight_sum > 0.0:
                            cx = int(np.sum(xs * region) / weight_sum)
                            strength = weight_sum
                            candidates.append((cx, strength))

                    start = None

        if start is not None:
            end = len(histogram)
            region = histogram[start:end]

            if len(region) > 0:
                xs = np.arange(start, end)
                weight_sum = float(np.sum(region))

                if weight_sum > 0.0:
                    cx = int(np.sum(xs * region) / weight_sum)
                    strength = weight_sum
                    candidates.append((cx, strength))

        return candidates

    def extract_multi_band_candidates(self, binary, image_center):
        h, w = binary.shape

        all_candidates = []

        if not USE_MULTI_BAND_HISTOGRAM:
            histogram = np.sum(binary[int(h * 0.55):, :], axis=0)
            return self.extract_raw_histogram_candidates(histogram)

        for y0_ratio, y1_ratio in HISTOGRAM_BANDS:
            y0 = int(h * y0_ratio)
            y1 = int(h * y1_ratio)

            y0 = clamp(y0, 0, h - 1)
            y1 = clamp(y1, y0 + 1, h)

            histogram = np.sum(binary[y0:y1, :], axis=0)

            candidates = self.extract_raw_histogram_candidates(histogram)

            # 중간/위쪽 후보도 쓰되, 아래쪽 후보보다 약간 낮은 가중치
            band_height = max(y1 - y0, 1)
            band_weight = band_height / float(h)

            for cx, strength in candidates:
                all_candidates.append((cx, strength * band_weight))

        if len(all_candidates) == 0:
            return []

        # 여러 band에서 같은 차선이 반복 검출되면 하나로 병합
        all_candidates = self.merge_tiny_peak_regions(all_candidates, image_center)

        return all_candidates

    def merge_tiny_peak_regions(self, candidates, image_center):
        """
        아주 가까운 peak들은 노이즈 또는 한 선의 두꺼운 영역으로 보고 하나로 병합한다.
        """
        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda item: item[0])

        merged = []

        for cx, strength in candidates:
            if len(merged) == 0:
                merged.append((cx, strength))
                continue

            prev_cx, prev_strength = merged[-1]

            if abs(cx - prev_cx) < PEAK_REGION_MERGE_PX:
                # 같은 선 안의 작은 peak면 더 강한 쪽 또는 중심 가까운 쪽을 유지
                if strength > prev_strength:
                    merged[-1] = (cx, strength)
                elif strength == prev_strength:
                    if abs(cx - image_center) < abs(prev_cx - image_center):
                        merged[-1] = (cx, strength)
            else:
                merged.append((cx, strength))

        return merged

    def suppress_close_outer_lines(self, candidates, image_center):
        """
        핵심 필터:
        가까운 흰색 선들은 같은 차선/옆 차선 간섭으로 보고 묶는다.
        군집 안에서는 이미지 중심에 가장 가까운 선만 남긴다.
        """
        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda item: item[0])

        clusters = []
        current_cluster = [candidates[0]]

        for cand in candidates[1:]:
            prev_cx = current_cluster[-1][0]
            cx = cand[0]

            if abs(cx - prev_cx) < CLOSE_LINE_REJECT_PX:
                current_cluster.append(cand)
            else:
                clusters.append(current_cluster)
                current_cluster = [cand]

        clusters.append(current_cluster)

        filtered = []

        for cluster in clusters:
            # 군집 안에서는 중심에 가까운 선만 유지
            best = min(cluster, key=lambda item: abs(item[0] - image_center))
            filtered.append(best)

        filtered.sort(key=lambda item: abs(item[0] - image_center))

        return filtered[:MAX_CANDIDATES_TOTAL]

    def split_candidates_by_side(self, candidates, image_center):
        left = []
        right = []

        for cx, strength in candidates:
            if cx < image_center:
                left.append((cx, strength))
            else:
                right.append((cx, strength))

        left.sort(key=lambda item: abs(item[0] - image_center))
        right.sort(key=lambda item: abs(item[0] - image_center))

        return left[:MAX_CANDIDATES_PER_SIDE], right[:MAX_CANDIDATES_PER_SIDE]

    # =========================
    # Sliding window
    # =========================
    def fit_line_from_base(self, binary, base_x):
        h, w = binary.shape
        nonzero_y, nonzero_x = binary.nonzero()

        current_x = int(base_x)
        window_h = h // N_WINDOWS
        lane_inds = []

        for window in range(N_WINDOWS):
            win_y_low = h - (window + 1) * window_h
            win_y_high = h - window * window_h

            win_x_low = current_x - WINDOW_MARGIN
            win_x_high = current_x + WINDOW_MARGIN

            good_inds = (
                (nonzero_y >= win_y_low) &
                (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_low) &
                (nonzero_x < win_x_high)
            ).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > MIN_PIXELS_TO_RECENTER:
                current_x = int(np.mean(nonzero_x[good_inds]))

        if len(lane_inds) == 0:
            return None, 0

        lane_inds = np.concatenate(lane_inds)

        if len(lane_inds) < MIN_LINE_PIXELS:
            return None, len(lane_inds)

        x = nonzero_x[lane_inds]
        y = nonzero_y[lane_inds]

        try:
            fit = np.polyfit(y, x, 2)
        except Exception:
            return None, len(lane_inds)

        return fit, len(lane_inds)

    def build_fit_candidates(self, binary, candidates, height):
        fit_candidates = []

        for base_x, strength in candidates:
            fit, count = self.fit_line_from_base(binary, base_x)

            if fit is None:
                continue

            x_bottom = poly_x(fit, height - 1)

            fit_candidates.append({
                'fit': fit,
                'count': count,
                'base_x': base_x,
                'strength': strength,
                'x_bottom': x_bottom,
            })

        return fit_candidates

    def estimate_fit_heading(self, fit, height):
        y_bottom = height - 1
        y_ahead = int(height * LOOKAHEAD_Y_RATIO)

        x_bottom = poly_x(fit, y_bottom)
        x_ahead = poly_x(fit, y_ahead)

        lane_width_px = max(float(self.last_lane_width), 1.0)
        xm_per_pix = LANE_WIDTH_M / lane_width_px
        ym_per_pix = BEV_FORWARD_M / float(height)

        dx_m = (x_bottom - x_ahead) * xm_per_pix
        dy_m = (y_bottom - y_ahead) * ym_per_pix

        return math.atan2(dx_m, max(dy_m, 1e-6))

    def choose_best_pair(self, left_candidates, right_candidates, image_center):
        best_pair = None
        best_cost = None

        for left in left_candidates:
            for right in right_candidates:
                left_bottom = left['x_bottom']
                right_bottom = right['x_bottom']

                lane_width = right_bottom - left_bottom

                self.get_logger().info(
                    f"pair_check: left={left_bottom:.1f}, right={right_bottom:.1f}, "
                    f"width={lane_width:.1f}, allowed=({PAIR_MIN_LANE_WIDTH_PX}, {PAIR_MAX_LANE_WIDTH_PX})"
                )

                if not (PAIR_MIN_LANE_WIDTH_PX <= lane_width <= PAIR_MAX_LANE_WIDTH_PX):
                    continue

                pair_center = (left_bottom + right_bottom) / 2.0

                center_cost = abs(pair_center - image_center)
                width_cost = abs(lane_width - self.last_lane_width)

                count_bonus = 0.002 * (left['count'] + right['count'])

                cost = center_cost + 0.45 * width_cost - count_bonus

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_pair = left, right, lane_width

        return best_pair

    def choose_cone_outer_pair(
        self,
        fit_candidates,
        target_lane,
        image_center,
        reference_x=None,
        strict_outer=False,
    ):
        if target_lane not in ('left', 'right'):
            return None

        if reference_x is None:
            reference_x = image_center

        ordered = sorted(fit_candidates, key=lambda item: item['x_bottom'])
        valid_pairs = []

        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                left = ordered[i]
                right = ordered[j]
                lane_width = right['x_bottom'] - left['x_bottom']

                if not (PAIR_MIN_LANE_WIDTH_PX <= lane_width <= PAIR_MAX_LANE_WIDTH_PX):
                    continue

                pair_center = (left['x_bottom'] + right['x_bottom']) / 2.0
                count_bonus = 0.001 * (left['count'] + right['count'])
                valid_pairs.append((left, right, lane_width, pair_center, count_bonus))

        if len(valid_pairs) == 0:
            return None

        if strict_outer:
            if target_lane == 'right':
                best = max(valid_pairs, key=lambda pair: pair[3] + pair[4])
            else:
                best = min(valid_pairs, key=lambda pair: pair[3] - pair[4])
            return best[0], best[1], best[2]

        side_tolerance = max(25.0, float(self.last_lane_width) * 0.15)
        max_reference_jump = max(70.0, float(self.last_lane_width) * 0.65)
        near_reference = [
            pair for pair in valid_pairs
            if abs(pair[3] - reference_x) <= max_reference_jump
        ]

        if target_lane == 'right':
            preferred = [
                pair for pair in near_reference
                if pair[3] >= reference_x - side_tolerance
            ]
        else:
            preferred = [
                pair for pair in near_reference
                if pair[3] <= reference_x + side_tolerance
            ]

        if len(preferred) > 0:
            source = preferred
        elif len(near_reference) > 0:
            source = near_reference
        else:
            source = valid_pairs

        best = min(
            source,
            key=lambda pair: abs(pair[3] - reference_x) - pair[4]
        )

        return best[0], best[1], best[2]

    def choose_cone_inner_single(
        self,
        fit_candidates,
        target_lane,
        image_center,
        reference_x=None,
        strict_outer=False,
    ):
        if target_lane not in ('left', 'right') or len(fit_candidates) == 0:
            return None, None

        outer_pair = self.choose_cone_outer_pair(
            fit_candidates,
            target_lane,
            image_center,
            reference_x,
            strict_outer
        )
        if outer_pair is not None:
            left, right, _ = outer_pair
            if target_lane == 'right':
                return left, 'left'
            return right, 'right'

        ordered = sorted(fit_candidates, key=lambda item: item['x_bottom'])
        if target_lane == 'right':
            single = ordered[-2] if len(ordered) >= 2 else ordered[-1]
            return single, 'left'

        single = ordered[1] if len(ordered) >= 2 else ordered[0]
        return single, 'right'

    def choose_best_single(self, left_candidates, right_candidates, image_center):
        """
        한쪽 차선만 보이는 경우:
        이전에 보던 쪽 차선이 있으면 그쪽 후보를 우선 사용한다.
        없으면 이미지 중심에 가까운 후보를 사용한다.
        """
        if self.last_single_side == 'left' and len(left_candidates) > 0:
            source = left_candidates
        elif self.last_single_side == 'right' and len(right_candidates) > 0:
            source = right_candidates
        else:
            source = left_candidates + right_candidates

        if len(source) == 0:
            return None

        best = min(
            source,
            key=lambda item: abs(item['x_bottom'] - image_center) - 0.002 * item['count']
        )

        return best

    def infer_single_side(self, fit, width, height):
        x_bottom = poly_x(fit, height - 1)
        center_x = width // 2

        if self.last_single_side is not None:
            if abs(x_bottom - center_x) < SINGLE_SIDE_DEADBAND_PX:
                return self.last_single_side

        if x_bottom < center_x:
            return 'left'
        else:
            return 'right'

    def update_lane_width(self, lane_width):
        if LANE_WIDTH_UPDATE_MIN_PX <= lane_width <= LANE_WIDTH_UPDATE_MAX_PX:
            self.last_lane_width = int(
                0.92 * self.last_lane_width + 0.08 * lane_width
            )

    def get_lane_by_sliding_window(self, mask):
        h, w = mask.shape
        image_center = w // 2

        binary = (mask > 0).astype(np.uint8)

        binary[:int(h * TOP_IGNORE_RATIO), :] = 0

        total_pixels = int(np.count_nonzero(binary))

        if total_pixels < MIN_WHITE_PIXELS:
            return None

        raw_candidates = self.extract_multi_band_candidates(binary, image_center)
        merged_candidates = self.merge_tiny_peak_regions(raw_candidates, image_center)
        filtered_candidates = self.suppress_close_outer_lines(merged_candidates, image_center)

        left_base_candidates, right_base_candidates = self.split_candidates_by_side(
            filtered_candidates,
            image_center
        )

        left_fit_candidates = self.build_fit_candidates(
            binary,
            left_base_candidates,
            h
        )

        right_fit_candidates = self.build_fit_candidates(
            binary,
            right_base_candidates,
            h
        )
        all_fit_candidates = left_fit_candidates + right_fit_candidates

        center_fit = None
        left_fit = None
        right_fit = None
        confidence = 0.0
        mode = 'none'
        side = None
        measured_lane_width = None
        cone_pair_target = None
        cone_pair_reference_x = image_center
        cone_lidar_active = False

        if self.prev_center_fit is not None:
            prev_x = poly_x(self.prev_center_fit, h - 1)
            if -w * 0.25 <= prev_x <= w * 1.25:
                cone_pair_reference_x = prev_x

        if (
            not self.cone_simple_force_enabled and
            self.state == Mission.CONE and
            self.cone_latched and
            self.cone_target_lane in ('left', 'right')
        ):
            cone_lidar_active, _, _, _ = self.cone_lidar_side_obstacle()

        if (
            not self.cone_simple_force_enabled and
            self.state == Mission.CONE and
            self.cone_latched and
            self.cone_recover_start_time is not None and
            not cone_lidar_active and
            self.cone_target_lane in ('left', 'right')
        ):
            single, single_side = self.choose_cone_inner_single(
                all_fit_candidates,
                self.cone_target_lane,
                image_center,
                cone_pair_reference_x,
                strict_outer=False
            )

            if single is not None:
                single_fit = single['fit']
                center_fit = np.array(single_fit, dtype=np.float64)
                side = single_side
                self.last_single_side = side

                if side == 'left':
                    center_fit[2] += self.last_lane_width / 2.0
                    mode = f'cone_recover_inner_{self.cone_target_lane}'
                    left_fit = single_fit
                else:
                    center_fit[2] -= self.last_lane_width / 2.0
                    mode = f'cone_recover_inner_{self.cone_target_lane}'
                    right_fit = single_fit

                confidence = 0.82

        best_pair = None
        if (
            center_fit is None and
            not self.cone_simple_force_enabled and
            self.state == Mission.CONE and
            self.cone_latched and
            self.cone_target_lane in ('left', 'right')
        ):
            cone_force_outer = self.cone_recover_start_time is None or cone_lidar_active
            best_pair = self.choose_cone_outer_pair(
                all_fit_candidates,
                self.cone_target_lane,
                image_center,
                cone_pair_reference_x,
                strict_outer=cone_force_outer
            )
            if best_pair is not None:
                cone_pair_target = self.cone_target_lane

        if (
            center_fit is None and
            best_pair is None and
            not self.cone_simple_force_enabled and
            self.state == Mission.CONE and
            self.cone_latched and
            self.cone_target_lane in ('left', 'right')
        ):
            single, single_side = self.choose_cone_inner_single(
                all_fit_candidates,
                self.cone_target_lane,
                image_center,
                cone_pair_reference_x,
                strict_outer=True
            )

            if single is not None:
                single_fit = single['fit']
                center_fit = np.array(single_fit, dtype=np.float64)
                side = single_side
                self.last_single_side = side

                if side == 'left':
                    center_fit[2] += self.last_lane_width / 2.0
                    left_fit = single_fit
                else:
                    center_fit[2] -= self.last_lane_width / 2.0
                    right_fit = single_fit

                mode = f'cone_force_single_{self.cone_target_lane}'
                confidence = 0.76

        if center_fit is None and best_pair is None:
            best_pair = self.choose_best_pair(
                left_fit_candidates,
                right_fit_candidates,
                image_center
            )

        if center_fit is None and best_pair is not None:
            left, right, lane_width = best_pair

            left_fit = left['fit']
            right_fit = right['fit']
            measured_lane_width = lane_width

            self.update_lane_width(lane_width)

            avg_center_fit = (left_fit + right_fit) / 2.0

            y_bottom = h - 1
            y_ahead = int(h * LOOKAHEAD_Y_RATIO)

            left_bottom = poly_x(left_fit, y_bottom)
            right_bottom = poly_x(right_fit, y_bottom)

            left_ahead = poly_x(left_fit, y_ahead)
            right_ahead = poly_x(right_fit, y_ahead)

            width_bottom = right_bottom - left_bottom
            width_ahead = right_ahead - left_ahead
            width_diff = abs(width_bottom - width_ahead)

            curve_heading = self.estimate_fit_heading(avg_center_fit, h)

            use_outer_lane = (
                cone_pair_target is None and
                USE_OUTER_LANE_IN_BOTH_CURVE and
                (
                    abs(curve_heading) > BOTH_CURVE_HEADING_TH or
                    width_diff > BOTH_WIDTH_DIFF_TH_PX
                )
            )

            if cone_pair_target is not None:
                center_fit = avg_center_fit
                confidence = 1.0
                mode = f'cone_pair_{cone_pair_target}'
            elif use_outer_lane:
                # heading < 0이면 오른쪽 커브로 판단
                # 오른쪽 커브에서는 오른쪽 차선이 안쪽 차선일 가능성이 크므로
                # 왼쪽 차선을 기준으로 중앙선을 만든다.
                if curve_heading < 0.0:
                    center_fit = np.array(left_fit, dtype=np.float64)
                    center_fit[2] += self.last_lane_width / 2.0
                    mode = 'both_curve_outer_left'

                # heading > 0이면 왼쪽 커브로 판단
                # 왼쪽 커브에서는 왼쪽 차선이 안쪽 차선일 가능성이 크므로
                # 오른쪽 차선을 기준으로 중앙선을 만든다.
                else:
                    center_fit = np.array(right_fit, dtype=np.float64)
                    center_fit[2] -= self.last_lane_width / 2.0
                    mode = 'both_curve_outer_right'

                confidence = 0.90
            else:
                center_fit = avg_center_fit
                confidence = 1.0
                mode = 'both'

            side = 'both'

        if center_fit is None:
            single = self.choose_best_single(
                left_fit_candidates,
                right_fit_candidates,
                image_center
            )

            if single is not None:
                single_fit = single['fit']

                side = self.infer_single_side(single_fit, w, h)
                self.last_single_side = side

                center_fit = np.array(single_fit, dtype=np.float64)

                if side == 'left':
                    center_fit[2] += self.last_lane_width / 2.0
                    mode = 'single_left'
                    left_fit = single_fit
                else:
                    center_fit[2] -= self.last_lane_width / 2.0
                    mode = 'single_right'
                    right_fit = single_fit

                confidence = 0.68

        if center_fit is None:
            return None

        bottom_x = poly_x(center_fit, h - 1)
        ahead_y = int(h * LOOKAHEAD_Y_RATIO)
        ahead_x = poly_x(center_fit, ahead_y)

        if bottom_x < -w * 0.25 or bottom_x > w * 1.25:
            confidence *= 0.45

        if ahead_x < -w * 0.35 or ahead_x > w * 1.35:
            confidence *= 0.55

        return {
            'center_fit': center_fit,
            'left_fit': left_fit,
            'right_fit': right_fit,
            'confidence': confidence,
            'mode': mode,
            'side': side,
            'total_pixels': total_pixels,
            'measured_lane_width': measured_lane_width,
            'lane_width_calibrated': self.lane_width_calibrated,
            'lane_width_sample_count': len(self.lane_width_samples),
            'raw_candidates': raw_candidates,
            'filtered_candidates': filtered_candidates,
            'left_candidates': left_base_candidates,
            'right_candidates': right_base_candidates,
        }

    # =========================
    # Ackermann conversion
    # =========================
    def steering_angle_to_yaw_rate(self, speed, steering_angle):
        if abs(speed) < 1e-4 or abs(steering_angle) < 1e-4:
            return 0.0

        yaw_rate_cmd = speed * math.tan(steering_angle) / WHEELBASE_M

        yaw_rate_cmd = clamp(
            yaw_rate_cmd,
            -MAX_YAW_RATE,
            MAX_YAW_RATE
        )

        return STEER_SIGN * yaw_rate_cmd

    # =========================
    # Stanley control
    # =========================
    def compute_stanley_control(self, center_fit, confidence, img_shape):
        h, w = img_shape
        image_center = w // 2
        desired_center = image_center + self.current_lane_bias_px

        y_bottom = h - 1
        y_ahead = int(h * LOOKAHEAD_Y_RATIO)

        x_bottom = poly_x(center_fit, y_bottom)
        x_ahead = poly_x(center_fit, y_ahead)

        lane_width_px = max(float(self.last_lane_width), 1.0)
        xm_per_pix = LANE_WIDTH_M / lane_width_px
        ym_per_pix = BEV_FORWARD_M / float(h)

        cte_px = desired_center - x_bottom
        cte_m = cte_px * xm_per_pix
        cte_norm = cte_px / float(image_center)

        dx_m = (x_bottom - x_ahead) * xm_per_pix
        dy_m = (y_bottom - y_ahead) * ym_per_pix
        heading_error = math.atan2(dx_m, max(dy_m, 1e-6))

        current_speed = max(self.prev_speed, 0.10)

        stanley_term = math.atan2(
            K_STANLEY * cte_m,
            current_speed + STANLEY_SOFTENING
        )

        d_cte = cte_norm - self.prev_cte_norm
        self.prev_cte_norm = cte_norm

        raw_steering_angle = (
            K_HEADING * heading_error +
            stanley_term +
            K_DAMPING * d_cte
        )

        raw_steering_angle = clamp(
            raw_steering_angle,
            -MAX_STEERING_ANGLE,
            MAX_STEERING_ANGLE
        )

        steer_delta = raw_steering_angle - self.prev_steer
        steer_delta = clamp(
            steer_delta,
            -STEER_RATE_LIMIT,
            STEER_RATE_LIMIT
        )

        rate_limited_steering_angle = self.prev_steer + steer_delta

        steering_angle = (
            STEER_SMOOTH_ALPHA * rate_limited_steering_angle +
            (1.0 - STEER_SMOOTH_ALPHA) * self.prev_steer
        )

        steering_angle = clamp(
            steering_angle,
            -MAX_STEERING_ANGLE,
            MAX_STEERING_ANGLE
        )

        self.prev_steer = steering_angle

        steer_ratio = min(abs(steering_angle) / MAX_STEERING_ANGLE, 1.0)
        heading_ratio = min(abs(heading_error) / 0.65, 1.0)

        curve_ratio = max(0.65 * steer_ratio, 0.35 * heading_ratio)

        target_speed = STRAIGHT_SPEED - (STRAIGHT_SPEED - CURVE_MIN_SPEED) * curve_ratio
        target_speed = clamp(target_speed, CURVE_MIN_SPEED, MAX_SPEED)

        if confidence < 0.75:
            target_speed = min(target_speed, 0.55)

        if confidence < 0.45:
            target_speed = min(target_speed, BAD_CONF_SPEED)

        if AUTO_CALIBRATE_LANE_WIDTH and not self.lane_width_calibrated:
            target_speed = min(target_speed, CALIBRATION_SPEED_LIMIT)

        speed = self.smooth_speed(target_speed)

        yaw_rate_cmd = self.steering_angle_to_yaw_rate(
            speed,
            steering_angle
        )

        return speed, yaw_rate_cmd, {
            'cte_px': cte_px,
            'cte_m': cte_m,
            'heading_error': heading_error,
            'stanley_term': stanley_term,
            'x_bottom': x_bottom,
            'x_ahead': x_ahead,
            'target_speed': target_speed,
            'lane_width_px': lane_width_px,
            'xm_per_pix': xm_per_pix,
            'raw_steer': raw_steering_angle,
            'steering_angle': steering_angle,
            'yaw_rate_cmd': yaw_rate_cmd,
        }

    def smooth_speed(self, target_speed):
        if target_speed > self.prev_speed:
            speed = min(self.prev_speed + MAX_ACCEL_STEP, target_speed)
        else:
            speed = max(self.prev_speed - MAX_DECEL_STEP, target_speed)

        self.prev_speed = speed
        return speed

    # =========================
    # Debug draw
    # =========================
    def draw_polyline(self, img, fit, color, thickness=3):
        if fit is None:
            return

        h, w = img.shape[:2]
        y_values = np.linspace(int(h * 0.35), h - 1, 60)
        points = []

        for y in y_values:
            x = poly_x(fit, y)
            if -w <= x <= 2 * w:
                points.append([int(x), int(y)])

        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(img, [pts], False, color, thickness)

    def draw_candidates(self, img, candidates, color, radius=5):
        h = img.shape[0]

        for cx, strength in candidates:
            cv2.circle(img, (int(cx), h - 20), radius, color, -1)

    def publish_debug(self, warp_img, mask, lane_data, control_info, avg_brightness, white_pixels):
        debug = warp_img.copy()
        h, w = mask.shape
        image_center = w // 2
        desired_center = image_center + self.current_lane_bias_px

        colored_mask = np.zeros_like(debug)
        colored_mask[:, :, 1] = mask
        debug = cv2.addWeighted(debug, 0.75, colored_mask, 0.25, 0)

        cv2.line(debug, (image_center, 0), (image_center, h), (255, 0, 0), 2)
        cv2.line(debug, (desired_center, 0), (desired_center, h), (0, 255, 255), 2)

        y_ahead = int(h * LOOKAHEAD_Y_RATIO)
        cv2.line(debug, (0, y_ahead), (w, y_ahead), (0, 255, 255), 1)

        if lane_data is not None:
            self.draw_polyline(debug, lane_data.get('left_fit'), (255, 0, 0), 2)
            self.draw_polyline(debug, lane_data.get('right_fit'), (0, 255, 0), 2)
            self.draw_polyline(debug, lane_data.get('center_fit'), (0, 0, 255), 4)

            # 원본 후보: 회색
            self.draw_candidates(
                debug,
                lane_data.get('raw_candidates', []),
                (120, 120, 120),
                radius=4
            )

            # 필터 후 후보: 노란색
            self.draw_candidates(
                debug,
                lane_data.get('filtered_candidates', []),
                (0, 255, 255),
                radius=7
            )

            measured_lane_width = lane_data.get('measured_lane_width')

            calib_state = 'CALIB_ON' if AUTO_CALIBRATE_LANE_WIDTH else 'CALIB_OFF'

            if control_info is not None:
                xb = int(control_info['x_bottom'])
                xa = int(control_info['x_ahead'])

                cv2.circle(debug, (xb, h - 1), 8, (0, 0, 255), -1)
                cv2.circle(debug, (xa, y_ahead), 8, (0, 255, 255), -1)
                cv2.line(debug, (xb, h - 1), (xa, y_ahead), (0, 0, 255), 2)

                text1 = (
                    f"mode={lane_data['mode']} conf={lane_data['confidence']:.2f} "
                    f"lane_w={self.last_lane_width}px {calib_state}"
                )

                text2 = (
                    f"cte={control_info['cte_px']:.1f}px "
                    f"head={control_info['heading_error']:.2f} "
                    f"angle={control_info['steering_angle']:.2f} "
                    f"wz={control_info['yaw_rate_cmd']:.2f}"
                )

                if measured_lane_width is not None:
                    text3 = (
                        f"measured_w={measured_lane_width:.1f}px "
                        f"target_v={control_info['target_speed']:.2f}"
                    )
                else:
                    text3 = (
                        f"measured_w=None "
                        f"target_v={control_info['target_speed']:.2f}"
                    )

                cv2.putText(debug, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(debug, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(debug, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(
            debug,
            f"brightness={avg_brightness:.1f} white={white_pixels}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(debug, dst_format='jpg')
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'debug publish error: {e}')

        if SHOW_DEBUG:
            try:
                cv2.imshow('fast_stanley_debug', debug)
                cv2.imshow('white_mask', mask)
                cv2.waitKey(1)
            except Exception:
                pass


    # ========================================================
    # Mission state helpers
    # ========================================================
    def state_elapsed(self):
        return time.monotonic() - self.state_enter_time

    def reset_state_local_vars(self, new_state):
        self.current_lane_bias_px = 0.0

        self.ped_stop_line_was_seen = False
        self.ped_waiting_after_line = False
        self.ped_stop_line_ratio = 0.0
        self.ped_stop_line_peak_ratio = 0.0
        self.pedestrian_was_seen = False
        self.last_ped_clear_time = None
        self.ped_db.reset()
        self.ped_line_db.reset()

        self.box_avoid_started = False
        self.box_clear_start = None
        self.box_seen_time = None
        self.last_box_side = 0.0
        self.box_track_candidate = None
        self.box_db.reset()

        self.tunnel_was_seen = False
        self.tunnel_recover_count = 0
        self.tunnel_candidate_count = 0
        self.tunnel_seen_time = None
        self.tunnel_exit_start_time = None

        self.rotary_clear_start = None
        self.rotary_vehicle_distance = 9.9
        self.rotary_follow_status = 'WAIT_VEHICLE'
        self.rotary_db.reset()
        if new_state == Mission.ROTARY:
            self.rotary_vehicle_seen = False
            self.reset_simple_cone_vars()
            # 핵심: ROTARY에 들어오는 순간 이전 구간에서 잘못 들어온 고깔 latch를 모두 지운다.
            # 고깔 정보는 CONE 상태에서만 저장한다.
            self.cone_latched = False
            self.cone_latched_lanes = set()
            self.cone_target_lane = 'center'
            self.cone_target_locked = False
            self.cone_target_votes = []
            self.cone_first_latch_time = None
            self.last_cone_msg_time = 0.0
            self.cone_center_only_start_time = None
            self.cone_latch_frozen = False
            self.cone_recover_start_time = None
            self.cone_lidar_clear_count = 0

        # CONE에 들어오는 순간부터 고깔 정보를 새로 받는다.
        # ROTARY에서 너무 일찍 들어온 오판단을 보존하지 않는다.
        if new_state == Mission.CONE:
            self.cone_latched = False
            self.cone_latched_lanes = set()
            self.cone_target_lane = 'center'
            self.cone_target_locked = False
            self.cone_target_votes = []
            self.cone_first_latch_time = None
            self.last_cone_msg_time = 0.0
            self.cone_center_only_start_time = None
            self.cone_latch_frozen = False
            self.cone_recover_start_time = None
            self.cone_lidar_clear_count = 0
            self.reset_simple_cone_vars()

        self.parking_started = False
        self.parking_step_index = 0
        self.parking_step_start = 0.0

    def set_state(self, new_state, reason=''):
        if new_state not in MISSION_ORDER_ALL:
            self.get_logger().warn(f'unknown mission state: {new_state}')
            return

        if (not self.use_traffic_light) and new_state == Mission.WAIT_TRAFFIC:
            new_state = Mission.PEDESTRIAN

        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state
        self.state_enter_time = time.monotonic()
        self.reset_state_local_vars(new_state)

        self.get_logger().warn(f'MISSION {old_state} -> {new_state} {reason}')

    def next_state(self, reason=''):
        try:
            idx = self.mission_order.index(self.state)
            next_idx = min(idx + 1, len(self.mission_order) - 1)
            self.set_state(self.mission_order[next_idx], reason)
        except ValueError:
            self.set_state(self.mission_order[0], reason)

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
        elif cmd == 'RESET_CONE':
            self.cone_latched = False
            self.cone_latched_lanes = set()
            self.cone_target_lane = 'center'
            self.cone_target_locked = False
            self.cone_target_votes = []
            self.cone_first_latch_time = None
            self.last_cone_msg_time = 0.0
            self.cone_center_only_start_time = None
            self.cone_latch_frozen = False
            self.cone_recover_start_time = None
            self.cone_lidar_clear_count = 0
            self.reset_simple_cone_vars()
            self.get_logger().warn('cone latch reset by manual command')
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
        front_points = []

        # 보행자/박스 판단은 사용자가 요청한 것처럼 각도 기반 ROI를 메인으로 사용한다.
        ped_front_hit = 0
        ped_front_min = 9.9

        box_front_hit = 0
        box_front_min = 9.9
        box_left_hit = 0
        box_left_min = 9.9
        box_right_hit = 0
        box_right_min = 9.9

        for i, dist in enumerate(msg.ranges):
            if not valid_range(dist):
                continue
            if dist < 0.05 or dist > 2.3:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            while angle_deg > 180.0:
                angle_deg -= 360.0
            while angle_deg < -180.0:
                angle_deg += 360.0

            x = dist * math.cos(angle)
            y = dist * math.sin(angle)

            # 회전교차로 차량 추종용 클러스터는 기존처럼 전방 영역을 유지한다.
            if -0.20 <= x <= 1.70 and abs(y) <= 0.95:
                points.append((x, y, dist, angle))
                if x > 0.0:
                    front_points.append((x, y, dist, angle))

            # 보행자: 전방 ±20도, 0.50m 이내만 정지 판단.
            if -self.ped_front_angle_deg <= angle_deg <= self.ped_front_angle_deg:
                if dist <= self.ped_front_distance:
                    ped_front_hit += 1
                    ped_front_min = min(ped_front_min, dist)

            # 박스: ±20도 안쪽은 회피 판단이 아니라 충돌 안전 정지용.
            if -self.box_front_angle_deg <= angle_deg <= self.box_front_angle_deg:
                if dist <= self.box_front_stop_distance:
                    box_front_hit += 1
                    box_front_min = min(box_front_min, dist)

            # 박스 회피 메인: ±20도 바깥의 좌/우 측면 감지.
            elif self.box_side_angle_min_deg < angle_deg <= self.box_side_angle_max_deg:
                if dist <= self.box_side_distance:
                    box_left_hit += 1
                    box_left_min = min(box_left_min, dist)

            elif -self.box_side_angle_max_deg <= angle_deg < -self.box_side_angle_min_deg:
                if dist <= self.box_side_distance:
                    box_right_hit += 1
                    box_right_min = min(box_right_min, dist)

        self.lidar_points = points
        self.lidar_clusters = self.make_clusters(front_points)
        self.last_scan_time = time.monotonic()

        self.ped_front_hit, self.ped_front_min_dist = self.pedestrian_front_cluster_hits()
        self.ped_front_obs = self.ped_front_hit >= self.ped_front_min_hits

        self.box_front_hit = box_front_hit
        self.box_front_min_dist = box_front_min
        self.box_front_obs = box_front_hit >= self.box_front_min_hits

        self.box_left_hit = box_left_hit
        self.box_left_obs_dist = box_left_min
        self.box_left_obs = box_left_hit >= self.box_side_min_hits

        self.box_right_hit = box_right_hit
        self.box_right_obs_dist = box_right_min
        self.box_right_obs = box_right_hit >= self.box_side_min_hits

    def make_clusters(self, scan_points):
        if len(scan_points) == 0:
            return []

        scan_points.sort(key=lambda p: p[3])
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
        if cluster.width > self.rotary_wall_width_threshold:
            return True
        if (cluster.max_x - cluster.min_x) > 0.75:
            return True
        return False

    def pedestrian_front_cluster_hits(self):
        hit_count = 0
        min_dist = 9.9

        for cluster in self.lidar_clusters:
            if cluster.nearest > self.ped_front_distance:
                continue
            if cluster.width > self.ped_front_wall_width_threshold:
                continue

            cluster_hits = 0
            cluster_min = 9.9

            for _, _, dist, angle in cluster.points:
                if dist > self.ped_front_distance:
                    continue

                angle_deg = math.degrees(angle)
                while angle_deg > 180.0:
                    angle_deg -= 360.0
                while angle_deg < -180.0:
                    angle_deg += 360.0

                if -self.ped_front_angle_deg <= angle_deg <= self.ped_front_angle_deg:
                    cluster_hits += 1
                    cluster_min = min(cluster_min, dist)

            if cluster_hits > 0:
                hit_count += cluster_hits
                min_dist = min(min_dist, cluster_min)

        return hit_count, min_dist

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

    def obstacle_in_pedestrian_roi(self):
        # 사용자가 요구한 좁은 보행자 기준:
        # LiDAR 전방 ±20도, 0.50m 이내 점이 일정 개수 이상 있을 때만 보행자 후보로 인정한다.
        if self.ped_front_obs:
            return {
                'type': 'front',
                'nearest': self.ped_front_min_dist,
                'hits': self.ped_front_hit,
            }
        return None

    def cluster_has_l_shape(self, cluster):
        """
        박스 L-shape 판정.
        라이다에서 박스 모서리가 보이면 보통 전면 + 측면 두 방향의 점들이 함께 찍힌다.
        그래서 아래 조건을 모두 만족할 때 L-shape 후보로 본다.

        1) 10cm 이내의 매우 가까운 점이 있어야 함.
        2) 전체 점 개수가 충분해야 함.
        3) x 방향, y 방향 span이 모두 일정 이상이어야 함.
        4) min_x 근처 전면 leg와 min_y 또는 max_y 근처 측면 leg가 각각 충분히 있어야 함.
        """
        if cluster is None:
            return False, {}

        x_span = cluster.max_x - cluster.min_x
        y_span = cluster.max_y - cluster.min_y

        near_points = [p for p in cluster.points if p[2] <= self.box_lshape_near_distance]
        if len(near_points) == 0:
            return False, {
                'reason': 'near_fail',
                'near_points': 0,
                'x_span': x_span,
                'y_span': y_span,
            }

        if cluster.count < self.box_lshape_min_points:
            return False, {
                'reason': 'count_fail',
                'near_points': len(near_points),
                'x_span': x_span,
                'y_span': y_span,
            }

        if x_span < self.box_lshape_min_x_span or y_span < self.box_lshape_min_y_span:
            return False, {
                'reason': 'span_fail',
                'near_points': len(near_points),
                'x_span': x_span,
                'y_span': y_span,
            }

        band = self.box_lshape_corner_band

        # 전면 leg: 차량에서 가장 가까운 x면 근처의 점들
        front_leg_count = 0
        # 좌/우 측면 leg: y 최소/최대 경계 근처 점들 중 더 강한 쪽
        side_min_count = 0
        side_max_count = 0

        for x, y, _, _ in cluster.points:
            if x <= cluster.min_x + band:
                front_leg_count += 1
            if y <= cluster.min_y + band:
                side_min_count += 1
            if y >= cluster.max_y - band:
                side_max_count += 1

        side_leg_count = max(side_min_count, side_max_count)

        l_shape = (
            front_leg_count >= self.box_lshape_min_leg_points and
            side_leg_count >= self.box_lshape_min_leg_points
        )

        info = {
            'reason': 'ok' if l_shape else 'leg_fail',
            'near_points': len(near_points),
            'x_span': x_span,
            'y_span': y_span,
            'front_leg_count': front_leg_count,
            'side_leg_count': side_leg_count,
            'side_min_count': side_min_count,
            'side_max_count': side_max_count,
        }

        return l_shape, info

    def box1_close_side_obstacle(self):
        best = None

        for cluster in self.lidar_clusters:
            if cluster.count < self.box1_close_side_min_hits:
                continue
            if cluster.nearest > self.box1_close_side_distance:
                continue
            if cluster.width > self.box1_close_wall_width_threshold:
                continue
            if (cluster.max_x - cluster.min_x) > self.box1_close_wall_x_span:
                continue

            left_hits = 0
            right_hits = 0
            left_min = 9.9
            right_min = 9.9

            for _, y, dist, angle in cluster.points:
                if dist > self.box1_close_side_distance:
                    continue

                angle_deg = math.degrees(angle)
                while angle_deg > 180.0:
                    angle_deg -= 360.0
                while angle_deg < -180.0:
                    angle_deg += 360.0

                if not (self.box1_angle_min_deg <= angle_deg <= self.box1_angle_max_deg):
                    continue

                right_hits += 1
                right_min = min(right_min, dist)

            if left_hits < self.box1_close_side_min_hits and right_hits < self.box1_close_side_min_hits:
                continue

            if left_hits >= self.box1_close_side_min_hits and right_hits >= self.box1_close_side_min_hits:
                sector = 'left' if left_min <= right_min else 'right'
                nearest = min(left_min, right_min)
                hits = max(left_hits, right_hits)
            elif left_hits >= self.box1_close_side_min_hits:
                sector = 'left'
                nearest = left_min
                hits = left_hits
            else:
                sector = 'right'
                nearest = right_min
                hits = right_hits

            candidate = {
                'sector': sector,
                'nearest': nearest,
                'hits': hits,
                'cluster': cluster,
                'cx': cluster.cx,
                'cy': cluster.cy,
                'width': cluster.width,
                'path_y': self.path_center_y_m,
                'l_shape': {'reason': 'box1_close_side'},
            }

            if best is None or candidate['nearest'] < best['nearest']:
                best = candidate

        return best

    def box_cluster_obstacle(self, box_profile='box1'):
        """
        박스 판단 기준(v6):
        1) 현재 주행 중인 트랙 corridor에 걸친 LiDAR 클러스터여야 한다.
        2) 설정한 LiDAR 각도 영역과 겹쳐야 한다.
        3) 10cm 이내의 가까운 점이 있어야 한다.
        4) L-shape 특징이 있어야 한다.
        5) 이 L-shape 후보가 사라져야 박스 회피 완료로 판단한다.
        """
        best = None
        path_y = self.path_center_y_m
        is_box2 = box_profile == 'box2'

        if not is_box2:
            close_candidate = self.box1_close_side_obstacle()
            if close_candidate is not None:
                self.box_track_candidate = close_candidate
                return close_candidate

        track_half_width = self.box2_track_half_width if is_box2 else self.box_track_half_width
        front_angle_deg = self.box2_front_angle_deg if is_box2 else self.box_front_angle_deg
        front_min_hits = self.box2_front_min_hits if is_box2 else self.box_front_min_hits
        front_stop_distance = self.box_front_stop_distance if is_box2 else self.box1_front_stop_distance
        side_min_hits = self.box2_side_min_hits if is_box2 else self.box_side_min_hits

        for cluster in self.lidar_clusters:
            if cluster.count < self.box_track_min_points:
                continue
            if cluster.width > self.box_wall_width_threshold:
                continue
            if (cluster.max_x - cluster.min_x) > 0.70:
                continue
            if not (self.box_track_x_min <= cluster.cx <= self.box_track_x_max):
                continue

            # 현재 주행 경로 corridor와 실제로 겹치는지 확인한다.
            corridor_min = path_y - track_half_width
            corridor_max = path_y + track_half_width
            overlaps_corridor = not (cluster.max_y < corridor_min or cluster.min_y > corridor_max)
            center_in_corridor = abs(cluster.cy - path_y) <= track_half_width
            if not (overlaps_corridor or center_in_corridor):
                continue

            left_hits = 0
            right_hits = 0
            front_hits = 0
            left_min = 9.9
            right_min = 9.9
            front_min = 9.9

            for _, _, dist, angle in cluster.points:
                angle_deg = math.degrees(angle)
                while angle_deg > 180.0:
                    angle_deg -= 360.0
                while angle_deg < -180.0:
                    angle_deg += 360.0

                if not is_box2 and not (self.box1_angle_min_deg <= angle_deg <= self.box1_angle_max_deg):
                    continue

                if -front_angle_deg <= angle_deg <= front_angle_deg:
                    if dist <= front_stop_distance:
                        front_hits += 1
                        front_min = min(front_min, dist)
                elif self.box_side_angle_min_deg < angle_deg <= self.box_side_angle_max_deg:
                    if dist <= self.box_side_distance:
                        left_hits += 1
                        left_min = min(left_min, dist)
                elif -self.box_side_angle_max_deg <= angle_deg < -self.box_side_angle_min_deg:
                    if dist <= self.box_side_distance:
                        right_hits += 1
                        right_min = min(right_min, dist)

            l_shape, l_info = self.cluster_has_l_shape(cluster)
            if not l_shape:
                if (
                    is_box2 and
                    self.box2_left_fallback_enabled and
                    cluster.cy >= path_y and
                    left_hits >= side_min_hits
                ):
                    candidate = {
                        'sector': 'left',
                        'nearest': left_min,
                        'hits': left_hits,
                        'cluster': cluster,
                        'cx': cluster.cx,
                        'cy': cluster.cy,
                        'width': cluster.width,
                        'path_y': path_y,
                        'l_shape': {'reason': 'box2_left_side_fallback'},
                    }

                    if best is None or candidate['nearest'] < best['nearest']:
                        best = candidate

                continue

            sector = None
            nearest = cluster.nearest
            hits = 0

            # 좌/우 각도 조건이 있으면 회피 방향 판단에 우선 사용한다.
            if left_hits >= side_min_hits or right_hits >= side_min_hits:
                if left_hits >= side_min_hits and right_hits >= side_min_hits:
                    sector = 'left' if left_min <= right_min else 'right'
                    nearest = min(left_min, right_min)
                    hits = max(left_hits, right_hits)
                elif left_hits >= side_min_hits:
                    sector = 'left'
                    nearest = left_min
                    hits = left_hits
                else:
                    sector = 'right'
                    nearest = right_min
                    hits = right_hits
            elif front_hits >= front_min_hits:
                # 전방에만 걸린 경우도 L-shape이면 박스 후보로 인정한다.
                # 방향은 cluster가 현재 경로 중심보다 어느 쪽에 치우쳤는지로 정한다.
                if not is_box2:
                    sector = 'right'
                elif cluster.cy >= path_y:
                    sector = 'left'
                else:
                    sector = 'right'
                nearest = front_min
                hits = front_hits
            else:
                continue

            candidate = {
                'sector': sector,
                'nearest': nearest,
                'hits': hits,
                'cluster': cluster,
                'cx': cluster.cx,
                'cy': cluster.cy,
                'width': cluster.width,
                'path_y': path_y,
                'l_shape': l_info,
            }

            if best is None or candidate['nearest'] < best['nearest']:
                best = candidate

        self.box_track_candidate = best
        return best

    def box_sector_obstacle(self, box_profile='box1'):
        # v5부터는 단순 각도 sector만으로 박스를 판단하지 않는다.
        # 현재 주행 트랙 corridor에 걸친 클러스터 + 각도 조건을 동시에 만족해야 한다.
        return self.box_cluster_obstacle(box_profile)
    def obstacle_in_box_roi(self, box_profile='box1'):
        return self.box_sector_obstacle(box_profile)

    def rotary_vehicle_candidate(self):
        # 사용자가 보낸 TrackVehicleFollowNode의 find_lead_vehicle 구조를 반영한다.
        # 즉, 넓은 고정 y ROI가 아니라 현재 주행 경로 중심(path_center_y_m) 주변의
        # 전방 차량 후보만 선행 차량으로 인정한다.
        best = None

        for cluster in self.lidar_clusters:
            if cluster.count < self.rotary_cluster_min_points:
                continue
            if self.is_wall_like(cluster):
                continue
            if not (self.rotary_front_x_min <= cluster.cx <= self.rotary_front_x_max):
                continue
            cluster_angle_deg = math.degrees(math.atan2(cluster.cy, max(cluster.cx, 1e-6)))
            if not (-self.rotary_front_angle_deg <= cluster_angle_deg <= self.rotary_front_angle_deg):
                continue
            if abs(cluster.cy - self.path_center_y_m) > self.rotary_front_half_width:
                continue
            if not (0.04 <= cluster.width <= 0.58):
                continue

            if best is None or cluster.nearest < best.nearest:
                best = cluster

        return best

    def compute_rotary_speed_limit(self, cluster):
        if time.monotonic() - self.last_scan_time > 0.6:
            self.rotary_follow_status = 'NO_SCAN'
            return 0.0

        if cluster is None:
            self.rotary_follow_status = 'CLEAR'
            return self.rotary_clear_speed_limit

        d = cluster.nearest

        if d <= self.rotary_emergency_stop_distance:
            self.rotary_follow_status = 'LEAD_STOP'
            return 0.0

        if d <= self.rotary_follow_distance:
            ratio = (d - self.rotary_emergency_stop_distance) / max(
                self.rotary_follow_distance - self.rotary_emergency_stop_distance,
                1e-3,
            )
            self.rotary_follow_status = 'LEAD_FOLLOW'
            return clamp(
                self.rotary_min_moving_speed + ratio * (self.rotary_follow_speed_limit - self.rotary_min_moving_speed),
                self.rotary_min_moving_speed,
                self.rotary_follow_speed_limit,
            )

        if d <= self.rotary_slow_distance:
            ratio = (d - self.rotary_follow_distance) / max(
                self.rotary_slow_distance - self.rotary_follow_distance,
                1e-3,
            )
            self.rotary_follow_status = 'LEAD_SLOW'
            return clamp(
                self.rotary_follow_speed_limit + ratio * (self.rotary_slow_speed_limit - self.rotary_follow_speed_limit),
                self.rotary_follow_speed_limit,
                self.rotary_slow_speed_limit,
            )

        self.rotary_follow_status = 'LEAD_AHEAD'
        return self.rotary_clear_speed_limit

    # ========================================================
    # Traffic / finish / cone
    # ========================================================
    def detect_traffic_light(self, img):
        h, w = img.shape[:2]

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
        white_pixels = int(cv2.countNonZero(band))
        ratio = white_pixels / float(band.shape[0] * band.shape[1])
        detected = ratio > self.parking_finish_ratio
        return self.finish_db.update(detected), ratio

    def detect_pedestrian_stop_line(self, mask, lane_data):
        if mask is None:
            self.ped_stop_line_ratio = 0.0
            return False

        h, w = mask.shape
        y0 = int(h * self.ped_line_roi_y_min_ratio)
        y1 = int(h * self.ped_line_roi_y_max_ratio)
        y0 = clamp(y0, 0, h - 1)
        y1 = clamp(y1, y0 + 1, h)

        max_fill_ratio = 0.0
        max_run = 0
        run = 0

        for y in range(y0, y1):
            x0, x1 = self.pedestrian_stop_line_x_range(y, w, lane_data)
            width = x1 - x0

            if width < self.ped_line_min_width_px:
                run = 0
                continue

            row = mask[y, x0:x1]
            fill_ratio = cv2.countNonZero(row) / float(width)
            max_fill_ratio = max(max_fill_ratio, fill_ratio)

            if fill_ratio >= self.ped_line_fill_ratio:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0

        self.ped_stop_line_ratio = max_fill_ratio
        return max_run >= self.ped_line_min_rows

    def pedestrian_stop_line_x_range(self, y, image_width, lane_data):
        if lane_data is not None:
            left_fit = lane_data.get('left_fit')
            right_fit = lane_data.get('right_fit')

            if left_fit is not None and right_fit is not None:
                left_x = int(poly_x(left_fit, y))
                right_x = int(poly_x(right_fit, y))

                if right_x < left_x:
                    left_x, right_x = right_x, left_x

                margin = int(max(10, self.last_lane_width * 0.08))
                x0 = clamp(left_x + margin, 0, image_width - 1)
                x1 = clamp(right_x - margin, x0 + 1, image_width)
                return int(x0), int(x1)

            center_fit = lane_data.get('center_fit')
            if center_fit is not None:
                center_x = int(poly_x(center_fit, y))
            else:
                center_x = image_width // 2
        else:
            center_x = image_width // 2

        half_width = int(max(self.ped_line_min_width_px / 2.0, self.last_lane_width * 0.34))
        x0 = clamp(center_x - half_width, 0, image_width - 1)
        x1 = clamp(center_x + half_width, x0 + 1, image_width)
        return int(x0), int(x1)

    def cone_callback(self, msg):
        # CONE 상태에 들어오기 전까지 들어오는 고깔 정보는 전부 버린다.
        # ROTARY에서는 차량 헤딩이 틀어져 있거나 노란 기준선이 안정적으로 잡히기 전일 수 있으므로
        # 고깔 위치를 latch/freeze하지 않는다.
        if self.state not in self.cone_accept_states:
            return

        raw = msg.data.strip().lower()
        lanes = set()

        for token in raw.replace(';', ',').replace(' ', ',').split(','):
            token = token.strip()
            if token in ('left', 'center', 'right'):
                lanes.add(token)

        if len(lanes) == 0:
            return

        now = time.monotonic()

        # 테스트 코드에서 정상 동작했던 방식과 동일하게,
        # center+right -> target left, center+left -> target right를 직접 lock한다.
        # 한 번 target이 left/right로 lock되면 이후 차량 진동으로 들어오는 추가 검출은 무시한다.
        if self.cone_target_locked and self.cone_target_lane in ('left', 'right'):
            self.last_cone_msg_time = now
            self.get_logger().warn(
                f'cone direct latch ignore after lock: raw={raw}, '
                f'latched={sorted(list(self.cone_latched_lanes))}, '
                f'target={self.cone_target_lane}'
            )
            return

        before_target = self.cone_target_lane

        self.cone_latched = True
        self.last_cone_msg_time = now

        if self.cone_first_latch_time is None:
            self.cone_first_latch_time = now

        center_only_assumed = False

        # 핵심 1: center + right cone -> 왼쪽 회피
        if 'center' in lanes and 'right' in lanes and 'left' not in lanes:
            self.cone_latched_lanes = {'center', 'right'}
            self.cone_target_lane = 'left'
            self.cone_target_locked = True
            self.cone_target_votes = ['left']
            self.cone_latch_frozen = bool(self.cone_freeze_after_pair)
            self.cone_center_only_start_time = None

        # 핵심 2: center + left cone -> 오른쪽 회피
        elif 'center' in lanes and 'left' in lanes and 'right' not in lanes:
            self.cone_latched_lanes = {'center', 'left'}
            self.cone_target_lane = 'right'
            self.cone_target_locked = True
            self.cone_target_votes = ['right']
            self.cone_latch_frozen = bool(self.cone_freeze_after_pair)
            self.cone_center_only_start_time = None

        # YOLO 노드가 right만 보냈을 때도 center+right로 간주해서 왼쪽 회피
        elif 'right' in lanes and 'left' not in lanes:
            self.cone_latched_lanes = {'center', 'right'}
            self.cone_target_lane = 'left'
            self.cone_target_locked = True
            self.cone_target_votes = ['left']
            self.cone_latch_frozen = bool(self.cone_freeze_after_pair)
            self.cone_center_only_start_time = None

        # YOLO 노드가 left만 보냈을 때도 center+left로 간주해서 오른쪽 회피
        elif 'left' in lanes and 'right' not in lanes:
            self.cone_latched_lanes = {'center', 'left'}
            self.cone_target_lane = 'right'
            self.cone_target_locked = True
            self.cone_target_votes = ['right']
            self.cone_latch_frozen = bool(self.cone_freeze_after_pair)
            self.cone_center_only_start_time = None

        # center만 들어온 경우:
        # 기본값에서는 노란선 기반 center 판단을 사용하므로 left로 자동 가정하지 않는다.
        # 필요할 때만 cone_center_only_auto_assume_left:=true로 켜서 기존 보정 로직을 사용한다.
        elif lanes == {'center'}:
            self.cone_latched_lanes = {'center'}
            self.cone_latch_frozen = False

            if self.cone_center_only_auto_assume_left:
                if self.cone_center_only_start_time is None:
                    self.cone_center_only_start_time = now

                center_elapsed = now - self.cone_center_only_start_time

                if center_elapsed >= self.cone_center_only_assume_left_time:
                    self.cone_latched_lanes = {'center', 'left'}
                    self.cone_target_lane = 'right'
                    self.cone_target_locked = True
                    self.cone_target_votes = ['right']
                    self.cone_latch_frozen = bool(self.cone_freeze_after_pair)
                    self.cone_center_only_start_time = None
                    center_only_assumed = True

                    self.get_logger().warn(
                        'cone center-only timeout: assume center+left -> target=right'
                    )
                else:
                    self.cone_target_lane = 'center'
                    self.cone_target_locked = False
            else:
                self.cone_center_only_start_time = None
                self.cone_target_lane = 'center'
                self.cone_target_locked = False
                self.cone_target_votes = ['center']

        else:
            # left, center, right가 동시에 들어오면 모호하므로 target 확정 전에는 무시한다.
            self.get_logger().warn(
                f'cone direct latch ambiguous ignore: raw={raw}, '
                f'lanes={sorted(list(lanes))}, '
                f'latched={sorted(list(self.cone_latched_lanes))}, '
                f'target={self.cone_target_lane}'
            )
            return

        self.get_logger().warn(
            f'cone direct latch update: raw={raw}, '
            f'latched={sorted(list(self.cone_latched_lanes))}, '
            f'target={self.cone_target_lane}, '
            f'locked={self.cone_target_locked}, '
            f'frozen={self.cone_latch_frozen}, '
            f'center_only_assumed={center_only_assumed}, '
            f'votes={self.cone_target_votes}'
        )

        # 테스트 코드에서 정상 동작했던 핵심 구조:
        # target이 left/right로 확정되는 순간 simple force 상태를 즉시 reset/start한다.
        if self.simple_cone_avoid_active() and before_target != self.cone_target_lane:
            self.reset_simple_cone_vars()
            self.ensure_simple_cone_started()

    def cone_latched_has_center_side_pair(self):
        return (
            'center' in self.cone_latched_lanes and
            ('left' in self.cone_latched_lanes or 'right' in self.cone_latched_lanes)
        )

    def filter_cone_lanes_for_latch(self, lanes):
        """
        고깔 latch 안정화 필터.

        목적:
          - center+right 또는 center+left가 이미 확정되면 새로운 위치를 추가하지 않는다.
          - 아직 center만 있는 상태에서 left/right 둘 다 동시에 들어오면 진동에 의한 오검출로 보고
            side 추가를 보류한다.
          - center만 1초 이상 유지되는 경우는 cone_callback에서 center+left로 보정한다.
        """
        sides = {'left', 'right'}
        current_sides = self.cone_latched_lanes & sides
        raw_sides = lanes & sides

        if self.cone_latched_has_center_side_pair():
            return set()

        filtered = set()

        if 'center' in lanes:
            filtered.add('center')

        # 이미 side 하나가 latch되어 있다면, 반대편 side는 추가하지 않는다.
        # 이후 center가 들어오면 기존 side와 center로 pair를 완성한다.
        if len(current_sides) == 1:
            existing_side = next(iter(current_sides))
            if existing_side in lanes:
                filtered.add(existing_side)
            return filtered

        if len(current_sides) >= 2:
            return filtered

        # 아직 side가 없다.
        if len(raw_sides) == 1:
            filtered |= raw_sides
        elif len(raw_sides) >= 2:
            # left/right가 동시에 들어오면 모호하므로 side는 추가하지 않는다.
            pass

        return filtered

    def update_cone_center_only_timeout(self):
        """
        cone_callback이 한 번만 들어오고 이후 같은 center-only 메시지가 반복되지 않아도,
        CONE 상태에서 center-only가 1초 이상 유지되면 center+left로 보정한다.

        현재 기본 설정에서는 노란선 기반 center 판단을 사용하므로 이 자동 보정은 끈다.
        """
        if not self.cone_center_only_auto_assume_left:
            return

        if self.cone_latch_frozen:
            return

        if self.cone_latched_lanes != {'center'}:
            return

        if self.cone_center_only_start_time is None:
            self.cone_center_only_start_time = time.monotonic()
            return

        now = time.monotonic()
        if now - self.cone_center_only_start_time < self.cone_center_only_assume_left_time:
            return

        self.cone_latched_lanes.add('left')
        self.cone_latch_frozen = bool(self.cone_freeze_after_pair)
        self.cone_target_lane = 'right'
        self.cone_target_locked = True
        self.cone_target_votes.append('right')
        self.last_cone_msg_time = now

        self.get_logger().warn(
            f'cone center-only timeout in handler: assume left cone. '
            f'latched={sorted(list(self.cone_latched_lanes))}, target={self.cone_target_lane}'
        )

    def infer_cone_target_from_lanes(self, lanes):
        if 'center' in lanes and 'left' in lanes and 'right' not in lanes:
            return 'right'
        if 'center' in lanes and 'right' in lanes and 'left' not in lanes:
            return 'left'
        if 'left' in lanes and 'right' not in lanes:
            return 'right'
        if 'right' in lanes and 'left' not in lanes:
            return 'left'
        return 'center'

    def record_cone_target_vote(self, target_lane):
        if target_lane not in ('left', 'right', 'center'):
            return

        self.cone_target_votes.append(target_lane)
        max_len = max(self.cone_target_vote_window, 1)
        if len(self.cone_target_votes) > max_len:
            self.cone_target_votes = self.cone_target_votes[-max_len:]

    def choose_cone_target_from_votes(self):
        left_count = self.cone_target_votes.count('left')
        right_count = self.cone_target_votes.count('right')

        if left_count > right_count:
            return 'left'
        if right_count > left_count:
            return 'right'

        for target in reversed(self.cone_target_votes):
            if target in ('left', 'right'):
                return target

        return 'center'

    def choose_cone_target_lane(self, lanes):
        if 'left' in lanes and 'right' in lanes:
            return self.choose_cone_target_from_votes()
        if 'center' in lanes and 'left' in lanes:
            return 'right'
        if 'center' in lanes and 'right' in lanes:
            return 'left'
        if 'left' in lanes and 'right' not in lanes:
            return 'right'
        if 'right' in lanes and 'left' not in lanes:
            return 'left'
        return 'center'

    def cone_lidar_side_obstacle(self):
        if self.cone_target_lane not in ('left', 'right'):
            self.cone_lidar_clear_count = 0
            return False, 'none', 0, 9.9

        # simple force-shift 모드에서는 사용자가 지정한 sector를 쓴다.
        # 왼쪽 회피: 오른쪽 0~120도, 오른쪽 회피: 왼쪽 -120~0도.
        if self.cone_simple_force_enabled:
            side, cluster_count, point_count, nearest = self.side_cone_clusters()
            active = cluster_count >= self.cone_side_min_clusters
            return active, side, cluster_count, nearest

        # target_lane은 주행할 빈 방향이다. 콘은 그 반대편 옆 섹터에 남는다.
        if self.cone_target_lane == 'left':
            side = 'right'
            angle_min = -self.cone_lidar_side_angle_max_deg
            angle_max = -self.cone_lidar_side_angle_min_deg
        else:
            side = 'left'
            angle_min = self.cone_lidar_side_angle_min_deg
            angle_max = self.cone_lidar_side_angle_max_deg

        hits = 0
        nearest = 9.9

        for _, _, dist, angle in self.lidar_points:
            if dist > self.cone_lidar_clear_distance:
                continue

            angle_deg = math.degrees(angle)
            while angle_deg > 180.0:
                angle_deg -= 360.0
            while angle_deg < -180.0:
                angle_deg += 360.0

            if angle_min <= angle_deg <= angle_max:
                hits += 1
                nearest = min(nearest, dist)

        active = hits >= self.cone_lidar_min_hits
        return active, side, hits, nearest

    def reset_simple_cone_vars(self):
        self.simple_phase = 'WAIT_TARGET'
        self.simple_target = None
        self.simple_force_start_time = None
        self.simple_side_seen = False
        self.simple_side_clear_count = 0
        self.simple_last_cluster_count = 0
        self.simple_last_nearest = 9.9
        self.simple_last_side_name = 'none'

    def simple_cone_avoid_active(self):
        return (
            self.cone_simple_force_enabled and
            self.state == Mission.CONE and
            self.cone_latched and
            self.cone_target_lane in ('left', 'right')
        )

    def ensure_simple_cone_started(self):
        if not self.simple_cone_avoid_active():
            return

        target_changed = self.simple_target != self.cone_target_lane
        if self.simple_force_start_time is None or target_changed:
            self.simple_target = self.cone_target_lane
            self.simple_force_start_time = time.monotonic()
            self.simple_phase = 'FORCE_SHIFT'
            self.simple_side_seen = False
            self.simple_side_clear_count = 0
            self.simple_last_cluster_count = 0
            self.simple_last_nearest = 9.9
            self.simple_last_side_name = 'right' if self.simple_target == 'left' else 'left'
            self.prev_speed = 0.0
            self.prev_cte_norm = 0.0

            self.get_logger().warn(
                f'SIMPLE_CONE_START target={self.simple_target} '
                f'watch_side={self.simple_last_side_name} '
                f'latched={sorted(list(self.cone_latched_lanes))}'
            )

    def forced_cone_linear_speed(self):
        speed = float(self.cone_force_speed)
        if self.cone_after_force_speed_limit > 0.0:
            speed = min(speed, self.cone_after_force_speed_limit)
        return max(0.0, speed)

    def forced_cone_yaw_for_target(self):
        yaw = abs(self.cone_force_yaw)

        if self.cone_target_lane == 'left':
            return yaw if self.cone_left_yaw_positive else -yaw

        if self.cone_target_lane == 'right':
            return -yaw if self.cone_left_yaw_positive else yaw

        return 0.0

    def watched_side_for_cone_target(self):
        # 왼쪽 도로로 회피하면 콘은 오른쪽에 남아 있으므로 오른쪽 sector를 본다.
        # 오른쪽 도로로 회피하면 콘은 왼쪽에 남아 있으므로 왼쪽 sector를 본다.
        if self.cone_target_lane == 'left':
            return 'right'

        if self.cone_target_lane == 'right':
            return 'left'

        return 'none'

    def cone_sector_limits_for_side(self, side):
        if side == 'right':
            return self.cone_right_angle_min_deg, self.cone_right_angle_max_deg

        if side == 'left':
            return self.cone_left_angle_min_deg, self.cone_left_angle_max_deg

        return 0.0, 0.0

    def side_cone_clusters(self):
        side = self.watched_side_for_cone_target()
        angle_min, angle_max = self.cone_sector_limits_for_side(side)

        pts = []
        for x, y, dist, angle in self.lidar_points:
            if dist <= 0.0 or dist > self.cone_side_distance:
                continue

            angle_deg = normalize_deg(math.degrees(angle))
            if not angle_in_range(angle_deg, angle_min, angle_max):
                continue

            pts.append((angle_deg, x, y, dist))

        if len(pts) == 0:
            return side, 0, 0, 9.9

        pts.sort(key=lambda p: p[0])

        clusters = []
        current = [pts[0]]
        prev = pts[0]

        for p in pts[1:]:
            _, x, y, _ = p
            _, px, py, _ = prev
            gap = math.hypot(x - px, y - py)

            if gap <= self.cone_side_cluster_gap_m:
                current.append(p)
            else:
                if len(current) >= self.cone_side_cluster_min_points:
                    clusters.append(current)

                current = [p]

            prev = p

        if len(current) >= self.cone_side_cluster_min_points:
            clusters.append(current)

        nearest = min(p[3] for p in pts)
        return side, len(clusters), len(pts), nearest

    def update_simple_cone_pass_state(self):
        side, cluster_count, point_count, nearest = self.side_cone_clusters()

        self.simple_last_side_name = side
        self.simple_last_cluster_count = cluster_count
        self.simple_last_nearest = nearest

        active = cluster_count >= self.cone_side_min_clusters

        if active:
            self.simple_side_seen = True
            self.simple_side_clear_count = 0
        else:
            if self.simple_side_seen:
                self.simple_side_clear_count += 1
            else:
                self.simple_side_clear_count = 0

        passed = (
            self.simple_side_seen and
            self.simple_side_clear_count >= self.cone_side_clear_frames
        )

        return {
            'side': side,
            'cluster_count': cluster_count,
            'point_count': point_count,
            'nearest': nearest,
            'active': active,
            'seen': self.simple_side_seen,
            'clear_count': self.simple_side_clear_count,
            'passed': passed,
        }

    # ========================================================
    # Command helpers
    # ========================================================
    def stop_cmd(self):
        self.prev_speed = 0.0
        self.prev_cte_norm = 0.0
        cmd = Twist()
        return cmd

    def apply_cmd_limit(self, speed, yaw_rate, speed_limit=None, extra_yaw=0.0):
        if speed_limit is not None:
            speed = min(speed, speed_limit)
            self.prev_speed = min(self.prev_speed, speed_limit)

        yaw_rate = clamp(yaw_rate + extra_yaw, -MAX_YAW_RATE, MAX_YAW_RATE)

        cmd = Twist()
        cmd.linear.x = float(max(0.0, speed))
        cmd.angular.z = float(yaw_rate)
        return cmd

    def stanley_cmd_from_lane(self, lane_data, speed_limit=None, extra_yaw=0.0, lane_bias_px=0.0):
        self.current_lane_bias_px = float(lane_bias_px)

        if lane_data is None:
            return self.memory_lane_cmd(False)

        speed, yaw_rate_cmd, control_info = self.compute_stanley_control(
            lane_data['center_fit'],
            lane_data['confidence'],
            (IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        # camera-lidar decision fusion:
        # 차선 중심선이 현재 base_link 기준으로 좌우 어디에 있는지 아주 단순하게 근사한다.
        # LiDAR ROI가 벽이 아니라 주행 가능 영역 안쪽 점만 보도록 쓰는 값이다.
        y_bottom = IMAGE_HEIGHT - 1
        x_bottom = poly_x(lane_data['center_fit'], y_bottom)
        image_center = IMAGE_WIDTH // 2
        xm_per_px = self.track_width_m / max(float(self.last_lane_width), 1.0)
        self.path_center_y_m = clamp((image_center - x_bottom) * xm_per_px, -0.25, 0.25)

        cmd = self.apply_cmd_limit(speed, yaw_rate_cmd, speed_limit, extra_yaw)
        return cmd, control_info

    def memory_lane_cmd(self, tunnel_like=False):
        self.current_lane_bias_px = 0.0
        self.lost_frames += 1

        if self.prev_center_fit is not None and self.lost_frames <= MAX_LOST_FRAMES:
            target_speed = TUNNEL_SPEED if tunnel_like else LOST_LANE_SPEED
            speed = self.smooth_speed(target_speed)

            steering_angle = self.prev_steer * 0.82
            steering_angle = clamp(
                steering_angle,
                -MAX_STEERING_ANGLE,
                MAX_STEERING_ANGLE
            )

            self.prev_steer = steering_angle
            yaw_rate_cmd = self.steering_angle_to_yaw_rate(speed, steering_angle)

            cmd = Twist()
            cmd.linear.x = speed
            cmd.angular.z = yaw_rate_cmd
            return cmd

        return self.stop_cmd()

    # ========================================================
    # Mission handlers
    # ========================================================
    def handle_wait_traffic(self, lane_data, traffic):
        red_active, green_active, _, _, _ = traffic
        if green_active:
            self.next_state('(green light)')
            cmd, _ = self.stanley_cmd_from_lane(lane_data)
            return cmd, 'WAIT_GREEN_GO'

        return self.stop_cmd(), 'WAIT_RED_OR_UNKNOWN'

    def handle_pedestrian(self, lane_data, mask=None):
        obstacle = self.obstacle_in_pedestrian_roi()
        ped_active = self.ped_db.update(obstacle is not None)
        now = time.monotonic()

        if ped_active:
            self.pedestrian_was_seen = True
            self.last_ped_clear_time = None
            return self.stop_cmd(), f'PED_STOP_FRONT_{self.ped_front_min_dist:.2f}m_H{self.ped_front_hit}'

        if self.pedestrian_was_seen:
            if self.last_ped_clear_time is None:
                self.last_ped_clear_time = now
            elif now - self.last_ped_clear_time >= self.ped_clear_time:
                self.next_state('(pedestrian front sector clear stable)')

            cmd, _ = self.stanley_cmd_from_lane(lane_data, speed_limit=self.ped_speed_limit)
            return cmd, 'PED_CLEAR_GO'

        cmd, _ = self.stanley_cmd_from_lane(lane_data, speed_limit=self.ped_speed_limit)
        return cmd, 'PED_SEARCH_FRONT_ONLY'

    def handle_box(self, lane_data, next_state_after_clear=True, box_profile='box1'):
        now = time.monotonic()
        is_box2 = box_profile == 'box2'
        avoid_hold_time = self.box2_avoid_hold_time if is_box2 else self.box_avoid_hold_time
        avoid_yaw = self.box2_extra_yaw if is_box2 else self.box1_extra_yaw

        def box_cmd(extra_yaw, fallback_speed=0.24):
            if lane_data is None:
                cmd = Twist()
                cmd.linear.x = fallback_speed
                cmd.angular.z = clamp(extra_yaw, -MAX_YAW_RATE, MAX_YAW_RATE)
                return cmd

            cmd, _ = self.stanley_cmd_from_lane(lane_data, extra_yaw=extra_yaw)
            return cmd

        if not self.box_avoid_started:
            obstacle = self.obstacle_in_box_roi(box_profile)
            box_active = self.box_db.update(obstacle is not None)

            # 박스가 차선을 가리면 box_db가 active 되기 전에도 lane_data가 끊길 수 있다.
            # 이 경우 정지로 빠지지 않도록, 유효한 박스 후보가 있으면 바로 latch한다.
            if obstacle is not None and lane_data is None:
                box_active = True

            if not (box_active and obstacle is not None):
                if lane_data is None:
                    return self.memory_lane_cmd(False), 'BOX_SEARCH_LANE_MEMORY'

                cmd, _ = self.stanley_cmd_from_lane(lane_data)
                return cmd, 'BOX_SEARCH_TRACK_CORRIDOR'

            sector = obstacle['sector']
            self.box_seen_time = now
            self.box_avoid_started = True
            self.box_clear_start = None

            if sector == 'left':
                # 현재 트랙 corridor에 걸친 박스가 좌측/좌전방에 걸림 → 오른쪽 회피
                self.last_box_side = 1.0
                status = (
                    f'{box_profile.upper()}_TRACK_AVOID_RIGHT d={obstacle["nearest"]:.2f} '
                    f'cy={obstacle["cy"]:.2f} path={obstacle["path_y"]:.2f}'
                )
            else:
                # 현재 트랙 corridor에 걸친 박스가 우측/우전방에 걸림 → 왼쪽 회피
                self.last_box_side = -1.0
                status = (
                    f'{box_profile.upper()}_TRACK_AVOID_LEFT d={obstacle["nearest"]:.2f} '
                    f'cy={obstacle["cy"]:.2f} path={obstacle["path_y"]:.2f}'
                )

            extra_yaw = -avoid_yaw if self.last_box_side > 0.0 else avoid_yaw
            cmd = box_cmd(extra_yaw, fallback_speed=0.24)
            return cmd, status

        avoid_elapsed = now - self.box_seen_time if self.box_seen_time is not None else 0.0

        if avoid_elapsed < avoid_hold_time:
            extra_yaw = -avoid_yaw if self.last_box_side > 0.0 else avoid_yaw
            cmd = box_cmd(extra_yaw, fallback_speed=0.24)
            direction = 'RIGHT' if self.last_box_side > 0.0 else 'LEFT'
            return cmd, f'{box_profile.upper()}_LATCH_AVOID_{direction} {avoid_elapsed:.1f}/{avoid_hold_time:.1f}s'

        if self.obstacle_in_box_roi(box_profile) is None:
            self.next_state(f'({box_profile} close obstacle passed)')
            recover_yaw = 0.18 * self.last_box_side
            cmd = box_cmd(recover_yaw, fallback_speed=0.20)
            return cmd, f'{box_profile.upper()}_PASSED_FAST_CLEAR'

        if self.box_clear_start is None:
            self.box_clear_start = now

        clear_ok = (now - self.box_clear_start) >= self.box_clear_time

        # 회피 후 즉시 직진으로 복귀하지 않고 반대 조향으로 천천히 복귀한다.
        recover_yaw = 0.18 * self.last_box_side
        cmd = box_cmd(recover_yaw, fallback_speed=0.20)

        if next_state_after_clear and clear_ok:
            self.next_state('(box latch avoid and recover complete)')

        return cmd, f'BOX_RECOVER clear={0.0 if self.box_clear_start is None else now - self.box_clear_start:.1f}s'
    def tunnel_wall_state(self):
        left_hits = 0
        right_hits = 0
        left_min = 9.9
        right_min = 9.9

        for x, y, dist, _ in self.lidar_points:
            if not (self.tunnel_wall_x_min <= x <= self.tunnel_wall_x_max):
                continue
            if abs(y) < self.tunnel_wall_min_y:
                continue
            if abs(y) > self.tunnel_wall_distance:
                continue

            if y > 0.0:
                left_hits += 1
                left_min = min(left_min, abs(y))
            else:
                right_hits += 1
                right_min = min(right_min, abs(y))

        left_wall = left_hits >= self.tunnel_wall_min_hits
        right_wall = right_hits >= self.tunnel_wall_min_hits

        return {
            'inside': left_wall and right_wall,
            'left_hits': left_hits,
            'right_hits': right_hits,
            'left_min': left_min,
            'right_min': right_min,
        }

    def tunnel_exit_front_wall_state(self):
        """
        터널 탈출 판단용 벽 상태.

        기존 방식은 전방 -90~90도 전체를 검사했기 때문에
        터널을 빠져나온 뒤에도 정면/모서리 점 때문에 clear가 잘 안 되는 문제가 있었다.

        수정 방식:
          - 전방 전체가 아니라 측면 밴드만 검사한다.
          - 기본 검사 각도: -90~-60도, 60~90도.
          - 이 측면 밴드에 벽 점이 없으면 터널을 빠져나온 것으로 판단한다.
        """
        left_hits = 0
        right_hits = 0

        min_abs = min(
            abs(self.tunnel_exit_side_angle_min_abs_deg),
            abs(self.tunnel_exit_side_angle_max_abs_deg),
        )
        max_abs = max(
            abs(self.tunnel_exit_side_angle_min_abs_deg),
            abs(self.tunnel_exit_side_angle_max_abs_deg),
        )

        for x, y, _, angle in self.lidar_points:
            angle_deg = normalize_deg(math.degrees(angle))

            in_left_side_band = min_abs <= angle_deg <= max_abs
            in_right_side_band = -max_abs <= angle_deg <= -min_abs

            if not (in_left_side_band or in_right_side_band):
                continue
            if x < self.tunnel_wall_x_min:
                continue
            if abs(y) < self.tunnel_wall_min_y:
                continue
            if abs(y) > self.tunnel_wall_distance:
                continue

            if y > 0.0:
                left_hits += 1
            else:
                right_hits += 1

        total_hits = left_hits + right_hits
        return {
            'clear': total_hits == 0,
            'hits': total_hits,
            'left_hits': left_hits,
            'right_hits': right_hits,
        }

    def tunnel_center_cmd_from_walls(self):
        left_y = []
        right_y = []

        for x, y, _, angle in self.lidar_points:
            angle_deg = math.degrees(angle)
            while angle_deg > 180.0:
                angle_deg -= 360.0
            while angle_deg < -180.0:
                angle_deg += 360.0

            if not (-90.0 <= angle_deg <= 90.0):
                continue
            if not (self.tunnel_center_x_min <= x <= self.tunnel_center_x_max):
                continue
            if abs(y) < self.tunnel_wall_min_y:
                continue
            if abs(y) > self.tunnel_wall_distance:
                continue

            if y > 0.0:
                left_y.append(y)
            else:
                right_y.append(y)

        if len(left_y) < self.tunnel_wall_min_hits or len(right_y) < self.tunnel_wall_min_hits:
            return None, None

        left_mid = float(np.median(left_y))
        right_mid = float(np.median(right_y))
        wall_center_y = (left_mid + right_mid) / 2.0

        speed = self.smooth_speed(TUNNEL_SPEED)
        yaw_rate = clamp(
            self.tunnel_wall_center_gain * wall_center_y,
            -MAX_YAW_RATE,
            MAX_YAW_RATE
        )
        self.prev_steer = clamp(
            math.atan2(yaw_rate * WHEELBASE_M, max(speed, 0.10)),
            -MAX_STEERING_ANGLE,
            MAX_STEERING_ANGLE
        )

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(yaw_rate)

        return cmd, {
            'center_y': wall_center_y,
            'left_mid': left_mid,
            'right_mid': right_mid,
            'left_hits': len(left_y),
            'right_hits': len(right_y),
        }

    def handle_tunnel(self, lane_data, tunnel_like):
        """
        터널 판단:
        카메라 밝기/차선 소실 대신 LiDAR 좌우 30cm 이내 벽을 기준으로 진입/탈출을 판단한다.
        터널이 짧으므로 기본 2프레임만 안정화하고, 양쪽 벽이 사라지면 빠르게 다음 상태로 넘긴다.
        """
        now = time.monotonic()
        wall = self.tunnel_wall_state()
        wall_inside = wall['inside']
        exit_wall = self.tunnel_exit_front_wall_state()

        if not self.tunnel_was_seen:
            if wall_inside and self.state_elapsed() >= self.tunnel_enter_min_time:
                self.tunnel_candidate_count += 1
            else:
                self.tunnel_candidate_count = 0

            if self.tunnel_candidate_count >= self.tunnel_enter_frames:
                self.tunnel_was_seen = True
                self.tunnel_seen_time = now
                self.tunnel_recover_count = 0
                return self.memory_lane_cmd(tunnel_like=True), (
                    f'TUNNEL_WALL_ENTER L{wall["left_hits"]} R{wall["right_hits"]}'
                )

            if lane_data is None:
                cmd = self.memory_lane_cmd(tunnel_like=False)
            else:
                cmd, _ = self.stanley_cmd_from_lane(lane_data)
            return cmd, (
                f'TUNNEL_WALL_APPROACH {self.tunnel_candidate_count}/{self.tunnel_enter_frames} '
                f'L{wall["left_hits"]} R{wall["right_hits"]}'
            )

        if not exit_wall['clear']:
            self.tunnel_recover_count = 0
            self.tunnel_exit_start_time = None
            cmd, center = self.tunnel_center_cmd_from_walls()
            if cmd is not None and center is not None:
                return cmd, (
                    f'TUNNEL_WALL_CENTER cy={center["center_y"]:.3f} '
                    f'L={center["left_mid"]:.2f}/{center["left_hits"]} '
                    f'R={center["right_mid"]:.2f}/{center["right_hits"]} '
                    f'front={exit_wall["hits"]}({exit_wall["left_hits"]}/{exit_wall["right_hits"]})'
                )

            return self.memory_lane_cmd(tunnel_like=True), (
                f'TUNNEL_WALL_MEMORY_FALLBACK L{wall["left_hits"]} R{wall["right_hits"]} '
                f'front={exit_wall["hits"]}({exit_wall["left_hits"]}/{exit_wall["right_hits"]})'
            )

        self.tunnel_recover_count += 1

        memory_time_ok = True
        if self.tunnel_seen_time is not None:
            memory_time_ok = (now - self.tunnel_seen_time) >= self.tunnel_memory_min_time

        cmd = self.memory_lane_cmd(tunnel_like=True)

        if memory_time_ok and self.tunnel_recover_count >= self.tunnel_exit_frames:
            if self.tunnel_exit_start_time is None:
                self.tunnel_exit_start_time = now

            exit_drive_time = self.tunnel_exit_drive_distance / max(TUNNEL_SPEED, 0.10)
            exit_elapsed = now - self.tunnel_exit_start_time

            if exit_elapsed >= exit_drive_time:
                self.next_state('(tunnel exit drive distance reached)')
                return self.stop_cmd(), 'TUNNEL_EXIT_DELAY_DONE_STOP'

            cmd.linear.x = min(cmd.linear.x, TUNNEL_SPEED)
            return cmd, (
                f'TUNNEL_EXIT_DELAY {exit_elapsed:.1f}/{exit_drive_time:.1f}s '
                f'L{wall["left_hits"]} R{wall["right_hits"]}'
            )

        return cmd, (
            f'TUNNEL_WALL_EXIT {self.tunnel_recover_count}/{self.tunnel_exit_frames} '
            f'L{wall["left_hits"]} R{wall["right_hits"]}'
        )
    def rotary_cmd_from_lane(self, lane_data, speed_limit, extra_yaw=0.0):
        """
        사용자가 보낸 TrackVehicleFollowNode의 apply_follow_limit 구조를 ROTARY 상태에만 적용한다.
        CLEAR/LEAD_AHEAD에서는 기본 Stanley 속도에 clear_speed_gain을 곱한 뒤 speed_limit로 제한한다.
        """
        speed_limit = min(speed_limit, self.rotary_speed_limit)

        if speed_limit <= 0.0:
            return self.stop_cmd(), None

        if lane_data is None:
            cmd = self.memory_lane_cmd(tunnel_like=False)
            cmd.linear.x = min(cmd.linear.x, speed_limit)
            return cmd, None

        speed, yaw_rate_cmd, control_info = self.compute_stanley_control(
            lane_data['center_fit'],
            lane_data['confidence'],
            (IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        if lane_data['confidence'] < 0.45:
            speed = min(speed, BAD_CONF_SPEED)
        elif lane_data['confidence'] < 0.75:
            speed = min(speed, CURVE_MIN_SPEED)

        if self.rotary_follow_status in ('CLEAR', 'LEAD_AHEAD'):
            speed *= self.rotary_clear_speed_gain

        speed = min(speed, speed_limit)
        self.prev_speed = min(self.prev_speed, speed_limit)
        yaw_rate_cmd = clamp(yaw_rate_cmd + extra_yaw, -MAX_YAW_RATE, MAX_YAW_RATE)

        cmd = Twist()
        cmd.linear.x = float(max(0.0, speed))
        cmd.angular.z = float(yaw_rate_cmd)
        return cmd, control_info

    def handle_rotary(self, lane_data):
        cluster = self.rotary_vehicle_candidate()
        active = self.rotary_db.update(cluster is not None)
        now = time.monotonic()

        # 요구사항 반영:
        # 1) 회전교차로에서는 차량을 반드시 한 번 인식해야 출발한다.
        # 2) 이후 회전 구간을 rotary_min_time(기본 4초) 이상 지속하면 회전교차로 완료로 판단한다.
        # 고깔 위치 latch는 ROTARY가 아니라 CONE 상태에 들어간 뒤부터 시작한다.
        rotary_time_ok = self.state_elapsed() >= self.rotary_min_time
        complete_ok = self.rotary_vehicle_seen and rotary_time_ok

        if complete_ok:
            if cluster is not None:
                self.rotary_vehicle_seen = True
                self.rotary_clear_start = None
                self.rotary_vehicle_distance = cluster.nearest
                self.rotary_follow_status = 'ROTARY_DONE_WAIT_VEHICLE'
                return self.stop_cmd(), (
                    f'ROTARY_DONE_WAIT_VEHICLE_{cluster.nearest:.2f}m '
                    f't={self.state_elapsed():.1f}s'
                )

            self.next_state('(rotary >= 4s and lead vehicle clear)')
            # 상태 전환 직후에도 이번 frame의 cmd는 안전하게 Stanley 기반으로 한 번 더 보낸다.
            speed_limit = self.compute_rotary_speed_limit(cluster if active else None)
            if speed_limit <= 0.0:
                return self.stop_cmd(), 'ROTARY_DONE_BUT_LEAD_STOP'
            cmd, _ = self.rotary_cmd_from_lane(lane_data, speed_limit=speed_limit)
            return cmd, 'ROTARY_DONE'

        # 차량을 보기 전에는 정지 또는 설정한 아주 낮은 속도로만 creep한다.
        if not self.rotary_vehicle_seen and not active:
            self.rotary_follow_status = 'WAIT_VEHICLE'
            if self.rotary_wait_before_seen_speed <= 0.0:
                return self.stop_cmd(), 'ROTARY_WAIT_VEHICLE'

            cmd, _ = self.rotary_cmd_from_lane(
                lane_data,
                speed_limit=self.rotary_wait_before_seen_speed,
            )
            return cmd, 'ROTARY_CREEP_WAIT_VEHICLE'

        if active and cluster is not None:
            self.rotary_vehicle_seen = True
            self.rotary_clear_start = None
            self.rotary_vehicle_distance = cluster.nearest

            speed_limit = self.compute_rotary_speed_limit(cluster)
            if speed_limit <= 0.0:
                return self.stop_cmd(), (
                    f'ROTARY_{self.rotary_follow_status}_{cluster.nearest:.2f}m '
                    f'cone={self.cone_latched} t={self.state_elapsed():.1f}s'
                )

            cmd, _ = self.rotary_cmd_from_lane(lane_data, speed_limit=speed_limit)
            return cmd, (
                f'ROTARY_{self.rotary_follow_status}_{cluster.nearest:.2f}m '
                f'cone={self.cone_latched} t={self.state_elapsed():.1f}s'
            )

        # 여기까지 왔다는 것은 차량을 한 번 봤고, 현재는 차량 후보가 clear라는 뜻이다.
        if self.rotary_clear_start is None:
            self.rotary_clear_start = now

        speed_limit = self.compute_rotary_speed_limit(None)
        cmd, _ = self.rotary_cmd_from_lane(lane_data, speed_limit=speed_limit)
        return cmd, (
            f'ROTARY_AFTER_SEEN_WAIT_CONE clear={now - self.rotary_clear_start:.1f}s '
            f'cone={self.cone_latched} t={self.state_elapsed():.1f}s'
        )

    def handle_cone(self, lane_data):
        self.update_cone_center_only_timeout()

        if not self.cone_simple_force_enabled:
            lane_bias_px = 0.0
            extra_yaw = 0.0

            if self.cone_latched:
                if not self.cone_target_locked:
                    self.cone_target_lane = self.choose_cone_target_lane(self.cone_latched_lanes)
                    self.cone_target_locked = self.cone_target_lane in ('left', 'right')

                if self.cone_target_lane == 'left':
                    lane_bias_px = self.last_lane_width * self.cone_lane_bias_ratio
                    extra_yaw = self.cone_extra_yaw
                elif self.cone_target_lane == 'right':
                    lane_bias_px = -self.last_lane_width * self.cone_lane_bias_ratio
                    extra_yaw = -self.cone_extra_yaw

            if lane_data is not None and (
                str(lane_data.get('mode', '')).startswith('cone_pair_') or
                str(lane_data.get('mode', '')).startswith('cone_recover_inner_') or
                str(lane_data.get('mode', '')).startswith('cone_force_single_')
            ):
                lane_bias_px = 0.0

            close_cluster = self.obstacle_in_corridor(
                0.12,
                0.55,
                self.drivable_half_width,
                min_points=3,
                ignore_wall=True,
            )
            if close_cluster is not None and close_cluster.nearest < 0.25:
                return self.stop_cmd(), 'CONE_SAFE_STOP'

            if self.cone_latched and self.cone_first_latch_time is not None:
                latch_time_ok = (time.monotonic() - self.cone_first_latch_time) >= self.cone_latched_min_time
                state_time_ok = self.state_elapsed() >= self.cone_min_time

                if latch_time_ok and state_time_ok:
                    if self.cone_recover_start_time is None:
                        self.cone_recover_start_time = time.monotonic()

                    lane_bias_px = 0.0
                    extra_yaw = 0.0

            cmd, _ = self.stanley_cmd_from_lane(
                lane_data,
                speed_limit=self.cone_speed_limit,
                extra_yaw=extra_yaw,
                lane_bias_px=lane_bias_px,
            )

            if self.cone_recover_start_time is not None:
                recover_elapsed = time.monotonic() - self.cone_recover_start_time
                lane_ok = lane_data is not None and lane_data.get('confidence', 0.0) >= self.cone_recover_confidence
                cone_side_active, cone_side, cone_hits, cone_nearest = self.cone_lidar_side_obstacle()

                if cone_side_active:
                    self.cone_lidar_clear_count = 0
                else:
                    self.cone_lidar_clear_count += 1

                lidar_clear_ok = self.cone_lidar_clear_count >= self.cone_lidar_clear_frames

                if recover_elapsed >= self.cone_recover_time and lane_ok and lidar_clear_ok:
                    self.next_state('(cone latched and lane shift done)')
                return cmd, (
                    f'CONE_RECOVER_CENTER {recover_elapsed:.1f}/{self.cone_recover_time:.1f}s '
                    f'mode={lane_data.get("mode", "none") if lane_data is not None else "none"} '
                    f'target={self.cone_target_lane} lidar_{cone_side}='
                    f'{cone_hits}/{self.cone_lidar_min_hits} clear='
                    f'{self.cone_lidar_clear_count}/{self.cone_lidar_clear_frames} '
                    f'd={cone_nearest:.2f}'
                )

            lane_mode = lane_data.get('mode', 'none') if lane_data is not None else 'none'
            return cmd, (
                f'CONE_LATCH_{self.cone_target_lane.upper()}_{sorted(list(self.cone_latched_lanes))} '
                f'mode={lane_mode}'
            )

        # -------------------------
        # Simple force-shift cone avoidance
        # -------------------------
        # target이 아직 정해지지 않았으면 정상 차선추종으로 대기한다.
        if self.cone_latched and not self.cone_target_locked:
            self.cone_target_lane = self.choose_cone_target_lane(self.cone_latched_lanes)
            self.cone_target_locked = self.cone_target_lane in ('left', 'right')

        if not self.simple_cone_avoid_active():
            cmd, _ = self.stanley_cmd_from_lane(
                lane_data,
                speed_limit=self.cone_after_force_speed_limit,
                extra_yaw=0.0,
                lane_bias_px=0.0,
            )
            return cmd, (
                f'SIMPLE_CONE_WAIT_TARGET_LANE_FOLLOW '
                f'latched={sorted(list(self.cone_latched_lanes))} target={self.cone_target_lane}'
            )

        self.ensure_simple_cone_started()
        side_info = self.update_simple_cone_pass_state()

        # 1) FORCE_SHIFT: 일정 시간 동안 차선을 무시하고 강제 조향한다.
        force_elapsed = time.monotonic() - self.simple_force_start_time

        if force_elapsed < self.cone_force_duration:
            cmd = Twist()
            cmd.linear.x = float(self.forced_cone_linear_speed())
            cmd.angular.z = float(
                clamp(self.forced_cone_yaw_for_target(), -MAX_YAW_RATE, MAX_YAW_RATE)
            )

            self.prev_speed = cmd.linear.x
            self.current_lane_bias_px = 0.0
            self.simple_phase = 'FORCE_SHIFT'

            if side_info['passed']:
                self.next_state('(simple cone passed during force shift)')

            return cmd, (
                f'SIMPLE_FORCE_SHIFT target={self.cone_target_lane} '
                f'v={cmd.linear.x:.2f} wz={cmd.angular.z:.2f} '
                f't={force_elapsed:.2f}/{self.cone_force_duration:.2f}s '
                f'{side_info["side"]}_clusters={side_info["cluster_count"]}/{self.cone_side_min_clusters} '
                f'points={side_info["point_count"]} seen={side_info["seen"]} '
                f'clear={side_info["clear_count"]}/{self.cone_side_clear_frames} '
                f'd={side_info["nearest"]:.2f}'
            )

        # 2) force 이후: 다시 정상 차선 추종으로 복귀한다.
        self.simple_phase = 'LANE_FOLLOW_AFTER_FORCE'

        cmd, _ = self.stanley_cmd_from_lane(
            lane_data,
            speed_limit=self.cone_after_force_speed_limit,
            extra_yaw=0.0,
            lane_bias_px=0.0,
        )

        if side_info['passed']:
            self.next_state('(simple cone side clusters appeared and disappeared)')
            return cmd, (
                f'SIMPLE_CONE_PASSED target={self.cone_target_lane} '
                f'{side_info["side"]}_clusters={side_info["cluster_count"]}/{self.cone_side_min_clusters} '
                f'clear={side_info["clear_count"]}/{self.cone_side_clear_frames}'
            )

        return cmd, (
            f'SIMPLE_LANE_FOLLOW_AFTER_FORCE target={self.cone_target_lane} '
            f't={force_elapsed:.2f}/{self.cone_force_duration:.2f}s '
            f'{side_info["side"]}_clusters={side_info["cluster_count"]}/{self.cone_side_min_clusters} '
            f'points={side_info["point_count"]} seen={side_info["seen"]} '
            f'clear={side_info["clear_count"]}/{self.cone_side_clear_frames} '
            f'd={side_info["nearest"]:.2f}'
        )

    def handle_parking(self, lane_data, finish_detected):
        now = time.monotonic()

        if not self.parking_started:
            if finish_detected:
                self.parking_started = True
                self.parking_step_index = 0
                self.parking_step_start = now
                self.get_logger().warn('parking sequence start')
            else:
                cmd, _ = self.stanley_cmd_from_lane(
                    lane_data,
                )
                return cmd, 'WAIT_FINISH_LINE'

        duration, v, w = self.parking_sequence[self.parking_step_index]
        if now - self.parking_step_start >= duration:
            self.parking_step_index = min(self.parking_step_index + 1, len(self.parking_sequence) - 1)
            self.parking_step_start = now
            duration, v, w = self.parking_sequence[self.parking_step_index]

        if self.parking_step_index == len(self.parking_sequence) - 1:
            self.set_state(Mission.FINISHED, '(parking complete)')
            return self.stop_cmd(), 'PARK_DONE'

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        return cmd, f'PARK_STEP_{self.parking_step_index}'

    # ========================================================
    # Mission debug
    # ========================================================
    def draw_mission_overlay(self, warp_img, mask, lane_data, status, finish_ratio):
        debug = warp_img.copy()
        h, w = mask.shape
        image_center = w // 2
        desired_center = int(image_center + self.current_lane_bias_px)

        colored_mask = np.zeros_like(debug)
        colored_mask[:, :, 1] = mask
        debug = cv2.addWeighted(debug, 0.75, colored_mask, 0.25, 0)

        cv2.line(debug, (image_center, 0), (image_center, h), (255, 0, 0), 2)
        cv2.line(debug, (desired_center, 0), (desired_center, h), (0, 255, 255), 2)

        if lane_data is not None:
            self.draw_polyline(debug, lane_data.get('left_fit'), (255, 0, 0), 2)
            self.draw_polyline(debug, lane_data.get('right_fit'), (0, 255, 0), 2)
            self.draw_polyline(debug, lane_data.get('center_fit'), (0, 0, 255), 4)

        text1 = f'state={self.state} status={status}'
        text2 = f'v={self.prev_speed:.2f} lane_w={self.last_lane_width} bias={self.current_lane_bias_px:.1f}'
        text3 = (
            f'cones_latched={sorted(list(self.cone_latched_lanes))} '
            f'target={self.cone_target_lane} simple={self.simple_phase} finish={finish_ratio:.2f}'
        )
        box_text = 'box=None'
        if self.box_track_candidate is not None:
            box_text = (
                f'box={self.box_track_candidate["sector"]} '
                f'd={self.box_track_candidate["nearest"]:.2f} '
                f'cy={self.box_track_candidate["cy"]:.2f}'
            )
        text4 = (
            f'pedF={self.ped_front_obs}/{self.ped_front_min_dist:.2f} '
            f'{box_text} rot={self.rotary_follow_status} '
            f'tun={self.tunnel_candidate_count}/{self.tunnel_enter_frames} '
            f'elapsed={self.state_elapsed():.1f}s'
        )

        cv2.putText(debug, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug, text4, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(debug, dst_format='jpg')
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'debug publish error: {e}')

        if SHOW_DEBUG:
            try:
                cv2.imshow('mission_stanley_debug', debug)
                cv2.imshow('white_mask', mask)
                cv2.waitKey(1)
            except Exception:
                pass

    # ========================================================
    # Main callback
    # ========================================================
    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            warp_img = self.img_warp(img)
            mask = self.detect_white(warp_img)

            h, w = mask.shape

            gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
            avg_brightness = float(np.mean(gray))

            look_band = mask[int(h * 0.45):int(h * 0.65), :]
            white_pixels = int(cv2.countNonZero(look_band))

            tunnel_like = (
                avg_brightness < TUNNEL_BRIGHTNESS and
                white_pixels < MIN_WHITE_PIXELS
            )

            lane_data = self.get_lane_by_sliding_window(mask)

            if lane_data is not None:
                self.lost_frames = 0
                self.prev_center_fit = lane_data['center_fit']

            if self.use_traffic_light:
                traffic = self.detect_traffic_light(img)
            else:
                traffic = (False, False, 0, 0, (0, 0, 0, 0))

            finish_detected, finish_ratio = self.detect_finish_line(mask)

            status = 'IDLE'

            if self.state == Mission.WAIT_TRAFFIC:
                if self.use_traffic_light:
                    cmd, status = self.handle_wait_traffic(lane_data, traffic)
                else:
                    self.set_state(Mission.PEDESTRIAN, '(traffic disabled)')
                    cmd, status = self.handle_pedestrian(lane_data, mask)

            elif self.state == Mission.PEDESTRIAN:
                cmd, status = self.handle_pedestrian(lane_data, mask)

            elif self.state == Mission.BOX1:
                cmd, status = self.handle_box(lane_data, next_state_after_clear=True, box_profile='box1')

            elif self.state == Mission.TUNNEL:
                cmd, status = self.handle_tunnel(lane_data, tunnel_like)

            elif self.state == Mission.ROTARY:
                cmd, status = self.handle_rotary(lane_data)

            elif self.state == Mission.CONE:
                cmd, status = self.handle_cone(lane_data)

            elif self.state == Mission.BOX2:
                cmd, status = self.handle_box(lane_data, next_state_after_clear=True, box_profile='box2')

            elif self.state == Mission.PARKING:
                cmd, status = self.handle_parking(lane_data, finish_detected)

            elif self.state == Mission.FINISHED:
                cmd = self.stop_cmd()
                status = 'FINISHED_STOP'

            else:
                cmd = self.stop_cmd()
                status = 'UNKNOWN_STOP'

            self.cmd_pub.publish(cmd)

            self.draw_mission_overlay(
                warp_img,
                mask,
                lane_data,
                status,
                finish_ratio
            )

        except Exception as e:
            self.get_logger().error(f'image callback error: {e}')
            self.cmd_pub.publish(Twist())

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
    node = StanleyMissionFSMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()