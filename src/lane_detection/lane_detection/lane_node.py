import math
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


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
MAX_SPEED = 0.75
STRAIGHT_SPEED = 0.65
CURVE_MIN_SPEED = 0.30
BAD_CONF_SPEED = 0.25
LOST_LANE_SPEED = 0.22
TUNNEL_SPEED = 0.30

CALIBRATION_SPEED_LIMIT = 0.35

MAX_ACCEL_STEP = 0.035
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


class FastStanleyWhiteLineControl(Node):
    def __init__(self):
        super().__init__('fast_stanley_white_line_control_node')

        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.debug_pub = self.create_publisher(
            CompressedImage,
            '/white/debug/compressed',
            10
        )

        self.sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.img_callback,
            qos_profile_sensor_data
        )

        self.prev_speed = 0.0
        self.prev_steer = 0.0
        self.prev_cte_norm = 0.0

        self.last_lane_width = INITIAL_LANE_WIDTH_PX
        self.prev_center_fit = None
        self.last_single_side = None
        self.lost_frames = 0

        self.lane_width_samples = []
        self.lane_width_calibrated = False

        self.get_logger().info('fast stanley white line control node start')

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

        center_fit = None
        left_fit = None
        right_fit = None
        confidence = 0.0
        mode = 'none'
        side = None
        measured_lane_width = None

        best_pair = self.choose_best_pair(
            left_fit_candidates,
            right_fit_candidates,
            image_center
        )

        if best_pair is not None:
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
                USE_OUTER_LANE_IN_BOTH_CURVE and
                (
                    abs(curve_heading) > BOTH_CURVE_HEADING_TH or
                    width_diff > BOTH_WIDTH_DIFF_TH_PX
                )
            )

            if use_outer_lane:
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
        desired_center = image_center + CENTER_BIAS_PX

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
        desired_center = image_center + CENTER_BIAS_PX

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

    # =========================
    # Main callback
    # =========================
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

            cmd = Twist()
            control_info = None

            if lane_data is not None:
                self.lost_frames = 0
                self.prev_center_fit = lane_data['center_fit']

                speed, yaw_rate_cmd, control_info = self.compute_stanley_control(
                    lane_data['center_fit'],
                    lane_data['confidence'],
                    mask.shape
                )

                cmd.linear.x = speed
                cmd.angular.z = yaw_rate_cmd

                measured_lane_width = lane_data.get('measured_lane_width')

                self.get_logger().info(
                    f"v={speed:.2f}, wz={yaw_rate_cmd:.2f}, "
                    f"mode={lane_data['mode']}, conf={lane_data['confidence']:.2f}, "
                    f"lane_w={self.last_lane_width}, "
                    f"measured_w={measured_lane_width}, "
                    f"cte={control_info['cte_px']:.1f}, "
                    f"head={control_info['heading_error']:.2f}, "
                    f"raw_angle={control_info['raw_steer']:.2f}, "
                    f"steer_angle={control_info['steering_angle']:.2f}, "
                    f"raw_cands={len(lane_data.get('raw_candidates', []))}, "
                    f"filtered_cands={len(lane_data.get('filtered_candidates', []))}, "
                    f"brightness={avg_brightness:.1f}, white={white_pixels}"
                )

            else:
                self.lost_frames += 1

                if self.prev_center_fit is not None and self.lost_frames <= MAX_LOST_FRAMES:
                    if tunnel_like:
                        target_speed = TUNNEL_SPEED
                    else:
                        target_speed = LOST_LANE_SPEED

                    speed = self.smooth_speed(target_speed)

                    steering_angle = self.prev_steer * 0.82
                    steering_angle = clamp(
                        steering_angle,
                        -MAX_STEERING_ANGLE,
                        MAX_STEERING_ANGLE
                    )

                    self.prev_steer = steering_angle

                    yaw_rate_cmd = self.steering_angle_to_yaw_rate(
                        speed,
                        steering_angle
                    )

                    cmd.linear.x = speed
                    cmd.angular.z = yaw_rate_cmd

                    lane_data = {
                        'center_fit': self.prev_center_fit,
                        'left_fit': None,
                        'right_fit': None,
                        'confidence': 0.35,
                        'mode': 'memory_tunnel' if tunnel_like else 'memory_lost',
                        'side': self.last_single_side,
                        'total_pixels': 0,
                        'measured_lane_width': None,
                        'lane_width_calibrated': self.lane_width_calibrated,
                        'lane_width_sample_count': len(self.lane_width_samples),
                        'raw_candidates': [],
                        'filtered_candidates': [],
                        'left_candidates': [],
                        'right_candidates': [],
                    }

                    self.get_logger().warn(
                        f"lane lost memory mode: v={speed:.2f}, "
                        f"wz={yaw_rate_cmd:.2f}, "
                        f"steer_angle={steering_angle:.2f}, "
                        f"lost={self.lost_frames}, "
                        f"lane_w={self.last_lane_width}, "
                        f"brightness={avg_brightness:.1f}, white={white_pixels}"
                    )

                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.prev_speed = 0.0
                    self.prev_steer = 0.0
                    self.prev_cte_norm = 0.0

                    self.get_logger().warn(
                        f"line not found. stop. "
                        f"lane_w={self.last_lane_width}, "
                        f"brightness={avg_brightness:.1f}, white={white_pixels}"
                    )

            self.cmd_pub.publish(cmd)

            self.publish_debug(
                warp_img,
                mask,
                lane_data,
                control_info,
                avg_brightness,
                white_pixels
            )

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
    node = FastStanleyWhiteLineControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()