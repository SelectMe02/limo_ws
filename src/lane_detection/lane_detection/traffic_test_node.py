#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class TrafficLightLaneTestNode(Node):
    """
    CV 기반 신호등 + 라인 추종 단독 테스트 노드.

    목적:
      - mission_fsm_node 사용 안 함
      - handle_cone 사용 안 함
      - CONE 상태 사용 안 함
      - YOLO 사용 안 함
      - OpenCV만 사용
      - 기본적으로 라인 추종 주행
      - 빨간불 감지 순간 정지
      - 정지 중 초록불 감지 시 다시 라인 추종 주행

    신호등 조건:
      - 신호등은 차량 왼쪽, 화면 좌측 하단에 위치
      - 휴대폰 화면에 띄운 신호등이라 조명 반사, 흰색 날림, 낮은 채도 가능
      - HSV + Lab + normalized RGB + excess color + 원형 밝은 blob 후보를 함께 사용
    """

    def __init__(self):
        super().__init__("traffic_light_lane_test_node")

        # ============================================================
        # Topic parameters
        # ============================================================
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("use_compressed_image", True)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.image_topic = self.get_parameter("image_topic").value
        self.use_compressed_image = bool(self.get_parameter("use_compressed_image").value)
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        # ============================================================
        # Control parameters
        # ============================================================
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("image_timeout_sec", 0.5)

        self.declare_parameter("base_speed", 0.22)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("max_steer", 0.42)

        self.declare_parameter("kp", 0.85)
        self.declare_parameter("kd", 0.18)

        self.declare_parameter("steer_smooth_alpha", 0.65)
        self.declare_parameter("steer_rate_limit", 0.08)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.image_timeout_sec = float(self.get_parameter("image_timeout_sec").value)

        self.base_speed = float(self.get_parameter("base_speed").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_steer = float(self.get_parameter("max_steer").value)

        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)

        self.steer_smooth_alpha = float(self.get_parameter("steer_smooth_alpha").value)
        self.steer_rate_limit = float(self.get_parameter("steer_rate_limit").value)

        # ============================================================
        # Lane detection parameters
        # 흰색 차선 기반 라인 추종
        # ============================================================
        self.declare_parameter("lane_roi_y1_ratio", 0.55)
        self.declare_parameter("lane_roi_y2_ratio", 0.96)

        self.declare_parameter("white_s_max", 90)
        self.declare_parameter("white_v_min", 145)

        self.declare_parameter("lane_min_pixels", 120)
        self.declare_parameter("lane_lost_stop_frames", 8)

        self.declare_parameter("initial_lane_width_px", 250.0)
        self.declare_parameter("min_lane_width_px", 130.0)
        self.declare_parameter("max_lane_width_px", 430.0)
        self.declare_parameter("lane_width_smooth_alpha", 0.85)

        self.lane_roi_y1_ratio = float(self.get_parameter("lane_roi_y1_ratio").value)
        self.lane_roi_y2_ratio = float(self.get_parameter("lane_roi_y2_ratio").value)

        self.white_s_max = int(self.get_parameter("white_s_max").value)
        self.white_v_min = int(self.get_parameter("white_v_min").value)

        self.lane_min_pixels = int(self.get_parameter("lane_min_pixels").value)
        self.lane_lost_stop_frames = int(self.get_parameter("lane_lost_stop_frames").value)

        self.lane_width_px = float(self.get_parameter("initial_lane_width_px").value)
        self.min_lane_width_px = float(self.get_parameter("min_lane_width_px").value)
        self.max_lane_width_px = float(self.get_parameter("max_lane_width_px").value)
        self.lane_width_smooth_alpha = float(self.get_parameter("lane_width_smooth_alpha").value)

        # ============================================================
        # Traffic light ROI parameters
        # 사진 기준: 좌측 하단 휴대폰 신호등
        # ============================================================
        self.declare_parameter("tl_roi_x1_ratio", 0.00)
        self.declare_parameter("tl_roi_x2_ratio", 0.42)
        self.declare_parameter("tl_roi_y1_ratio", 0.36)
        self.declare_parameter("tl_roi_y2_ratio", 0.98)

        self.tl_roi_x1_ratio = float(self.get_parameter("tl_roi_x1_ratio").value)
        self.tl_roi_x2_ratio = float(self.get_parameter("tl_roi_x2_ratio").value)
        self.tl_roi_y1_ratio = float(self.get_parameter("tl_roi_y1_ratio").value)
        self.tl_roi_y2_ratio = float(self.get_parameter("tl_roi_y2_ratio").value)

        # ============================================================
        # Traffic light detection parameters
        # 흔히 쓰는 CV 신호등 검출 방식:
        # HSV 색상 분리 + Lab 색차 + normalized RGB + 밝은 원형 blob 후보
        # ============================================================
        self.declare_parameter("tl_process_scale", 5.0)

        # 밝은 신호등 후보 검출
        self.declare_parameter("light_v_min", 105)
        self.declare_parameter("light_l_min", 100)
        self.declare_parameter("light_s_min", 5)

        # HSV 색상 기준
        self.declare_parameter("red_h1_max", 25)
        self.declare_parameter("red_h2_min", 145)
        self.declare_parameter("red_s_min", 5)
        self.declare_parameter("red_v_min", 20)

        self.declare_parameter("green_h_min", 30)
        self.declare_parameter("green_h_max", 115)
        self.declare_parameter("green_s_min", 5)
        self.declare_parameter("green_v_min", 20)

        # Lab 색차 기준
        # OpenCV Lab에서 a 채널은 128 근처가 중립, 크면 빨강, 작으면 초록
        self.declare_parameter("lab_red_a_delta", 2.0)
        self.declare_parameter("lab_green_a_delta", 2.0)

        # normalized RGB / excess color 기준
        self.declare_parameter("red_norm_min", 0.335)
        self.declare_parameter("green_norm_min", 0.315)
        self.declare_parameter("rgb_margin", 2.5)
        self.declare_parameter("excess_min", 2.0)

        # 작은 휴대폰 불빛 blob 기준
        self.declare_parameter("tl_min_blob_area", 2.0)
        self.declare_parameter("tl_max_blob_area", 3500.0)
        self.declare_parameter("tl_min_score", 1.0)
        self.declare_parameter("tl_candidate_expand_px", 10)

        # 후보 blob 모양 필터
        self.declare_parameter("tl_min_aspect", 0.20)
        self.declare_parameter("tl_max_aspect", 5.00)
        self.declare_parameter("tl_min_fill_ratio", 0.04)

        # 바닥 반사광 억제
        # ROI 내부 아래쪽은 반사광일 가능성이 커서 가중치를 줄임
        self.declare_parameter("floor_reflection_y_ratio", 0.78)
        self.declare_parameter("floor_reflection_weight", 0.35)

        # 색 판정 확정 프레임
        self.declare_parameter("red_confirm_frames", 1)
        self.declare_parameter("green_confirm_frames", 2)

        # RED/GREEN 동시 검출 시 점수 비교
        self.declare_parameter("color_score_ratio", 1.08)

        self.tl_process_scale = float(self.get_parameter("tl_process_scale").value)

        self.light_v_min = int(self.get_parameter("light_v_min").value)
        self.light_l_min = int(self.get_parameter("light_l_min").value)
        self.light_s_min = int(self.get_parameter("light_s_min").value)

        self.red_h1_max = int(self.get_parameter("red_h1_max").value)
        self.red_h2_min = int(self.get_parameter("red_h2_min").value)
        self.red_s_min = int(self.get_parameter("red_s_min").value)
        self.red_v_min = int(self.get_parameter("red_v_min").value)

        self.green_h_min = int(self.get_parameter("green_h_min").value)
        self.green_h_max = int(self.get_parameter("green_h_max").value)
        self.green_s_min = int(self.get_parameter("green_s_min").value)
        self.green_v_min = int(self.get_parameter("green_v_min").value)

        self.lab_red_a_delta = float(self.get_parameter("lab_red_a_delta").value)
        self.lab_green_a_delta = float(self.get_parameter("lab_green_a_delta").value)

        self.red_norm_min = float(self.get_parameter("red_norm_min").value)
        self.green_norm_min = float(self.get_parameter("green_norm_min").value)
        self.rgb_margin = float(self.get_parameter("rgb_margin").value)
        self.excess_min = float(self.get_parameter("excess_min").value)

        self.tl_min_blob_area = float(self.get_parameter("tl_min_blob_area").value)
        self.tl_max_blob_area = float(self.get_parameter("tl_max_blob_area").value)
        self.tl_min_score = float(self.get_parameter("tl_min_score").value)
        self.tl_candidate_expand_px = int(self.get_parameter("tl_candidate_expand_px").value)

        self.tl_min_aspect = float(self.get_parameter("tl_min_aspect").value)
        self.tl_max_aspect = float(self.get_parameter("tl_max_aspect").value)
        self.tl_min_fill_ratio = float(self.get_parameter("tl_min_fill_ratio").value)

        self.floor_reflection_y_ratio = float(self.get_parameter("floor_reflection_y_ratio").value)
        self.floor_reflection_weight = float(self.get_parameter("floor_reflection_weight").value)

        self.red_confirm_frames = int(self.get_parameter("red_confirm_frames").value)
        self.green_confirm_frames = int(self.get_parameter("green_confirm_frames").value)
        self.color_score_ratio = float(self.get_parameter("color_score_ratio").value)

        # ============================================================
        # Debug parameters
        # ============================================================
        self.declare_parameter("show_debug_window", True)
        self.declare_parameter("show_mask_window", True)
        self.declare_parameter("print_log", True)
        self.declare_parameter("log_interval", 0.30)

        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)
        self.show_mask_window = bool(self.get_parameter("show_mask_window").value)
        self.print_log = bool(self.get_parameter("print_log").value)
        self.log_interval = float(self.get_parameter("log_interval").value)

        # ============================================================
        # State variables
        # ============================================================
        self.latest_frame = None
        self.last_image_time = None
        self.last_log_time = 0.0

        self.traffic_state = "GO"
        self.traffic_detected_color = "UNKNOWN"
        self.red_count = 0
        self.green_count = 0

        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.lane_lost_count = 0

        self.latest_speed_cmd = 0.0
        self.latest_steer_cmd = 0.0
        self.latest_lane_info = self.empty_lane_info()
        self.latest_tl_info = self.empty_tl_info((0, 0, 1, 1))

        # ============================================================
        # ROS pub/sub/timer
        # ============================================================
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        if self.use_compressed_image:
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.compressed_image_callback,
                qos_profile_sensor_data,
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.raw_image_callback,
                qos_profile_sensor_data,
            )

        timer_period = 1.0 / max(self.control_hz, 1.0)
        self.control_timer = self.create_timer(timer_period, self.control_timer_callback)

        self.get_logger().warn("traffic_light_lane_test_node started.")
        self.get_logger().warn("mission_fsm_node, handle_cone, YOLO 모두 사용하지 않는다.")
        self.get_logger().warn("기본 라인 추종 주행 -> RED 정지 -> GREEN 재출발")
        self.get_logger().warn(
            f"image_topic={self.image_topic}, use_compressed_image={self.use_compressed_image}"
        )
        self.get_logger().warn(
            f"TL ROI x=({self.tl_roi_x1_ratio:.2f},{self.tl_roi_x2_ratio:.2f}), "
            f"y=({self.tl_roi_y1_ratio:.2f},{self.tl_roi_y2_ratio:.2f})"
        )

    # ============================================================
    # Image callbacks
    # ============================================================

    def compressed_image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("compressed image decode failed")
            return

        self.handle_frame(frame)

    def raw_image_callback(self, msg):
        try:
            frame = self.ros_image_to_bgr(msg)
            if frame is None:
                return

            self.handle_frame(frame)

        except Exception as e:
            self.get_logger().warn(f"raw image conversion failed: {e}")

    def ros_image_to_bgr(self, msg):
        if msg.encoding == "bgr8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = arr.reshape((msg.height, msg.step // 3, 3))
            frame = frame[:, :msg.width, :]
            return frame.copy()

        if msg.encoding == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            rgb = arr.reshape((msg.height, msg.step // 3, 3))
            rgb = rgb[:, :msg.width, :]
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if msg.encoding == "bgra8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgra = arr.reshape((msg.height, msg.step // 4, 4))
            bgra = bgra[:, :msg.width, :]
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        if msg.encoding == "rgba8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            rgba = arr.reshape((msg.height, msg.step // 4, 4))
            rgba = rgba[:, :msg.width, :]
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        if msg.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            gray = arr.reshape((msg.height, msg.step))
            gray = gray[:, :msg.width]
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        self.get_logger().warn(f"unsupported image encoding: {msg.encoding}")
        return None

    # ============================================================
    # Main frame process
    # ============================================================

    def handle_frame(self, frame):
        self.latest_frame = frame
        self.last_image_time = time.monotonic()

        lane_info = self.compute_lane_follow(frame)
        tl_info = self.detect_traffic_light(frame)

        self.latest_lane_info = lane_info
        self.latest_tl_info = tl_info
        self.traffic_detected_color = tl_info["detected_color"]

        self.update_traffic_state(self.traffic_detected_color, tl_info)

        if self.traffic_state == "STOP_RED":
            self.latest_speed_cmd = 0.0
            self.latest_steer_cmd = 0.0
        else:
            self.latest_speed_cmd = lane_info["speed"]
            self.latest_steer_cmd = lane_info["steer"]

        if self.print_log:
            now = time.monotonic()
            if now - self.last_log_time >= self.log_interval:
                self.last_log_time = now
                self.get_logger().warn(
                    f"TL={tl_info['detected_color']} state={self.traffic_state} "
                    f"R(score={tl_info['red_score']:.2f}, valid={tl_info['red_valid']}) "
                    f"G(score={tl_info['green_score']:.2f}, valid={tl_info['green_valid']}) "
                    f"cnt={self.red_count}/{self.green_count} "
                    f"lane={lane_info['valid']} "
                    f"cmd=({self.latest_speed_cmd:.2f}, {self.latest_steer_cmd:.2f})"
                )

        if self.show_debug_window:
            debug_img = self.make_debug_image(frame, lane_info, tl_info)
            cv2.imshow("traffic_light_lane_debug", debug_img)
            cv2.waitKey(1)

        if self.show_mask_window:
            mask_debug = self.make_mask_debug_image(tl_info)
            if mask_debug is not None:
                cv2.imshow("traffic_light_mask_debug", mask_debug)
                cv2.waitKey(1)

    # ============================================================
    # Control timer
    # ============================================================

    def control_timer_callback(self):
        if self.last_image_time is None:
            self.publish_cmd(0.0, 0.0)
            return

        dt = time.monotonic() - self.last_image_time

        if dt > self.image_timeout_sec:
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(self.latest_speed_cmd, self.latest_steer_cmd)

    def publish_cmd(self, speed, steer):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(steer)
        self.cmd_pub.publish(cmd)

    # ============================================================
    # Lane following
    # ============================================================

    def compute_lane_follow(self, frame):
        h, w = frame.shape[:2]

        y1 = int(h * self.lane_roi_y1_ratio)
        y2 = int(h * self.lane_roi_y2_ratio)

        y1 = int(clamp(y1, 0, h - 1))
        y2 = int(clamp(y2, y1 + 1, h))

        roi = frame[y1:y2, 0:w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_v_min]),
            np.array([179, self.white_s_max, 255]),
        )

        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        ys, xs = np.where(white_mask > 0)

        if len(xs) < self.lane_min_pixels:
            return self.handle_lane_lost(w, y1, y2, white_mask)

        roi_h = y2 - y1

        # 아래쪽 픽셀에 더 큰 가중치를 줘서 가까운 차선 기준으로 주행
        row_weights = 0.5 + 0.5 * (ys.astype(np.float32) / max(float(roi_h - 1), 1.0))

        mid_x = w // 2

        left_mask = xs < mid_x
        right_mask = xs >= mid_x

        left_xs = xs[left_mask]
        right_xs = xs[right_mask]

        left_ws = row_weights[left_mask]
        right_ws = row_weights[right_mask]

        left_valid = len(left_xs) >= self.lane_min_pixels
        right_valid = len(right_xs) >= self.lane_min_pixels

        left_line_x = None
        right_line_x = None

        if left_valid:
            left_line_x = float(np.average(left_xs, weights=left_ws))

        if right_valid:
            right_line_x = float(np.average(right_xs, weights=right_ws))

        lane_center_x = None

        if left_valid and right_valid:
            measured_width = right_line_x - left_line_x

            if self.min_lane_width_px <= measured_width <= self.max_lane_width_px:
                self.lane_width_px = (
                    self.lane_width_smooth_alpha * self.lane_width_px
                    + (1.0 - self.lane_width_smooth_alpha) * measured_width
                )

            lane_center_x = (left_line_x + right_line_x) / 2.0

        elif left_valid:
            lane_center_x = left_line_x + self.lane_width_px / 2.0

        elif right_valid:
            lane_center_x = right_line_x - self.lane_width_px / 2.0

        else:
            return self.handle_lane_lost(w, y1, y2, white_mask)

        lane_center_x = clamp(lane_center_x, 0.0, float(w - 1))

        error = (lane_center_x - (w / 2.0)) / (w / 2.0)
        d_error = error - self.prev_error

        raw_steer = -(self.kp * error + self.kd * d_error)
        raw_steer = clamp(raw_steer, -self.max_steer, self.max_steer)

        smooth_steer = (
            self.steer_smooth_alpha * self.prev_steer
            + (1.0 - self.steer_smooth_alpha) * raw_steer
        )

        steer_delta = smooth_steer - self.prev_steer
        steer_delta = clamp(
            steer_delta,
            -self.steer_rate_limit,
            self.steer_rate_limit,
        )

        steer = self.prev_steer + steer_delta
        steer = clamp(steer, -self.max_steer, self.max_steer)

        steer_ratio = abs(steer) / max(self.max_steer, 1e-6)
        speed = self.base_speed - (self.base_speed - self.min_speed) * steer_ratio
        speed = clamp(speed, self.min_speed, self.base_speed)

        self.prev_error = error
        self.prev_steer = steer
        self.lane_lost_count = 0

        return {
            "valid": True,
            "speed": speed,
            "steer": steer,
            "lane_roi": (0, y1, w, y2),
            "left_line_x": left_line_x,
            "right_line_x": right_line_x,
            "lane_center_x": lane_center_x,
            "frame_center_x": w / 2.0,
            "error": error,
            "lost_count": self.lane_lost_count,
            "white_mask": white_mask,
        }

    def handle_lane_lost(self, w, y1, y2, white_mask=None):
        self.lane_lost_count += 1

        if self.lane_lost_count <= self.lane_lost_stop_frames:
            speed = self.min_speed
            steer = self.prev_steer
        else:
            speed = 0.0
            steer = 0.0

        return {
            "valid": False,
            "speed": speed,
            "steer": steer,
            "lane_roi": (0, y1, w, y2),
            "left_line_x": None,
            "right_line_x": None,
            "lane_center_x": None,
            "frame_center_x": w / 2.0,
            "error": None,
            "lost_count": self.lane_lost_count,
            "white_mask": white_mask,
        }

    def empty_lane_info(self):
        return {
            "valid": False,
            "speed": 0.0,
            "steer": 0.0,
            "lane_roi": (0, 0, 1, 1),
            "left_line_x": None,
            "right_line_x": None,
            "lane_center_x": None,
            "frame_center_x": 0.0,
            "error": None,
            "lost_count": 0,
            "white_mask": None,
        }

    # ============================================================
    # Traffic light detection
    # ============================================================

    def detect_traffic_light(self, frame):
        h, w = frame.shape[:2]

        x1 = int(w * self.tl_roi_x1_ratio)
        x2 = int(w * self.tl_roi_x2_ratio)
        y1 = int(h * self.tl_roi_y1_ratio)
        y2 = int(h * self.tl_roi_y2_ratio)

        x1 = int(clamp(x1, 0, w - 1))
        x2 = int(clamp(x2, x1 + 1, w))
        y1 = int(clamp(y1, 0, h - 1))
        y2 = int(clamp(y2, y1 + 1, h))

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return self.empty_tl_info((x1, y1, x2, y2))

        scale = max(1.0, self.tl_process_scale)

        proc = cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

        proc = cv2.GaussianBlur(proc, (3, 3), 0)

        masks = self.build_traffic_masks(proc)

        candidate_result = self.classify_by_light_candidates(
            proc,
            masks,
            scale,
        )

        fallback_result = self.classify_by_color_masks(
            masks,
            scale,
        )

        # 후보 기반 결과가 약하면 색상 마스크 전체 결과를 사용
        if candidate_result["best_score"] >= fallback_result["best_score"]:
            result = candidate_result
        else:
            result = fallback_result

        red_score = result["red_score"]
        green_score = result["green_score"]

        red_valid = red_score >= self.tl_min_score
        green_valid = green_score >= self.tl_min_score

        detected_color = "UNKNOWN"

        if red_valid and green_valid:
            if red_score >= green_score * self.color_score_ratio:
                detected_color = "RED"
            elif green_score >= red_score * self.color_score_ratio:
                detected_color = "GREEN"
            else:
                detected_color = "UNKNOWN"

        elif red_valid:
            detected_color = "RED"

        elif green_valid:
            detected_color = "GREEN"

        red_bbox = self.bbox_to_frame(result["red_bbox"], x1, y1, scale)
        green_bbox = self.bbox_to_frame(result["green_bbox"], x1, y1, scale)

        return {
            "detected_color": detected_color,
            "roi": (x1, y1, x2, y2),
            "scale": scale,

            "red_valid": red_valid,
            "green_valid": green_valid,

            "red_score": red_score,
            "green_score": green_score,

            "red_bbox": red_bbox,
            "green_bbox": green_bbox,

            "proc_roi": proc,
            "red_mask": masks["red_mask"],
            "green_mask": masks["green_mask"],
            "light_mask": masks["light_mask"],
        }

    def build_traffic_masks(self, proc_bgr):
        hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2LAB)

        h, s, v = cv2.split(hsv)
        l_ch, a_ch, b_lab = cv2.split(lab)

        clahe_v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        v_eq = clahe_v.apply(v)

        clahe_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_eq = clahe_l.apply(l_ch)

        b, g, r = cv2.split(proc_bgr)

        r_f = r.astype(np.float32)
        g_f = g.astype(np.float32)
        b_f = b.astype(np.float32)
        v_f = v.astype(np.float32)
        v_eq_f = v_eq.astype(np.float32)
        l_eq_f = l_eq.astype(np.float32)
        s_f = s.astype(np.float32)
        h_f = h.astype(np.float32)
        a_f = a_ch.astype(np.float32)

        rgb_sum = r_f + g_f + b_f + 1.0
        r_norm = r_f / rgb_sum
        g_norm = g_f / rgb_sum

        # ------------------------------------------------------------
        # 1. HSV 기반 색 검출
        # ------------------------------------------------------------
        red_hsv = (
            ((h_f <= self.red_h1_max) | (h_f >= self.red_h2_min))
            & (s_f >= self.red_s_min)
            & (v_eq_f >= self.red_v_min)
        )

        green_hsv = (
            (h_f >= self.green_h_min)
            & (h_f <= self.green_h_max)
            & (s_f >= self.green_s_min)
            & (v_eq_f >= self.green_v_min)
        )

        # ------------------------------------------------------------
        # 2. Lab 기반 색차 검출
        # 빨강: a 채널이 128보다 큼
        # 초록: a 채널이 128보다 작음
        # ------------------------------------------------------------
        red_lab = (
            ((a_f - 128.0) >= self.lab_red_a_delta)
            & (l_eq_f >= self.light_l_min)
        )

        green_lab = (
            ((128.0 - a_f) >= self.lab_green_a_delta)
            & (l_eq_f >= self.light_l_min)
        )

        # ------------------------------------------------------------
        # 3. normalized RGB 기반 검출
        # 조명 반사로 HSV 채도가 낮게 잡혀도 채널 비율로 보완
        # ------------------------------------------------------------
        red_norm = (
            (r_norm >= self.red_norm_min)
            & ((r_f - g_f) >= self.rgb_margin)
            & ((r_f - b_f) >= self.rgb_margin * 0.2)
            & (v_f >= self.green_v_min)
        )

        green_norm = (
            (g_norm >= self.green_norm_min)
            & ((g_f - r_f) >= self.rgb_margin * 0.2)
            & (g_f >= b_f * 0.65)
            & (v_f >= self.green_v_min)
        )

        # ------------------------------------------------------------
        # 4. excess color 기반 검출
        # ------------------------------------------------------------
        red_excess = np.maximum(0.0, 2.0 * r_f - g_f - b_f)
        green_excess = np.maximum(0.0, 2.0 * g_f - r_f - b_f)

        red_excess_mask = (
            (red_excess >= self.excess_min)
            & (v_f >= self.red_v_min)
        )

        green_excess_mask = (
            (green_excess >= self.excess_min)
            & (v_f >= self.green_v_min)
        )

        red_bool = red_hsv | red_lab | red_norm | red_excess_mask
        green_bool = green_hsv | green_lab | green_norm | green_excess_mask

        red_mask = red_bool.astype(np.uint8) * 255
        green_mask = green_bool.astype(np.uint8) * 255

        # ------------------------------------------------------------
        # 5. 밝은 원형 신호 후보
        # 신호등 불빛이 흰색으로 날아가도 후보 blob은 밝게 잡힘
        # ------------------------------------------------------------
        bright_bool = (
            ((v_eq_f >= self.light_v_min) | (l_eq_f >= self.light_l_min))
            & (s_f >= self.light_s_min)
        )

        light_bool = bright_bool | red_bool | green_bool
        light_mask = light_bool.astype(np.uint8) * 255

        kernel2 = np.ones((2, 2), np.uint8)
        kernel3 = np.ones((3, 3), np.uint8)

        # 작은 신호등이 사라지면 안 되므로 open은 하지 않음
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel2)
        red_mask = cv2.dilate(red_mask, kernel2, iterations=1)

        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel2)
        green_mask = cv2.dilate(green_mask, kernel2, iterations=1)

        light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel3)
        light_mask = cv2.dilate(light_mask, kernel2, iterations=1)

        # score map
        red_score_map = np.zeros_like(v_f, dtype=np.float32)
        green_score_map = np.zeros_like(v_f, dtype=np.float32)

        red_score_map += np.clip(red_excess, 0, 255) / 255.0
        green_score_map += np.clip(green_excess, 0, 255) / 255.0

        red_score_map[red_hsv] += 1.0
        green_score_map[green_hsv] += 1.0

        red_score_map[red_lab] += np.clip((a_f[red_lab] - 128.0) / 30.0, 0.0, 1.0)
        green_score_map[green_lab] += np.clip((128.0 - a_f[green_lab]) / 30.0, 0.0, 1.0)

        red_score_map[red_norm] += 0.7
        green_score_map[green_norm] += 0.7

        red_score_map = np.clip(red_score_map, 0.0, 4.0)
        green_score_map = np.clip(green_score_map, 0.0, 4.0)

        return {
            "red_mask": red_mask,
            "green_mask": green_mask,
            "light_mask": light_mask,
            "red_score_map": red_score_map,
            "green_score_map": green_score_map,
        }

    def classify_by_light_candidates(self, proc_bgr, masks, scale):
        light_mask = masks["light_mask"]
        red_mask = masks["red_mask"]
        green_mask = masks["green_mask"]
        red_score_map = masks["red_score_map"]
        green_score_map = masks["green_score_map"]

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            light_mask,
            connectivity=8,
        )

        best_red_score = 0.0
        best_green_score = 0.0
        best_red_bbox = None
        best_green_bbox = None
        best_score = 0.0

        weights = self.make_position_weight(light_mask.shape)

        h, w = light_mask.shape[:2]

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area <= 0:
                continue

            area_original = area / max(scale * scale, 1.0)

            if area_original < self.tl_min_blob_area:
                continue

            if area_original > self.tl_max_blob_area:
                continue

            aspect = bw / max(float(bh), 1.0)
            if aspect < self.tl_min_aspect or aspect > self.tl_max_aspect:
                continue

            fill_ratio = area / max(float(bw * bh), 1.0)
            if fill_ratio < self.tl_min_fill_ratio:
                continue

            ex = self.tl_candidate_expand_px

            x1 = int(clamp(x - ex, 0, w - 1))
            y1 = int(clamp(y - ex, 0, h - 1))
            x2 = int(clamp(x + bw + ex, x1 + 1, w))
            y2 = int(clamp(y + bh + ex, y1 + 1, h))

            red_patch_mask = red_mask[y1:y2, x1:x2] > 0
            green_patch_mask = green_mask[y1:y2, x1:x2] > 0

            red_patch_score = red_score_map[y1:y2, x1:x2]
            green_patch_score = green_score_map[y1:y2, x1:x2]
            patch_weights = weights[y1:y2, x1:x2]

            red_pixels = float(np.sum(patch_weights[red_patch_mask])) / max(scale * scale, 1.0)
            green_pixels = float(np.sum(patch_weights[green_patch_mask])) / max(scale * scale, 1.0)

            red_evidence = float(
                np.sum(red_patch_score[red_patch_mask] * patch_weights[red_patch_mask])
            ) / max(scale * scale, 1.0)

            green_evidence = float(
                np.sum(green_patch_score[green_patch_mask] * patch_weights[green_patch_mask])
            ) / max(scale * scale, 1.0)

            # 원형/덩어리 후보 가산
            compact_bonus = 0.5 + min(fill_ratio, 1.0)
            red_score = (red_pixels + red_evidence) * compact_bonus
            green_score = (green_pixels + green_evidence) * compact_bonus

            if red_score > best_red_score:
                best_red_score = red_score
                best_red_bbox = (x1, y1, x2 - x1, y2 - y1)

            if green_score > best_green_score:
                best_green_score = green_score
                best_green_bbox = (x1, y1, x2 - x1, y2 - y1)

            best_score = max(best_score, red_score, green_score)

        return {
            "red_score": best_red_score,
            "green_score": best_green_score,
            "red_bbox": best_red_bbox,
            "green_bbox": best_green_bbox,
            "best_score": best_score,
        }

    def classify_by_color_masks(self, masks, scale):
        red_mask = masks["red_mask"]
        green_mask = masks["green_mask"]
        red_score_map = masks["red_score_map"]
        green_score_map = masks["green_score_map"]

        weights = self.make_position_weight(red_mask.shape)

        red_bool = red_mask > 0
        green_bool = green_mask > 0

        red_score = 0.0
        green_score = 0.0

        if np.any(red_bool):
            red_pixels = float(np.sum(weights[red_bool])) / max(scale * scale, 1.0)
            red_evidence = float(
                np.sum(red_score_map[red_bool] * weights[red_bool])
            ) / max(scale * scale, 1.0)
            red_score = red_pixels + red_evidence

        if np.any(green_bool):
            green_pixels = float(np.sum(weights[green_bool])) / max(scale * scale, 1.0)
            green_evidence = float(
                np.sum(green_score_map[green_bool] * weights[green_bool])
            ) / max(scale * scale, 1.0)
            green_score = green_pixels + green_evidence

        red_bbox = self.mask_bbox(red_mask)
        green_bbox = self.mask_bbox(green_mask)

        return {
            "red_score": red_score,
            "green_score": green_score,
            "red_bbox": red_bbox,
            "green_bbox": green_bbox,
            "best_score": max(red_score, green_score),
        }

    def make_position_weight(self, shape):
        height, width = shape[:2]
        weights = np.ones((height, width), dtype=np.float32)

        floor_start = int(height * self.floor_reflection_y_ratio)
        floor_start = int(clamp(floor_start, 0, height))

        if floor_start < height:
            weights[floor_start:, :] *= self.floor_reflection_weight

        return weights

    def mask_bbox(self, mask):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        x1 = int(np.min(xs))
        x2 = int(np.max(xs))
        y1 = int(np.min(ys))
        y2 = int(np.max(ys))

        return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    def bbox_to_frame(self, bbox, offset_x, offset_y, scale):
        if bbox is None:
            return None

        x, y, w, h = bbox

        fx = int(offset_x + x / scale)
        fy = int(offset_y + y / scale)
        fw = max(1, int(w / scale))
        fh = max(1, int(h / scale))

        return (fx, fy, fw, fh)

    def empty_tl_info(self, roi):
        return {
            "detected_color": "UNKNOWN",
            "roi": roi,
            "scale": 1.0,
            "red_valid": False,
            "green_valid": False,
            "red_score": 0.0,
            "green_score": 0.0,
            "red_bbox": None,
            "green_bbox": None,
            "proc_roi": None,
            "red_mask": None,
            "green_mask": None,
            "light_mask": None,
        }

    # ============================================================
    # Traffic state
    # ============================================================

    def update_traffic_state(self, detected_color, tl_info):
        if detected_color == "RED":
            self.red_count += 1
            self.green_count = 0

        elif detected_color == "GREEN":
            self.green_count += 1
            self.red_count = 0

        else:
            self.red_count = max(0, self.red_count - 1)
            self.green_count = max(0, self.green_count - 1)

        # 주행 중 빨간불이면 즉시 정지
        if self.traffic_state == "GO":
            if self.red_count >= self.red_confirm_frames:
                self.traffic_state = "STOP_RED"
                self.red_count = 0
                self.green_count = 0
                self.prev_steer = 0.0
                self.get_logger().warn(
                    f"TRAFFIC STOP: red detected. "
                    f"R={tl_info['red_score']:.2f}, G={tl_info['green_score']:.2f}"
                )
            return

        # 정지 중에는 초록불이 확실히 잡힐 때만 출발
        if self.traffic_state == "STOP_RED":
            if self.green_count >= self.green_confirm_frames:
                self.traffic_state = "GO"
                self.red_count = 0
                self.green_count = 0
                self.prev_steer = 0.0
                self.get_logger().warn(
                    f"TRAFFIC GO: green detected. "
                    f"R={tl_info['red_score']:.2f}, G={tl_info['green_score']:.2f}"
                )
            return

    # ============================================================
    # Debug visualization
    # ============================================================

    def make_debug_image(self, frame, lane_info, tl_info):
        vis = frame.copy()

        # Traffic ROI
        x1, y1, x2, y2 = tl_info["roi"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

        red_bbox = tl_info["red_bbox"]
        green_bbox = tl_info["green_bbox"]

        if red_bbox is not None:
            bx, by, bw, bh = red_bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

        if green_bbox is not None:
            bx, by, bw, bh = green_bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        # Lane ROI
        lx1, ly1, lx2, ly2 = lane_info["lane_roi"]
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (255, 0, 255), 2)

        frame_center_x = int(lane_info["frame_center_x"])
        cv2.line(vis, (frame_center_x, ly1), (frame_center_x, ly2), (255, 255, 255), 2)

        if lane_info["left_line_x"] is not None:
            lx = int(lane_info["left_line_x"])
            cv2.line(vis, (lx, ly1), (lx, ly2), (255, 0, 0), 2)

        if lane_info["right_line_x"] is not None:
            rx = int(lane_info["right_line_x"])
            cv2.line(vis, (rx, ly1), (rx, ly2), (0, 255, 255), 2)

        if lane_info["lane_center_x"] is not None:
            cx = int(lane_info["lane_center_x"])
            cv2.line(vis, (cx, ly1), (cx, ly2), (0, 255, 0), 3)

        detected = tl_info["detected_color"]

        if detected == "RED":
            text_color = (0, 0, 255)
        elif detected == "GREEN":
            text_color = (0, 255, 0)
        else:
            text_color = (180, 180, 180)

        lane_status = "OK" if lane_info["valid"] else "LOST"

        cv2.putText(
            vis,
            f"TL={detected}  STATE={self.traffic_state}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            text_color,
            2,
        )

        cv2.putText(
            vis,
            f"R score={tl_info['red_score']:.2f} valid={tl_info['red_valid']} / "
            f"G score={tl_info['green_score']:.2f} valid={tl_info['green_valid']}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"count R/G={self.red_count}/{self.green_count}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"LANE={lane_status} lost={lane_info['lost_count']}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"cmd linear.x={self.latest_speed_cmd:.2f}, angular.z={self.latest_steer_cmd:.2f}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            "DEFAULT: LANE FOLLOW / RED: STOP / GREEN: GO",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        return vis

    def make_mask_debug_image(self, tl_info):
        proc_roi = tl_info["proc_roi"]
        red_mask = tl_info["red_mask"]
        green_mask = tl_info["green_mask"]
        light_mask = tl_info["light_mask"]

        if proc_roi is None or red_mask is None or green_mask is None or light_mask is None:
            return None

        h, w = proc_roi.shape[:2]

        red_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        green_bgr = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        light_bgr = cv2.cvtColor(light_mask, cv2.COLOR_GRAY2BGR)

        red_bgr[:, :, 0] = 0
        red_bgr[:, :, 1] = 0

        green_bgr[:, :, 0] = 0
        green_bgr[:, :, 2] = 0

        light_bgr[:, :, 0] = light_mask
        light_bgr[:, :, 1] = light_mask
        light_bgr[:, :, 2] = 0

        target_h = 180
        scale = target_h / max(h, 1)
        target_w = int(w * scale)

        roi_small = cv2.resize(proc_roi, (target_w, target_h))
        red_small = cv2.resize(red_bgr, (target_w, target_h))
        green_small = cv2.resize(green_bgr, (target_w, target_h))
        light_small = cv2.resize(light_bgr, (target_w, target_h))

        cv2.putText(roi_small, "ROI", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(red_small, "RED MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(green_small, "GREEN MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(light_small, "LIGHT BLOB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        top = np.hstack([roi_small, red_small])
        bottom = np.hstack([green_small, light_small])
        mosaic = np.vstack([top, bottom])

        return mosaic

    # ============================================================
    # Shutdown
    # ============================================================

    def destroy_node(self):
        try:
            self.publish_cmd(0.0, 0.0)
        except Exception:
            pass

        if self.show_debug_window or self.show_mask_window:
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightLaneTestNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            node.publish_cmd(0.0, 0.0)
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()