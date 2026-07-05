import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

try:
    from torch import cuda
except Exception:
    cuda = None

try:
    from interfaces_pkg.msg import Detection, DetectionArray
    HAS_INTERFACES = True
except Exception:
    Detection = None
    DetectionArray = None
    HAS_INTERFACES = False


class Yolov8ConeLatchNode(Node):
    """
    Cone-only YOLO node for the mission FSM.

    Main outputs:
      /cone/blocked_lanes  std_msgs/String
        - left,center
        - center,right
        - left
        - center
        - right

      /detections interfaces_pkg/DetectionArray
        - optional output for debug_pkg/yolov8_visualizer_node.

    Main idea of this modified version:
      1. YOLO detects cones from /camera/color/image_raw.
      2. CV detects the lower yellow horizontal line from the same camera image.
      3. The cone whose x position is closest to the detected yellow-line reference
         is treated as the center cone.
      4. Other cones are classified as left/right relative to that center cone.

    Important:
      The yellow line and the cones are both yellow. To prevent the cone body from
      being misdetected as the yellow horizontal line, this node masks out YOLO cone
      bounding boxes first, then searches only the lower ROI for wide horizontal
      yellow components.
    """

    def __init__(self):
        super().__init__('yolov8_cone_latch_node')

        self.bridge = CvBridge()

        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('model_path', '/home/wego/limo_ws/src/lane_detection/models/best_cone.pt')
        self.declare_parameter('blocked_lanes_topic', '/cone/blocked_lanes')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('reset_topic', '/cone/reset_latch')
        self.declare_parameter('mission_state_topic', '/mission/state')
        self.declare_parameter('gate_by_mission_state', True)
        self.declare_parameter('latch_enable_states', ['CONE'])
        self.declare_parameter('clear_latch_when_disabled', True)

        self.declare_parameter('device', 'auto')          # auto, cpu, 0, cuda:0
        self.declare_parameter('conf_th', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('inference_period', 0.25)  # seconds. 0.25 = 4 Hz
        self.declare_parameter('publish_empty', True)

        # Old fallback parameters. These are still used when the yellow line is not reliable.
        self.declare_parameter('left_max_ratio', 0.38)
        self.declare_parameter('center_max_ratio', 0.62)
        self.declare_parameter('side_decision_min_cones', 2)
        self.declare_parameter('side_decision_min_ratio', 0.30)
        self.declare_parameter('side_decision_max_ratio', 0.70)
        self.declare_parameter('side_decision_deadband', 0.04)
        self.declare_parameter('freeze_after_pair', True)
        self.declare_parameter('center_only_assume_left_time', 1.00)

        # New yellow horizontal line based center-reference parameters.
        self.declare_parameter('use_yellow_line_center_reference', True)
        self.declare_parameter('yellow_line_roi_y_min_ratio', 0.58)
        self.declare_parameter('yellow_line_roi_y_max_ratio', 0.96)
        self.declare_parameter('yellow_hsv_lower_h', 15)
        self.declare_parameter('yellow_hsv_lower_s', 70)
        self.declare_parameter('yellow_hsv_lower_v', 80)
        self.declare_parameter('yellow_hsv_upper_h', 45)
        self.declare_parameter('yellow_hsv_upper_s', 255)
        self.declare_parameter('yellow_hsv_upper_v', 255)
        self.declare_parameter('yellow_line_min_width_ratio', 0.18)
        self.declare_parameter('yellow_line_min_area_ratio', 0.0008)
        self.declare_parameter('yellow_line_min_aspect', 3.0)
        self.declare_parameter('yellow_line_max_height_ratio', 0.12)
        self.declare_parameter('yellow_line_max_angle_deg', 35.0)
        self.declare_parameter('yellow_line_hough_min_length_ratio', 0.16)
        self.declare_parameter('cone_mask_expand_ratio', 0.10)
        self.declare_parameter('reference_smoothing_alpha', 0.35)
        self.declare_parameter('line_reference_stale_time', 1.00)
        self.declare_parameter('center_match_max_dx_ratio', 0.24)
        self.declare_parameter('side_cone_dx_deadband_ratio', 0.05)
        self.declare_parameter('debug_yellow_line', False)

        camera_topic = self.get_parameter('camera_topic').value
        blocked_topic = self.get_parameter('blocked_lanes_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        reset_topic = self.get_parameter('reset_topic').value
        mission_state_topic = self.get_parameter('mission_state_topic').value

        self.model_path = self.get_parameter('model_path').value
        self.conf_th = float(self.get_parameter('conf_th').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.inference_period = float(self.get_parameter('inference_period').value)
        self.publish_empty = bool(self.get_parameter('publish_empty').value)
        self.left_max_ratio = float(self.get_parameter('left_max_ratio').value)
        self.center_max_ratio = float(self.get_parameter('center_max_ratio').value)
        self.side_decision_min_cones = int(self.get_parameter('side_decision_min_cones').value)
        self.side_decision_min_ratio = float(self.get_parameter('side_decision_min_ratio').value)
        self.side_decision_max_ratio = float(self.get_parameter('side_decision_max_ratio').value)
        self.side_decision_deadband = float(self.get_parameter('side_decision_deadband').value)
        self.freeze_after_pair = bool(self.get_parameter('freeze_after_pair').value)
        self.center_only_assume_left_time = float(
            self.get_parameter('center_only_assume_left_time').value
        )
        self.gate_by_mission_state = bool(self.get_parameter('gate_by_mission_state').value)
        self.latch_enable_states = set([
            str(x).strip().upper()
            for x in self.get_parameter('latch_enable_states').value
        ])
        self.clear_latch_when_disabled = bool(self.get_parameter('clear_latch_when_disabled').value)

        self.use_yellow_line_center_reference = bool(
            self.get_parameter('use_yellow_line_center_reference').value
        )
        self.yellow_line_roi_y_min_ratio = float(
            self.get_parameter('yellow_line_roi_y_min_ratio').value
        )
        self.yellow_line_roi_y_max_ratio = float(
            self.get_parameter('yellow_line_roi_y_max_ratio').value
        )
        self.yellow_hsv_lower = np.array([
            int(self.get_parameter('yellow_hsv_lower_h').value),
            int(self.get_parameter('yellow_hsv_lower_s').value),
            int(self.get_parameter('yellow_hsv_lower_v').value),
        ], dtype=np.uint8)
        self.yellow_hsv_upper = np.array([
            int(self.get_parameter('yellow_hsv_upper_h').value),
            int(self.get_parameter('yellow_hsv_upper_s').value),
            int(self.get_parameter('yellow_hsv_upper_v').value),
        ], dtype=np.uint8)
        self.yellow_line_min_width_ratio = float(
            self.get_parameter('yellow_line_min_width_ratio').value
        )
        self.yellow_line_min_area_ratio = float(
            self.get_parameter('yellow_line_min_area_ratio').value
        )
        self.yellow_line_min_aspect = float(
            self.get_parameter('yellow_line_min_aspect').value
        )
        self.yellow_line_max_height_ratio = float(
            self.get_parameter('yellow_line_max_height_ratio').value
        )
        self.yellow_line_max_angle_deg = float(
            self.get_parameter('yellow_line_max_angle_deg').value
        )
        self.yellow_line_hough_min_length_ratio = float(
            self.get_parameter('yellow_line_hough_min_length_ratio').value
        )
        self.cone_mask_expand_ratio = float(
            self.get_parameter('cone_mask_expand_ratio').value
        )
        self.reference_smoothing_alpha = float(
            self.get_parameter('reference_smoothing_alpha').value
        )
        self.line_reference_stale_time = float(
            self.get_parameter('line_reference_stale_time').value
        )
        self.center_match_max_dx_ratio = float(
            self.get_parameter('center_match_max_dx_ratio').value
        )
        self.side_cone_dx_deadband_ratio = float(
            self.get_parameter('side_cone_dx_deadband_ratio').value
        )
        self.debug_yellow_line = bool(self.get_parameter('debug_yellow_line').value)

        self.device = self.resolve_device(str(self.get_parameter('device').value))

        self.blocked_pub = self.create_publisher(String, blocked_topic, 10)

        if HAS_INTERFACES:
            self.detections_pub = self.create_publisher(DetectionArray, detections_topic, 10)
        else:
            self.detections_pub = None
            self.get_logger().warn(
                'interfaces_pkg is not available. /detections will not be published. '
                'debug_pkg/yolov8_visualizer_node requires interfaces_pkg.'
            )

        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.reset_sub = self.create_subscription(
            String,
            reset_topic,
            self.reset_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            String,
            mission_state_topic,
            self.mission_state_callback,
            10,
        )

        self.latest_msg = None
        self.latest_stamp = None
        self.current_mission_state = ''
        self.latch_armed_once = False
        self.model = YOLO(self.model_path)

        self.latched_lanes = set()
        self.last_detected_lanes = set()
        self.last_inference_time = 0.0
        self.latch_frozen = False
        self.center_only_start_time = None

        # Smoothed x reference of the lower yellow horizontal line.
        self.line_reference_x = None
        self.last_line_reference_time = None
        self.last_line_reference_source = 'none'

        self.timer = self.create_timer(self.inference_period, self.timer_callback)

        self.get_logger().info(
            f'yolov8 cone latch node start. model={self.model_path}, '
            f'camera={camera_topic}, device={self.device}, imgsz={self.imgsz}, '
            f'inference_period={self.inference_period}s, gate_by_state={self.gate_by_mission_state}, '
            f'latch_enable_states={sorted(list(self.latch_enable_states))}, '
            f'use_yellow_line_center_reference={self.use_yellow_line_center_reference}'
        )

    def resolve_device(self, device_param):
        device_param = device_param.strip().lower()

        if device_param == 'auto':
            if cuda is not None and cuda.is_available():
                return '0'
            return 'cpu'

        if device_param == 'cuda':
            return '0'

        if device_param == 'gpu':
            return '0'

        return device_param

    def image_callback(self, msg):
        # Keep only the newest frame. This prevents inference backlog.
        self.latest_msg = msg
        self.latest_stamp = msg.header.stamp

    def reset_latch_state(self):
        self.latched_lanes = set()
        self.last_detected_lanes = set()
        self.latch_frozen = False
        self.center_only_start_time = None
        self.line_reference_x = None
        self.last_line_reference_time = None
        self.last_line_reference_source = 'none'

    def reset_callback(self, msg):
        # Any message resets latch.
        self.reset_latch_state()
        self.get_logger().warn(f'cone latch reset: {msg.data}')

    def mission_state_callback(self, msg):
        state = msg.data.strip().upper()
        prev_state = self.current_mission_state
        self.current_mission_state = state

        if self.is_latch_enabled():
            if not self.latch_armed_once:
                self.latch_armed_once = True
                self.reset_latch_state()
                self.get_logger().warn(
                    f'cone latch ARMED at mission state={state}. Old cone detections cleared.'
                )
        else:
            self.latch_armed_once = False
            if self.clear_latch_when_disabled and (self.latched_lanes or self.last_detected_lanes):
                self.reset_latch_state()
                self.get_logger().warn(
                    f'cone latch disabled at mission state={state}. Cone detections cleared.'
                )

        if state != prev_state:
            self.get_logger().info(f'mission state update: {prev_state} -> {state}')

    def is_latch_enabled(self):
        if not self.gate_by_mission_state:
            return True
        return self.current_mission_state in self.latch_enable_states

    def timer_callback(self):
        if not self.is_latch_enabled():
            # Before ROTARY/CONE, ignore all YOLO detections.
            # This prevents cones from other track sections from being latched.
            if self.clear_latch_when_disabled:
                self.reset_latch_state()
            self.publish_latched_lanes()
            return

        if self.latest_msg is None:
            self.publish_latched_lanes()
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            self.publish_latched_lanes()
            return

        try:
            use_half = str(self.device) != 'cpu'
            results_list = self.model.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.conf_th,
                device=self.device,
                imgsz=self.imgsz,
                half=use_half,
            )
        except Exception as e:
            self.get_logger().error(f'YOLO predict error: {e}')
            self.publish_latched_lanes()
            return

        detected_lanes = set()
        cone_infos = []
        detections_msg = None

        if HAS_INTERFACES:
            detections_msg = DetectionArray()
            detections_msg.header = self.latest_msg.header

        height = cv_image.shape[0]
        width = cv_image.shape[1]

        if results_list:
            results = results_list[0].cpu()
            height, width = results.orig_img.shape[:2]
            class_names = results.names

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls.item())
                    score = float(box.conf.item())

                    xywh = box.xywh[0]
                    cx = float(xywh[0])
                    cy = float(xywh[1])
                    bw = float(xywh[2])
                    bh = float(xywh[3])

                    xyxy = box.xyxy[0]
                    x1 = float(xyxy[0])
                    y1 = float(xyxy[1])
                    x2 = float(xyxy[2])
                    y2 = float(xyxy[3])

                    cone_infos.append({
                        'class_id': cls_id,
                        'class_name': str(class_names.get(cls_id, f'class_{cls_id}')),
                        'score': score,
                        'cx': cx,
                        'cy': cy,
                        'bw': bw,
                        'bh': bh,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                    })

                    if HAS_INTERFACES:
                        det = Detection()
                        det.class_id = cls_id
                        det.class_name = str(class_names.get(cls_id, f'class_{cls_id}'))
                        det.score = score
                        det.bbox.center.position.x = cx
                        det.bbox.center.position.y = cy
                        det.bbox.size.x = bw
                        det.bbox.size.y = bh
                        detections_msg.detections.append(det)

        raw_line_ref_x = None
        center_ref_x = None
        center_ref_source = 'none'

        if self.use_yellow_line_center_reference:
            raw_line_ref_x = self.detect_yellow_line_reference_x(cv_image, cone_infos)
            center_ref_x, center_ref_source = self.update_line_reference(raw_line_ref_x)

        detected_lanes = self.classify_lanes_from_cones(
            cone_infos=cone_infos,
            width=width,
            center_ref_x=center_ref_x,
            center_ref_source=center_ref_source,
        )

        self.last_detected_lanes = detected_lanes

        # Latch with freeze logic.
        # Once center+left or center+right is fixed, later opposite-side detections are ignored.
        # However, when yellow-line center reference is enabled, center-only must not be
        # automatically converted to center+left.
        changed = self.update_cone_latch(detected_lanes)

        if changed:
            self.get_logger().warn(
                f'cone latch update: detected={self.format_lanes(detected_lanes)}, '
                f'latched={self.format_lanes(self.latched_lanes)}, '
                f'frozen={self.latch_frozen}, '
                f'line_ref={center_ref_x if center_ref_x is not None else -1:.1f}, '
                f'line_ref_source={center_ref_source}'
            )

        if self.debug_yellow_line and raw_line_ref_x is not None:
            self.get_logger().info(
                f'yellow line raw_ref_x={raw_line_ref_x:.1f}, '
                f'smoothed_ref_x={center_ref_x if center_ref_x is not None else -1:.1f}, '
                f'source={center_ref_source}'
            )

        if detections_msg is not None and self.detections_pub is not None:
            self.detections_pub.publish(detections_msg)

        self.publish_latched_lanes()

    def detect_yellow_line_reference_x(self, cv_image, cone_infos):
        """
        Detect the lower yellow horizontal line and return its x reference.

        Because cones are yellow too, this method:
          1. uses only the lower ROI,
          2. masks out YOLO cone boxes,
          3. keeps only long and thin horizontal yellow components,
          4. allows moderate line angle so heading/perspective changes are tolerated.
        """
        height, width = cv_image.shape[:2]
        if height <= 0 or width <= 0:
            return None

        y0 = int(height * self.yellow_line_roi_y_min_ratio)
        y1 = int(height * self.yellow_line_roi_y_max_ratio)
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))

        roi = cv_image[y0:y1, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_hsv_lower, self.yellow_hsv_upper)

        # Remove cone regions from the yellow mask so the cone body/base is not
        # used as the horizontal-line reference.
        self.erase_cone_boxes_from_mask(mask, cone_infos, y_offset=y0, width=width, height=height)

        # Horizontal morphology: it suppresses narrow/vertical yellow regions and
        # preserves wide lane/stop-line-like regions.
        horizontal_kernel_w = max(15, int(width * 0.045))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_w, 3))
        mask_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)

        close_kernel_w = max(15, int(width * 0.035))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_w, 5))
        mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_CLOSE, close_kernel)

        contour_ref = self.find_line_reference_from_contours(mask_h, width, height, y0)
        if contour_ref is not None:
            return contour_ref

        # Fallback: Hough line detection is useful when the yellow line is split
        # by lighting, shadows, or motion blur.
        return self.find_line_reference_from_hough(mask_h, width)

    def erase_cone_boxes_from_mask(self, mask, cone_infos, y_offset, width, height):
        roi_height = mask.shape[0]
        expand = max(0.0, self.cone_mask_expand_ratio)

        for cone in cone_infos:
            bw = max(float(cone['bw']), 1.0)
            bh = max(float(cone['bh']), 1.0)

            x1 = int(cone['x1'] - bw * expand)
            y1 = int(cone['y1'] - bh * expand)
            x2 = int(cone['x2'] + bw * expand)
            y2 = int(cone['y2'] + bh * expand)

            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height, y2))

            ry1 = max(0, y1 - y_offset)
            ry2 = min(roi_height, y2 - y_offset)

            if x2 > x1 and ry2 > ry1:
                mask[ry1:ry2, x1:x2] = 0

    def find_line_reference_from_contours(self, mask_h, image_width, image_height, y_offset):
        contours, _ = cv2.findContours(mask_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_width = image_width * self.yellow_line_min_width_ratio
        min_area = image_width * image_height * self.yellow_line_min_area_ratio
        max_height = image_height * self.yellow_line_max_height_ratio

        best_candidate = None
        best_score = -1.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            area = float(cv2.contourArea(contour))
            aspect = float(w) / max(float(h), 1.0)

            if float(w) < min_width:
                continue
            if area < min_area:
                continue
            if aspect < self.yellow_line_min_aspect:
                continue
            if float(h) > max_height:
                continue

            rect = cv2.minAreaRect(contour)
            (_, _), (rw, rh), angle = rect
            long_angle = self.normalize_min_area_rect_angle(rw, rh, angle)
            if abs(long_angle) > self.yellow_line_max_angle_deg:
                continue

            # Prefer wide, lower, thin, and near-horizontal yellow components.
            lower_bonus = (float(y_offset + y) / max(float(image_height), 1.0)) * 50.0
            score = (float(w) * 2.0) + area + (aspect * 10.0) + lower_bonus - (abs(long_angle) * 3.0)

            if score > best_score:
                best_score = score
                best_candidate = (x + (w * 0.5))

        return best_candidate

    def find_line_reference_from_hough(self, mask_h, image_width):
        min_len = max(20, int(image_width * self.yellow_line_hough_min_length_ratio))
        lines = cv2.HoughLinesP(
            mask_h,
            rho=1,
            theta=np.pi / 180.0,
            threshold=30,
            minLineLength=min_len,
            maxLineGap=20,
        )

        if lines is None:
            return None

        best_ref_x = None
        best_score = -1.0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue

            angle = np.degrees(np.arctan2(dy, dx))
            while angle < -90.0:
                angle += 180.0
            while angle > 90.0:
                angle -= 180.0

            if abs(angle) > self.yellow_line_max_angle_deg:
                continue

            score = length - (abs(angle) * 2.0)
            if score > best_score:
                best_score = score
                best_ref_x = (float(x1) + float(x2)) * 0.5

        return best_ref_x

    def normalize_min_area_rect_angle(self, rw, rh, angle):
        """
        Convert cv2.minAreaRect angle to the long-axis angle in degrees.
        Returned angle is normalized into [-90, 90].
        """
        if rw < rh:
            angle = angle + 90.0

        while angle < -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0

        return float(angle)

    def update_line_reference(self, raw_ref_x):
        now = time.monotonic()
        alpha = min(1.0, max(0.0, self.reference_smoothing_alpha))

        if raw_ref_x is not None:
            raw_ref_x = float(raw_ref_x)
            if self.line_reference_x is None:
                self.line_reference_x = raw_ref_x
            else:
                self.line_reference_x = (alpha * raw_ref_x) + ((1.0 - alpha) * self.line_reference_x)

            self.last_line_reference_time = now
            self.last_line_reference_source = 'detected'
            return self.line_reference_x, 'detected'

        if (
            self.line_reference_x is not None and
            self.last_line_reference_time is not None and
            (now - self.last_line_reference_time) <= self.line_reference_stale_time
        ):
            self.last_line_reference_source = 'stale'
            return self.line_reference_x, 'stale'

        self.last_line_reference_source = 'none'
        return None, 'none'

    def classify_lanes_from_cones(self, cone_infos, width, center_ref_x=None, center_ref_source='none'):
        """
        Classify blocked lanes from YOLO cone boxes.

        If a yellow-line reference is available, it is used first:
          - nearest cone to the yellow-line x reference => center cone
          - cones left/right of that center cone => left/right

        If yellow-line center reference is enabled but the reference is missing,
        the old image-ratio logic is allowed to publish center only.
        It must not decide left/right or freeze a side without a valid line reference.
        """
        if not cone_infos:
            return set()

        img_width = max(float(width), 1.0)

        if self.use_yellow_line_center_reference:
            if center_ref_x is not None:
                lanes = self.classify_lanes_with_center_reference(cone_infos, img_width, center_ref_x)
                if lanes:
                    return lanes

                if self.debug_yellow_line:
                    self.get_logger().warn(
                        f'yellow-line reference rejected. ref_x={center_ref_x:.1f}, '
                        f'source={center_ref_source}. side fallback disabled.'
                    )

            fallback_lanes = self.classify_lanes_by_old_ratio_logic(cone_infos, img_width)

            if 'center' in fallback_lanes:
                return {'center'}

            return set()

        return self.classify_lanes_by_old_ratio_logic(cone_infos, img_width)

    def classify_lanes_with_center_reference(self, cone_infos, img_width, center_ref_x):
        center_ref_x = float(center_ref_x)
        max_center_dx = img_width * self.center_match_max_dx_ratio
        side_deadband_px = img_width * self.side_cone_dx_deadband_ratio

        # The center cone is the YOLO cone closest to the yellow-line x reference.
        center_cone = min(cone_infos, key=lambda cone: abs(float(cone['cx']) - center_ref_x))
        center_dx = abs(float(center_cone['cx']) - center_ref_x)

        # If the nearest cone is too far from the yellow line, the line reference is probably wrong.
        # In that case, use the old fallback instead of latching a wrong lane.
        if center_dx > max_center_dx:
            return set()

        lanes = {'center'}
        center_x = float(center_cone['cx'])

        left_score = 0.0
        right_score = 0.0

        for cone in cone_infos:
            if cone is center_cone:
                continue

            dx = float(cone['cx']) - center_x
            if abs(dx) < side_deadband_px:
                continue

            # Larger and closer-to-center side cone gets slightly higher priority.
            # This helps avoid a small far false-positive from changing the side decision.
            area = max(float(cone['bw']) * float(cone['bh']), 1.0)
            score = (abs(dx) / img_width) + (0.00001 * area) + (0.1 * float(cone['score']))

            if dx < 0:
                left_score += score
            else:
                right_score += score

        if left_score > 0.0 or right_score > 0.0:
            if right_score >= left_score:
                lanes.add('right')
            else:
                lanes.add('left')

        return lanes

    def classify_lanes_by_old_ratio_logic(self, cone_infos, img_width):
        cone_centers = [float(cone['cx']) for cone in cone_infos]
        ratios = [float(cx) / img_width for cx in cone_centers]

        # center-only fallback을 위해, 콘이 1개만 보이더라도 화면 중앙 영역이면 center로 인정한다.
        # 좌/우 판단은 기존처럼 side_decision_min_cones 개수 이상일 때만 수행한다.
        if len(cone_centers) < self.side_decision_min_cones:
            for ratio in ratios:
                if self.side_decision_min_ratio <= ratio <= self.side_decision_max_ratio:
                    return {'center'}
            return set()

        lanes = {'center'}
        center_gate_ratios = [
            ratio for ratio in ratios
            if self.side_decision_min_ratio <= ratio <= self.side_decision_max_ratio
        ]

        if len(center_gate_ratios) >= self.side_decision_min_cones:
            pair_ratios = center_gate_ratios
        else:
            sorted_by_center = sorted(ratios, key=lambda ratio: abs(ratio - 0.5))
            pair_ratios = sorted_by_center[:self.side_decision_min_cones]

        pair_center_ratio = sum(pair_ratios) / float(len(pair_ratios))
        if pair_center_ratio < 0.5 - self.side_decision_deadband:
            lanes.add('left')
        elif pair_center_ratio > 0.5 + self.side_decision_deadband:
            lanes.add('right')

        return lanes

    def has_center_pair(self, lanes):
        return (
            'center' in lanes and
            (
                ('left' in lanes and 'right' not in lanes) or
                ('right' in lanes and 'left' not in lanes)
            )
        )

    def finalize_pair_if_possible(self, lanes):
        """
        center+left 또는 center+right 중 하나가 확정되면 그 2개만 저장한다.
        left/right가 둘 다 섞여 있으면, 이미 저장된 쪽을 우선한다.
        """
        lanes = set(lanes)

        if 'center' not in lanes:
            return None

        # 이미 center+right가 먼저 잡혀 있었다면 right 유지
        if 'right' in self.latched_lanes and 'left' not in self.latched_lanes:
            return {'center', 'right'}

        # 이미 center+left가 먼저 잡혀 있었다면 left 유지
        if 'left' in self.latched_lanes and 'right' not in self.latched_lanes:
            return {'center', 'left'}

        if 'right' in lanes and 'left' not in lanes:
            return {'center', 'right'}

        if 'left' in lanes and 'right' not in lanes:
            return {'center', 'left'}

        return None

    def update_cone_latch(self, detected_lanes):
        now = time.monotonic()

        # 이미 center+left 또는 center+right가 확정되면 이후 검출은 무시
        if self.freeze_after_pair and self.latch_frozen:
            return False

        before = set(self.latched_lanes)

        if detected_lanes:
            combined = self.latched_lanes | detected_lanes

            pair = self.finalize_pair_if_possible(combined)
            if pair is not None:
                self.latched_lanes = pair
                self.latch_frozen = True
                self.center_only_start_time = None
            else:
                self.latched_lanes = combined

                if self.latched_lanes == {'center'}:
                    if self.center_only_start_time is None:
                        self.center_only_start_time = now
                else:
                    self.center_only_start_time = None

        # 노란선 기반 center reference를 사용하는 경우:
        # center cone은 항상 존재한다는 전제를 쓰므로, center만 보인다고 left로 가정하지 않는다.
        # 즉, center-only 상태에서는 freeze하지 않고 다음 프레임에서 side cone이 잡히기를 기다린다.
        if self.use_yellow_line_center_reference:
            return self.latched_lanes != before

        # 노란선 기반 center reference를 끈 경우에만 기존 fallback 유지:
        # center만 1초 이상 지속되면 center+left로 확정
        if (
            self.freeze_after_pair and
            not self.latch_frozen and
            self.latched_lanes == {'center'}
        ):
            if self.center_only_start_time is None:
                self.center_only_start_time = now

            elapsed = now - self.center_only_start_time
            if elapsed >= self.center_only_assume_left_time:
                self.latched_lanes = {'center', 'left'}
                self.latch_frozen = True
                self.get_logger().warn(
                    f'cone center-only timeout: assume left. '
                    f'latched={self.format_lanes(self.latched_lanes)}'
                )

        return self.latched_lanes != before

    def format_lanes(self, lanes):
        order = ['left', 'center', 'right']
        return ','.join([lane for lane in order if lane in lanes])

    def publish_latched_lanes(self):
        if not self.latched_lanes and not self.publish_empty:
            return

        msg = String()
        msg.data = self.format_lanes(self.latched_lanes)
        self.blocked_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8ConeLatchNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()