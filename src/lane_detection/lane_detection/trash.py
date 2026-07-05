import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO

try:
    from torch import cuda
except Exception:
    cuda = None


class ConeYoloAngleDebugNode(Node):
    def __init__(self):
        super().__init__('cone_yolo_angle_debug_node')

        self.bridge = CvBridge()

        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('model_path', '/home/wego/limo_ws/src/lane_detection/models/best_cone.pt')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('conf_th', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('inference_period', 0.25)
        self.declare_parameter('show_window', True)

        self.declare_parameter('left_max_ratio', 0.38)
        self.declare_parameter('center_max_ratio', 0.62)
        self.declare_parameter('side_decision_min_cones', 2)
        self.declare_parameter('side_decision_min_ratio', 0.30)
        self.declare_parameter('side_decision_max_ratio', 0.70)
        self.declare_parameter('side_decision_deadband', 0.04)

        camera_topic = self.get_parameter('camera_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.device = self.resolve_device(str(self.get_parameter('device').value))
        self.conf_th = float(self.get_parameter('conf_th').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.inference_period = float(self.get_parameter('inference_period').value)
        self.show_window = bool(self.get_parameter('show_window').value)

        self.left_max_ratio = float(self.get_parameter('left_max_ratio').value)
        self.center_max_ratio = float(self.get_parameter('center_max_ratio').value)
        self.side_decision_min_cones = int(self.get_parameter('side_decision_min_cones').value)
        self.side_decision_min_ratio = float(self.get_parameter('side_decision_min_ratio').value)
        self.side_decision_max_ratio = float(self.get_parameter('side_decision_max_ratio').value)
        self.side_decision_deadband = float(self.get_parameter('side_decision_deadband').value)

        self.latest_msg = None
        self.last_inference_time = 0.0

        self.model = YOLO(self.model_path)

        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(self.inference_period, self.timer_callback)

        self.get_logger().info(
            f'cone yolo angle debug start. camera={camera_topic}, model={self.model_path}, '
            f'device={self.device}, conf={self.conf_th}, imgsz={self.imgsz}'
        )

    def resolve_device(self, device_param):
        device_param = device_param.strip().lower()

        if device_param == 'auto':
            if cuda is not None and cuda.is_available():
                return '0'
            return 'cpu'
        if device_param in ('cuda', 'gpu'):
            return '0'
        return device_param

    def image_callback(self, msg):
        self.latest_msg = msg

    def x_to_lane_by_split(self, x_center, width):
        ratio = float(x_center) / max(float(width), 1.0)
        if ratio < self.left_max_ratio:
            return 'left'
        if ratio < self.center_max_ratio:
            return 'center'
        return 'right'

    def classify_lanes_from_cones(self, cone_centers, width):
        if len(cone_centers) < self.side_decision_min_cones:
            return set(), None, []

        lanes = {'center'}
        img_width = max(float(width), 1.0)
        ratios = [float(cx) / img_width for cx in cone_centers]
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

        return lanes, pair_center_ratio, pair_ratios

    def format_lanes(self, lanes):
        order = ['left', 'center', 'right']
        return ','.join([lane for lane in order if lane in lanes])

    def timer_callback(self):
        if self.latest_msg is None:
            return

        now = time.monotonic()
        if now - self.last_inference_time < self.inference_period:
            return
        self.last_inference_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        try:
            use_half = str(self.device) != 'cpu'
            results_list = self.model.predict(
                source=frame,
                verbose=False,
                stream=False,
                conf=self.conf_th,
                device=self.device,
                imgsz=self.imgsz,
                half=use_half,
            )
        except Exception as e:
            self.get_logger().error(f'YOLO predict error: {e}')
            return

        debug = frame.copy()
        h, w = debug.shape[:2]
        cone_centers = []
        per_box = []

        if results_list:
            results = results_list[0].cpu()
            class_names = results.names

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls)
                    score = float(box.conf)
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    xywh = box.xywh[0]
                    cx = float(xywh[0])
                    cy = float(xywh[1])
                    lane = self.x_to_lane_by_split(cx, w)
                    ratio = cx / max(float(w), 1.0)

                    cone_centers.append(cx)
                    per_box.append((lane, ratio, score))

                    color = (0, 255, 255)
                    if lane == 'left':
                        color = (255, 80, 80)
                    elif lane == 'right':
                        color = (80, 255, 80)

                    cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(debug, (int(cx), int(cy)), 4, color, -1)
                    label = f'{str(class_names.get(cls_id, cls_id))} {score:.2f} {lane} x={ratio:.2f}'
                    cv2.putText(
                        debug,
                        label,
                        (int(x1), max(18, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        detected_lanes, pair_center_ratio, pair_ratios = self.classify_lanes_from_cones(cone_centers, w)

        self.draw_guides(debug, pair_center_ratio)

        if pair_center_ratio is None:
            pair_text = 'pair_center=none'
        else:
            pair_text = f'pair_center={pair_center_ratio:.3f}'

        lane_text = self.format_lanes(detected_lanes) if detected_lanes else 'none'
        box_text = ', '.join([f'{lane}:{ratio:.2f}/{score:.2f}' for lane, ratio, score in per_box])
        if not box_text:
            box_text = 'no boxes'

        cv2.putText(debug, f'blocked_lanes_like_yolo_node: {lane_text}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(debug, pair_text, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug, f'boxes: {box_text}', (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        self.get_logger().warn(
            f'blocked_lanes_like_yolo_node={lane_text}, {pair_text}, '
            f'pair_ratios={[round(x, 3) for x in pair_ratios]}, boxes=[{box_text}]'
        )

        if self.show_window:
            cv2.imshow('cone_yolo_angle_debug', debug)
            cv2.waitKey(1)

    def draw_guides(self, image, pair_center_ratio):
        h, w = image.shape[:2]
        left_x = int(w * self.left_max_ratio)
        center_x = int(w * 0.5)
        right_x = int(w * self.center_max_ratio)
        min_x = int(w * self.side_decision_min_ratio)
        max_x = int(w * self.side_decision_max_ratio)

        cv2.line(image, (left_x, 0), (left_x, h), (255, 80, 80), 1)
        cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        cv2.line(image, (right_x, 0), (right_x, h), (80, 255, 80), 1)
        cv2.line(image, (min_x, 0), (min_x, h), (120, 120, 120), 1)
        cv2.line(image, (max_x, 0), (max_x, h), (120, 120, 120), 1)

        if pair_center_ratio is not None:
            pair_x = int(w * pair_center_ratio)
            cv2.line(image, (pair_x, 0), (pair_x, h), (0, 255, 255), 2)


def main(args=None):
    rclpy.init(args=args)
    node = ConeYoloAngleDebugNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
