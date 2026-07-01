import os
from collections import deque, Counter

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool

from ultralytics import YOLO

try:
    from torch import cuda
except Exception:
    cuda = None


class Yolov8ConeNode(Node):
    """
    Cone-only YOLO node for the LIMO track mission.

    Input:
      /camera/color/image_raw        sensor_msgs/Image

    Output:
      /cone/blocked_lanes            std_msgs/String
          examples: "center,left", "center,right", "left", "right", ""

      /cone/debug/compressed         sensor_msgs/CompressedImage

    The mission_fsm_node subscribes /cone/blocked_lanes and selects the free lane.
    """

    def __init__(self):
        super().__init__('yolov8_cone_node')

        self.bridge = CvBridge()

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'model_path',
            '/home/wego/limo_ws/src/lane_detection/models/best_cone.pt'
        )
        self.declare_parameter('device', 'cpu')        # auto, cpu, cuda, 0
        self.declare_parameter('conf_th', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('enable', True)

        self.declare_parameter('blocked_lanes_topic', '/cone/blocked_lanes')
        self.declare_parameter('debug_topic', '/cone/debug/compressed')

        # Image lane split ratio: left | center | right
        self.declare_parameter('left_max_ratio', 0.33)
        self.declare_parameter('center_max_ratio', 0.67)

        # Ignore tiny far/false boxes.
        self.declare_parameter('min_box_area', 250.0)
        self.declare_parameter('min_box_height', 12.0)

        # Optional ROI. Cone mission cones usually appear in lower/middle image.
        # 0.0 means top of image, 1.0 means bottom.
        self.declare_parameter('roi_y_min_ratio', 0.25)
        self.declare_parameter('roi_y_max_ratio', 1.00)

        # Debounce: lane must appear repeatedly before it is treated as blocked.
        self.declare_parameter('history_size', 3)
        self.declare_parameter('min_votes', 2)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.device_param = self.get_parameter('device').value
        self.conf_th = float(self.get_parameter('conf_th').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.enable = bool(self.get_parameter('enable').value)

        self.blocked_lanes_topic = self.get_parameter('blocked_lanes_topic').value
        self.debug_topic = self.get_parameter('debug_topic').value

        self.left_max_ratio = float(self.get_parameter('left_max_ratio').value)
        self.center_max_ratio = float(self.get_parameter('center_max_ratio').value)
        self.min_box_area = float(self.get_parameter('min_box_area').value)
        self.min_box_height = float(self.get_parameter('min_box_height').value)
        self.roi_y_min_ratio = float(self.get_parameter('roi_y_min_ratio').value)
        self.roi_y_max_ratio = float(self.get_parameter('roi_y_max_ratio').value)

        history_size = int(self.get_parameter('history_size').value)
        self.min_votes = int(self.get_parameter('min_votes').value)
        self.history = deque(maxlen=max(1, history_size))

        self.device = self.resolve_device(self.device_param)

        if not os.path.exists(self.model_path):
            self.get_logger().warn(
                f'model file not found now: {self.model_path}. '
                'If the path is wrong, pass -p model_path:=...'
            )

        self.get_logger().info(f'loading cone model: {self.model_path}')
        self.model = YOLO(self.model_path)

        self.blocked_pub = self.create_publisher(String, self.blocked_lanes_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, 10)

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.enable_srv = self.create_service(SetBool, '~/enable', self.enable_callback)

        self.get_logger().info(
            f'yolov8 cone node start: camera={self.camera_topic}, '
            f'blocked_topic={self.blocked_lanes_topic}, device={self.device}'
        )

    def resolve_device(self, value):
        text = str(value).strip().lower()
        if text == 'auto':
            if cuda is not None and cuda.is_available():
                return 0
            return 'cpu'
        if text in ('cuda', 'gpu'):
            if cuda is not None and cuda.is_available():
                return 0
            self.get_logger().warn('CUDA requested but unavailable. fallback to cpu.')
            return 'cpu'
        if text in ('0', 'cuda:0', 'gpu:0'):
            return 0
        return text

    def enable_callback(self, request, response):
        self.enable = bool(request.data)
        response.success = True
        response.message = f'enable={self.enable}'
        return response

    def x_to_lane(self, x_center, image_width):
        ratio = x_center / max(float(image_width), 1.0)
        if ratio < self.left_max_ratio:
            return 'left'
        if ratio < self.center_max_ratio:
            return 'center'
        return 'right'

    def stable_lanes(self, current_lanes):
        self.history.append(tuple(sorted(current_lanes)))

        votes = Counter()
        for lanes in self.history:
            for lane in lanes:
                votes[lane] += 1

        stable = set()
        for lane in ('left', 'center', 'right'):
            if votes[lane] >= self.min_votes:
                stable.add(lane)

        return stable

    def publish_lanes(self, lanes):
        order = ['left', 'center', 'right']
        msg = String()
        msg.data = ','.join([lane for lane in order if lane in lanes])
        self.blocked_pub.publish(msg)
        return msg.data

    def image_callback(self, msg):
        if not self.enable:
            self.publish_lanes(set())
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        h, w = img.shape[:2]
        y_min = int(h * self.roi_y_min_ratio)
        y_max = int(h * self.roi_y_max_ratio)

        current_lanes = set()
        debug = img.copy()

        try:
            results_list = self.model.predict(
                source=img,
                verbose=False,
                stream=False,
                conf=self.conf_th,
                imgsz=self.imgsz,
                device=self.device,
            )
        except Exception as e:
            self.get_logger().error(f'yolo predict error: {e}')
            self.publish_lanes(set())
            return

        if results_list:
            results = results_list[0].cpu()
            boxes = results.boxes

            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [float(v) for v in xyxy]

                    bw = x2 - x1
                    bh = y2 - y1
                    area = bw * bh
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5

                    if area < self.min_box_area:
                        continue
                    if bh < self.min_box_height:
                        continue
                    if not (y_min <= cy <= y_max):
                        continue

                    lane = self.x_to_lane(cx, w)
                    current_lanes.add(lane)

                    color = (0, 255, 255)
                    if lane == 'left':
                        color = (255, 0, 0)
                    elif lane == 'center':
                        color = (0, 255, 255)
                    elif lane == 'right':
                        color = (0, 255, 0)

                    cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        debug,
                        f'{lane} {conf:.2f}',
                        (int(x1), max(20, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

        stable = self.stable_lanes(current_lanes)
        lane_text = self.publish_lanes(stable)

        # Draw lane split and ROI.
        x_left = int(w * self.left_max_ratio)
        x_right = int(w * self.center_max_ratio)
        cv2.line(debug, (x_left, 0), (x_left, h), (255, 255, 255), 1)
        cv2.line(debug, (x_right, 0), (x_right, h), (255, 255, 255), 1)
        cv2.rectangle(debug, (0, y_min), (w - 1, y_max - 1), (255, 255, 255), 1)
        cv2.putText(
            debug,
            f'blocked={lane_text if lane_text else "none"}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        try:
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug, dst_format='jpg')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'debug publish error: {e}')

        try:
            cv2.imshow('yolov8_cone_debug', debug)
            cv2.waitKey(1)
        except Exception:
            pass

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8ConeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()