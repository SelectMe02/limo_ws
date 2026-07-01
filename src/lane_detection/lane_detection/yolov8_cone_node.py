import time

import cv2
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

    Important:
      This node latches cone lanes only after the mission FSM reaches ROTARY
      or CONE. Earlier detections are ignored and cleared, so cones seen in
      other parts of the track do not affect the cone mission.
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
        self.declare_parameter('latch_enable_states', ['ROTARY', 'CONE'])
        self.declare_parameter('clear_latch_when_disabled', True)

        self.declare_parameter('device', 'auto')          # auto, cpu, 0, cuda:0
        self.declare_parameter('conf_th', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('inference_period', 0.25)  # seconds. 0.25 = 4 Hz
        self.declare_parameter('publish_empty', True)

        # Image x-axis lane split ratios.
        # Tune these in RViz if cone lane classification is shifted.
        self.declare_parameter('left_max_ratio', 0.38)
        self.declare_parameter('center_max_ratio', 0.62)
        self.declare_parameter('side_decision_min_cones', 2)
        self.declare_parameter('side_decision_min_ratio', 0.30)
        self.declare_parameter('side_decision_max_ratio', 0.70)
        self.declare_parameter('side_decision_deadband', 0.04)

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
        self.gate_by_mission_state = bool(self.get_parameter('gate_by_mission_state').value)
        self.latch_enable_states = set([
            str(x).strip().upper()
            for x in self.get_parameter('latch_enable_states').value
        ])
        self.clear_latch_when_disabled = bool(self.get_parameter('clear_latch_when_disabled').value)

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

        self.timer = self.create_timer(self.inference_period, self.timer_callback)

        self.get_logger().info(
            f'yolov8 cone latch node start. model={self.model_path}, '
            f'camera={camera_topic}, device={self.device}, imgsz={self.imgsz}, '
            f'inference_period={self.inference_period}s, gate_by_state={self.gate_by_mission_state}, '
            f'latch_enable_states={sorted(list(self.latch_enable_states))}'
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

    def reset_callback(self, msg):
        # Any message resets latch.
        self.latched_lanes = set()
        self.last_detected_lanes = set()
        self.get_logger().warn(f'cone latch reset: {msg.data}')

    def mission_state_callback(self, msg):
        state = msg.data.strip().upper()
        prev_state = self.current_mission_state
        self.current_mission_state = state

        if self.is_latch_enabled():
            if not self.latch_armed_once:
                self.latch_armed_once = True
                self.latched_lanes = set()
                self.last_detected_lanes = set()
                self.get_logger().warn(
                    f'cone latch ARMED at mission state={state}. Old cone detections cleared.'
                )
        else:
            self.latch_armed_once = False
            if self.clear_latch_when_disabled and (self.latched_lanes or self.last_detected_lanes):
                self.latched_lanes = set()
                self.last_detected_lanes = set()
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
                self.latched_lanes = set()
                self.last_detected_lanes = set()
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
        cone_centers = []
        detections_msg = None

        if HAS_INTERFACES:
            detections_msg = DetectionArray()
            detections_msg.header = self.latest_msg.header

        if results_list:
            results = results_list[0].cpu()
            height, width = results.orig_img.shape[:2]
            class_names = results.names

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls)
                    score = float(box.conf)

                    xywh = box.xywh[0]
                    cx = float(xywh[0])
                    cy = float(xywh[1])
                    bw = float(xywh[2])
                    bh = float(xywh[3])

                    cone_centers.append(cx)

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

            detected_lanes = self.classify_lanes_from_cones(cone_centers, width)

        self.last_detected_lanes = detected_lanes

        # Latch: once seen, keep it.
        if detected_lanes:
            before = set(self.latched_lanes)
            self.latched_lanes |= detected_lanes
            if self.latched_lanes != before:
                self.get_logger().warn(
                    f'cone latch update: detected={self.format_lanes(detected_lanes)}, '
                    f'latched={self.format_lanes(self.latched_lanes)}'
                )

        if detections_msg is not None and self.detections_pub is not None:
            self.detections_pub.publish(detections_msg)

        self.publish_latched_lanes()

    def classify_lanes_from_cones(self, cone_centers, width):
        if len(cone_centers) < self.side_decision_min_cones:
            return set()

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

        return lanes

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
