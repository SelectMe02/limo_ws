import cv2
import rclpy
from sensor_msgs.msg import CompressedImage

from lane_detection.yolov8_cone_node import (
    Detection,
    DetectionArray,
    HAS_INTERFACES,
    Yolov8ConeLatchNode,
)


class ConeTestYoloNode(Yolov8ConeLatchNode):
    """YOLO cone node that always publishes cone lane detections for testing."""

    def __init__(self):
        super().__init__()

        self.declare_parameter('debug_topic', '/cone/debug/compressed')
        debug_topic = self.get_parameter('debug_topic').value
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)

        self.gate_by_mission_state = False
        self.clear_latch_when_disabled = False
        self.current_mission_state = 'CONE_TEST'
        self.latch_armed_once = True

        self.get_logger().warn(
            'cone_test_yolo start: mission-state gating disabled. '
            f'Publishing /cone/blocked_lanes and {debug_topic} whenever cones are detected.'
        )

    def timer_callback(self):
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
        debug_boxes = []
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
                    x1 = float(box.xyxy[0][0])
                    y1 = float(box.xyxy[0][1])
                    x2 = float(box.xyxy[0][2])
                    y2 = float(box.xyxy[0][3])
                    class_name = str(class_names.get(cls_id, f'class_{cls_id}'))

                    cone_centers.append(cx)
                    debug_boxes.append((x1, y1, x2, y2, score, class_name))

                    if HAS_INTERFACES:
                        det = Detection()
                        det.class_id = cls_id
                        det.class_name = class_name
                        det.score = score
                        det.bbox.center.position.x = cx
                        det.bbox.center.position.y = cy
                        det.bbox.size.x = bw
                        det.bbox.size.y = bh
                        detections_msg.detections.append(det)

            detected_lanes = self.classify_lanes_from_cones(cone_centers, width)

        self.last_detected_lanes = detected_lanes

        if detected_lanes:
            before = set(self.latched_lanes)
            self.latched_lanes |= detected_lanes
            if self.latched_lanes != before:
                self.get_logger().warn(
                    f'cone test latch update: detected={self.format_lanes(detected_lanes)}, '
                    f'latched={self.format_lanes(self.latched_lanes)}'
                )

        if detections_msg is not None and self.detections_pub is not None:
            self.detections_pub.publish(detections_msg)

        self.publish_debug_image(cv_image, debug_boxes, detected_lanes)
        self.publish_latched_lanes()

    def publish_debug_image(self, cv_image, debug_boxes, detected_lanes):
        debug = cv_image.copy()
        h, w = debug.shape[:2]

        x_left = int(w * self.left_max_ratio)
        x_right = int(w * self.center_max_ratio)
        cv2.line(debug, (x_left, 0), (x_left, h), (255, 255, 255), 1)
        cv2.line(debug, (x_right, 0), (x_right, h), (255, 255, 255), 1)

        gate_left = int(w * self.side_decision_min_ratio)
        gate_right = int(w * self.side_decision_max_ratio)
        cv2.line(debug, (gate_left, 0), (gate_left, h), (180, 180, 0), 1)
        cv2.line(debug, (gate_right, 0), (gate_right, h), (180, 180, 0), 1)

        for x1, y1, x2, y2, score, class_name in debug_boxes:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(debug, p1, p2, (0, 255, 255), 2)
            cv2.putText(
                debug,
                f'{class_name} {score:.2f}',
                (p1[0], max(18, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        detected_text = self.format_lanes(detected_lanes) or 'none'
        latched_text = self.format_lanes(self.latched_lanes) or 'none'
        cv2.rectangle(debug, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.putText(
            debug,
            f'cone_test_yolo detected={detected_text} latched={latched_text}',
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            'mission gate disabled',
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            2,
            cv2.LINE_AA,
        )

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(debug, dst_format='jpg')
            msg.header = self.latest_msg.header
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'cone test debug publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ConeTestYoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
