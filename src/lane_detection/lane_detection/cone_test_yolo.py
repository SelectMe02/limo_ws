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
    """
    YOLO cone test node.

    - mission/state와 관계없이 항상 YOLO를 동작시킨다.
    - /cone/blocked_lanes를 항상 publish한다.
    - /cone/debug/compressed에 디버그 이미지를 publish한다.
    - yolov8_cone_node의 노란색 하단 가로선 기반 center cone 기준 알고리즘을 그대로 사용한다.

    핵심 로직:
      1. YOLO로 cone bbox 검출
      2. 같은 camera_raw 이미지에서 하단 노란색 가로선 검출
      3. 노란색 가로선의 x 기준 위치와 가장 가까운 cone을 center cone으로 판단
      4. center cone 기준으로 나머지 cone이 왼쪽이면 left, 오른쪽이면 right로 판단
      5. 테스트 노드에서는 latch/freeze를 사용하지 않고 현재 프레임 판단값을 바로 publish
    """

    def __init__(self):
        super().__init__()

        self.declare_parameter('debug_topic', '/cone/debug/compressed')
        debug_topic = self.get_parameter('debug_topic').value
        self.debug_pub = self.create_publisher(CompressedImage, debug_topic, 10)

        # 테스트 노드에서는 mission state gate를 완전히 끈다.
        # 따라서 ROTARY/CONE 상태가 아니어도 항상 YOLO가 동작한다.
        self.gate_by_mission_state = False
        self.clear_latch_when_disabled = False
        self.current_mission_state = 'CONE_TEST'
        self.latch_armed_once = True

        # 노란선 기반 center 판단을 사용할 때는 center-only -> center,left 자동 확정을 끈다.
        # 테스트 노드에서는 현재 프레임 판단값만 바로 publish하므로 frozen도 계속 False로 유지한다.
        self.center_only_assume_left_time = 999.0
        self.latch_frozen = False

        self.get_logger().warn(
            'cone_test_yolo start: mission-state gating disabled. '
            f'Publishing /cone/blocked_lanes and {debug_topic} whenever cones are detected.'
        )

    def timer_callback(self):
        if self.latest_msg is None:
            self.publish_latched_lanes()
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                self.latest_msg,
                desired_encoding='bgr8'
            )
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

        height, width = cv_image.shape[:2]

        if HAS_INTERFACES:
            detections_msg = DetectionArray()
            detections_msg.header = self.latest_msg.header

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

                    class_name = str(class_names.get(cls_id, f'class_{cls_id}'))

                    cone_infos.append({
                        'class_id': cls_id,
                        'class_name': class_name,
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
                        det.class_name = class_name
                        det.score = score
                        det.bbox.center.position.x = cx
                        det.bbox.center.position.y = cy
                        det.bbox.size.x = bw
                        det.bbox.size.y = bh
                        detections_msg.detections.append(det)

        raw_line_ref_x = None
        center_ref_x = None
        center_ref_source = 'none'

        # yolov8_cone_node의 노란색 하단 가로선 기반 center 기준 로직을 그대로 사용한다.
        if self.use_yellow_line_center_reference:
            raw_line_ref_x = self.detect_yellow_line_reference_x(
                cv_image,
                cone_infos
            )

            center_ref_x, center_ref_source = self.update_line_reference(
                raw_line_ref_x
            )

        detected_lanes = self.classify_lanes_from_cones(
            cone_infos=cone_infos,
            width=width,
            center_ref_x=center_ref_x,
            center_ref_source=center_ref_source,
        )

        self.last_detected_lanes = detected_lanes

        # 테스트 노드에서는 update_cone_latch()를 사용하지 않는다.
        # 이전 프레임에서 잘못 latch된 left/center/right가 남지 않도록
        # 현재 프레임의 판단 결과만 바로 /cone/blocked_lanes로 publish한다.
        before = set(self.latched_lanes)

        if detected_lanes:
            self.latched_lanes = set(detected_lanes)
        else:
            self.latched_lanes = set()

        self.latch_frozen = False
        self.center_only_start_time = None

        changed = self.latched_lanes != before

        if changed:
            self.get_logger().warn(
                f'cone test current-frame update: '
                f'detected={self.format_lanes(detected_lanes)}, '
                f'published={self.format_lanes(self.latched_lanes)}, '
                f'frozen={self.latch_frozen}, '
                f'line_ref={center_ref_x if center_ref_x is not None else -1:.1f}, '
                f'line_ref_source={center_ref_source}'
            )

        if detections_msg is not None and self.detections_pub is not None:
            self.detections_pub.publish(detections_msg)

        self.publish_debug_image(
            cv_image=cv_image,
            cone_infos=cone_infos,
            detected_lanes=detected_lanes,
            raw_line_ref_x=raw_line_ref_x,
            center_ref_x=center_ref_x,
            center_ref_source=center_ref_source,
        )

        self.publish_latched_lanes()

    def publish_debug_image(
        self,
        cv_image,
        cone_infos,
        detected_lanes,
        raw_line_ref_x,
        center_ref_x,
        center_ref_source,
    ):
        debug = cv_image.copy()
        h, w = debug.shape[:2]

        # 기존 화면 비율 기반 fallback 기준선 표시
        x_left = int(w * self.left_max_ratio)
        x_right = int(w * self.center_max_ratio)
        cv2.line(debug, (x_left, 0), (x_left, h), (255, 255, 255), 1)
        cv2.line(debug, (x_right, 0), (x_right, h), (255, 255, 255), 1)

        gate_left = int(w * self.side_decision_min_ratio)
        gate_right = int(w * self.side_decision_max_ratio)
        cv2.line(debug, (gate_left, 0), (gate_left, h), (180, 180, 0), 1)
        cv2.line(debug, (gate_right, 0), (gate_right, h), (180, 180, 0), 1)

        # 노란색 가로선 검출 ROI 표시
        roi_y0 = int(h * self.yellow_line_roi_y_min_ratio)
        roi_y1 = int(h * self.yellow_line_roi_y_max_ratio)
        roi_y0 = max(0, min(h - 1, roi_y0))
        roi_y1 = max(roi_y0 + 1, min(h, roi_y1))
        cv2.rectangle(
            debug,
            (0, roi_y0),
            (w - 1, roi_y1),
            (80, 80, 80),
            1,
        )

        # raw line reference 표시
        if raw_line_ref_x is not None:
            x_raw = int(raw_line_ref_x)
            cv2.line(debug, (x_raw, roi_y0), (x_raw, roi_y1), (0, 180, 255), 2)
            cv2.putText(
                debug,
                'raw yellow-line ref',
                (max(0, x_raw - 80), max(20, roi_y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

        # smoothed/stale center reference 표시
        if center_ref_x is not None:
            x_ref = int(center_ref_x)
            cv2.line(debug, (x_ref, 0), (x_ref, h), (0, 255, 255), 2)
            cv2.putText(
                debug,
                f'center ref: {center_ref_source}',
                (max(0, x_ref - 80), h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        center_cone_index = self.find_debug_center_cone_index(
            cone_infos,
            center_ref_x,
            w,
        )

        for idx, cone in enumerate(cone_infos):
            x1 = cone['x1']
            y1 = cone['y1']
            x2 = cone['x2']
            y2 = cone['y2']
            score = cone['score']
            class_name = cone['class_name']
            cx = cone['cx']
            cy = cone['cy']

            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))

            if idx == center_cone_index:
                box_color = (0, 255, 0)
                label_prefix = 'CENTER'
            else:
                box_color = (0, 255, 255)
                label_prefix = 'CONE'

            cv2.rectangle(debug, p1, p2, box_color, 2)
            cv2.circle(debug, (int(cx), int(cy)), 4, box_color, -1)

            cv2.putText(
                debug,
                f'{label_prefix} {class_name} {score:.2f}',
                (p1[0], max(18, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                box_color,
                2,
                cv2.LINE_AA,
            )

        detected_text = self.format_lanes(detected_lanes) or 'none'
        latched_text = self.format_lanes(self.latched_lanes) or 'none'

        if center_ref_x is None:
            ref_text = 'line_ref=none'
        else:
            ref_text = f'line_ref={center_ref_x:.1f} ({center_ref_source})'

        cv2.rectangle(debug, (0, 0), (w, 82), (0, 0, 0), -1)

        cv2.putText(
            debug,
            f'cone_test_yolo detected={detected_text} latched={latched_text}',
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug,
            f'{ref_text}  frozen={self.latch_frozen}',
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug,
            'mission gate disabled / yellow-line center reference enabled',
            (10, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(
                debug,
                dst_format='jpg'
            )
            msg.header = self.latest_msg.header
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'cone test debug publish error: {e}')

    def find_debug_center_cone_index(self, cone_infos, center_ref_x, image_width):
        """
        디버그 이미지에서 center cone으로 판단된 bbox를 표시하기 위한 함수.
        실제 판단 로직은 부모 클래스의 classify_lanes_with_center_reference()에서 수행된다.
        여기서는 시각화만 한다.
        """
        if not cone_infos:
            return None

        if center_ref_x is None:
            return None

        max_center_dx = image_width * self.center_match_max_dx_ratio

        best_index = None
        best_dx = None

        for idx, cone in enumerate(cone_infos):
            dx = abs(float(cone['cx']) - float(center_ref_x))

            if best_dx is None or dx < best_dx:
                best_dx = dx
                best_index = idx

        if best_dx is None:
            return None

        if best_dx > max_center_dx:
            return None

        return best_index


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