import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class Yolov8VisualizerNode(Node):
    """
    Visualizer matched to the current yolov8_cone_node.

    Current yolov8_cone_node publishes:
      - /cone/debug/compressed   sensor_msgs/msg/CompressedImage
        Already contains YOLO bounding boxes, lane split lines, ROI, and blocked lane text.
      - /cone/blocked_lanes      std_msgs/msg/String
        Examples: "center,left", "center,right", "left", "right", ""

    This visualizer converts /cone/debug/compressed into /dbg_image so RViz2 can show it
    using the normal Image display.
    """

    def __init__(self):
        super().__init__('yolov8_visualizer_node')

        self.bridge = CvBridge()

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('debug_image_topic', '/cone/debug/compressed')
        self.declare_parameter('blocked_lanes_topic', '/cone/blocked_lanes')
        self.declare_parameter('output_image_topic', '/dbg_image')
        self.declare_parameter('output_compressed_topic', '/dbg_image/compressed')
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('show_window', False)
        self.declare_parameter('draw_extra_text', True)

        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.blocked_lanes_topic = self.get_parameter('blocked_lanes_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.output_compressed_topic = self.get_parameter('output_compressed_topic').value
        self.publish_compressed = bool(self.get_parameter('publish_compressed').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.draw_extra_text = bool(self.get_parameter('draw_extra_text').value)

        self.last_blocked_lanes = ''

        # -------------------------
        # Publishers
        # -------------------------
        self.dbg_pub = self.create_publisher(
            Image,
            self.output_image_topic,
            10,
        )

        self.dbg_compressed_pub = None
        if self.publish_compressed:
            self.dbg_compressed_pub = self.create_publisher(
                CompressedImage,
                self.output_compressed_topic,
                10,
            )

        # -------------------------
        # Subscribers
        # -------------------------
        self.debug_sub = self.create_subscription(
            CompressedImage,
            self.debug_image_topic,
            self.debug_image_callback,
            qos_profile_sensor_data,
        )

        self.blocked_sub = self.create_subscription(
            String,
            self.blocked_lanes_topic,
            self.blocked_lanes_callback,
            10,
        )

        self.get_logger().info(
            'yolov8 visualizer node start: '
            f'input_debug={self.debug_image_topic}, '
            f'blocked={self.blocked_lanes_topic}, '
            f'output={self.output_image_topic}'
        )

    def blocked_lanes_callback(self, msg: String):
        self.last_blocked_lanes = msg.data.strip()

    def draw_overlay(self, image):
        if not self.draw_extra_text:
            return image

        blocked_text = self.last_blocked_lanes if self.last_blocked_lanes else 'none'

        # Semi-transparent header background.
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

        cv2.putText(
            image,
            f'YOLO cone debug | blocked_lanes={blocked_text}',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return image

    def debug_image_callback(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
        except Exception as e:
            self.get_logger().error(f'compressed image bridge error: {e}')
            return

        cv_image = self.draw_overlay(cv_image)

        # Publish raw Image for RViz2 Image display.
        try:
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = msg.header
            self.dbg_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'raw debug publish error: {e}')

        # Optional compressed output for rqt_image_view or network saving.
        if self.dbg_compressed_pub is not None:
            try:
                compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                    cv_image,
                    dst_format='jpg',
                )
                compressed_msg.header = msg.header
                self.dbg_compressed_pub.publish(compressed_msg)
            except Exception as e:
                self.get_logger().warn(f'compressed debug publish error: {e}')

        if self.show_window:
            try:
                cv2.imshow('yolov8_visualizer_debug', cv_image)
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
    node = Yolov8VisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
