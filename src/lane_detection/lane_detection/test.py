import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist

from lane_detection.mission_fsm_node import (
    IMAGE_WIDTH,
    MAX_YAW_RATE,
    MIN_WHITE_PIXELS,
    Mission,
    StanleyMissionFSMNode,
    TOP_IGNORE_RATIO,
    poly_x,
)

try:
    from interfaces_pkg.msg import DetectionArray
    HAS_DETECTION_ARRAY = True
except Exception:
    DetectionArray = None
    HAS_DETECTION_ARRAY = False


class ConeOnlyTestNode(StanleyMissionFSMNode):
    """Run only the CONE avoidance behavior from the mission FSM."""

    def __init__(self):
        super().__init__()

        self.declare_parameter('cone_only_target_lane', 'right')
        self.declare_parameter('cone_only_use_yolo', True)
        self.declare_parameter('cone_only_finish_stop', False)
        self.declare_parameter('cone_detections_topic', '/detections')
        self.declare_parameter('cone_box_filter_margin_px', 30.0)
        self.declare_parameter('cone_box_filter_stale_sec', 0.45)
        self.declare_parameter('cone_box_filter_min_score', 0.0)
        self.declare_parameter('cone_initial_steer_time', 0.75)
        self.declare_parameter('cone_initial_steer_speed', 0.32)
        self.declare_parameter('cone_initial_steer_yaw', MAX_YAW_RATE)
        self.declare_parameter('cone_lidar_trigger_distance', 0.40)
        self.declare_parameter('cone_lidar_trigger_front_angle_deg', 50.0)
        self.declare_parameter('cone_approach_speed', 0.35)

        self.cone_only_target_lane = (
            str(self.get_parameter('cone_only_target_lane').value).strip().lower()
        )
        if self.cone_only_target_lane not in ('left', 'right'):
            self.cone_only_target_lane = 'right'

        self.cone_only_use_yolo = bool(self.get_parameter('cone_only_use_yolo').value)
        self.cone_only_finish_stop = bool(self.get_parameter('cone_only_finish_stop').value)
        self.cone_box_filter_margin_px = float(
            self.get_parameter('cone_box_filter_margin_px').value
        )
        self.cone_box_filter_stale_sec = float(
            self.get_parameter('cone_box_filter_stale_sec').value
        )
        self.cone_box_filter_min_score = float(
            self.get_parameter('cone_box_filter_min_score').value
        )
        self.cone_initial_steer_time = float(
            self.get_parameter('cone_initial_steer_time').value
        )
        self.cone_initial_steer_speed = float(
            self.get_parameter('cone_initial_steer_speed').value
        )
        self.cone_initial_steer_yaw = float(
            self.get_parameter('cone_initial_steer_yaw').value
        )
        self.cone_lidar_trigger_distance = float(
            self.get_parameter('cone_lidar_trigger_distance').value
        )
        self.cone_lidar_trigger_front_angle_deg = float(
            self.get_parameter('cone_lidar_trigger_front_angle_deg').value
        )
        self.cone_approach_speed = float(
            self.get_parameter('cone_approach_speed').value
        )
        self.cone_boxes = []
        self.cone_boxes_time = 0.0
        self.cone_box_filter_log_time = 0.0
        self.cone_test_completed = False
        self.cone_test_avoid_start_time = None
        self.cone_test_steer_start_time = None
        self.cone_test_lidar_triggered = False
        self.cone_test_single_acquired = False
        self.cone_test_active_target = None

        if HAS_DETECTION_ARRAY:
            detections_topic = self.get_parameter('cone_detections_topic').value
            self.cone_detection_sub = self.create_subscription(
                DetectionArray,
                detections_topic,
                self.cone_detections_callback,
                10
            )
        else:
            self.cone_detection_sub = None

        self.mission_order = [Mission.CONE]
        self.force_cone_state('startup')

        self.get_logger().warn(
            'cone only test node start. '
            f'target={self.cone_only_target_lane}, use_yolo={self.cone_only_use_yolo}, '
            f'finish_stop={self.cone_only_finish_stop}'
        )
        if not HAS_DETECTION_ARRAY:
            self.get_logger().warn(
                'interfaces_pkg DetectionArray not found. '
                'Cone bbox lane filtering is disabled.'
            )

    def cone_avoid_active(self):
        return (
            self.state == Mission.CONE and
            self.cone_latched and
            self.cone_target_lane in ('left', 'right') and
            not self.cone_test_completed
        )

    def reset_cone_test_avoid(self):
        self.cone_test_avoid_start_time = None
        self.cone_test_steer_start_time = None
        self.cone_test_lidar_triggered = False
        self.cone_test_single_acquired = False
        self.cone_test_active_target = None

    def ensure_cone_test_avoid_started(self):
        if not self.cone_avoid_active():
            return

        if (
            self.cone_test_avoid_start_time is None or
            self.cone_test_active_target != self.cone_target_lane
        ):
            self.cone_test_avoid_start_time = time.monotonic()
            self.cone_test_steer_start_time = None
            self.cone_test_lidar_triggered = False
            self.cone_test_single_acquired = False
            self.cone_test_active_target = self.cone_target_lane
            self.get_logger().warn(
                f'CONE_TEST_FORCE_START target={self.cone_target_lane}'
            )

    def update_cone_lidar_trigger(self):
        if not self.cone_avoid_active() or self.cone_test_lidar_triggered:
            return self.cone_test_lidar_triggered, 'front', 0, 9.9

        hits = 0
        nearest = 9.9
        angle_limit = self.cone_lidar_trigger_front_angle_deg

        for _, _, dist, angle in self.lidar_points:
            if dist > self.cone_lidar_trigger_distance:
                continue

            angle_deg = math.degrees(angle)
            while angle_deg > 180.0:
                angle_deg -= 360.0
            while angle_deg < -180.0:
                angle_deg += 360.0

            if -angle_limit <= angle_deg <= angle_limit:
                hits += 1
                nearest = min(nearest, dist)

        active = hits >= self.cone_lidar_min_hits
        if active and nearest <= self.cone_lidar_trigger_distance:
            self.cone_test_lidar_triggered = True
            self.cone_test_steer_start_time = time.monotonic()
            self.get_logger().warn(
                f'CONE_TEST_LIDAR_TRIGGER target={self.cone_target_lane} '
                f'front=-{angle_limit:.0f}~{angle_limit:.0f} '
                f'hits={hits} d={nearest:.2f}m'
            )

        return self.cone_test_lidar_triggered, 'front', hits, nearest

    def cone_detections_callback(self, msg):
        boxes = []
        for det in msg.detections:
            score = float(getattr(det, 'score', 0.0))
            if score < self.cone_box_filter_min_score:
                continue

            bbox = det.bbox
            cx = float(bbox.center.position.x)
            cy = float(bbox.center.position.y)
            bw = float(bbox.size.x)
            bh = float(bbox.size.y)
            if bw <= 1.0 or bh <= 1.0:
                continue

            boxes.append({
                'cx': cx,
                'cy': cy,
                'x1': cx - bw * 0.5,
                'x2': cx + bw * 0.5,
                'y1': cy - bh * 0.5,
                'y2': cy + bh * 0.5,
                'score': score,
            })

        self.cone_boxes = boxes
        self.cone_boxes_time = time.monotonic()

    def recent_cone_boxes(self):
        if time.monotonic() - self.cone_boxes_time > self.cone_box_filter_stale_sec:
            return []
        return self.cone_boxes

    def fit_near_cone_box(self, fit, height, boxes):
        margin = self.cone_box_filter_margin_px
        sample_ys = (
            height - 1,
            int(height * 0.78),
            int(height * 0.62),
            int(height * 0.48),
        )

        for box in boxes:
            x1 = box['x1'] - margin
            x2 = box['x2'] + margin
            for y in sample_ys:
                x = poly_x(fit, y)
                if x1 <= x <= x2:
                    return True

        return False

    def filter_cone_box_fit_candidates(self, fit_candidates, height):
        if self.state != Mission.CONE:
            return fit_candidates

        boxes = self.recent_cone_boxes()
        if len(boxes) == 0:
            return fit_candidates

        filtered = [
            item for item in fit_candidates
            if not self.fit_near_cone_box(item['fit'], height, boxes)
        ]
        removed = len(fit_candidates) - len(filtered)

        if removed > 0:
            now = time.monotonic()
            if now - self.cone_box_filter_log_time > 0.8:
                self.cone_box_filter_log_time = now
                self.get_logger().warn(
                    f'CONE_TEST_BOX_FILTER removed={removed} '
                    f'kept={len(filtered)} boxes={len(boxes)} '
                    f'margin={self.cone_box_filter_margin_px:.0f}px'
                )

        return filtered

    def build_fit_candidates(self, binary, candidates, height):
        fit_candidates = super().build_fit_candidates(binary, candidates, height)
        return self.filter_cone_box_fit_candidates(fit_candidates, height)

    def choose_cone_inner_single(
        self,
        fit_candidates,
        target_lane,
        image_center,
        reference_x=None,
        strict_outer=False,
    ):
        if (
            self.state != Mission.CONE or
            not strict_outer or
            target_lane not in ('left', 'right') or
            len(fit_candidates) == 0 or
            len(self.recent_cone_boxes()) == 0
        ):
            return super().choose_cone_inner_single(
                fit_candidates,
                target_lane,
                image_center,
                reference_x,
                strict_outer
            )

        ordered = sorted(fit_candidates, key=lambda item: item['x_bottom'])

        if target_lane == 'left':
            left_side = [item for item in ordered if item['x_bottom'] <= IMAGE_WIDTH * 0.5]
            single = left_side[0] if len(left_side) > 0 else ordered[0]
            return single, 'left'

        right_side = [item for item in ordered if item['x_bottom'] >= IMAGE_WIDTH * 0.5]
        single = right_side[-1] if len(right_side) > 0 else ordered[-1]
        return single, 'right'

    def get_lane_by_sliding_window(self, mask):
        if not self.cone_avoid_active():
            return super().get_lane_by_sliding_window(mask)

        self.ensure_cone_test_avoid_started()
        lidar_triggered, _, _, _ = self.update_cone_lidar_trigger()
        if not lidar_triggered:
            return None

        if (
            self.cone_test_steer_start_time is not None and
            not self.cone_test_single_acquired and
            time.monotonic() - self.cone_test_steer_start_time < self.cone_initial_steer_time
        ):
            return None

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

        if self.cone_target_lane == 'left':
            base_candidates = left_base_candidates
            lane_side = 'left'
        else:
            base_candidates = right_base_candidates
            lane_side = 'right'

        fit_candidates = self.build_fit_candidates(binary, base_candidates, h)
        if len(fit_candidates) == 0:
            return None

        if lane_side == 'left':
            side_candidates = [
                item for item in fit_candidates
                if item['x_bottom'] <= image_center
            ]
            single = min(
                side_candidates if len(side_candidates) > 0 else fit_candidates,
                key=lambda item: item['x_bottom'] - 0.002 * item['count']
            )
        else:
            side_candidates = [
                item for item in fit_candidates
                if item['x_bottom'] >= image_center
            ]
            single = max(
                side_candidates if len(side_candidates) > 0 else fit_candidates,
                key=lambda item: item['x_bottom'] + 0.002 * item['count']
            )

        single_fit = single['fit']
        center_fit = np.array(single_fit, dtype=np.float64)

        if lane_side == 'left':
            center_fit[2] += self.last_lane_width / 2.0
            left_fit = single_fit
            right_fit = None
        else:
            center_fit[2] -= self.last_lane_width / 2.0
            left_fit = None
            right_fit = single_fit

        if not self.cone_test_single_acquired:
            self.cone_test_single_acquired = True
            self.last_single_side = lane_side
            self.get_logger().warn(
                f'CONE_TEST_SINGLE_ACQUIRED side={lane_side} '
                f'x={single["x_bottom"]:.1f}'
            )

        return {
            'center_fit': center_fit,
            'left_fit': left_fit,
            'right_fit': right_fit,
            'confidence': 0.82,
            'mode': f'cone_test_single_{lane_side}',
            'side': lane_side,
            'total_pixels': total_pixels,
            'measured_lane_width': None,
            'lane_width_calibrated': self.lane_width_calibrated,
            'lane_width_sample_count': len(self.lane_width_samples),
            'raw_candidates': raw_candidates,
            'filtered_candidates': filtered_candidates,
            'left_candidates': left_base_candidates,
            'right_candidates': right_base_candidates,
        }

    def clear_cone_latch_for_continue(self):
        self.cone_latched = False
        self.cone_latched_lanes = set()
        self.cone_target_lane = 'center'
        self.cone_target_locked = False
        self.cone_target_votes = []
        self.cone_first_latch_time = None
        self.last_cone_msg_time = 0.0
        self.cone_recover_start_time = None
        self.cone_lidar_clear_count = 0
        self.reset_cone_test_avoid()

    def cone_callback(self, msg):
        if self.cone_test_completed and not self.cone_only_finish_stop:
            return

        before_target = self.cone_target_lane
        super().cone_callback(msg)
        if self.cone_avoid_active() and before_target != self.cone_target_lane:
            self.reset_cone_test_avoid()

    def handle_cone(self, lane_data):
        if self.cone_test_completed and not self.cone_only_finish_stop:
            cmd, _ = self.stanley_cmd_from_lane(lane_data)
            return cmd, 'CONE_ONLY_CONTINUE_LANE'

        if not self.cone_avoid_active():
            return super().handle_cone(lane_data)

        self.ensure_cone_test_avoid_started()
        lidar_triggered, cone_side, cone_hits, cone_nearest = self.update_cone_lidar_trigger()

        close_cluster = self.obstacle_in_corridor(
            0.12,
            0.55,
            self.drivable_half_width,
            min_points=3,
            ignore_wall=True,
        )
        if close_cluster is not None and close_cluster.nearest < 0.25:
            return self.stop_cmd(), 'CONE_SAFE_STOP'

        if not lidar_triggered:
            cmd = Twist()
            cmd.linear.x = float(max(0.0, min(self.cone_approach_speed, self.cone_speed_limit)))
            cmd.angular.z = 0.0
            self.prev_speed = cmd.linear.x
            self.current_lane_bias_px = 0.0
            return cmd, (
                f'CONE_TEST_APPROACH_STRAIGHT target={self.cone_target_lane} '
                f'lidar_{cone_side}={cone_hits}/{self.cone_lidar_min_hits} '
                f'd={cone_nearest:.2f}/{self.cone_lidar_trigger_distance:.2f}'
            )

        if not self.cone_test_single_acquired or lane_data is None:
            cmd = Twist()
            cmd.linear.x = float(max(0.0, min(self.cone_initial_steer_speed, self.cone_speed_limit)))
            steer_sign = 1.0 if self.cone_target_lane == 'left' else -1.0
            force_yaw = min(abs(self.cone_initial_steer_yaw), MAX_YAW_RATE)
            cmd.angular.z = float(steer_sign * force_yaw)
            self.prev_speed = cmd.linear.x
            self.current_lane_bias_px = 0.0
            return cmd, (
                f'CONE_TEST_FORCE_MAX_{self.cone_target_lane.upper()} '
                f'yaw={cmd.angular.z:.2f}'
            )

        if self.cone_first_latch_time is not None:
            latch_time_ok = (time.monotonic() - self.cone_first_latch_time) >= self.cone_latched_min_time
            state_time_ok = self.state_elapsed() >= self.cone_min_time

            if latch_time_ok and state_time_ok and self.cone_recover_start_time is None:
                self.cone_recover_start_time = time.monotonic()

        cmd, _ = self.stanley_cmd_from_lane(
            lane_data,
            speed_limit=self.cone_speed_limit,
            extra_yaw=0.0,
            lane_bias_px=0.0,
        )

        if self.cone_recover_start_time is not None:
            recover_elapsed = time.monotonic() - self.cone_recover_start_time
            lane_ok = lane_data.get('confidence', 0.0) >= self.cone_recover_confidence
            cone_side_active, cone_side, cone_hits, cone_nearest = self.cone_lidar_side_obstacle()

            if cone_side_active:
                self.cone_lidar_clear_count = 0
            else:
                self.cone_lidar_clear_count += 1

            lidar_clear_ok = self.cone_lidar_clear_count >= self.cone_lidar_clear_frames

            if recover_elapsed >= self.cone_recover_time and lane_ok and lidar_clear_ok:
                self.next_state('(cone test single lane done)')

            return cmd, (
                f'CONE_TEST_RECOVER_SINGLE_{self.cone_target_lane.upper()} '
                f'mode={lane_data.get("mode", "none")} '
                f'lidar_{cone_side}={cone_hits}/{self.cone_lidar_min_hits} '
                f'clear={self.cone_lidar_clear_count}/{self.cone_lidar_clear_frames} '
                f'd={cone_nearest:.2f}'
            )

        return cmd, (
            f'CONE_TEST_SINGLE_{self.cone_target_lane.upper()} '
            f'mode={lane_data.get("mode", "none")}'
        )

    def force_cone_state(self, reason=''):
        self.state = Mission.CONE
        self.state_enter_time = time.monotonic()
        self.cone_test_completed = False
        self.reset_cone_test_avoid()
        self.cone_recover_start_time = None
        self.cone_lidar_clear_count = 0

        if self.cone_only_use_yolo:
            self.cone_latched = False
            self.cone_latched_lanes = set()
            self.cone_target_lane = 'center'
            self.cone_target_locked = False
            self.cone_target_votes = []
            self.cone_first_latch_time = None
            self.last_cone_msg_time = 0.0
            self.get_logger().warn(f'CONE_ONLY_WAIT_YOLO {reason}')
            return

        if self.cone_only_target_lane == 'right':
            self.cone_latched_lanes = {'center', 'left'}
        else:
            self.cone_latched_lanes = {'center', 'right'}

        now = time.monotonic()
        self.cone_latched = True
        self.cone_target_lane = self.cone_only_target_lane
        self.cone_target_locked = True
        self.cone_target_votes = [self.cone_only_target_lane]
        self.cone_first_latch_time = now
        self.last_cone_msg_time = now
        self.ensure_cone_test_avoid_started()
        self.get_logger().warn(
            f'CONE_ONLY_FORCE target={self.cone_target_lane} '
            f'latched={sorted(list(self.cone_latched_lanes))} {reason}'
        )

    def set_state(self, new_state, reason=''):
        if new_state == Mission.CONE:
            self.force_cone_state(reason)
            return

        if new_state == Mission.FINISHED and self.cone_only_finish_stop:
            super().set_state(Mission.FINISHED, reason)
            return

        self.cone_test_completed = True
        self.clear_cone_latch_for_continue()
        self.get_logger().warn(f'CONE_ONLY_CONTINUE ignore transition to {new_state} {reason}')

    def next_state(self, reason=''):
        if self.cone_only_finish_stop:
            super().set_state(Mission.FINISHED, f'(cone only complete) {reason}')
        else:
            self.cone_test_completed = True
            self.clear_cone_latch_for_continue()
            self.get_logger().warn(f'CONE_ONLY_COMPLETE_CONTINUE {reason}')


def main(args=None):
    rclpy.init(args=args)
    node = ConeOnlyTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
