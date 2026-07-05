import math
import time

import rclpy
from geometry_msgs.msg import Twist

from lane_detection.mission_fsm_node import (
    MAX_YAW_RATE,
    Mission,
    StanleyMissionFSMNode,
)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_deg(angle_deg):
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def angle_in_range(angle_deg, min_deg, max_deg):
    """Return True when angle_deg is inside [min_deg, max_deg]. Supports wrap-around."""
    angle_deg = normalize_deg(angle_deg)
    min_deg = normalize_deg(min_deg)
    max_deg = normalize_deg(max_deg)

    if min_deg <= max_deg:
        return min_deg <= angle_deg <= max_deg

    return angle_deg >= min_deg or angle_deg <= max_deg


class ConeOnlyTestNode(StanleyMissionFSMNode):
    """
    아주 단순한 CONE 전용 테스트 노드.

    동작 요약:
      1. CONE 상태에서 YOLO latch로 cone_target_lane이 left/right가 되면 회피 시작.
         - center + right cone  -> cone_target_lane = left  -> 왼쪽으로 강제 조향
         - center + left cone   -> cone_target_lane = right -> 오른쪽으로 강제 조향
      2. force_duration 동안 현재 차선 추종을 무시하고 정해진 방향으로 cmd_vel을 직접 발행.
      3. force_duration 이후에는 다시 일반 차선 추종으로 복귀.
      4. 통과 판단:
         - 왼쪽으로 회피할 때는 오른쪽 LiDAR sector에서 50cm 이내 cone cluster 2개 이상을 본 적이 있고,
           이후 그 cluster가 사라진 상태가 clear_frames만큼 연속되면 통과.
         - 오른쪽으로 회피할 때는 왼쪽 LiDAR sector에 대해 동일하게 판단.
    """

    def declare_if_not_declared(self, name, value):
        if not self.has_parameter(name):
            self.declare_parameter(name, value)

    def __init__(self):
        super().__init__()

        # -------------------------
        # Test / compatibility params
        # -------------------------
        self.declare_if_not_declared('cone_only_target_lane', 'right')
        self.declare_if_not_declared('cone_only_use_yolo', True)
        self.declare_if_not_declared('cone_only_finish_stop', False)

        # -------------------------
        # Simple forced-shift params
        # -------------------------
        self.declare_if_not_declared('cone_force_duration', 2.00)
        self.declare_if_not_declared('cone_force_speed', 0.40)
        self.declare_if_not_declared('cone_force_yaw', 0.45)
        self.declare_if_not_declared('cone_after_force_speed_limit', 0.45)

        # 일반적으로 /cmd_vel.angular.z 양수가 왼쪽 회전이다.
        # 실제 차량이 반대로 움직이면 False로 바꿔서 테스트한다.
        self.declare_if_not_declared('cone_left_yaw_positive', True)

        # -------------------------
        # LiDAR pass-detection params
        # -------------------------
        # 사용자가 요청한 기준: 오른쪽 0~120도, 50cm 이내.
        # 반대 방향은 왼쪽 -120~0도로 둔다.
        # 실제 라이다 좌표계가 반대면 아래 angle parameter만 바꾸면 된다.
        self.declare_if_not_declared('cone_side_distance', 0.50)
        self.declare_if_not_declared('cone_side_clear_frames', 3)
        self.declare_if_not_declared('cone_side_min_clusters', 2)
        self.declare_if_not_declared('cone_side_cluster_gap_m', 0.12)
        self.declare_if_not_declared('cone_side_cluster_min_points', 1)

        self.declare_if_not_declared('cone_right_angle_min_deg', 0.0)
        self.declare_if_not_declared('cone_right_angle_max_deg', 120.0)
        self.declare_if_not_declared('cone_left_angle_min_deg', -120.0)
        self.declare_if_not_declared('cone_left_angle_max_deg', 0.0)

        # -------------------------
        # Read params
        # -------------------------
        self.cone_only_target_lane = str(
            self.get_parameter('cone_only_target_lane').value
        ).strip().lower()
        if self.cone_only_target_lane not in ('left', 'right'):
            self.cone_only_target_lane = 'right'

        self.cone_only_use_yolo = bool(self.get_parameter('cone_only_use_yolo').value)
        self.cone_only_finish_stop = bool(self.get_parameter('cone_only_finish_stop').value)

        self.cone_force_duration = float(self.get_parameter('cone_force_duration').value)
        self.cone_force_speed = float(self.get_parameter('cone_force_speed').value)
        self.cone_force_yaw = float(self.get_parameter('cone_force_yaw').value)
        self.cone_after_force_speed_limit = float(
            self.get_parameter('cone_after_force_speed_limit').value
        )

        # cone_force_speed를 0 이하로 주면 after_force 속도와 동일하게 사용한다.
        if self.cone_force_speed <= 0.0:
            self.cone_force_speed = self.cone_after_force_speed_limit

        self.cone_left_yaw_positive = bool(
            self.get_parameter('cone_left_yaw_positive').value
        )

        self.cone_side_distance = float(self.get_parameter('cone_side_distance').value)
        self.cone_side_clear_frames = int(
            self.get_parameter('cone_side_clear_frames').value
        )
        self.cone_side_min_clusters = int(
            self.get_parameter('cone_side_min_clusters').value
        )
        self.cone_side_cluster_gap_m = float(
            self.get_parameter('cone_side_cluster_gap_m').value
        )
        self.cone_side_cluster_min_points = int(
            self.get_parameter('cone_side_cluster_min_points').value
        )
        self.cone_right_angle_min_deg = float(
            self.get_parameter('cone_right_angle_min_deg').value
        )
        self.cone_right_angle_max_deg = float(
            self.get_parameter('cone_right_angle_max_deg').value
        )
        self.cone_left_angle_min_deg = float(
            self.get_parameter('cone_left_angle_min_deg').value
        )
        self.cone_left_angle_max_deg = float(
            self.get_parameter('cone_left_angle_max_deg').value
        )

        self.cone_test_completed = False
        self.reset_simple_cone_vars()

        # 이 파일은 CONE만 단독 테스트한다.
        self.mission_order = [Mission.CONE]
        self.force_cone_state('startup')

        self.get_logger().warn(
            'simple cone force-shift test start. '
            f'use_yolo={self.cone_only_use_yolo}, '
            f'target={self.cone_only_target_lane}, '
            f'force_duration={self.cone_force_duration:.2f}s, '
            f'force_speed={self.cone_force_speed:.2f}, '
            f'force_yaw={self.cone_force_yaw:.2f}, '
            f'side_dist={self.cone_side_distance:.2f}m, '
            f'min_clusters={self.cone_side_min_clusters}'
        )

    # ========================================================
    # State helpers
    # ========================================================
    def reset_simple_cone_vars(self):
        self.simple_phase = 'WAIT_TARGET'
        self.simple_target = None
        self.simple_force_start_time = None
        self.simple_side_seen = False
        self.simple_side_clear_count = 0
        self.simple_last_cluster_count = 0
        self.simple_last_nearest = 9.9
        self.simple_last_side_name = 'none'
        self.center_only_start_time = None

    def cone_avoid_active(self):
        return (
            self.state == Mission.CONE
            and self.cone_latched
            and self.cone_target_lane in ('left', 'right')
            and not self.cone_test_completed
        )

    def ensure_simple_started(self):
        if not self.cone_avoid_active():
            return

        target_changed = self.simple_target != self.cone_target_lane
        if self.simple_force_start_time is None or target_changed:
            self.simple_target = self.cone_target_lane
            self.simple_force_start_time = time.monotonic()
            self.simple_phase = 'FORCE_SHIFT'
            self.simple_side_seen = False
            self.simple_side_clear_count = 0
            self.simple_last_cluster_count = 0
            self.simple_last_nearest = 9.9
            self.simple_last_side_name = 'right' if self.simple_target == 'left' else 'left'
            self.prev_speed = 0.0
            self.prev_cte_norm = 0.0

            self.get_logger().warn(
                f'SIMPLE_CONE_START target={self.simple_target} '
                f'watch_side={self.simple_last_side_name} '
                f'latched={sorted(list(self.cone_latched_lanes))}'
            )

    def forced_linear_speed(self):
        """
        강제 조향 중 속도.
        이전 코드처럼 cone_speed_limit으로 다시 낮추지 않고,
        cone_force_speed를 그대로 사용하되 after_force_speed_limit까지만 제한한다.
        """
        speed = float(self.cone_force_speed)
        if self.cone_after_force_speed_limit > 0.0:
            speed = min(speed, self.cone_after_force_speed_limit)
        return max(0.0, speed)

    def forced_yaw_for_target(self):
        yaw = abs(self.cone_force_yaw)

        if self.cone_target_lane == 'left':
            return yaw if self.cone_left_yaw_positive else -yaw

        if self.cone_target_lane == 'right':
            return -yaw if self.cone_left_yaw_positive else yaw

        return 0.0

    # ========================================================
    # Lane detection override
    # ========================================================
    def get_lane_by_sliding_window(self, mask):
        """
        CONE 상태에서도 기존 cone-specific lane selection을 쓰지 않고 일반 차선 추종 결과만 사용한다.
        FORCE_SHIFT 동안에는 lane_data가 계산되어도 handle_cone에서 무시한다.
        """
        saved_latched = self.cone_latched
        saved_target = self.cone_target_lane
        saved_recover = self.cone_recover_start_time

        try:
            self.cone_latched = False
            self.cone_target_lane = 'center'
            self.cone_recover_start_time = None
            return super().get_lane_by_sliding_window(mask)
        finally:
            self.cone_latched = saved_latched
            self.cone_target_lane = saved_target
            self.cone_recover_start_time = saved_recover

    # ========================================================
    # LiDAR side cone pass detection
    # ========================================================
    def watched_side_for_target(self):
        # 왼쪽 도로로 회피하면 콘은 오른쪽에 남아 있으므로 오른쪽 sector를 본다.
        # 오른쪽 도로로 회피하면 콘은 왼쪽에 남아 있으므로 왼쪽 sector를 본다.
        if self.cone_target_lane == 'left':
            return 'right'

        if self.cone_target_lane == 'right':
            return 'left'

        return 'none'

    def sector_limits_for_side(self, side):
        if side == 'right':
            return self.cone_right_angle_min_deg, self.cone_right_angle_max_deg

        if side == 'left':
            return self.cone_left_angle_min_deg, self.cone_left_angle_max_deg

        return 0.0, 0.0

    def side_cone_clusters(self):
        side = self.watched_side_for_target()
        angle_min, angle_max = self.sector_limits_for_side(side)

        pts = []
        for x, y, dist, angle in self.lidar_points:
            if dist <= 0.0 or dist > self.cone_side_distance:
                continue

            angle_deg = normalize_deg(math.degrees(angle))
            if not angle_in_range(angle_deg, angle_min, angle_max):
                continue

            pts.append((angle_deg, x, y, dist))

        if len(pts) == 0:
            return side, 0, 0, 9.9

        pts.sort(key=lambda p: p[0])

        clusters = []
        current = [pts[0]]
        prev = pts[0]

        for p in pts[1:]:
            _, x, y, _ = p
            _, px, py, _ = prev
            gap = math.hypot(x - px, y - py)

            if gap <= self.cone_side_cluster_gap_m:
                current.append(p)
            else:
                if len(current) >= self.cone_side_cluster_min_points:
                    clusters.append(current)

                current = [p]

            prev = p

        if len(current) >= self.cone_side_cluster_min_points:
            clusters.append(current)

        nearest = min(p[3] for p in pts)
        return side, len(clusters), len(pts), nearest

    def update_side_pass_state(self):
        side, cluster_count, point_count, nearest = self.side_cone_clusters()

        self.simple_last_side_name = side
        self.simple_last_cluster_count = cluster_count
        self.simple_last_nearest = nearest

        active = cluster_count >= self.cone_side_min_clusters

        if active:
            self.simple_side_seen = True
            self.simple_side_clear_count = 0
        else:
            if self.simple_side_seen:
                self.simple_side_clear_count += 1
            else:
                self.simple_side_clear_count = 0

        passed = (
            self.simple_side_seen
            and self.simple_side_clear_count >= self.cone_side_clear_frames
        )

        return {
            'side': side,
            'cluster_count': cluster_count,
            'point_count': point_count,
            'nearest': nearest,
            'active': active,
            'seen': self.simple_side_seen,
            'clear_count': self.simple_side_clear_count,
            'passed': passed,
        }

    # ========================================================
    # CONE handler
    # ========================================================
    def handle_cone(self, lane_data):
        # 통과 완료 후에는 차선추종만 계속한다.
        if self.cone_test_completed and not self.cone_only_finish_stop:
            cmd, _ = self.stanley_cmd_from_lane(
                lane_data,
                speed_limit=self.cone_after_force_speed_limit,
                extra_yaw=0.0,
                lane_bias_px=0.0,
            )
            return cmd, 'SIMPLE_CONE_COMPLETE_CONTINUE_LANE'

        # YOLO latch 또는 강제 target이 아직 없으면 정상 차선추종으로 대기한다.
        if not self.cone_avoid_active():
            cmd, _ = self.stanley_cmd_from_lane(
                lane_data,
                speed_limit=self.cone_after_force_speed_limit,
                extra_yaw=0.0,
                lane_bias_px=0.0,
            )
            return cmd, 'SIMPLE_CONE_WAIT_TARGET_LANE_FOLLOW'

        self.ensure_simple_started()
        side_info = self.update_side_pass_state()

        # 1) FORCE_SHIFT: 일정 시간 동안 차선을 무시하고 강제 조향한다.
        force_elapsed = time.monotonic() - self.simple_force_start_time

        if force_elapsed < self.cone_force_duration:
            cmd = Twist()
            cmd.linear.x = float(self.forced_linear_speed())
            cmd.angular.z = float(
                clamp(self.forced_yaw_for_target(), -MAX_YAW_RATE, MAX_YAW_RATE)
            )

            self.prev_speed = cmd.linear.x
            self.current_lane_bias_px = 0.0
            self.simple_phase = 'FORCE_SHIFT'

            if side_info['passed']:
                self.next_state('(simple cone passed during force shift)')

            return cmd, (
                f'SIMPLE_FORCE_SHIFT target={self.cone_target_lane} '
                f'v={cmd.linear.x:.2f} wz={cmd.angular.z:.2f} '
                f't={force_elapsed:.2f}/{self.cone_force_duration:.2f}s '
                f'{side_info["side"]}_clusters={side_info["cluster_count"]}/{self.cone_side_min_clusters} '
                f'points={side_info["point_count"]} seen={side_info["seen"]} '
                f'clear={side_info["clear_count"]}/{self.cone_side_clear_frames} '
                f'd={side_info["nearest"]:.2f}'
            )

        # 2) force 이후: 다시 정상 차선 추종으로 복귀한다.
        self.simple_phase = 'LANE_FOLLOW_AFTER_FORCE'

        cmd, _ = self.stanley_cmd_from_lane(
            lane_data,
            speed_limit=self.cone_after_force_speed_limit,
            extra_yaw=0.0,
            lane_bias_px=0.0,
        )

        if side_info['passed']:
            self.next_state('(simple cone side clusters appeared and disappeared)')
            return cmd, (
                f'SIMPLE_CONE_PASSED target={self.cone_target_lane} '
                f'{side_info["side"]}_clusters={side_info["cluster_count"]}/{self.cone_side_min_clusters} '
                f'clear={side_info["clear_count"]}/{self.cone_side_clear_frames}'
            )

        return cmd, (
            f'SIMPLE_LANE_FOLLOW_AFTER_FORCE target={self.cone_target_lane} '
            f't={force_elapsed:.2f}/{self.cone_force_duration:.2f}s '
            f'{side_info["side"]}_clusters={side_info["cluster_count"]}/{self.cone_side_min_clusters} '
            f'points={side_info["point_count"]} seen={side_info["seen"]} '
            f'clear={side_info["clear_count"]}/{self.cone_side_clear_frames} '
            f'd={side_info["nearest"]:.2f}'
        )

    # ========================================================
    # Latch / test-state behavior
    # ========================================================
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
        self.reset_simple_cone_vars()

    def cone_callback(self, msg):
        """
        CONE 단독 테스트용 cone callback.

        mission_fsm_node의 cone_callback에 맡기지 않고,
        /cone/blocked_lanes 문자열을 직접 해석해서 바로 회피 방향을 확정한다.

        규칙:
          - center + right -> target left  -> 왼쪽 강제 조향
          - center + left  -> target right -> 오른쪽 강제 조향
          - center만 1초 이상 지속 -> center + left로 가정 -> target right
          - 한 번 target이 left/right로 lock되면 이후 추가 검출은 무시
        """
        if self.cone_test_completed and not self.cone_only_finish_stop:
            return

        raw = msg.data.strip().lower()
        lanes = set()

        for token in raw.replace(';', ',').replace(' ', ',').split(','):
            token = token.strip()
            if token in ('left', 'center', 'right'):
                lanes.add(token)

        if len(lanes) == 0:
            return

        # center+right 또는 center+left로 이미 회피 방향이 확정되었으면
        # 이후 차량 진동 때문에 들어오는 추가 검출은 무시한다.
        if self.cone_target_locked and self.cone_target_lane in ('left', 'right'):
            self.get_logger().warn(
                f'SIMPLE_CONE_IGNORE_AFTER_LOCK raw={raw} '
                f'latched={sorted(list(self.cone_latched_lanes))} '
                f'target={self.cone_target_lane}'
            )
            return

        before_target = self.cone_target_lane
        now = time.monotonic()

        self.cone_latched = True
        self.last_cone_msg_time = now

        if self.cone_first_latch_time is None:
            self.cone_first_latch_time = now

        # 혹시 이전 버전 상태로 실행되더라도 안전하게 생성한다.
        if not hasattr(self, 'center_only_start_time'):
            self.center_only_start_time = None

        # 핵심 1: center + right cone -> 왼쪽 회피
        if 'center' in lanes and 'right' in lanes and 'left' not in lanes:
            self.cone_latched_lanes = {'center', 'right'}
            self.cone_target_lane = 'left'
            self.cone_target_locked = True
            self.cone_target_votes = ['left']
            self.center_only_start_time = None

        # 핵심 2: center + left cone -> 오른쪽 회피
        elif 'center' in lanes and 'left' in lanes and 'right' not in lanes:
            self.cone_latched_lanes = {'center', 'left'}
            self.cone_target_lane = 'right'
            self.cone_target_locked = True
            self.cone_target_votes = ['right']
            self.center_only_start_time = None

        # YOLO 노드가 right만 보냈을 때도 center+right로 간주해서 왼쪽 회피
        elif 'right' in lanes and 'left' not in lanes:
            self.cone_latched_lanes = {'center', 'right'}
            self.cone_target_lane = 'left'
            self.cone_target_locked = True
            self.cone_target_votes = ['left']
            self.center_only_start_time = None

        # YOLO 노드가 left만 보냈을 때도 center+left로 간주해서 오른쪽 회피
        elif 'left' in lanes and 'right' not in lanes:
            self.cone_latched_lanes = {'center', 'left'}
            self.cone_target_lane = 'right'
            self.cone_target_locked = True
            self.cone_target_votes = ['right']
            self.center_only_start_time = None

        # 핵심 3: center만 들어온 경우 1초 이상 지속되면 left가 있다고 가정
        elif lanes == {'center'}:
            self.cone_latched_lanes = {'center'}

            if self.center_only_start_time is None:
                self.center_only_start_time = now

            center_elapsed = now - self.center_only_start_time

            if center_elapsed >= 1.0:
                self.cone_latched_lanes = {'center', 'left'}
                self.cone_target_lane = 'right'
                self.cone_target_locked = True
                self.cone_target_votes = ['right']
                self.center_only_start_time = None

                self.get_logger().warn(
                    'SIMPLE_CONE_CENTER_ONLY_TIMEOUT '
                    'assume center+left -> target=right'
                )
            else:
                self.cone_target_lane = 'center'
                self.cone_target_locked = False

        else:
            # left, center, right가 동시에 들어오면 단독 테스트에서는 애매하므로
            # target 확정 전에는 무시한다.
            self.get_logger().warn(
                f'SIMPLE_CONE_AMBIGUOUS_IGNORE raw={raw} '
                f'lanes={sorted(list(lanes))}'
            )
            return

        self.get_logger().warn(
            f'SIMPLE_CONE_DIRECT_LATCH raw={raw} '
            f'latched={sorted(list(self.cone_latched_lanes))} '
            f'target={self.cone_target_lane} '
            f'locked={self.cone_target_locked}'
        )

        if self.cone_avoid_active() and before_target != self.cone_target_lane:
            self.reset_simple_cone_vars()
            self.ensure_simple_started()

    def force_cone_state(self, reason=''):
        self.state = Mission.CONE
        self.state_enter_time = time.monotonic()
        self.cone_test_completed = False
        self.reset_simple_cone_vars()
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
            self.get_logger().warn(f'SIMPLE_CONE_WAIT_YOLO {reason}')
            return

        # 강제 테스트 모드.
        # right target = center+left blocked -> 오른쪽 회피
        # left target  = center+right blocked -> 왼쪽 회피
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
        self.ensure_simple_started()

        self.get_logger().warn(
            f'SIMPLE_CONE_FORCE_TARGET target={self.cone_target_lane} '
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
        self.get_logger().warn(
            f'SIMPLE_CONE_COMPLETE_CONTINUE ignore transition to {new_state} {reason}'
        )

    def next_state(self, reason=''):
        if self.cone_only_finish_stop:
            super().set_state(Mission.FINISHED, f'(simple cone complete) {reason}')
        else:
            self.cone_test_completed = True
            self.clear_cone_latch_for_continue()
            self.get_logger().warn(f'SIMPLE_CONE_COMPLETE_CONTINUE {reason}')


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