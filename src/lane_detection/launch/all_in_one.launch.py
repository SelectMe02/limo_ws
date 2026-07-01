from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    camera_topic = '/camera/color/image_raw'
    model_path = '/home/wego/limo_ws/src/lane_detection/models/best_cone.pt'

    # 1) YOLO 고깔 노드
    # 먼저 켜서 모델 로딩을 시작한다.
    yolo_cone_node = Node(
        package='lane_detection',
        executable='yolov8_cone_node',
        name='yolov8_cone_node',
        output='screen',
        parameters=[
            {
                'camera_topic': camera_topic,
                'model_path': model_path,
                'device': 'cpu',
                'conf_th': 0.45,
                'imgsz': 320,
            }
        ],
    )

    # 2) YOLO visualizer
    # debug_pkg visualizer는 기본적으로 /camera1/image_raw를 구독하므로
    # 네 카메라 토픽 /camera/color/image_raw로 remapping한다.
    visualizer_node = Node(
        package='debug_pkg',
        executable='yolov8_visualizer_node',
        name='yolov8_visualizer_node',
        output='screen',
        remappings=[
            ('/camera1/image_raw', camera_topic),
            ('/detections', '/detections'),
        ],
    )

    # 3) 미션 FSM 노드
    # 실제 /cmd_vel을 발행하는 주행 메인 노드.
    mission_fsm_node = Node(
        package='lane_detection',
        executable='mission_fsm_node',
        name='mission_fsm_node',
        output='screen',
        parameters=[
            {
                # 신호등 미션 제외 버전을 쓰는 경우
                'use_traffic_light': False,

                # 필요하면 여기서 속도도 같이 제한 가능
                'max_speed': 0.45,
                'tunnel_speed': 0.22,
                'avoid_speed': 0.22,
                'follow_speed': 0.18,
            }
        ],
    )

    return LaunchDescription([
        # 0초: YOLO 먼저 실행
        TimerAction(
            period=0.0,
            actions=[yolo_cone_node],
        ),

        # 2초 뒤: visualizer 실행
        TimerAction(
            period=4.0,
            actions=[visualizer_node],
        ),

        # 4초 뒤: 실제 주행 FSM 실행
        TimerAction(
            period=6.0,
            actions=[mission_fsm_node],
        ),
    ])