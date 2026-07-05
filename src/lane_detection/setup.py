import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='hanjuhyeong35@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_node = lane_detection.lane_node:main',
            'lidar_stop_node = lane_detection.lidar_stop_node:main',
            'mission_fsm_node = lane_detection.mission_fsm_node:main',
            'yolov8_cone_node = lane_detection.yolov8_cone_node:main',
            'cone_test_yolo = lane_detection.cone_test_yolo:main',
            'test_node = lane_detection.test:main',
            'practice_node = lane_detection.practice:main',
            'trash_node = lane_detection.trash:main',
            'parking_test_node = lane_detection.parking_test:main',
            'traffic_test_node = lane_detection.traffic_test_node:main',
        ],
    },
)
