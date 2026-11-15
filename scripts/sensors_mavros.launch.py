"""T3: bridge semua sensor (6 lidar + 2 kamera) + MAVROS, dalam satu perintah.

  ros2 launch ~/oprec/gazebo_sim/launch/sensors_mavros.launch.py

Tidak perlu colcon build, file ini dijalankan langsung dari path-nya.
Opsional: fcu_url:=udp://127.0.0.1:14550@
Ctrl+C mematikan bridge dan MAVROS sekaligus.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LIDARS = ['left', 'right', 'up', 'down', 'front', 'back']
CAMERAS = ['front', 'down']


def generate_launch_description():
    bridges = [f'/lidar_{n}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan' for n in LIDARS]
    bridges += [f'/camera_{n}@sensor_msgs/msg/Image[gz.msgs.Image' for n in CAMERAS]

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='sensor_bridge',
        arguments=bridges,
        output='screen',
    )

    mavros_launch = os.path.join(get_package_share_directory('mavros'), 'launch', 'apm.launch')
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_launch),
        launch_arguments={'fcu_url': LaunchConfiguration('fcu_url')}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://127.0.0.1:14550@'),
        bridge,
        mavros,
    ])
