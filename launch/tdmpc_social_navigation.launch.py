import os
import sys
from glob import glob
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    pkg_dir = get_package_share_directory('tdmpc_social_navigation')
    list = [
        Node(
            package='tdmpc_social_navigation',
            executable='tdmpc_social_navigation',
            namespace='',
            # remappings=[('image_raw', '/thetav/image_raw')],
            output="screen",
            respawn=True,
        ),
    ]

    return LaunchDescription(list)