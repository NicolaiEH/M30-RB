"""Launch profiling_frontend with basic parameters."""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mapping_pkg',
            executable='profiling_frontend',
            name='profiling_frontend',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'amp_min': 0.1},
                {'voxel_downsample': 0.05},
            ],
        ),
    ])
