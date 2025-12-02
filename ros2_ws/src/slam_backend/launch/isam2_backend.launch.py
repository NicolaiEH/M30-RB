"""Launch the ISAM2 backend node with basic parameters."""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='slam_pkg',
            executable='isam2_backend',
            name='isam2_backend',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'key_period': 0.25},
                {'imu_accel_sigma': 0.02},
                {'imu_gyro_sigma': 0.002},
                {'dvl_vel_sigma': 0.05},
                {'depth_sigma': 0.1},
                {'use_gt_factor': False},
            ],
        ),
    ])
