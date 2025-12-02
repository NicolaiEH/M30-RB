#!/usr/bin/env python3
"""Launch SLAM backend, profiling frontend, and RViz for bag playback."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.descriptions import ParameterValue
from launch.conditions import IfCondition

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config  = LaunchConfiguration('rviz_config')

    def bool_param(name: str):
        return ParameterValue(LaunchConfiguration(name), value_type=bool)

    def float_param(name: str):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    def base_to_profiling_static(context):
        xyz = LaunchConfiguration('base_to_profiling_xyz').perform(context).split()
        rpy = LaunchConfiguration('base_to_profiling_rpy').perform(context).split()

        if len(xyz) != 3:
            raise RuntimeError(
                f'base_to_profiling_xyz must contain 3 space-separated values, got: {xyz}'
            )
        if len(rpy) != 3:
            raise RuntimeError(
                f'base_to_profiling_rpy must contain 3 space-separated values, got: {rpy}'
            )

        return [
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='base_to_profiling_static',
                arguments=xyz + rpy + ['base_link', 'profiling_link']
            )
        ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('profiling_frontend'), 'rviz', 'slam_mapping.rviz'
            ])
        ),
        DeclareLaunchArgument('tf_timeout_sec', default_value='1.0'),
        DeclareLaunchArgument('tf_backoff_sec', default_value='0.40'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),

        DeclareLaunchArgument('slam_publish_map_to_odom_tf', default_value='true'),
        DeclareLaunchArgument('slam_dvl_staleness_sec', default_value='0.60'),
        DeclareLaunchArgument('slam_depth_staleness_sec', default_value='0.60'),
        DeclareLaunchArgument('slam_use_dvl_factor', default_value='true'),
        DeclareLaunchArgument('slam_dvl_use_huber', default_value='true'),

        DeclareLaunchArgument('profiling_voxel_downsample', default_value='0.05'),
        DeclareLaunchArgument('profiling_cfar_k', default_value='1.5'),

        DeclareLaunchArgument('base_to_profiling_xyz', default_value='0 0 0'),
        DeclareLaunchArgument('base_to_profiling_rpy', default_value='0 1.57079632679 0'),

        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '0'),

        OpaqueFunction(function=base_to_profiling_static),

        Node(
            package='slam_backend',
            executable='node',
            name='slam_backend',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,

                'publish_map_to_odom_tf': bool_param('slam_publish_map_to_odom_tf'),

                'key_period': 0.25,
                'dvl_vel_sigma': 0.10,
                'depth_sigma': 0.10,
                'dvl_staleness_sec': float_param('slam_dvl_staleness_sec'),
                'depth_staleness_sec': float_param('slam_depth_staleness_sec'),
                'use_dvl_factor': bool_param('slam_use_dvl_factor'),
                'dvl_use_huber': bool_param('slam_dvl_use_huber'),
                'use_gt_factor': False,
                'gt_pose_sigma': 3.0,
            }],
        ),

        Node(
            package='profiling_frontend',
            executable='node',
            name='profiling_frontend',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,

                'amp_min': 0.1,
                'voxel_downsample': float_param('profiling_voxel_downsample'),
                'tvg_enable': True,
                'tvg_alpha_db_per_m': 0.035,
                'cfar_win': 12,
                'cfar_guard': 2,
                'cfar_k': float_param('profiling_cfar_k'),
                'smooth_med_win': 5,

                'tf_timeout_sec': float_param('tf_timeout_sec'),
                'tf_backoff_sec': float_param('tf_backoff_sec')
            }],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('launch_rviz')),
        ),
    ])
