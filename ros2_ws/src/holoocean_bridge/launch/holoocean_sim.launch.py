"""Launch the HoloOcean bridge with optional imaging and profiling sonar."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    scenario = LaunchConfiguration('scenario')
    enable_imaging = LaunchConfiguration('enable_imaging_sonar')
    enable_profiling = LaunchConfiguration('enable_profiling_sonar')
    control_mode = LaunchConfiguration('control_mode')
    log_level = LaunchConfiguration('log_level')
    use_sim_time = LaunchConfiguration('use_sim_time')
    render_quality = LaunchConfiguration('render_quality')
    run_headless = LaunchConfiguration('run_headless')
    publish_gt_tf = LaunchConfiguration('publish_gt_tf')

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='custom_scenario.json'),
        DeclareLaunchArgument('enable_imaging_sonar', default_value='False'),
        DeclareLaunchArgument('enable_profiling_sonar', default_value='False'),
        DeclareLaunchArgument('control_mode', default_value='thrusters'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('render_quality', default_value='3'),
        DeclareLaunchArgument('run_headless', default_value='false'),
        DeclareLaunchArgument('publish_gt_tf', default_value='false'),

        Node(
            package='holoocean_bridge',
            executable='node',
            name='holoocean_bridge',
            output='screen',
            parameters=[{
                'scenario_path': PathJoinSubstitution([FindPackageShare('holoocean_bridge'), 'config', scenario]),
                'enable_imaging_sonar': ParameterValue(enable_imaging, value_type=bool),
                'enable_profiling_sonar': ParameterValue(enable_profiling, value_type=bool),
                'control_mode': ParameterValue(control_mode, value_type=str),
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'render_quality': ParameterValue(render_quality, value_type=int),
                'run_headless': ParameterValue(run_headless, value_type=bool),
                'publish_gt_tf': ParameterValue(publish_gt_tf, value_type=bool),
            }],
            arguments=['--ros-args', '--log-level', log_level]
        ),

        Node(
            package='holoocean_bridge',                
            executable='lawnmower_pose_node',        
            name='lawnmower_pose_surface',
            output='screen',
            parameters=[{
                'area': [-50.0, 50.0, -30.0, 30.0],
                'lane_spacing': 20.0,
                'waypoint_tol': 1.0,
                'depth_m': 5.0,                     
                'surface_z': 0.0,
                'cmd_pose_topic': 'cmd/pose',       
                'odom_frame_id': 'odom',
                'rate_hz': 10.0,
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            }],
            condition=IfCondition(PythonExpression(["'", control_mode, "' == 'pose'"]))
        ),
    ])
