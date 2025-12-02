from setuptools import setup
import os
from glob import glob

package_name = 'slam_backend'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # the Python package: slam_backend/
    data_files=[
        # Index this package with ament
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Install the package manifest
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # (Optional) install config files if you add them later
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nicolai Enstad Haraldseth',
    maintainer_email='nicolai.enstad.haraldseth@nmbu.no',
    description='Minimal iSAM2 backend for HoloOcean ROS bags (IMU + DVL + imaging odometry).',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run slam_backend node
            'node = slam_backend.node:main',
        ],
    },
)
