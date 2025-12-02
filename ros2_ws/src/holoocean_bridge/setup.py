from setuptools import setup
import os
from glob import glob

package_name = 'holoocean_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json') + glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
    ],
    zip_safe=True,
    maintainer='Nicolai Enstad Haraldseth',
    maintainer_email='nicolai.enstad.haraldseth@nmbu.no',
    description='ROS 2 <-> HoloOcean bridge (IMU, DVL, imaging sonar, cmd_vel)',
    license='Proprietary',
)
