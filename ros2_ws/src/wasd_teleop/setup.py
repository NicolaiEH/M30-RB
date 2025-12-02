from setuptools import setup
package_name = 'wasd_teleop'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nicolai Enstad Haraldseth',
    maintainer_email='nicolai.enstad.haraldseth@nmbu.no',
    description='WASD + Q/E teleop for /cmd_vel',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'teleop_wasd = wasd_teleop.teleop_wasd:main',
        ],
    },
)
