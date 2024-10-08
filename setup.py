import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'human_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aisl',
    maintainer_email='nishizawa@aisl.cs.tut.ac.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_detector = '+ package_name +'.human_detector:main',
            'depth_human_detector = '+ package_name +'.depth_human_detector:main',
        ],
    },
)
