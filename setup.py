from setuptools import setup, find_packages

setup(
    name='obstacle_avoidance_rl_drone',
    version='0.1.0',
    description='RL-based drone obstacle avoidance with PyBullet',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pybullet',
        'numpy',
        'matplotlib',
        'torch',
        'tk',
    ],
) 