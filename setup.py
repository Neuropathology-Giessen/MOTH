from setuptools import setup

setup(
    packages = ['mothi'],
    install_requires = [
        'tifffile',
        'shapely',
        'paquo',
        'opencv-python',
        'numpy',
        'tiffslide'
    ])
