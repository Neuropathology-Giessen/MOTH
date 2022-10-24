from setuptools import setup

setup(
    packages = ['moth'],
    install_requires = [
        'tifffile',
        'shapely',
        'paquo',
        'opencv-python',
        'numpy',
        'openslide-python'
    ])