from setuptools import setup

setup(
    name = 'moth',
    version = '0.1',
    packages = ['moth'],
    install_requires = [
        'tifffile',
        'shapely',
        'paquo',
        'opencv-python',
        'numpy',
        'openslide-python'
    ])