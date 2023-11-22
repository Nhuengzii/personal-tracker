from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.3'
DESCRIPTION = 'A simple object tracker for tracking target person using few images of target person'

setup(
    name="personal-tracker",
    version=VERSION,
    author="Nhuengzii (Anawat Moonmanee)",
    author_email="<32nhueng@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["ultralytics", "opencv-python","opencv-contrib-python", "clip", "gdown"],
)
