import pathlib

import setuptools
from pkg_resources import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in parse_requirements(requirements_txt)
    ]

setuptools.setup(
    name="yolov5",
    version="0.0.1",
    author="nanovare",
    author_email="vincent@nanovare.com",
    description="Integrate yolov5 for mojo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robin-maillot/yolov5",
    packages=setuptools.find_packages(include=['yolov5', 'yolov5.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires = install_requires
)