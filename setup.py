import pathlib

import setuptools
from pkg_resources import parse_requirements

import glob
import os
import shutil

MODULE_NAME = "yolov5"


def copy_to_module():
    if os.path.isdir(MODULE_NAME):
        shutil.rmtree(MODULE_NAME)

    for python_file_path in glob.glob("**/*.py", recursive=True):
        if python_file_path == "setup.py":
            continue

        out_path = f"{MODULE_NAME}/{python_file_path}"
        out_dir = os.path.dirname(out_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/__init__.py", 'a'):
                pass
        shutil.copyfile(python_file_path, out_path)


with open("README.md", "r") as fh:
    long_description = fh.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in parse_requirements(requirements_txt)
    ]


copy_to_module()
modules = setuptools.find_packages(include=['yolov5', 'yolov5.*'])

setuptools.setup(
    name=MODULE_NAME,
    version="0.0.1",
    author="nanovare",
    author_email="vincent@nanovare.com",
    description="yolov5 modifications for nanovare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robin-maillot/yolov5",
    packages=modules,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    dependency_links=[
            "https://download.pytorch.org/whl/torch_stable.html",
        ],
    install_requires=install_requires
)