import setuptools

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setuptools.setup(
    name="yolov5",
    version='6.1.0',
    author="",
    license="GPL",
    description="Packaged version of the Yolov5 object detector",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ultralytics/yolov5",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=read_requirements(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)