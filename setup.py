import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="ultralytics-yolov5",
    version='0.0.0',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ultralytics/yolov5",
    author="ultralytics",
    license='GPLv3',
    packages=['yolov5'],
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow",
        "PyYAML>=5.3.1",
        "scipy>=1.4.1",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm>=4.41.0"
    ],
    extras_require={
        'tests': [
            'pytest',
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License ::OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)