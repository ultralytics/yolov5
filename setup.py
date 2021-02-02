
from setuptools import setup, find_packages

#with open('README.rst') as readme_file:
#    README = readme_file.read()

#with open('HISTORY.rst') as history_file:
#    HISTORY = history_file.read()

REQUIREMENTS = [
    'gitpython',
    'requests',
    'tqdm',
    'requests_cache',
]

TEST_REQUIREMENTS = [
]

setup(
    name='yolov5pkg2',
    version='0',
    description="Downloads or clones a python project from github and allows to import it from anywhere. Very useful when the repo is not a package",
    long_description='yolo5 package',
    author="zara",
    author_email='',
    url='https://github.com/zaraalaverdyan-planorama/yolov5',
    packages=['yolov5'],
    install_requires=REQUIREMENTS,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
