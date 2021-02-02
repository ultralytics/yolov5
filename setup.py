
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
    name='yolov5',
    version='0',
    description="Downloads or clones a python project from github and allows to import it from anywhere. Very useful when the repo is not a package",
    long_description='yolo5 package',
    author="zara",
    author_email='',
    url='https://github.com/zaraalaverdyan-planorama/yolov5',
    packages=find_packages(),
    package_dir={'packyou':
                 'packyou'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    license="MIT license",
    zip_safe=False,
    keywords='packyou',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
)
