"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from os import path
from setuptools import setup, find_packages

import yolo5

PATH_HERE = path.abspath(path.dirname(__file__))

with open(path.join(PATH_HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]

# Get the long description from the README file
with open(path.join(PATH_HERE, 'README.md'), encoding='utf-8') as fp:
    long_description = fp.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install YOLOv5
    #
    # And where it will live on PyPI: https://pypi.org/project/YOLOv5/
    name='YOLOv5',  # Required
    version=yolo5.__version__,  # Required
    description=yolo5.__doc__,  # Optional

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    long_description=yolo5.__doc_long__,  # Optional

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    url='https://github.com/ultralytics/yolov5',  # Optional

    # This should be your name or the name of the organization which owns the project.
    author='Ultralytics',  # Optional
    # This should be a valid email address corresponding to the author listed above.
    author_email='glenn.jocher@ultralytics.com',  # Optional

    # Classifiers help users find your project by categorizing it.
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Operation system
        'Operating System :: OS Independent',

        # Topics
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='YOLO object-detection',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. If you
    # do not support Python 2, you can simplify this to '>=3.5' or similar, see
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.6, <4',
    install_requires=install_reqs,

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    # install_requires=[],  # Optional

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[('models',
                 [f'yolo5/models/yolov5{n}.yaml' for n in 'smlx']
                 + ['yolo5/models/hub/yolov3-spp.yaml']
                 + [f'yolo5/models/hub/yolov5-{n}.yaml' for n in ('fpn', 'panet')]
                 )],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sandbox=sandbox:main',
    #     ],
    # },

    # List additional URLs that are relevant to your project as a dict.
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/ultralytics/yolov5/issues',
        'Funding': 'https://www.ultralytics.com',
        'Source': 'https://github.com/ultralytics/yolov5/',
    },
)
