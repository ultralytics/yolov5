from setuptools import find_packages, setup


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))
    return requirements


# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import cached_path whilst setting up.
VERSION = {}  # type: ignore
with open("yolov5/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="yolov5",
    version=VERSION["VERSION"],
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts':[
        "yolov5 = yolov5.__main__:main"
        ]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    url="https://github.com/allenai/python-package-template",
    author="Allen Institute for Artificial Intelligence",
    author_email="contact@allenai.org",
    license="Apache",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "env"],
    ),
    package_data={"yolov5": ["py.typed"]},
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": read_requirements("dev-requirements.txt")},
    python_requires=">=3.7",
    include_package_data=True
)
