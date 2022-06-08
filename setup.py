from gzip import READ
from urllib import request

from setuptools import find_packages, setup

README = request.urlopen('https://raw.githubusercontent.com/ultralytics/yolov5/master/README.md').read().decode('utf-8')


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req)
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
with open("yolov5/version.py") as version_file:
    exec(version_file.read(), VERSION)

setup(name="ultralytics-yolov5",
      version=VERSION["VERSION"],
      description="",
      long_description=README,
      long_description_content_type="text/markdown",
      entry_points={'console_scripts': ["yolov5 = yolov5.__main__:main"]},
      classifiers=[
          "Intended Audience :: Science/Research",
          "Development Status :: 3 - Alpha",
          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",],
      keywords="",
      url="https://github.com/ultralytics/yolov5",
      author="Ultralytics",
      author_email="hello@ultralytics.com",
      license="Apache",
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "env"],),
      package_data={"yolov5": ["py.typed"]},
      install_requires=read_requirements("requirements.txt"),
      extras_require={"dev": read_requirements("dev-requirements.txt")},
      python_requires=">=3.7",
      include_package_data=True)
