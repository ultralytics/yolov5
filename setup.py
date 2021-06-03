import pathlib

import setuptools
from pkg_resources import parse_requirements

import shutil

MODULE_NAME = "nanovare_yolov5"


def copy_to_module():
    current_path = pathlib.Path(".").resolve()
    module_root_path = current_path / MODULE_NAME
    module_root_path = module_root_path.relative_to(current_path)

    if module_root_path.is_dir():
        shutil.rmtree(module_root_path)

    modules_moved = []
    python_files = list(current_path.glob("**/*.py"))
    for python_file_path in python_files:
        python_file_path = python_file_path.relative_to(current_path)
        if str(python_file_path) == "setup.py" or python_file_path.stem == "__init__":
            continue
        new_path = module_root_path / python_file_path
        modules_moved.append(".".join((list(map(str,python_file_path.parents))[::-1][1:] + [python_file_path.stem])))
        if not new_path.parent.is_dir():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            # make an init
            init_file = new_path.parent / "__init__.py"
            init_file.touch()

        shutil.copyfile(python_file_path, new_path)


    # Rename modules
    strings_to_replace = {}
    for module_name in modules_moved:
        strings_to_replace[f"from {module_name} "] = f"from {MODULE_NAME}.{module_name} "
        strings_to_replace[f"import {module_name}"] = f"import {MODULE_NAME}.{module_name} "
    strings_to_search_for = tuple(strings_to_replace)
    new_python_files = list(module_root_path.glob("**/*.py"))
    lines_changed = []

    for python_file_path in new_python_files:
        data = []
        with python_file_path.open("r") as f:
            for line in f:
                if line.startswith(strings_to_search_for):
                    lines_changed.append(line)
                    for key in strings_to_search_for:
                        if line.startswith(key):
                            line = line.replace(key, strings_to_replace[key], 1)
                data.append(line)
        with python_file_path.open("w") as f:
            for line in data:
                f.write(line)

    print(f"Changed {len(lines_changed)} lines to accommodate for imports.")


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in parse_requirements(requirements_txt)
    ]

copy_to_module()
modules = setuptools.find_packages(include=[MODULE_NAME, f"{MODULE_NAME}.*"])

setuptools.setup(
    name=MODULE_NAME,
    version="0.1.1",
    author="nanovare",
    author_email="vincent@nanovare.com, robin@nanovare.com",
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