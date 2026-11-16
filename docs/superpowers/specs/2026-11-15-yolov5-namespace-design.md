# YOLOv5 Python Namespace Design

## Intent

Give the editable workspace package an informative `yolov5` import namespace. Today its discovered packages are top-level `models` and `utils`, and the source imports those names throughout.

## Goals and non-goals

- Move YOLOv5 Python implementation under `yolov5/` and use `yolov5.*` imports.
- Keep dataset assets and download scripts in root `data/`.
- Keep Docker build contexts in root `utils/docker/` for the existing workflows.
- Keep existing command-line script paths working through thin root-level launchers.
- Keep classification and segmentation tutorial notebooks at their documented root paths.
- Keep thin `models.*` compatibility modules for checkpoint loading and the existing `models/yolo.py` command.
- Keep model YAML configurations with `yolov5.models` and include them as package data.
- Do not change model behavior or reorganize unrelated project code.

## Approaches

- A top-level facade would expose only selected imports and leave internal imports flat; this does not meet the requested full namespace.
- A full source package migration moves implementation into `yolov5/`, updates imports, and preserves current command-line entry paths with forwarders. This is the selected approach because it makes both the public and internal import structure consistent.

## Architecture and constraints

The implementation and internal imports use `yolov5.*`; minimal top-level `models.*` forwarders remain for legacy checkpoint pickle paths and the existing `models/yolo.py` command. Setuptools discovers `yolov5*` and those compatibility modules. Model configuration YAML files live under `yolov5/models`; dataset configuration and sample assets remain in root `data/`. Docker build contexts remain in root `utils/docker/` because existing workflows reference those paths. Classification and segmentation notebooks remain at their documented root paths. Package code resolves these roots explicitly so behavior does not depend on the caller's working directory. Root CLI scripts forward to package entry points.

## Verification

Verify editable package discovery and imports, then invoke the existing CLI launchers with `--help` from both the repository root and another working directory. Do not run the test suite as part of this request.
