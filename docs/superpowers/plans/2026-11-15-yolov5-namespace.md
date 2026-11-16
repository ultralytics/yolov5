# YOLOv5 Namespace Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move YOLOv5's Python implementation under `yolov5/` and expose it through `yolov5.*` imports.

**Architecture:** Move the existing `models`, `utils`, `classify`, and `segment` implementations, plus core scripts, into a regular `yolov5` package. Update internal imports, retain root CLI forwarders, keep dataset assets under root `data/`, and resolve model YAMLs from the package.

**Tech Stack:** Python 3.8+, setuptools editable installs, uv workspace.

**Spec:** `docs/superpowers/specs/2026-11-15-yolov5-namespace-design.md`

## Global Constraints

- Preserve `python train.py`, `python detect.py`, `python val.py`, `python export.py`, and classify/segment script entry paths.
- Keep dataset assets and download scripts in root `data/`.
- Keep Docker build contexts in root `utils/docker/` for the existing workflows.
- Keep `classify/tutorial.ipynb` and `segment/tutorial.ipynb` at their documented paths.
- Keep small `models.*` forwarding modules for old checkpoint pickle paths and the `models/yolo.py` CLI.
- Keep model YAMLs available to package code under `yolov5.models`.
- Keep all implementation imports namespaced with `yolov5.`.
- Do not modify model behavior or unrelated project files.

## Review Focus

- Importing from outside the checkout must resolve the editable package rather than a local flat `models` or `utils` package.
- CLI defaults must resolve root dataset assets when invoked from another working directory.
- Model construction must still find YAML configurations after they move under the package.
- Existing root and nested CLI paths must forward arguments and exit status correctly.
- Wheel/editable metadata must include the package modules and model YAML data without discovering unrelated root directories.

---

### Task 1: Create the namespaced package and move implementation

**Files:**

- Move: root `models/`, Python implementation in `utils/`, `classify/`, and `segment/` into `yolov5/`.
- Keep: root `utils/docker/` in place for Docker workflows.
- Keep: classification and segmentation tutorial notebooks at their current root paths.
- Create: top-level `models/` compatibility modules forwarding to `yolov5.models`.
- Move: root implementation modules `train.py`, `val.py`, `detect.py`, `export.py`, and `benchmarks.py` into `yolov5/`.
- Keep: root `hubconf.py` as a forwarding module for PyTorch Hub compatibility.
- Create: `yolov5/__init__.py` with package version metadata only.
- Modify: `pyproject.toml` for `yolov5*` and `models*` package discovery and model YAML package data.

- [x] Move the implementation while preserving directory-relative resource files.
- [x] Add the package initializer and setuptools discovery rules.
- [x] Preserve the `models/yolo.py` CLI and old `models.*` pickle module paths with forwarding modules.
- [x] Confirm the resulting tree has the expected module and YAML paths.

### Task 2: Namespace imports and preserve CLI paths

**Files:**

- Modify: Python files throughout `yolov5/` and existing `tests/` files to use `yolov5.models`, `yolov5.utils`, and `yolov5.classify` / `yolov5.segment` imports.
- Modify: existing root `tests/` imports and path setup.
- Create: thin root CLI forwarders and `classify/*` / `segment/*` forwarders for the existing script paths.
- Modify: root `hubconf.py` to forward PyTorch Hub entry points to `yolov5.hubconf`.

- [x] Rewrite absolute imports to the `yolov5.*` namespace, including dynamic imports and test helpers.
- [x] Preserve root CLI commands and `classify/*` / `segment/*` script paths through forwarding entry points.
- [x] Preserve the root `hubconf.py` command-line behavior and PyTorch Hub entry points.
- [x] Search all Python files for stale flat imports and resolve each remaining occurrence.

### Task 3: Repair resource roots and verify the editable package

**Files:**

- Modify: package path constants in `yolov5/utils/general.py`, model builders, logger integrations, and moved CLI modules as required.
- Modify: `pyproject.toml` only if verification identifies missing package data.

- [x] Keep dataset defaults rooted at the checkout's `data/` directory and model defaults rooted at `yolov5/models/`.
- [x] Install the package editable without dependencies into a temporary venv and inspect its editable metadata.
- [x] Verify representative imports from outside the checkout.
- [x] Run root and nested CLI `--help` commands from the checkout and from another working directory.
- [x] Run the existing `models/yolo.py --help`-equivalent model construction CLI with a package model YAML.
- [x] Review the diff for unrelated changes and stale flat imports.
