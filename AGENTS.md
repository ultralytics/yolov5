# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout master && git pull`.

## Commands

```bash
uv pip install -r requirements.txt                                                                    # install (CI adds --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
uv pip install pytest pytest-cov                                                                      # test dependencies (the "dev" extra in pyproject.toml)
python -m pytest tests/                                                                               # run all tests (python -m puts the repo root on sys.path; bare pytest fails imports)
python -m pytest tests/ -m "not network"                                                              # skip tests that hit the live network
python -m pytest tests/test_invariant_export.py::test_export_edgetpu_no_shell_true                    # run one test
python -m pytest tests/ --cov                                                                         # coverage (local convention only; CI runs no pytest or coverage)
ruff format . && ruff check --fix .                                                                   # format + lint (line-length 120 from pyproject.toml [tool.ruff])
python train.py --imgsz 64 --batch 32 --weights yolov5n.pt --cfg yolov5n.yaml --epochs 1 --device cpu # CI-style smoke train
```

- CI (`.github/workflows/ci-testing.yml`) runs end-to-end train/val/detect/export smoke scripts and `benchmarks.py` — not pytest — on push/PR to `master` plus a daily cron.
- Tests matrix: ubuntu-latest, windows-latest, macos-14 on latest Python 3.x, plus ubuntu on Python 3.8 with torch 1.8.0 (repo floors: `requires-python >=3.8`, `torch>=1.8.0`); the Benchmarks job is pinned to Python 3.11.

## Architecture

YOLOv5 is run from a repo clone, not as an installed package: `pyproject.toml` carries packaging metadata (static version 7.0.0) but there is no PyPI publish workflow, and pretrained weights download from the GitHub v7.0 release. Each task has a script triad — root `train.py`/`val.py`/`detect.py` for detection, mirrored in `segment/` and `classify/` (which use `predict.py` instead of `detect.py`) — with shared `export.py` (all export formats) and `benchmarks.py` (export + val across formats). `models/yolo.py` builds DetectionModel and SegmentationModel from YAML configs (`models/*.yaml`; P6 and experimental variants in `models/hub/`, segmentation in `models/segment/`), while ClassificationModel only wraps an existing detection model (its YAML construction is an unimplemented placeholder); `models/common.py` holds the layer zoo and `DetectMultiBackend` for multi-format inference, and `hubconf.py` is the PyTorch Hub entry point. `utils/` provides dataloaders, general helpers, torch utilities, and `utils/loggers/` (Comet, ClearML, W&B, TensorBoard). The repo depends on the `ultralytics` pip package for some utilities (e.g. `ultralytics.utils.patches.torch_load`). Publishing: `docker.yml` builds and pushes `ultralytics/yolov5` Docker Hub images (`latest`, `latest-cpu`, `latest-arm64` from `utils/docker/`) on every push to `master`, gated to the `ultralytics/yolov5` repository; `format.yml` (Ultralytics Actions) auto-formats, labels, and summarizes PRs by pushing commits to the PR branch.

## Conventions

- Source files carry the `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header as the first line (after the shebang in shell scripts) — Ultralytics Actions adds these headers automatically, so don't add or revert them manually.
- Google-style docstrings for larger classes and functions; a single-line docstring is fine for small functions and methods (docformatter wraps at 120, config in `pyproject.toml`).
- Formatting is enforced on PRs by Ultralytics Actions: Ruff + docformatter for Python, Prettier for YAML/JSON/Markdown, codespell (ignore list in `pyproject.toml [tool.codespell]`).
- Tests live in `tests/` as plain pytest; tests marked `@pytest.mark.network` hit the live network — deselect with `-m "not network"` when offline.
- The default branch is `master`, not `main`.
- No automated version bumps or releases: the version in `pyproject.toml` is fixed at 7.0.0 and releases are hand-cut GitHub tags with attached weights.
