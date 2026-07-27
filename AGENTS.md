# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `master`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout master && git pull`.

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
