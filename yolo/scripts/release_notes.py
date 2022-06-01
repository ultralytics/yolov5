# encoding: utf-8

"""
Prepares markdown release notes for GitHub releases.
"""

import os
from typing import List, Optional

import packaging.version

TAG = os.environ["TAG"]

ADDED_HEADER = "### Added 🎉"
CHANGED_HEADER = "### Changed ⚠️"
FIXED_HEADER = "### Fixed ✅"
REMOVED_HEADER = "### Removed 👋"


def get_change_log_notes() -> str:
    in_current_section = False
    current_section_notes: List[str] = []
    with open("CHANGELOG.md") as changelog:
        for line in changelog:
            if line.startswith("## "):
                if line.startswith("## Unreleased"):
                    continue
                if line.startswith(f"## [{TAG}]"):
                    in_current_section = True
                    continue
                break
            if in_current_section:
                if line.startswith("### Added"):
                    line = ADDED_HEADER + "\n"
                elif line.startswith("### Changed"):
                    line = CHANGED_HEADER + "\n"
                elif line.startswith("### Fixed"):
                    line = FIXED_HEADER + "\n"
                elif line.startswith("### Removed"):
                    line = REMOVED_HEADER + "\n"
                current_section_notes.append(line)
    assert current_section_notes
    return "## What's new\n\n" + "".join(current_section_notes).strip() + "\n"


def get_commit_history() -> str:
    new_version = packaging.version.parse(TAG)

    # Get all tags sorted by version, latest first.
    all_tags = os.popen("git tag -l --sort=-version:refname 'v*'").read().split("\n")

    # Out of `all_tags`, find the latest previous version so that we can collect all
    # commits between that version and the new version we're about to publish.
    # Note that we ignore pre-releases unless the new version is also a pre-release.
    last_tag: Optional[str] = None
    for tag in all_tags:
        if not tag.strip():  # could be blank line
            continue
        version = packaging.version.parse(tag)
        if new_version.pre is None and version.pre is not None:
            continue
        if version < new_version:
            last_tag = tag
            break
    if last_tag is not None:
        commits = os.popen(f"git log {last_tag}..{TAG}^ --oneline --first-parent").read()
    else:
        commits = os.popen("git log --oneline --first-parent").read()
    return "## Commits\n\n" + commits


def main():
    print(get_change_log_notes())
    print(get_commit_history())


if __name__ == "__main__":
    main()
