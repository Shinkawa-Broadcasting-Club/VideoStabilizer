# 入力収集・出力パス解決・衝突ポリシー

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

CollisionPolicy = Literal["overwrite", "skip", "rename"]
ResumeMode = Literal["skip_done", "retry_failed", "run_all"]


def collect_targets(
    paths: list[str],
    extensions: tuple[str, ...],
    *,
    recursive: bool = True,
) -> list[str]:
    """Expand files and folders into a sorted unique list of video paths."""
    ext_set = {e.lower() for e in extensions}
    found: dict[str, None] = {}

    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in ext_set:
                found[str(p.resolve())] = None
            continue
        if not p.is_dir():
            continue
        iterator = p.rglob("*") if recursive else p.glob("*")
        for child in iterator:
            if child.is_file() and child.suffix.lower() in ext_set:
                found[str(child.resolve())] = None

    return sorted(found.keys())


def build_output_path(
    input_path: str,
    *,
    output_dir: str | None,
    output_subdir: str,
    prefix: str,
    suffix: str,
) -> str:
    """Resolve final output path for one input file."""
    inp = Path(input_path)
    if output_dir:
        out_parent = Path(output_dir)
    else:
        out_parent = inp.parent / output_subdir
    out_parent.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    if prefix and stem.startswith(prefix):
        out_name = f"{stem}{suffix}{inp.suffix}"
    else:
        out_name = f"{prefix}{stem}{suffix}{inp.suffix}"
    return str(out_parent / out_name)


def resolve_collision(path: str, policy: CollisionPolicy) -> str | None:
    """Return output path to use, or None if processing should be skipped."""
    if policy == "overwrite" or not os.path.isfile(path):
        return path
    if policy == "skip":
        return None

    base = Path(path)
    parent = base.parent
    stem = base.stem
    ext = base.suffix
    n = 2
    while True:
        candidate = parent / f"{stem} ({n}){ext}"
        if not candidate.is_file():
            return str(candidate)
        n += 1
