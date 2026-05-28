# バッチ処理 manifest（中断復帰）

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

JobStatus = Literal["pending", "running", "done", "failed", "skipped", "cancelled"]


@dataclass
class JobEntry:
    input_path: str
    output_path: str
    status: JobStatus = "pending"
    error: str | None = None


@dataclass
class BatchManifest:
    version: int = 1
    created_at: str = ""
    config_hash: str = ""
    jobs: list[JobEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "jobs": [asdict(j) for j in self.jobs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchManifest:
        jobs = [
            JobEntry(
                input_path=j["input_path"],
                output_path=j["output_path"],
                status=j.get("status", "pending"),
                error=j.get("error"),
            )
            for j in data.get("jobs", [])
        ]
        return cls(
            version=int(data.get("version", 1)),
            created_at=str(data.get("created_at", "")),
            config_hash=str(data.get("config_hash", "")),
            jobs=jobs,
        )


def config_hash(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_manifest(path: str) -> BatchManifest | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return BatchManifest.from_dict(json.load(f))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def save_manifest(path: str, manifest: BatchManifest) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)


def new_manifest(jobs: list[JobEntry], config_hash_value: str) -> BatchManifest:
    return BatchManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash_value,
        jobs=jobs,
    )


def should_skip_job(
    entry: JobEntry,
    resume_mode: str,
    *,
    output_exists: bool,
) -> bool:
    # resume_mode ごとに「再実行すべきでない job」を判定する。
    # retry_failed: failed のみ再試行、done+出力ありはスキップ。
    if resume_mode == "run_all":
        return False
    if resume_mode == "skip_done":
        return entry.status == "done" and output_exists
    if resume_mode == "retry_failed":
        if entry.status == "failed":
            return False
        return entry.status == "done" and output_exists
    return False
