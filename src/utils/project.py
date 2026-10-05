"""Resolve paths from the repository, record configuration and provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path | None = None) -> tuple[dict, Path]:
    path = project_path(path or "configs/project.yaml")
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream), path


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                       encoding="utf-8").strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain"))}


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
