"""Keep the existing shared CSV schema and refuse accidental result replacement."""
import csv
import os
import re
from pathlib import Path

FIELDS = ["exp_id", "date", "owner", "git_commit", "config_path", "features", "model", "params",
          "train_period", "valid_period", "ic_mean", "ic_std", "icir", "ic_positive_ratio",
          "annual_excess", "top1_annual_ret", "mean_turnover", "final_score", "artifact_path", "notes"]


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Experiment log header differs from the shared schema.")
        return list(reader)


def check_experiment_id(exp_id: str, log_path: Path, directories: list[Path]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", exp_id):
        raise ValueError("Invalid experiment ID; use letters, numbers, underscore or dash.")
    if any(row["exp_id"] == exp_id for row in read_records(log_path)):
        raise ValueError(f"Experiment is already logged: {exp_id}")
    if any(path.exists() for path in directories):
        raise FileExistsError(f"Artifacts already exist for {exp_id}; choose a new experiment ID.")


def append_record(path: Path, record: dict) -> None:
    records = read_records(path)
    if any(row["exp_id"] == record["exp_id"] for row in records):
        raise ValueError(f"Duplicate experiment ID: {record['exp_id']}")
    if set(record) != set(FIELDS):
        raise ValueError("Experiment record fields differ from the shared schema.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records + [record])
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
