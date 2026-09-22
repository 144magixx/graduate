"""Read-only legacy capability audit. Never upgrades old satisfaction to new U."""
import csv
import hashlib
from pathlib import Path
from ..audit import sha256


def inspect_legacy(path):
    path = Path(path)
    report = {"source": "legacy_import", "filename": path.name,
              "sha256": sha256(path),
              "exact_step_replay_available": False, "new_utility_available": False,
              "limitations": ["原精度无法恢复", "旧满意度不能自动视为新版U", "未记录字段不可补造"]}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            report["recorded_fields"] = reader.fieldnames or []
            report["row_count"] = sum(1 for _ in reader)
    elif path.suffix.lower() == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            report["recorded_fields"] = list(archive.files)
            report["shapes"] = {name: list(archive[name].shape) for name in archive.files}
    else:
        raise ValueError("Only legacy CSV/NPZ are supported")
    return report
