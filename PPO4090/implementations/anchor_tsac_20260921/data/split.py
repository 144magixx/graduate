"""先按可追溯母场景谱系划分；旧数据只承诺hash隔离。"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ..audit import sha256


def manifest_hash(manifest):
    payload = {k:v for k,v in manifest.items() if k != "manifest_hash"}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()


def semantic_hash(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    canonical = [{k: float(v) for k, v in sorted(row.items()) if k and v} for row in rows]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def grouped_split(records, seed=20260920):
    """同root、split_group、内容hash的传递闭包都在一个split。"""
    records = [dict(r) for r in records]
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, record in enumerate(records):
        for field in ("root_scene_id", "split_group_id", "semantic_hash", "source_hash"):
            value = record.get(field)
            if value:
                key = (field, value)
                if key in seen:
                    parent[find(i)] = find(seen[key])
                else:
                    seen[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    ids = sorted(groups, key=lambda i: str(records[groups[i][0]].get("scenario_id", i)))
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    train_end = int(n * .70)
    val_end = train_end + int(n * .15)
    for j, group in enumerate(ids):
        split = "train" if j < train_end else "validation" if j < val_end else "test"
        for i in groups[group]:
            records[i]["split"] = split
    return records


def legacy_manifest(paths, seed=20260920):
    records = []
    for path in sorted(map(Path, paths)):
        records.append(dict(scenario_id=path.stem, file=path.name, source_hash=sha256(path),
                            semantic_hash=semantic_hash(path), root_scene_id=None,
                            split_group_id=None, generator_family_id=None,
                            provenance_status="unknown_legacy_aggregate"))
    records = grouped_split(records, seed)
    payload = {"dataset_version": "legacy_aggregate.v1", "split_seed": seed,
               "split_unit": "content_hash_unknown_lineage", "strict_root_independence": False,
               "limitations": ["无root/split_group谱系，hash隔离不等同真实业务实例独立"],
               "records": records}
    payload["manifest_hash"] = manifest_hash(payload)
    return payload


def validate_split(records):
    identifiers = set()
    for record in records:
        if record.get("split") not in ("train", "validation", "test"):
            raise ValueError("未知数据split；必须显式为train/validation/test")
        identifier = record.get("scenario_id")
        if not identifier or identifier in identifiers:
            raise ValueError("scenario_id必须非空且在清单内唯一")
        identifiers.add(identifier)
    for field in ("root_scene_id", "split_group_id", "semantic_hash", "source_hash"):
        memberships = {}
        for record in records:
            if record.get(field):
                memberships.setdefault(record[field], set()).add(record["split"])
        overlap = [key for key, values in memberships.items() if len(values) > 1]
        if overlap:
            raise ValueError(f"跨split泄漏：{field}={overlap[:5]}")
    return True

