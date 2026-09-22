"""Versioned SQLite contract and deterministic recorded-state reconstruction."""
import json
import hashlib
import math
import sqlite3
from pathlib import Path

DB_NAME = "运行记录.sqlite"
MANIFEST_NAME = "运行清单.json"
SCHEMA_VERSION = "horizon.telemetry.v1"


def sanitize(value, path="", invalid=None):
    invalid = [] if invalid is None else invalid
    if hasattr(value, "detach"):
        raise TypeError("Recorder requires an immutable CPU copy, not a tensor")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append({"path": path, "kind": "nan" if math.isnan(value) else "infinity"})
        return None
    if isinstance(value, dict):
        return {str(k): sanitize(v, f"{path}.{k}".strip("."), invalid) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v, f"{path}[{i}]", invalid) for i, v in enumerate(value)]
    if isinstance(value, Path):
        return str(value)
    return value


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def state_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def create_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, manifest TEXT NOT NULL, status TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS events (
      event_seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
      event_type TEXT NOT NULL, severity TEXT, component TEXT, phase TEXT,
      episode_id TEXT, scenario_id TEXT, beam_id TEXT, step_index INTEGER,
      env_step INTEGER, update_step INTEGER, elapsed_seconds REAL,
      occurred_at_utc TEXT, committed_at_utc TEXT, payload TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS events_context ON events(episode_id,step_index,event_seq);
    CREATE INDEX IF NOT EXISTS events_filter ON events(event_type,severity,beam_id,event_seq);
    CREATE TABLE IF NOT EXISTS episodes (
      episode_id TEXT PRIMARY KEY, scenario_id TEXT, phase TEXT, start_step INTEGER,
      last_step INTEGER, status TEXT, initial_event_seq INTEGER, final_event_seq INTEGER,
      summary TEXT, FOREIGN KEY(initial_event_seq) REFERENCES events(event_seq));
    CREATE TABLE IF NOT EXISTS keyframes (
      episode_id TEXT, step_index INTEGER, event_seq INTEGER, snapshot TEXT NOT NULL,
      PRIMARY KEY(episode_id,step_index));
    CREATE TABLE IF NOT EXISTS transitions (
      episode_id TEXT, step_index INTEGER, event_seq INTEGER, transition TEXT NOT NULL,
      delta TEXT NOT NULL, PRIMARY KEY(episode_id,step_index));
    CREATE TABLE IF NOT EXISTS state_hashes (
      episode_id TEXT,step_index INTEGER,state_hash TEXT NOT NULL,event_seq INTEGER,
      PRIMARY KEY(episode_id,step_index));
    CREATE TABLE IF NOT EXISTS beam_deltas (
      episode_id TEXT, step_index INTEGER, beam_id TEXT, event_seq INTEGER,
      before_value TEXT, after_value TEXT, PRIMARY KEY(episode_id,step_index,beam_id));
    CREATE TABLE IF NOT EXISTS metrics (
      metric_id INTEGER PRIMARY KEY AUTOINCREMENT, event_seq INTEGER, episode_id TEXT,
      phase TEXT, name TEXT, axis TEXT, axis_value REAL, value REAL,
      raw_available INTEGER, aggregation TEXT, count INTEGER, first_axis REAL,
      last_axis REAL, min REAL, max REAL, last REAL, anomaly INTEGER DEFAULT 0);
    CREATE INDEX IF NOT EXISTS metrics_native ON metrics(name,axis,phase,axis_value);
    CREATE INDEX IF NOT EXISTS metrics_native_order ON metrics(name,axis,axis_value,metric_id);
    CREATE TABLE IF NOT EXISTS artifacts (
      artifact_id TEXT PRIMARY KEY, event_seq INTEGER, path TEXT, sha256 TEXT,
      kind TEXT, status TEXT);
    """)


def state_delta(before, after):
    changed = {k: v for k, v in after.items() if k != "beams" and before.get(k) != v}
    removed = [k for k in before if k not in after]
    old = {str(b["beam_id"]): b for b in before.get("beams", [])}
    new = {str(b["beam_id"]): b for b in after.get("beams", [])}
    beams = []
    for key in dict.fromkeys([*old, *new]):
        left, right = old.get(key), new.get(key)
        if left == right:
            continue
        if left is None or right is None:
            beams.append({"beam_id": key, "before": left, "after": right, "mode": "replace"})
        else:
            fields = {field for field in left.keys() | right.keys() if left.get(field) != right.get(field)}
            # Always include the two business values, but do not repeatedly store
            # invariant polygons/coordinates for every interference update.
            fields.update(field for field in ("rate_bps", "satisfaction") if field in right)
            beams.append({"beam_id": key, "mode": "patch", "before": {f: left[f] for f in fields if f in left},
                          "after": {f: right[f] for f in fields if f in right}, "remove_fields": list(fields - right.keys())})
    return {"set": changed, "remove": removed, "beams": beams,
            "beam_order": [str(b["beam_id"]) for b in after.get("beams", [])]}


def apply_delta(state, delta):
    state.update(delta["set"])
    for key in delta["remove"]:
        state.pop(key, None)
    beams = {str(b["beam_id"]): b for b in state.get("beams", [])}
    for b in delta["beams"]:
        if b["after"] is None:
            beams.pop(b["beam_id"], None)
        elif b.get("mode") == "patch":
            beams[b["beam_id"]].update(b["after"])
            for field in b.get("remove_fields", []):
                beams[b["beam_id"]].pop(field, None)
        else:
            beams[b["beam_id"]] = b["after"]
    state["beams"] = [beams[k] for k in delta["beam_order"]]
    return state


def reconstruct(conn, episode_id, step):
    """Caller owns one short read transaction so all fields share one watermark."""
    episode = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
    if episode is None:
        return {"status": "unavailable", "reason": "unknown_episode"}
    bounds = [episode["start_step"], episode["last_step"]]
    if step < bounds[0] or step > bounds[1]:
        return {"status": "unavailable", "available_range": bounds, "requested_step": step}
    frame = conn.execute("SELECT * FROM keyframes WHERE episode_id=? AND step_index<=? ORDER BY step_index DESC LIMIT 1", (episode_id, step)).fetchone()
    if frame is None:
        return {"status": "unavailable", "reason": "missing_keyframe", "available_range": bounds}
    state = json.loads(frame["snapshot"])
    seq, at = frame["event_seq"], frame["step_index"]
    for row in conn.execute("SELECT * FROM transitions WHERE episode_id=? AND step_index>? AND step_index<=? ORDER BY step_index", (episode_id, at, step)):
        if row["step_index"] != at + 1:
            return {"status": "unavailable", "reason": "trace_gap", "last_complete_step": at, "available_range": bounds}
        apply_delta(state, json.loads(row["delta"]))
        at, seq = row["step_index"], row["event_seq"]
    if at != step:
        return {"status": "unavailable", "reason": "trace_gap", "last_complete_step": at}
    verified = False
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_hashes'").fetchone():
        checksum = conn.execute("SELECT state_hash FROM state_hashes WHERE episode_id=? AND step_index=?", (episode_id, step)).fetchone()
        if not checksum or checksum[0] != state_hash(state):
            return {"status": "corrupted", "reason": "state_hash_mismatch" if checksum else "state_hash_missing", "requested_step": step,
                    "available_range": bounds, "committed_event_seq": seq}
        verified = True
    manifest = json.loads(conn.execute("SELECT manifest FROM runs LIMIT 1").fetchone()[0])
    physics = state.get("physics_version", manifest.get("physics_version", "unspecified"))
    detail_available = bool(state.get("detail")) or any(beam.get("sinr_db") is not None for beam in state.get("beams", []))
    return {"status": "recorded", "source": "recorded", "state": state, "state_hash_verified": verified,
            "snapshot_id": f"{manifest['run_id']}:{episode_id}:{step}:{physics}",
            "committed_event_seq": seq, "watermark": conn.execute("SELECT COALESCE(MAX(event_seq),0) FROM events").fetchone()[0],
            "available_range": bounds, "episode_status": episode["status"],
            "detail": {"available": detail_available, "source": "recorded" if detail_available else "unavailable"}}

