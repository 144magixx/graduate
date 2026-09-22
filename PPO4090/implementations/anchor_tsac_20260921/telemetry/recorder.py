"""A bounded producer and independent single SQLite writer process.

Every core message is one FULL-synchronous transaction. A failed writer or bounded
enqueue timeout raises RecorderError: the controller must stop safely, never ignore it.
"""
import csv
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import sqlite3
import sys
import time
import traceback
import uuid

from .storage import (DB_NAME, MANIFEST_NAME, SCHEMA_VERSION, create_schema, dumps,
                      readonly, sanitize, state_delta, state_hash)
from ..audit import sha256


class RecorderError(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(dumps(value))
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _add_metrics(conn, seq, message, values, axis, axis_value, aggregation=None):
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (float, int, type(None))):
            continue
        agg = (aggregation or {}).get(name)
        conn.execute("""INSERT INTO metrics(event_seq,episode_id,phase,name,axis,axis_value,value,
            raw_available,aggregation,count,first_axis,last_axis,min,max,last,anomaly)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (seq, message.get("episode_id"), message.get("phase"), name, axis, axis_value, value,
             0 if agg else 1, "window" if agg else "raw", agg["count"] if agg else 1,
             agg["first_axis"] if agg else axis_value, agg["last_axis"] if agg else axis_value,
             agg["min"] if agg else value, agg["max"] if agg else value, agg["last"] if agg else value,
             int(value is None)))


def _commit(conn, message, keyframe_interval):
    # Check identity before applying any side effect, including step writes.
    existing = conn.execute("SELECT event_seq FROM events WHERE event_id=?", (message["event_id"],)).fetchone()
    if existing:
        return existing[0]
    context = [message.get(k) for k in ("event_id", "event_type", "severity", "component", "phase", "episode_id", "scenario_id", "beam_id", "step_index", "env_step", "update_step", "elapsed_seconds", "occurred_at_utc")]
    payload = message.get("payload", {})
    cursor = conn.execute("""INSERT INTO events(event_id,event_type,severity,component,phase,episode_id,
        scenario_id,beam_id,step_index,env_step,update_step,elapsed_seconds,occurred_at_utc,committed_at_utc,payload)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", [*context, utc(), dumps(payload)])
    seq, kind, ep = cursor.lastrowid, message["event_type"], message.get("episode_id")
    if kind in ("episode_started", "resume_anchor"):
        snapshot, step = payload["snapshot"], message["step_index"]
        conn.execute("INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?)", (ep, message.get("scenario_id"), message.get("phase"), step, step, "active", seq, None, None))
        conn.execute("INSERT INTO keyframes VALUES(?,?,?,?)", (ep, step, seq, dumps(snapshot)))
        conn.execute("INSERT INTO state_hashes VALUES(?,?,?,?)", (ep, step, state_hash(snapshot), seq))
    elif kind == "step_completed":
        step, after, delta = message["step_index"], payload.get("keyframe"), payload["delta"]
        episode = conn.execute("SELECT last_step,status FROM episodes WHERE episode_id=?", (ep,)).fetchone()
        if episode is None or episode[1] != "active" or step != episode[0] + 1:
            raise RecorderError(f"Non-contiguous episode step: {ep}/{step}")
        previous_hash = conn.execute("SELECT state_hash FROM state_hashes WHERE episode_id=? AND step_index=?", (ep, step - 1)).fetchone()
        if payload["after_step_index"] != step or not previous_hash or previous_hash[0] != payload["before_hash"]:
            raise RecorderError(f"Step state does not match the committed prefix: {ep}/{step}")
        conn.execute("INSERT INTO transitions VALUES(?,?,?,?,?)", (ep, step, seq, dumps(payload["transition"]), dumps(delta)))
        for change in delta["beams"]:
            conn.execute("INSERT INTO beam_deltas VALUES(?,?,?,?,?,?)", (ep, step, change["beam_id"], seq, dumps(change["before"]), dumps(change["after"])))
        terminal = payload["terminal"]
        if step % keyframe_interval == 0 or terminal:
            if after is None or state_hash(after) != payload["after_hash"]:
                raise RecorderError("Required keyframe missing or hash mismatch")
            conn.execute("INSERT INTO keyframes VALUES(?,?,?,?)", (ep, step, seq, dumps(after)))
        conn.execute("UPDATE episodes SET last_step=?,final_event_seq=? WHERE episode_id=?", (step, seq, ep))
        conn.execute("INSERT INTO state_hashes VALUES(?,?,?,?)", (ep, step, payload["after_hash"], seq))
        _add_metrics(conn, seq, message, payload.get("metrics", {}), "env_step", message.get("env_step"))
        # Full state is in the keyframe/delta tables, not duplicated in the event log.
        conn.execute("UPDATE events SET payload=? WHERE event_seq=?", (dumps({"transition": payload["transition"], "changed_beams": len(delta["beams"]),
                     "snapshot_nonfinite_fields": payload.get("snapshot_nonfinite_fields", []),
                     "nonfinite_fields": payload.get("nonfinite_fields", []), "recent_steps": payload.get("recent_steps", []),
                     "state_hash": payload["after_hash"]}), seq))
    elif kind == "episode_completed":
        status = "truncated" if payload.get("truncated") else "completed" if payload.get("terminated") else "interrupted"
        conn.execute("UPDATE episodes SET status=?,summary=?,final_event_seq=? WHERE episode_id=?", (status, dumps(payload["metrics"]), seq, ep))
        _add_metrics(conn, seq, message, payload["metrics"], "episode", payload["episode_axis"])
    elif kind in ("optimizer_update", "optimizer_window"):
        _add_metrics(conn, seq, message, payload["values"], "update_step", message.get("update_step"), payload.get("aggregation"))
    elif kind == "performance":
        _add_metrics(conn, seq, message, payload, "elapsed_seconds", message.get("elapsed_seconds"))
    elif kind == "run_finished":
        conn.execute("UPDATE runs SET status=?", (payload["status"],))
        manifest = json.loads(conn.execute("SELECT manifest FROM runs LIMIT 1").fetchone()[0])
        manifest.update(status=payload["status"], durable_event_seq=seq)
        conn.execute("UPDATE runs SET manifest=?", (dumps(manifest),))
        conn.execute("UPDATE episodes SET status='interrupted' WHERE status='active'")
    elif kind == "artifact_published":
        conn.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?)", (payload["artifact_id"], seq, payload["path"], payload["sha256"], payload.get("kind"), "available"))
    return seq


def _writer(run_dir, manifest, inbox, replies, errors, queued_bytes, keyframe_interval, flush_interval):
    conn = None
    mirror = None
    mirror_index, mirrored_seq, mirror_flushed = 0, 0, time.monotonic()
    try:
        conn = sqlite3.connect(str(Path(run_dir) / DB_NAME), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        create_schema(conn)
        with conn:
            conn.execute("INSERT INTO runs VALUES(?,?,?)", (manifest["run_id"], dumps(manifest), "active"))
        mirror_path = Path(run_dir) / "诊断日志.jsonl"
        try:
            mirror = mirror_path.open("a", encoding="utf-8")
        except OSError as error:
            print(f"Diagnostic mirror unavailable; SQLite remains authoritative: {error}", file=sys.stderr, flush=True)
        while True:
            packed = inbox.get()
            try:
                message = json.loads(packed)
                if message.get("command") in ("flush", "stop"):
                    watermark = conn.execute("SELECT COALESCE(MAX(event_seq),0) FROM events").fetchone()[0]
                    conn.commit()
                    if mirror:
                        try:
                            mirror.flush()
                        except OSError as error:
                            print(f"Diagnostic mirror flush failed; SQLite prefix committed: {error}", file=sys.stderr, flush=True)
                            try:
                                mirror.close()
                            except OSError:
                                pass
                            mirror = None
                    if message["command"] == "stop":
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    replies.put((message["barrier_id"], watermark))
                    if message["command"] == "stop":
                        break
                else:
                    with conn:
                        seq = _commit(conn, message, keyframe_interval)
                    if mirror and seq > mirrored_seq:
                        try:
                            event = conn.execute("SELECT event_id,event_type,payload,committed_at_utc FROM events WHERE event_seq=?", (seq,)).fetchone()
                            mirror.write(dumps({"event_seq": seq, "event_id": event[0], "event_type": event[1],
                                               "payload": json.loads(event[2]), "committed_at_utc": event[3], "run_id": manifest["run_id"]}) + "\n")
                            mirrored_seq = seq
                            if time.monotonic() - mirror_flushed >= flush_interval:
                                mirror.flush()
                                mirror_flushed = time.monotonic()
                            if mirror.tell() >= 10 * 1024 * 1024:
                                mirror.close()
                                mirror_index += 1
                                mirror_path.replace(Path(run_dir) / f"诊断日志-{mirror_index:04d}.jsonl")
                                mirror = mirror_path.open("a", encoding="utf-8")
                        except OSError as error:
                            print(f"Diagnostic mirror degraded; committed SQLite data intact: {error}", file=sys.stderr, flush=True)
                            try:
                                mirror.close()
                            except OSError:
                                pass
                            mirror = None
            finally:
                with queued_bytes.get_lock():
                    queued_bytes.value -= len(packed)
    except BaseException:
        detail = traceback.format_exc()
        errors.put(detail)
        print("Recorder writer failure; training must stop:\n" + detail, file=sys.stderr, flush=True)
        try:
            atomic_json(Path(run_dir) / "记录失败后备.json", {"occurred_at_utc": utc(), "error": detail,
                        "note": "尽力写入；磁盘失败时不能保证后备文件可用"})
        except OSError:
            pass
    finally:
        if conn is not None:
            conn.close()
        if mirror:
            try:
                mirror.close()
            except OSError:
                pass


class Recorder:
    def __init__(self, run_dir, manifest, config=None):
        from ..config import TelemetryConfig
        self.config = config or TelemetryConfig()
        if isinstance(self.config, dict):
            self.config = TelemetryConfig(**self.config)
        if (self.config.queue_max_messages < 1 or self.config.queue_max_bytes < 1
                or self.config.keyframe_interval < 1 or self.config.update_window < 1
                or self.config.enqueue_timeout <= 0 or self.config.flush_interval <= 0):
            raise ValueError("Telemetry queue limits, intervals and timeouts must be positive")
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / DB_NAME).exists():
            raise RecorderError("Existing run database is immutable; resume into a new run directory")
        self.manifest = dict(sanitize(manifest))
        for key in ("run_id", "trial_id", "attempt_id"):
            self.manifest.setdefault(key, str(uuid.uuid4()))
        self.manifest.update(telemetry_schema=SCHEMA_VERSION, created_at_utc=utc(),
                             storage={"journal_mode": "WAL", "synchronous": "FULL", "sqlite_version": sqlite3.sqlite_version,
                                      "keyframe_interval": self.config.keyframe_interval, "commit_policy": "one_core_message_per_transaction"})
        atomic_json(self.run_dir / MANIFEST_NAME, self.manifest)
        ctx = mp.get_context("spawn")
        self._inbox = ctx.Queue(maxsize=self.config.queue_max_messages)
        self._replies, self._errors = ctx.Queue(), ctx.Queue()
        self._bytes = ctx.Value("q", 0)
        self._process = ctx.Process(target=_writer, args=(str(self.run_dir), self.manifest, self._inbox,
                                    self._replies, self._errors, self._bytes, self.config.keyframe_interval, self.config.flush_interval), daemon=True)
        self._started = time.monotonic()
        self._closed = False
        self._failure = None
        self._window = []
        self._recent_steps = deque(maxlen=20)
        self._episode_axis = 0
        self._process.start()
        self.event("run_started", {"manifest": self.manifest})
        self.flush()

    @property
    def run_id(self):
        return self.manifest["run_id"]

    def _healthy(self):
        if self._failure:
            raise RecorderError(self._failure)
        try:
            self._failure = self._errors.get_nowait()
        except queue.Empty:
            pass
        if self._failure or not self._process.is_alive():
            self._failure = self._failure or f"Recorder writer exited with code {self._process.exitcode}"
            raise RecorderError(self._failure)
        if self._closed:
            raise RecorderError("Recorder already closed")

    def _enqueue(self, message, optional=False):
        self._healthy()
        packed = dumps(message).encode("utf-8")
        size = len(packed)
        if size > self.config.queue_max_bytes:
            if optional:
                return False
            raise RecorderError(f"Core telemetry message {size} bytes exceeds queue byte limit")
        deadline = time.monotonic() + self.config.enqueue_timeout
        while True:
            self._healthy()
            reserved = False
            with self._bytes.get_lock():
                if self._bytes.value + size <= self.config.queue_max_bytes:
                    self._bytes.value += size
                    reserved = True
            if reserved:
                try:
                    self._inbox.put(packed, block=False)
                    return True
                except queue.Full:
                    with self._bytes.get_lock():
                        self._bytes.value -= size
            if optional:
                return False
            if time.monotonic() >= deadline:
                self._failure = "Core telemetry queue backpressure timed out; controlled stop required"
                raise RecorderError(self._failure)
            time.sleep(0.01)

    def event(self, event_type, payload=None, *, event_id=None, optional=False, **context):
        invalid = []
        safe = sanitize(payload or {}, invalid=invalid)
        if invalid:
            safe = dict(safe, nonfinite_fields=invalid)
        if (invalid or context.get("severity") == "ERROR") and self._recent_steps:
            safe["recent_steps"] = list(self._recent_steps)
        message = {"event_id": event_id or str(uuid.uuid4()), "event_type": event_type,
                   "severity": "ERROR" if invalid else "INFO", "component": "controller",
                   "occurred_at_utc": utc(), "elapsed_seconds": time.monotonic() - self._started,
                   **context, "payload": safe}
        accepted = self._enqueue(message, optional=optional)
        if accepted and event_type == "step_completed":
            self._recent_steps.append({"episode_id": context.get("episode_id"), "step_index": context.get("step_index"),
                                       "transition": safe.get("transition")})
        return message["event_id"] if accepted else None

    def start_episode(self, episode_id, scenario_id, snapshot, phase="train", env_step=None):
        step = int(snapshot.get("step_index", 0))
        self.event("resume_anchor" if step else "episode_started", {"snapshot": snapshot}, episode_id=episode_id,
                   scenario_id=scenario_id, step_index=step, phase=phase, env_step=env_step)

    def record_step(self, episode_id, scenario_id, step_index, before, after, transition, env_step=None, phase="train", event_id=None):
        invalid = []
        try:
            before_hash, after_hash = state_hash(before), state_hash(after)
        except (TypeError, ValueError):
            before, after = sanitize(before, invalid=invalid), sanitize(after, invalid=invalid)
            before_hash, after_hash = state_hash(before), state_hash(after)
        terminal = bool(after.get("terminal") or after.get("terminated") or after.get("truncated") or transition.get("terminated") or transition.get("truncated"))
        keyframe = after if step_index % self.config.keyframe_interval == 0 or terminal else None
        self.event("step_completed", {"keyframe": keyframe, "before_hash": before_hash, "after_hash": after_hash,
                   "after_step_index": after.get("step_index"), "terminal": terminal, "metrics": after.get("metrics", {}),
                   "delta": state_delta(before, after), "transition": transition,
                   "snapshot_nonfinite_fields": invalid}, event_id=event_id, episode_id=episode_id,
                   scenario_id=scenario_id, step_index=step_index, env_step=env_step, phase=phase, severity="ERROR" if invalid else "INFO",
                   beam_id=str(transition.get("acted_beam_id")) if transition.get("acted_beam_id") is not None else None)

    def record_update(self, diagnostics, update_step, env_step=None, phase="train"):
        if self._window and self._window[-1][3] != phase:
            self._flush_window()
        invalid = []
        values = sanitize(diagnostics, invalid=invalid)
        if update_step == 1 or invalid:
            self.event("optimizer_update", {"values": values, "nonfinite_fields": invalid},
                       update_step=update_step, env_step=env_step, phase=phase, severity="ERROR" if invalid else "INFO", component="optimizer")
            return
        self._window.append((update_step, values, env_step, phase))
        if len(self._window) >= self.config.update_window:
            self._flush_window()

    def _flush_window(self):
        if not self._window:
            return
        aggregation = {}
        keys = set().union(*(value.keys() for _, value, _, _ in self._window))
        for key in keys:
            pairs = [(step, value[key]) for step, value, _, _ in self._window if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)]
            if pairs:
                vals = [value for _, value in pairs]
                aggregation[key] = dict(count=len(vals), mean=sum(vals) / len(vals), min=min(vals), max=max(vals), last=vals[-1], first_axis=pairs[0][0], last_axis=pairs[-1][0])
        step, _, env_step, phase = self._window[-1]
        references = [{"update_step": u, "sample_references": values.get("sample_references", [])}
                      for u, values, _, _ in self._window if values.get("sample_references")]
        self.event("optimizer_window", {"values": {k: v["mean"] for k, v in aggregation.items()}, "aggregation": aggregation,
                   "batch_references": references},
                   update_step=step, env_step=env_step, phase=phase, component="optimizer")
        self._window.clear()

    def end_episode(self, episode_id, metrics, *, scenario_id=None, env_step=None, terminated=True, truncated=False, phase="train"):
        self._episode_axis += 1
        self.event("episode_completed", {"metrics": metrics, "episode_axis": self._episode_axis,
                   "terminated": terminated, "truncated": truncated}, episode_id=episode_id,
                   scenario_id=scenario_id, env_step=env_step, phase=phase)
        return self.flush()

    def publish_artifact(self, path, kind="checkpoint", artifact_id=None):
        path = Path(path).resolve()
        relative = path.relative_to(self.run_dir)
        digest = sha256(path)
        artifact_id = artifact_id or str(uuid.uuid4())
        self.event("artifact_published", dict(artifact_id=artifact_id, path=relative.as_posix(), sha256=digest, kind=kind))
        self.flush()
        return artifact_id

    def _barrier(self, command):
        token = str(uuid.uuid4())
        self._enqueue({"command": command, "barrier_id": token})
        deadline = time.monotonic() + max(30, self.config.enqueue_timeout)
        while time.monotonic() < deadline:
            try:
                received, watermark = self._replies.get(timeout=.1)
                if received == token:
                    return watermark
            except queue.Empty:
                self._healthy()
        raise RecorderError("Recorder flush barrier timed out")

    def flush(self):
        self._flush_window()
        return self._barrier("flush")

    def close(self, status="completed"):
        if self._closed:
            return
        try:
            self._flush_window()
            self.event("run_finished", {"status": status})
            watermark = self._barrier("stop")
            self._process.join(timeout=5)
            self.manifest.update(status=status, durable_event_seq=watermark)
            atomic_json(self.run_dir / MANIFEST_NAME, self.manifest)
        finally:
            self._closed = True
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            for channel in (self._inbox, self._replies, self._errors):
                channel.cancel_join_thread()
                channel.close()

    def export(self, destination):
        if not self._closed:
            self.flush()
        return export_run(self.run_dir, destination)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        self.close("failed" if typ else "completed")


def export_run(run_dir, destination):
    """Online backup fixes one committed watermark; copy only its referenced artifacts."""
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if destination == run_dir or run_dir in destination.parents:
        raise ValueError("Export destination must be outside its source run")
    destination.mkdir(parents=True, exist_ok=False)
    source = readonly(run_dir / DB_NAME)
    target = sqlite3.connect(str(destination / DB_NAME))
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    with closing(readonly(destination / DB_NAME)) as conn:
        run = conn.execute("SELECT manifest,status FROM runs").fetchone()
        manifest = json.loads(run[0])
        manifest["status"] = run[1]
        manifest["export_watermark"] = conn.execute("SELECT COALESCE(MAX(event_seq),0) FROM events").fetchone()[0]
        manifest["exported_at_utc"] = utc()
        for artifact in conn.execute("SELECT * FROM artifacts WHERE status='available'"):
            path = (run_dir / artifact["path"]).resolve()
            if run_dir not in path.parents or sha256(path) != artifact["sha256"]:
                raise RecorderError(f"Artifact missing, unsafe or hash mismatch: {artifact['artifact_id']}")
            out = destination / artifact["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            if sha256(out) != artifact["sha256"]:
                raise RecorderError(f"Artifact changed while exporting: {artifact['artifact_id']}")
        exports = {"训练回合指标.csv": "SELECT * FROM episodes WHERE phase='train'", "优化更新指标.csv": "SELECT * FROM metrics WHERE axis='update_step'",
                   "分配步骤.csv": "SELECT * FROM transitions", "波束状态变化.csv": "SELECT * FROM beam_deltas",
                   "评估场景指标.csv": "SELECT * FROM episodes WHERE phase IN ('validation','test')"}
        for filename, sql in exports.items():
            result = conn.execute(sql)
            with (destination / filename).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                context_keys = ("run_id", "physics_version", "reward_version", "metric_version", "export_watermark")
                context = [manifest.get(key, manifest.get("versions", {}).get(key)) for key in context_keys]
                writer.writerow([*context_keys, *[column[0] for column in result.description]])
                writer.writerows([*context, *tuple(row)] for row in result)
    try:
        from project_paths import ANCHOR_FRONTEND_DIR
        built = ANCHOR_FRONTEND_DIR / "dist"
        if built.exists():
            shutil.copytree(built, destination / "静态页面")
    except ImportError:
        pass
    atomic_json(destination / MANIFEST_NAME, manifest)
    hashes = {p.relative_to(destination).as_posix(): sha256(p)
              for p in destination.rglob("*") if p.is_file()}
    atomic_json(destination / "数据哈希.json", hashes)
    return destination
