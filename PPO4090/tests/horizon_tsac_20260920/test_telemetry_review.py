"""独立交叉审查回归；独立诊断输入不冒充训练记录。"""
import asyncio
import copy
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from implementations.horizon_tsac_20260920.dashboard.api import create_app
from implementations.horizon_tsac_20260920.telemetry.recorder import Recorder, _commit, export_run
from implementations.horizon_tsac_20260920.telemetry.storage import DB_NAME, MANIFEST_NAME, create_schema, dumps, readonly, reconstruct


def make_database(folder, identifier="review"):
    from implementations.horizon_tsac_20260920.config import Config
    from implementations.horizon_tsac_20260920.env.action import ActionSpec
    folder.mkdir(parents=True)
    manifest = {"run_id": identifier, "trial_id": identifier, "attempt_id": identifier, "source": "diagnostic_fixture",
                "physics_version": "review.v1", "reward_version": "review.v1", "metric_version": "review.v1",
                "dataset_version": "review.v1", "split_hash": "review", "power_budget_w": 6000,
                "action_version": "review.v1", "seed": 0}
    manifest["config"] = Config().to_dict()
    manifest["action_spec_signature"] = ActionSpec().signature()
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    conn = sqlite3.connect(folder / DB_NAME)
    conn.row_factory = sqlite3.Row
    create_schema(conn)
    with conn:
        conn.execute("INSERT INTO runs VALUES(?,?,?)", (identifier, dumps(manifest), "completed"))
    return conn


def message(event_id, kind, payload, **context):
    return {"event_id": event_id, "event_type": kind, "phase": "train", "payload": payload, **context}


def snapshot(step):
    return {"scenario_id": "diagnostic", "step_index": step, "terminal": step == 2,
            "beams": [{"beam_id": 10, "rate_bps": step*1e8}], "resources": {"power_used_w": step*5},
            "metrics": {"mean_satisfaction": step*.1}}


def step_payload(before, after):
    from implementations.horizon_tsac_20260920.telemetry.storage import state_delta, state_hash
    return {"keyframe": after, "before_hash": state_hash(before), "after_hash": state_hash(after),
            "after_step_index": after["step_index"], "terminal": after["terminal"], "metrics": after["metrics"],
            "delta": state_delta(before, after), "transition": {}}


class TelemetryReviewTests(unittest.TestCase):
    def test_integrity_hashes_new_files_only_after_closing_read_transaction(self):
        import hashlib
        from unittest.mock import patch
        import implementations.horizon_tsac_20260920.dashboard.api as api
        from implementations.horizon_tsac_20260920.audit import sha256
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"run"
            conn = make_database(folder)
            first = folder/"first.bin"
            first.write_bytes(b"first")
            with conn:
                _commit(conn, message("artifact-one", "artifact_published", {"artifact_id": "one", "path": first.name,
                        "sha256": hashlib.sha256(b"first").hexdigest(), "kind": "diagnostic"}), 20)
            conn.close()
            client = TestClient(create_app(Path(temp)))
            calls, opened = [], []
            original_readonly = api.readonly
            def track_readonly(path):
                connection = original_readonly(path)
                opened.append(connection)
                return connection
            def verified_hash(path):
                with self.assertRaises(sqlite3.ProgrammingError):
                    opened[-1].execute("SELECT 1")  # 最新读连接须在文件I/O前已关闭
                calls.append(Path(path).name)
                return sha256(path)
            with patch.object(api, "readonly", side_effect=track_readonly), patch.object(api, "sha256", side_effect=verified_hash):
                self.assertEqual(client.get("/api/v1/runs/review/integrity").json()["status"], "verified")
                second = folder/"second.bin"
                second.write_bytes(b"second")
                conn = sqlite3.connect(folder/DB_NAME)
                with conn:
                    _commit(conn, message("artifact-two", "artifact_published", {"artifact_id": "two", "path": second.name,
                            "sha256": hashlib.sha256(b"second").hexdigest(), "kind": "diagnostic"}), 20)
                conn.close()
                self.assertEqual(client.get("/api/v1/runs/review/integrity").json()["status"], "verified")
                self.assertEqual(calls, ["first.bin", "second.bin"])

    def test_integrity_reports_orphan_hash_damage_and_missing_reference(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"run"
            with Recorder(folder, {"run_id": "integrity-review"}) as recorder:
                snapshots = folder/"场景快照"
                snapshots.mkdir()
                artifact = snapshots/"实际场景.json"
                artifact.write_text("A", encoding="utf-8")
                recorder.publish_artifact(artifact, kind="scenario")
            client = TestClient(create_app(Path(temp)))
            endpoint = "/api/v1/runs/integrity-review/integrity"
            self.assertEqual(client.get(endpoint).json()["status"], "verified")
            orphan = snapshots/"未发布场景.json"
            orphan.write_text("uncommitted", encoding="utf-8")
            body = client.get(endpoint).json()
            self.assertEqual(body["status"], "orphan_warning")
            self.assertIn("场景快照/未发布场景.json", body["unreferenced_snapshots"])
            stat = artifact.stat()
            artifact.write_text("B", encoding="utf-8")
            os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns+10000000))
            body = client.get(endpoint).json()
            self.assertEqual(body["status"], "corrupted")
            self.assertEqual(body["failures"][0]["reason"], "hash_mismatch")
            artifact.unlink()
            body = client.get(endpoint).json()
            self.assertEqual(body["failures"][0]["reason"], "missing")
            self.assertEqual(client.get("/api/v1/runs").json()["items"][0]["integrity"]["status"], "corrupted")
            self.assertTrue(orphan.is_file())  # 审计不删除/修补原始记录

    def test_before_hash_mismatch_rolls_back_and_keyframe_corruption_detected(self):
        from implementations.horizon_tsac_20260920.telemetry.recorder import RecorderError
        with tempfile.TemporaryDirectory() as temp:
            conn = make_database(Path(temp)/"run")
            try:
                with conn:
                    _commit(conn, message("start", "episode_started", {"snapshot": snapshot(0)}, episode_id="ep", step_index=0), 20)
                wrong_before = snapshot(0)
                wrong_before["resources"]["power_used_w"] = 123
                with self.assertRaises(RecorderError), conn:
                    _commit(conn, message("bad-prefix", "step_completed", step_payload(wrong_before, snapshot(1)), episode_id="ep", step_index=1), 20)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM events WHERE event_id='bad-prefix'").fetchone()[0], 0)
                with conn:
                    corrupted = snapshot(0)
                    corrupted["beams"][0]["rate_bps"] = 999.
                    conn.execute("UPDATE keyframes SET snapshot=? WHERE episode_id='ep'", (dumps(corrupted),))
                self.assertEqual(reconstruct(conn, "ep", 0)["status"], "corrupted")
            finally:
                conn.close()

    def test_comparison_uses_common_evaluation_phase_and_actual_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for identifier, train_utility, test_utility in (("left", .9, .2), ("right", .1, .4)):
                conn = make_database(root/identifier, identifier)
                with conn:
                    for phase, utility in (("train", train_utility), ("test", test_utility)):
                        conn.execute("INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?)", (phase, "same-scene", phase, 0, 1, "completed", 0, 0,
                                     dumps({"mean_satisfaction": utility})))
                conn.close()
            client = TestClient(create_app(root))
            body = client.get("/api/v1/comparisons?run_ids=left,right").json()
            self.assertTrue(body["compatible"])
            self.assertEqual(body["phase"], "test")
            self.assertAlmostEqual(body["paired_differences"][0]["delta_U"], .2)
            self.assertEqual(client.get("/api/v1/comparisons?run_ids=left,right&phase=train").status_code, 422)
            path = root/"right"/MANIFEST_NAME
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["config"]["data"]["traffic_scale"] = .5
            path.write_text(json.dumps(manifest), encoding="utf-8")
            body = client.get("/api/v1/comparisons?run_ids=left,right").json()
            self.assertFalse(body["compatible"])
            self.assertEqual(body["paired_differences"], [])

    def test_export_hash_inventory_refers_only_to_persistent_files(self):
        import gc
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with Recorder(root/"run", {"run_id": "export-review"}) as recorder:
                artifact = root/"run"/"实际场景.json"
                artifact.write_text('{"seed":17}', encoding="utf-8")
                recorder.publish_artifact(artifact, kind="scenario")
                out = recorder.export(root/"export")
            gc.collect()
            hashes = json.loads((out/"数据哈希.json").read_text(encoding="utf-8"))
            self.assertTrue((out/"实际场景.json").is_file())
            for relative, digest in hashes.items():
                path = out/relative
                self.assertTrue(path.is_file(), relative)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_snapshot_nonfinite_is_visible_as_error_event(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"run"
            with Recorder(folder, {"run_id": "nan-review"}) as recorder:
                recorder.start_episode("ep", "diagnostic", snapshot(0))
                after = snapshot(1)
                after["beams"][0]["rate_bps"] = float("nan")
                recorder.record_step("ep", "diagnostic", 1, snapshot(0), after, {})
                recorder.flush()
            conn = readonly(folder/DB_NAME)
            try:
                row = conn.execute("SELECT severity,payload FROM events WHERE event_type='step_completed'").fetchone()
                self.assertEqual(row["severity"], "ERROR")
                self.assertTrue(json.loads(row["payload"]).get("snapshot_nonfinite_fields"))
            finally:
                conn.close()

    def test_null_is_not_zero_in_viewport_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"run"
            conn = make_database(folder)
            with conn:
                for step, value in enumerate((1., None, 3.)):
                    _commit(conn, message(str(step), "optimizer_update", {"values": {"actor_loss": value}}, update_step=step), 20)
            conn.close()
            client = TestClient(create_app(Path(temp)))
            body = client.get("/api/v1/runs/review/metrics?name=actor_loss&axis=update_step&limit=1").json()
            self.assertEqual(body["items"][0]["value"], 2.)
            self.assertGreater(body["items"][0]["anomaly"], 0)

    def test_optimizer_window_keeps_phase_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"run"
            with Recorder(folder, {"run_id": "phase-review"}) as recorder:
                recorder.record_update({"actor_loss": 1.}, 2, phase="train")
                recorder.record_update({"actor_loss": 9.}, 3, phase="validation")
                recorder.flush()
            with readonly(folder/DB_NAME) as conn:
                rows = conn.execute("SELECT phase,value FROM metrics WHERE name='actor_loss' ORDER BY metric_id").fetchall()
                self.assertEqual([(r["phase"], r["value"]) for r in rows], [("train", 1.), ("validation", 9.)])
            conn.close()

    def test_rollback_leaves_no_partial_event_or_transition(self):
        from implementations.horizon_tsac_20260920.telemetry.storage import state_delta
        with tempfile.TemporaryDirectory() as temp:
            conn = make_database(Path(temp)/"run")
            with conn:
                _commit(conn, message("start", "episode_started", {"snapshot": snapshot(0)}, episode_id="ep", step_index=0), 20)
            with self.assertRaises(Exception), conn:
                _commit(conn, message("bad", "step_completed", step_payload(snapshot(0), snapshot(2)), episode_id="ep", step_index=2), 20)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM events WHERE event_id='bad'").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0], 0)
            self.assertEqual(reconstruct(conn, "ep", 0)["state"], snapshot(0))
            conn.close()

    def test_trace_gap_is_reported_and_resume_anchor_has_local_bounds(self):
        from implementations.horizon_tsac_20260920.telemetry.storage import state_delta
        with tempfile.TemporaryDirectory() as temp:
            conn = make_database(Path(temp)/"run")
            with conn:
                _commit(conn, message("start", "resume_anchor", {"snapshot": snapshot(10)}, episode_id="ep", step_index=10), 20)
                for step in (11, 12):
                    _commit(conn, message(str(step), "step_completed", step_payload(snapshot(step-1), snapshot(step)), episode_id="ep", step_index=step), 20)
            self.assertEqual(reconstruct(conn, "ep", 9)["available_range"], [10, 12])
            with conn:
                conn.execute("DELETE FROM transitions WHERE episode_id='ep' AND step_index=11")
            self.assertEqual(reconstruct(conn, "ep", 12)["reason"], "trace_gap")
            conn.close()

    def test_sse_cursor_ahead_resync(self):
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as temp:
            conn = make_database(Path(temp)/"run")
            with conn:
                _commit(conn, message("one", "test", {}), 20)
            conn.close()
            app = create_app(Path(temp))
            endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "").endswith("/stream"))
            class Request:
                headers = {"last-event-id": "999"}
                async def is_disconnected(self):
                    return False
            async def first():
                response = await endpoint(Request(), "review", 0)
                value = await response.body_iterator.__anext__()
                await response.body_iterator.aclose()
                return value
            self.assertIn("event: resync", asyncio.run(first()))


def benchmark():
    """10万事件+10万指标的诊断查询基准；写入吞吐单独测量。"""
    from project_paths import HORIZON_REPORT_DIR
    import platform
    import sys
    from implementations.horizon_tsac_20260920.env.environment import Environment
    from tests.horizon_tsac_20260920.test_data import fixture
    import numpy as np
    result = {"kind": "diagnostic_microbenchmark_not_training_performance", "python": sys.version,
              "platform": platform.platform(), "sqlite": sqlite3.sqlite_version}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        folder = root/"large"
        conn = make_database(folder)
        count = 100000
        with conn:
            conn.executemany("INSERT INTO events(event_id,event_type,severity,component,phase,scenario_id,beam_id,env_step,payload) VALUES(?,?,?,?,?,?,?,?,?)",
                ((str(i), "diagnostic_event", "ERROR" if i%997==0 else "INFO", "fixture", "train", f"scenario-{i%100}", str(i%220), i, dumps({"text": "诊断事件", "step": i})) for i in range(count)))
            conn.executemany("INSERT INTO metrics(event_seq,phase,name,axis,axis_value,value,raw_available,aggregation,count,first_axis,last_axis,min,max,last,anomaly) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ((i+1, "train", "diagnostic_metric", "env_step", i, float(i%100)/100, 1, "raw", 1, i, i, float(i%100)/100, float(i%100)/100, float(i%100)/100, 0) for i in range(count)))
        conn.close()
        client = TestClient(create_app(root))
        paths = {"event_pagination": "/api/v1/runs/review/events?cursor=50000&limit=200",
                 "event_filter": "/api/v1/runs/review/events?beam_id=17&severity=ERROR&limit=200",
                 "event_text": "/api/v1/runs/review/events?text=诊断&limit=200",
                 "metrics_viewport": "/api/v1/runs/review/metrics?name=diagnostic_metric&axis=env_step&limit=500"}
        timings = {}
        for name, path in paths.items():
            samples = []
            for repeat in range(12):
                started = time.perf_counter()
                response = client.get(path)
                response.raise_for_status()
                elapsed = (time.perf_counter()-started)*1000
                if repeat:
                    samples.append(elapsed)
            timings[name] = {"p50_ms": float(np.quantile(samples, .5)), "p95_ms": float(np.quantile(samples, .95)), "repeats": len(samples)}
        result.update(events=count, metric_records=count, query_timings=timings, database_bytes=(folder/DB_NAME).stat().st_size)

        # 同一220正需求合法固定动作序列，纯环境和全部基本轨迹分别计时；不以此替代训练开销百分比。
        scenario = fixture(220)
        env = Environment()
        env.reset(scenario)
        started = time.perf_counter()
        for _ in range(220):
            env.step(0)
        baseline = time.perf_counter()-started
        logged = root/"write"
        recorder = Recorder(logged, {"run_id": "write-review", "source": "diagnostic_fixture"})
        env.reset(scenario)
        recorder.start_episode("ep", scenario.scenario_id, env.snapshot())
        started = time.perf_counter()
        for step in range(1, 221):
            before = env.snapshot()
            _, reward, terminated, truncated, info = env.step(0)
            recorder.record_step("ep", scenario.scenario_id, step, before, env.snapshot(), {"reward": reward, "terminated": terminated, "truncated": truncated, **info}, env_step=step)
        recorder.end_episode("ep", env.evaluate().metrics)
        with_logger = time.perf_counter()-started
        recorder.close()
        result["write_measurement"] = {"positive_beams": 220, "env_steps": 220, "baseline_environment_seconds": baseline,
           "environment_with_snapshots_and_durable_logger_seconds": with_logger, "incremental_seconds_per_step": (with_logger-baseline)/220,
           "committed_steps_per_second": 220/with_logger, "database_bytes": (logged/DB_NAME).stat().st_size,
           "note": "全SKIP可重放压力轨迹；开销含快照构建/JSON/SQLite FULL，不代表含模型更新的训练吞吐损耗。"}
    HORIZON_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = HORIZON_REPORT_DIR/"日志查询与写入性能实测.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
