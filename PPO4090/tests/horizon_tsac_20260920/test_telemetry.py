"""Durability, replay and read-only API acceptance with explicit diagnostic fixtures."""
import asyncio
import copy
import json
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from implementations.horizon_tsac_20260920.config import TelemetryConfig
from implementations.horizon_tsac_20260920.dashboard.api import create_app
from implementations.horizon_tsac_20260920.telemetry import Recorder, RecorderError
from implementations.horizon_tsac_20260920.telemetry.legacy import inspect_legacy
from implementations.horizon_tsac_20260920.telemetry.recorder import export_run
from implementations.horizon_tsac_20260920.telemetry.storage import DB_NAME, readonly, reconstruct


def snapshot(step=0, terminal=False):
    return {"scenario_id": "diagnostic-fixture", "step_index": step, "terminal": terminal,
            "beams": [{"beam_id": 10, "longitude_deg": 120., "latitude_deg": 30.,
                       "status": "allocated" if step else "pending", "rate_bps": float(step),
                       "satisfaction": step / 100, "sinr_db": None}],
            "resources": {"remaining_power_w": 100 - step}, "metrics": {"U": step / 100}}


@pytest.fixture
def run(tmp_path):
    folder = tmp_path / "runs" / "验收运行"
    recorder = Recorder(folder, {"run_id": "验收运行", "source": "diagnostic_fixture", "physics_version": "test.v1"})
    recorder.start_episode("ep", "diagnostic-fixture", snapshot())
    for step in range(1, 24):
        recorder.record_step("ep", "diagnostic-fixture", step, snapshot(step - 1), snapshot(step, step == 23),
                             {"acted_beam_id": 10, "reward": 1., "terminated": step == 23}, env_step=step,
                             event_id=f"step-{step}")
    recorder.end_episode("ep", {"U": .23, "return": 23.})
    recorder.close()
    return folder


def test_replay_all_steps_and_terminal_keyframe(run):
    with readonly(run / DB_NAME) as conn:
        for step in (0, 1, 19, 20, 21, 23, 4):
            result = reconstruct(conn, "ep", step)
            assert result["status"] == "recorded"
            assert result["state"] == snapshot(step, step == 23)
        assert reconstruct(conn, "ep", 24)["status"] == "unavailable"
        assert [row[0] for row in conn.execute("SELECT step_index FROM keyframes ORDER BY step_index")] == [0, 20, 23]
        assert conn.execute("SELECT COUNT(*) FROM beam_deltas").fetchone()[0] == 23


def test_dedup_nonfinite_full_precision_and_window(tmp_path):
    with Recorder(tmp_path / "run", {"run_id": "dedup"}) as rec:
        rec.event("中文事件", {"full": .123456789012345, "nan": float("nan"), "trace": "第一行\n第二行"}, event_id="same")
        rec.event("中文事件", {"duplicate": True}, event_id="same")
        rec.record_update({"actor_loss": 2.}, 1)
        for step in range(2, 6):
            rec.record_update({"actor_loss": float(step)}, step)
        high = rec.flush()
        assert high > 0
    with readonly(tmp_path / "run" / DB_NAME) as conn:
        rows = conn.execute("SELECT * FROM events WHERE event_id='same'").fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["nan"] is None and payload["nonfinite_fields"][0]["kind"] == "nan"
        assert payload["full"] == .123456789012345
        window = conn.execute("SELECT * FROM metrics WHERE aggregation='window'").fetchone()
        assert (window["count"], window["first_axis"], window["last_axis"], window["raw_available"]) == (4, 2, 5, 0)
        seq = [row[0] for row in conn.execute("SELECT event_seq FROM events ORDER BY event_seq")]
        assert seq == sorted(set(seq))


def test_duplicate_step_is_atomic(tmp_path):
    with Recorder(tmp_path / "run", {"run_id": "atomic"}) as rec:
        rec.start_episode("ep", "s", snapshot())
        for _ in range(2):
            rec.record_step("ep", "s", 1, snapshot(), snapshot(1), {"reward": 1}, event_id="same-step")
        rec.flush()
    with readonly(tmp_path / "run" / DB_NAME) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM beam_deltas").fetchone()[0] == 1


def test_writer_failure_preserves_committed_prefix(tmp_path):
    rec = Recorder(tmp_path / "run", {"run_id": "failure"})
    rec.start_episode("ep", "s", snapshot())
    rec.flush()
    rec.record_step("ep", "s", 2, snapshot(), snapshot(2), {})
    with pytest.raises(RecorderError):
        rec.flush()
    with pytest.raises(RecorderError):
        rec.close()
    with readonly(tmp_path / "run" / DB_NAME) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='step_completed'").fetchone()[0] == 0
        assert reconstruct(conn, "ep", 0)["status"] == "recorded"


def test_queue_bytes_never_silently_drop_core(tmp_path):
    rec = Recorder(tmp_path / "run", {"run_id": "bounded"}, TelemetryConfig(queue_max_bytes=4096))
    try:
        with pytest.raises(RecorderError, match="exceeds queue byte limit"):
            rec.event("too_large", {"value": "a" * 10000})
    finally:
        rec.close()


def test_writer_death_is_detected(tmp_path):
    rec = Recorder(tmp_path / "run", {"run_id": "dead"})
    rec._process.terminate()
    rec._process.join()
    with pytest.raises(RecorderError, match="exited"):
        rec.event("must_fail")
    with pytest.raises(RecorderError):
        rec.close()


def test_live_backup_and_hashes(tmp_path):
    with Recorder(tmp_path / "run", {"run_id": "export"}) as rec:
        rec.start_episode("ep", "s", snapshot())
        high = rec.flush()
        out = rec.export(tmp_path / "运行包")
        with readonly(out / DB_NAME) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert reconstruct(conn, "ep", 0)["status"] == "recorded"
        manifest = json.loads((out / "运行清单.json").read_text(encoding="utf-8"))
        assert manifest["export_watermark"] >= high
        hashes = json.loads((out / "数据哈希.json").read_text(encoding="utf-8"))
        assert DB_NAME in hashes and "分配步骤.csv" in hashes


def test_api_pagination_filter_snapshot_and_readonly(run):
    client = TestClient(create_app(run.parent))
    prefix = "/api/v1/runs/验收运行"
    assert len(client.get("/api/v1/runs").json()["items"]) == 1
    first = client.get(prefix + "/events?limit=2").json()
    second = client.get(prefix + f"/events?limit=2&cursor={first['next_cursor']}").json()
    assert first["items"][-1]["event_seq"] < second["items"][0]["event_seq"]
    assert client.get(prefix + "/events?limit=1001").status_code == 422
    assert client.get(prefix + "/events?event_type=step_completed&beam_id=10").json()["items"]
    assert client.get(prefix + "/episodes/ep/state?step=23").json()["state"]["terminal"]
    assert client.get(prefix + "/episodes/ep/state?step=24").json()["status"] == "unavailable"
    metrics = client.get(prefix + "/metrics?name=U&axis=env_step&limit=3").json()
    assert len(metrics["items"]) <= 3 and metrics["aggregation"] == "min_max_count_viewport"
    assert max(item["max"] for item in metrics["items"]) == .23
    assert client.post(prefix + "/events", json={}).status_code == 405
    assert client.get("/api/v1/artifacts/../../project_paths.py").status_code == 404
    assert client.get("/api/v1/datasets/unknown/coverage").json()["status"] == "unavailable"


def test_resume_anchor_independent_of_parent(tmp_path):
    with Recorder(tmp_path / "child", {"run_id": "child", "parent_run_id": "absent", "resume_env_step": 10}) as rec:
        rec.start_episode("ep-child", "s", snapshot(10))
        rec.record_step("ep-child", "s", 11, snapshot(10), snapshot(11), {})
        rec.flush()
    with readonly(tmp_path / "child" / DB_NAME) as conn:
        assert reconstruct(conn, "ep-child", 11)["status"] == "recorded"
        assert reconstruct(conn, "ep-child", 9)["available_range"] == [10, 11]


def test_legacy_readonly_capabilities(tmp_path):
    path = tmp_path / "旧日志.csv"
    original = "episode,satisfaction\n1,0.12\n"
    path.write_text(original, encoding="utf-8")
    report = inspect_legacy(path)
    assert report["source"] == "legacy_import" and not report["exact_step_replay_available"]
    assert path.read_text(encoding="utf-8") == original


def test_sse_committed_cursor_resume(run):
    app = create_app(run.parent)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "").endswith("/stream"))
    class Request:
        headers = {"last-event-id": "2"}
        async def is_disconnected(self):
            return False
    async def read_one():
        response = await endpoint(Request(), "验收运行", 0)
        value = await response.body_iterator.__anext__()
        await response.body_iterator.aclose()
        return value
    assert "id: 3\n" in asyncio.run(read_one())


def test_evaluation_phase_and_optimizer_window_boundaries(tmp_path):
    with Recorder(tmp_path / "phase", {"run_id": "phase"}) as rec:
        rec.start_episode("evaluation", "s", snapshot(), phase="validation")
        rec.end_episode("evaluation", {"mean_satisfaction": .5}, phase="validation")
        rec.record_update({"loss": 2.}, 2, phase="train")
        rec.record_update({"loss": 9.}, 3, phase="validation")
    with readonly(tmp_path / "phase" / DB_NAME) as conn:
        metric = conn.execute("SELECT phase FROM metrics WHERE axis='episode'").fetchone()
        assert metric[0] == "validation"
        rows = conn.execute("SELECT phase,value,count FROM metrics WHERE axis='update_step' ORDER BY first_axis").fetchall()
        assert [tuple(row) for row in rows] == [("train", 2., 1), ("validation", 9., 1)]


def test_viewport_null_is_not_zero(tmp_path):
    with Recorder(tmp_path / "null", {"run_id": "null"}) as rec:
        for index, value in enumerate((1., None, 3.), start=1):
            rec.event("optimizer_update", {"values": {"loss": value}}, update_step=index, phase="train")
    client = TestClient(create_app(tmp_path))
    result = client.get("/api/v1/runs/null/metrics?name=loss&axis=update_step&limit=1").json()
    assert result["items"][0]["value"] == 2.
    assert result["items"][0]["missing_count"] == 1
    assert result["items"][0]["anomaly"] == 1


def test_comparison_uses_only_named_evaluation_phase(tmp_path):
    from implementations.horizon_tsac_20260920.config import Config
    manifest = {"physics_version": "v1", "metric_version": "v1", "dataset_version": "v1",
                "data_manifest_hash": "hash", "power_budget_w": 100., "action_spec_signature": "same",
                "config": Config().to_dict(), "reward_version": "v1", "seed": 42}
    for index in (1, 2):
        with Recorder(tmp_path / f"run{index}", {**manifest, "run_id": f"run{index}"}) as rec:
            rec.start_episode("training", "same-scene", snapshot(), phase="train")
            rec.end_episode("training", {"mean_satisfaction": 100.}, phase="train")
            rec.start_episode("testing", "same-scene", snapshot(), phase="test")
            rec.end_episode("testing", {"mean_satisfaction": .1 * index}, phase="test")
    client = TestClient(create_app(tmp_path))
    result = client.get("/api/v1/comparisons?run_ids=run1,run2").json()
    assert result["compatible"]
    assert result["paired_differences"] == [{"scenario_id": "same-scene", "delta_U": .1}]
    assert len(result["items"][0]["episodes"]) == 1
    assert client.get("/api/v1/comparisons?run_ids=run1,run2&phase=train").status_code == 422


def test_sparse_delta_does_not_repeat_static_geometry(tmp_path):
    from implementations.horizon_tsac_20260920.telemetry.storage import state_delta, apply_delta
    before = snapshot()
    before["beams"][0]["coverage_polygon"] = {"coordinates": [[[100., 30.], [101., 30.]]]}
    after = copy.deepcopy(before)
    after["beams"][0]["rate_bps"] = 123.
    delta = state_delta(before, after)
    assert "coverage_polygon" not in delta["beams"][0]["after"]
    assert apply_delta(copy.deepcopy(before), delta) == after


def test_diagnostic_mirror_recent_context_and_manifest_status(tmp_path):
    folder = tmp_path / "mirror"
    with Recorder(folder, {"run_id": "mirror"}) as rec:
        rec.start_episode("ep", "s", snapshot())
        for step in range(1, 22):
            rec.record_step("ep", "s", step, snapshot(step - 1), snapshot(step), {"action": "SKIP"}, env_step=step)
        rec.event("diagnostic_error", {"reason": "fixture"}, severity="ERROR")
    manifest = json.loads((folder / "运行清单.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["durable_event_seq"] > 0
    mirror = [json.loads(line) for line in (folder / "诊断日志.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({event["event_id"] for event in mirror}) == len(mirror)
    context = next(event for event in mirror if event["event_type"] == "diagnostic_error")["payload"]["recent_steps"]
    assert len(context) == 20 and context[0]["step_index"] == 2 and context[-1]["step_index"] == 21
