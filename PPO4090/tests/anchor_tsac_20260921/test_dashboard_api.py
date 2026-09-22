"""Anchor read-only dashboard contract tests; no training or CSV sampling."""
from fastapi.testclient import TestClient


def test_implementation_endpoint_reports_capabilities_without_fabricating_a_trial(tmp_path):
    from implementations.anchor_tsac_20260921.dashboard.api import create_app

    client = TestClient(create_app(runs_root=tmp_path, datasets_root=tmp_path))
    response = client.get("/api/v1/implementation")
    assert response.status_code == 200
    payload = response.json()
    assert payload["implementation"] == "anchor_tsac_20260921"
    assert payload["algorithm"] == {
        "id": "anchor_tsac", "model": "pure_transformer_14", "token_count": 14,
        "d_model": 128, "num_heads": 8, "num_layers": 2,
        "actor": "independent", "critic": "additive",
    }
    assert payload["observation"]["input"] == "anchor205"
    assert payload["observation"]["adapter_modes"] == ["legacy_scaled205", "modern205"]
    assert payload["lineage_modes"] == ["from_scratch", "weights_only_not_exact_resume", "exact_resume"]
    assert payload["trial_status"] == "not_run"
    assert payload["is_demo"] is False
    assert payload["read_only"] is True
    assert client.get("/api/implementation").json() == payload
    assert client.get("/api/v1/runs").json()["items"] == []
    assert client.post("/api/v1/runs").status_code == 405


def test_engineering_demo_exercises_read_only_five_view_data_without_training(tmp_path):
    from implementations.anchor_tsac_20260921.dashboard.api import create_app
    from implementations.anchor_tsac_20260921.dashboard.demo import create_demo_run

    create_demo_run(tmp_path)
    client = TestClient(create_app(runs_root=tmp_path, datasets_root=tmp_path))
    run = client.get("/api/v1/runs").json()["items"]
    assert len(run) == 1
    item = run[0]
    assert item["is_demo"] is True and item["mode"] == "demo"
    assert item["experiment_authorized"] is False
    assert item["source"] == "diagnostic_fixture"
    assert item["initialization"] == "engineering_fixture_not_applicable"
    assert item["lineage"]["kind"] == "engineering_fixture_not_applicable"
    # These are the manifest inputs rendered by the current-run lineage banner.
    assert item["observation_adapter"] == "legacy_scaled205"
    assert item["observation_adapter_version"] == "legacy_scaled205_new_physics.v1"
    assert item["source_hash"] == "synthetic-engineering-fixture-not-a-dataset"
    assert item["trial_id"] == "anchor-engineering-demo"
    run_id = item["run_id"]
    episodes = client.get(f"/api/v1/runs/{run_id}/episodes").json()["items"]
    assert len(episodes) == 1 and episodes[0]["last_step"] == 3
    episode_id = episodes[0]["episode_id"]
    state = client.get(f"/api/v1/runs/{run_id}/episodes/{episode_id}/state?step=3").json()
    assert state["status"] == "recorded" and state["state"]["terminal"] is True
    assert len(state["state"]["beams"]) == 3
    transitions = client.get(f"/api/v1/runs/{run_id}/episodes/{episode_id}/transitions?from=1").json()["items"]
    assert len(transitions) == 3 and all(step["transition"]["manual_action"] for step in transitions)
    available = client.get(f"/api/v1/runs/{run_id}/metrics").json()["available"]
    assert available and all(metric["axis"] != "update_step" for metric in available)
    metric_name = next(metric["name"] for metric in available if metric["axis"] == "env_step")
    metrics = client.get(f"/api/v1/runs/{run_id}/metrics?name={metric_name}&axis=env_step").json()
    assert metrics["items"] and metrics["name"] == metric_name
    events = client.get(f"/api/v1/runs/{run_id}/events?limit=20").json()["items"]
    assert any(event["event_type"] == "step_completed" for event in events)
    assert client.get(f"/api/v1/runs/{run_id}/integrity").json()["status"] == "verified"
    comparison = client.get(f"/api/v1/comparisons?run_ids={run_id}")
    assert comparison.status_code == 422
    assert "excluded" in comparison.json()["detail"]


def test_lineage_display_inputs_keep_all_declared_modes_distinct():
    examples = [
        {"initialization": "from_scratch", "parent_checkpoint_id": None},
        {"initialization": "weights_only_not_exact_resume", "parent_checkpoint_id": "parent-weights"},
        {"initialization": "exact_resume", "parent_checkpoint_id": "parent-run-checkpoint"},
    ]
    assert [item["initialization"] for item in examples] == [
        "from_scratch", "weights_only_not_exact_resume", "exact_resume"
    ]
    assert examples[1]["parent_checkpoint_id"] != examples[2]["parent_checkpoint_id"]
