"""Evaluation contract tests use no CSV, model rollout, checkpoint, or optimizer."""
import pytest

from implementations.anchor_tsac_20260921 import evaluate


def test_request_validation_happens_before_manifest_or_output_access(tmp_path):
    missing = tmp_path / "missing" / "运行清单.json"
    with pytest.raises(ValueError, match="validation"):
        evaluate.evaluate_run(missing, split="train", allow_experiment=True)
    with pytest.raises(ValueError, match="策略"):
        evaluate.evaluate_run(missing, policies=("paper_random",), allow_experiment=True)
    with pytest.raises(ValueError, match="limit"):
        evaluate.evaluate_run(missing, limit=True, allow_experiment=True)
    assert not missing.parent.exists()


def test_real_rollout_helpers_require_explicit_authorization(tmp_path):
    with pytest.raises(RuntimeError, match="未授权"):
        evaluate.evaluate_run(tmp_path / "运行清单.json")
    with pytest.raises(RuntimeError, match="未授权"):
        evaluate.allocate_candidate(None, object(), "random")


def test_evaluation_identity_keeps_anchor_policy_and_marks_heuristics_non_training():
    base = {
        "run_id": "source-run", "trial_id": "trial-1", "algorithm_id": "anchor_tsac",
        "algorithm": "Anchor T-SAC", "algorithm_spec": {"model": "pure_transformer_14"},
        "paper_baseline": {"must_not": "survive"}, "baseline_settings": {"must_not": "survive"},
    }
    policy = evaluate.build_evaluation_manifest(base, "policy", "eval-1", "checkpoint-1", {"episodes": 3}, "validation")
    assert policy["algorithm"] == "Anchor T-SAC"
    assert policy["algorithm_id"] == "anchor_tsac"
    assert policy["algorithm_spec"] == {"model": "pure_transformer_14"}
    assert policy["phase"] == "validation"
    for name in ("random", "greedy", "equal_power"):
        heuristic = evaluate.build_evaluation_manifest(base, name, "eval-1", "checkpoint-1", {}, "validation")
        assert heuristic["algorithm"] == f"{name} heuristic (non-training)"
        assert heuristic["algorithm_id"] == f"heuristic_{name}"
        assert heuristic["model_not_applicable"] is True and heuristic["training_required"] is False
        assert "algorithm_spec" not in heuristic and "paper_baseline" not in heuristic


def test_export_verify_only_round_trip_uses_only_the_synthetic_engineering_fixture(tmp_path):
    from implementations.anchor_tsac_20260921.config import Config
    from implementations.anchor_tsac_20260921.dashboard.demo import _scenario
    from implementations.anchor_tsac_20260921.env.action import Action
    from implementations.anchor_tsac_20260921.env.environment import Environment
    from implementations.anchor_tsac_20260921.export import export_allocation, reload_and_evaluate

    environment = Environment(Config())
    environment.reset(_scenario(), seed=0)
    for action in (Action("ALLOC", 0, 1, 0), Action("ALLOC", 1, 1, 0), Action("ALLOC", 2, 1, 0)):
        environment.step(action)
    _, verified = export_allocation(environment, tmp_path, run_id="synthetic-only")
    assert verified["passed"] is True
    assert reload_and_evaluate(tmp_path)["rate_recomputation"] is True
