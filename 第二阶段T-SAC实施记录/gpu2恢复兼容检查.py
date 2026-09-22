"""在现有 GPU 分配内只读加载可信检查点；不运行环境或优化更新。"""
import argparse
import json
import os
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import numpy as np
    from project_paths import COVER_OUTPUT_DIR
    from implementations.spectrum_tsac_20260921.config import load_config
    from implementations.spectrum_tsac_20260921.artifacts import load_checkpoint, restore_rng, atomic_json
    from implementations.spectrum_tsac_20260921.audit import sha256
    from implementations.spectrum_tsac_20260921.train import prepare_dataset
    from implementations.spectrum_tsac_20260921.data.loader import load_scenario
    from implementations.spectrum_tsac_20260921.data.domain import DomainSampler
    from implementations.spectrum_tsac_20260921.env.environment import Environment
    from implementations.spectrum_tsac_20260921.rl.sac import DiscreteSAC
    from implementations.spectrum_tsac_20260921.rl.replay import ReplayBuffer

    assert platform.node() == "gpu2"
    assert os.environ.get("SLURM_JOB_ID") == "206357"
    assert torch.cuda.is_available()
    config = load_config(plan["config_path"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(config.train.deterministic)
    assert sha256(plan["resume_checkpoint"]) == plan["checkpoint_sha256"]
    manifest = prepare_dataset(config)
    payload = load_checkpoint(plan["resume_checkpoint"], config, manifest["manifest_hash"])
    for key, expected in [("run_id", plan["source_run_id"]), ("trial_id", plan["source_trial_id"]),
                          ("checkpoint_id", plan["source_checkpoint_id"])]:
        assert payload[key] == expected, (key, payload.get(key), expected)
    assert payload["counters"] == {"episodes": 150, "env_step": 21294, "update_step": 20295}
    assert payload.get("initialization")
    record = next(r for r in manifest["records"] if r["split"] == "train")
    scenario = load_scenario(Path(record["scenario_path"]) if record.get("scenario_path") else COVER_OUTPUT_DIR / record["file"],
                             config, rng=np.random.default_rng(config.train.seed + 59))
    env = Environment(config)
    example, _ = env.reset(scenario, config.train.seed)
    agent = DiscreteSAC(config, env.action_spec, example)
    agent.load_state_dict(payload["agent"])
    assert agent.update_step == 20295
    replay = ReplayBuffer(config.train.replay_capacity, config.semantic_versions(),
                          action_spec_signature=agent.action_signature(), mode=config.train.replay_mode,
                          seed=config.train.seed + 41)
    replay.load_state_dict(payload["replay"])
    assert len(replay) > 0
    sampler = DomainSampler(manifest["records"], config.train.seed + 11)
    sampler.load_state_dict(payload["sampler"])
    for state in payload["local_rngs"].values():
        rng = np.random.default_rng()
        rng.bit_generator.state = state
    restore_rng(payload["rng"])
    probability = agent.probabilities(example)
    assert np.isfinite(probability).all() and np.isclose(probability.sum(), 1.0, atol=1e-5)
    assert float(probability[~example["valid_action_mask"]].sum()) == 0.0
    report = dict(status="passed", host=platform.node(), allocation=os.environ["SLURM_JOB_ID"],
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  checkpoint_id=payload["checkpoint_id"], checkpoint_sha256=plan["checkpoint_sha256"],
                  trial_id=payload["trial_id"], counters=payload["counters"], replay_size=len(replay),
                  alpha=float(agent.alpha.detach().cpu()), probability_sum=float(probability.sum()),
                  data_manifest_hash=manifest["manifest_hash"], initialization=payload["initialization"],
                  checks=["frozen_data_hashes", "checkpoint_sha256", "strict_config_compatibility", "five_networks",
                          "optimizers_and_alpha", "replay", "sampler", "global_and_local_rng", "finite_legal_policy"],
                  environment_steps_added=0, optimizer_updates_added=0,
                  note="跨节点完整状态恢复兼容检查通过，不宣称跨节点逐位一致。")
    atomic_json(Path(plan["control_dir"]) / "gpu2恢复兼容检查结果.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
