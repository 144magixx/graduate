"""在gpu4分配内验证独立部署；不启动正式长训练。"""
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback


def main():
    plan_path, release = Path(sys.argv[1]), Path(sys.argv[2])
    report_path = release / "SpectrumGPU预检结果.json"
    report = {"status": "running", "experiments": [], "formal_training_started": False}
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        import torch
        assert socket.gethostname().split(".")[0] == "gpu4"
        assert torch.cuda.is_available()
        plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        assert plan["test_set_allowed"] is False and plan["paired_transfer"] is True
        source = Path(plan["experiments"][0]["initialize_from"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == plan["source_checkpoint_sha256"]
        report["hardware"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
        suite = subprocess.run([sys.executable, "-m", "pytest", "tests/spectrum_tsac_20260921", "-q",
                                "--junitxml=" + str(release / "SpectrumGPU回归测试.xml")])
        report["pytest_exit_code"] = suite.returncode
        save()
        if suite.returncode:
            raise RuntimeError("GPU环境回归测试未通过，不进入真实预检或正式训练")
        directory = release / "GPU预检"
        directory.mkdir(exist_ok=True)
        seen = set()
        for experiment in plan["experiments"]:
            mode = experiment["config"]["model"]["encoder"]
            if mode in seen:
                continue
            seen.add(mode)
            config = copy.deepcopy(experiment["config"])
            config["train"].update(episodes=1, warmup_steps=0, batch_size=2, microbatch_size=2,
                                   checkpoint_every=1, max_wall_seconds=600)
            path = directory / (mode + "配置.json")
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            trained = subprocess.run([sys.executable, "-X", "utf8", "-m",
                                      "implementations.spectrum_tsac_20260921.train", "--mode", "preflight",
                                      "--config", str(path), "--initialize-from", str(source)],
                                     capture_output=True, text=True, timeout=900)
            (directory / (mode + "训练日志.txt")).write_text(trained.stdout + trained.stderr, encoding="utf-8")
            if trained.returncode:
                raise RuntimeError(mode + "真实GPU预检训练失败")
            result = json.loads(trained.stdout.strip().splitlines()[-1])
            run = Path(result["run_dir"])
            manifest = json.loads((run / "运行清单.json").read_text())
            summary = json.loads((run / "训练结果.json").read_text())
            assert manifest["mode"] == "preflight"
            assert manifest["initialization"]["source_sha256"] == plan["source_checkpoint_sha256"]
            assert result["counters"] == {"episodes": 1, "env_step": 139, "update_step": 1}
            assert all(row["constraint_violation_count"] == 0 for row in summary["episodes"])
            evaluated = subprocess.run([sys.executable, "-X", "utf8", "-m",
                                        "implementations.spectrum_tsac_20260921.evaluate", "--manifest",
                                        str(run / "运行清单.json"), "--split", "validation", "--limit", "1",
                                        "--policies", "policy"], capture_output=True, text=True, timeout=600)
            (directory / (mode + "评价日志.txt")).write_text(evaluated.stdout + evaluated.stderr, encoding="utf-8")
            if evaluated.returncode:
                raise RuntimeError(mode + "真实GPU检查点CPU评价失败")
            evaluation = json.loads(evaluated.stdout.strip().splitlines()[-1])
            score = evaluation["summary"]["policy"]
            assert score["scenarios"] == 1 and score["constraint_violations"] == 0 and score["mean_U"] >= .7
            report["experiments"].append({"encoder": mode, "status": "completed", "run_dir": str(run),
                                           "counters": result["counters"], "first_episode_U": summary["episodes"][0]["mean_satisfaction"],
                                           "evaluation": evaluation, "research_data": False})
            save()
        assert len(report["experiments"]) == 2
        assert abs(report["experiments"][0]["first_episode_U"] - report["experiments"][1]["first_episode_U"]) < 1e-10
        report["status"] = "completed"
        save()
        print(json.dumps(report, ensure_ascii=False))
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
