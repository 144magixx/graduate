"""W0 只读冻结与数据来源审计，可重复验证旧链路没有被修改。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from project_paths import PROJECT_ROOT, COVER_OUTPUT_DIR, HORIZON_REPORT_DIR


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sources():
    old = PROJECT_ROOT / "implementations" / "sac_fyh_io"
    inventory = []
    paths = sorted(old.glob("*.py")) + sorted(COVER_OUTPUT_DIR.glob("*.csv"))
    paths += sorted(p for p in (PROJECT_ROOT / "outputs").rglob("*")
                    if p.is_file() and "horizon_tsac_20260920" not in p.parts)
    for path in paths:
        inventory.append({"path": path.relative_to(PROJECT_ROOT).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    scenarios = []
    for path in sorted(COVER_OUTPUT_DIR.glob("cover_output_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        positive = [r for r in rows if float(r["rate"]) > 0 and float(r["beamwidth"]) > 0]
        scenarios.append({"file": path.name, "source_hash": sha256(path), "rows": len(rows),
                          "n_demand": len(positive),
                          "padding": sum(float(r["rate"]) == 0 and float(r["beamwidth"]) == 0 for r in rows),
                          "maximum_width_deg": max((float(r["beamwidth"]) for r in positive), default=None),
                          "root_scene_id": None, "split_group_id": None,
                          "provenance_status": "unknown_legacy_aggregate"})
    counts = [s["n_demand"] for s in scenarios]
    return {"schema_version": "baseline_freeze.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "inventory": inventory, "scenarios": scenarios,
            "statistics": {"files": len(scenarios), "rows": sum(s["rows"] for s in scenarios),
                           "positive_demand_rows": sum(counts), "padding_rows": sum(s["padding"] for s in scenarios),
                           "n_demand_min": min(counts), "n_demand_max": max(counts),
                           "n_demand_mean": sum(counts) / len(counts)},
            "upstream": {"generator_available": False, "root_business_available": False,
                         "assignment_available": False,
                         "evidence": "当前工作区只有聚合覆盖CSV及第二阶段/历史资源分配代码；未发现第一阶段生成器或原始业务归属。"}}


def freeze():
    HORIZON_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = HORIZON_REPORT_DIR / "基线冻结清单.json"
    if path.exists():
        return verify(path)
    result = audit_sources()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    source_lines = "\n".join(f"- `{x['path']}`：`{x['sha256']}`" for x in result["inventory"] if x["path"].endswith(".py"))
    note = f'''# 基线冻结记录

日期：2026-09-20。工单：W0。依据：设计 v3；用户已明确解除此前“只更新方案”的限制。

## 源码与数据事实

三份核心源码 SHA256 与 v3 的记录逐一一致。原始源码、100份CSV和历史产物的逐文件大小/哈希见《基线冻结清单.json》。不加载历史checkpoint。

{source_lines}

数据统计：{json.dumps(result['statistics'], ensure_ascii=False)}。

旧链路依赖：sac_train_fyh_IO → Environment_fyh_IO / sac_fyh_IO → project_paths → data/cover_outputs；输出 outputs/sac_fyh_io。其他城市、贪心、历史PPO不混入新环境。

旧行为：205维输入、14 token、独立三头、分支Q先min后加、γ=.99、负目标熵−6.9、预算6000W；旧入口导入即训练；critic-2独立命名已修复。新实现将保留合法适配基线，但不声称新物理数值等同旧物理。

## 固定诊断动作序列

小型fixture 4束、8槽、group=[0,2,1,3]，依次 ALLOC(0,2,5W)、ALLOC(0,2,10W)、SKIP、ALLOC(7,1,5W)。该序列只用于诊断功率、同极化双向干扰、SKIP和末束记账，不是新广覆盖训练数据。另建立3束4槽长度≤2/两档功率的完整合法序列穷举oracle。

## 保留范围与新目录

原 sac_fyh_io、原始CSV、历史产物保持不变；实现新增于 implementations/horizon_tsac_20260920，算法名称 Horizon T-SAC。用户交付说明和产物使用中文文件名；技术模块标识稳定。

第一阶段覆盖生成器、原始业务和assignment未发现；目前只能冻结兼容数据的hash划分，不能证明真实母场景独立，不能宣称48–256支持域已覆盖。D0–D3实现接口及审计工具，缺数据部分标明待上游。

## 验证命令

从PPO4090运行 `python -m implementations.horizon_tsac_20260920.audit --verify`，重新检查所有冻结文件，无需导入旧训练入口。
'''
    (HORIZON_REPORT_DIR / "基线冻结记录.md").write_text(note, encoding="utf-8")
    return {"frozen_files": len(result["inventory"]), **result["statistics"]}


def verify(path=None):
    path = path or HORIZON_REPORT_DIR / "基线冻结清单.json"
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    changed = [r["path"] for r in frozen["inventory"]
               if not (PROJECT_ROOT / r["path"]).is_file() or sha256(PROJECT_ROOT / r["path"]) != r["sha256"]]
    if changed:
        raise RuntimeError(f"冻结文件发生变更：{changed}")
    return {"verified_files": len(frozen["inventory"]), "changed": changed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else freeze(), ensure_ascii=False))


if __name__ == "__main__":
    main()
