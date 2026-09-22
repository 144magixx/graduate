"""离线绘制真实训练记录；不导入训练入口，不读取test，不修改运行数据。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import unicodedata
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FuncFormatter

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
RAW = OUT / "绘图数据原件"
SMOOTH = 10
DPI = 240
COLORS = ["#176080", "#bd6132", "#42805c", "#846298", "#72644b"]
for filename in ("msyh.ttc", "msyhbd.ttc"):
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        font_manager.fontManager.addfont(str(path))
plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11,
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#a5afb8", "axes.labelcolor": "#34424d", "text.color": "#263743",
    "xtick.color": "#53606a", "ytick.color": "#53606a", "grid.color": "#e4e9ed",
    "grid.linewidth": .7, "savefig.facecolor": "white"})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(REPORT.resolve())).replace("\\", "/")


def source(path):
    return {"path": relative(path), "sha256": digest(path)}


def wrapped(value, width=51):
    lines, line, size = [], "", 0
    for char in value:
        n = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if size+n > width:
            lines.append(line)
            line, size = "", 0
        line += char
        size += n
    return lines + ([line] if line else [])


def number(v):
    return f"{v:g}"


def lr(v):
    return f"{v:.0e}".replace("e-0", "e-")


def make_run(name, group, config_path, manifest_path, result_path, *, color, transfer=False):
    config, manifest, result = read(config_path), read(manifest_path), read(result_path)
    assert manifest["config"] == config
    episodes = result["episodes"]
    rows = [{"episode": i+1, "env_step": e["env_step"], "update_step": e["update_step"],
             "U": e["mean_satisfaction"], "return": e["episode_return"], "segment": 0}
            for i, e in enumerate(episodes)]
    assert len(rows) == result["counters"]["episodes"]
    assert rows[-1]["env_step"] == result["counters"]["env_step"]
    return {"name": name, "group": group, "config": config, "manifest": manifest,
            "algorithm_id": result.get("algorithm", ("cnn_sac" if config["model"]["encoder"] == "cnn_local" else "spectrum_tsac") if transfer else "horizon_tsac"),
            "run_ids": [manifest["run_id"]], "rows": rows, "color": color, "transfer": transfer,
            "status": "已完成", "elapsed_seconds": result["elapsed_seconds"],
            "sources": [source(p) for p in (config_path, manifest_path, result_path)]}


def parse_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # 日志快照末行可能未写完；仅收录完整逐回合JSON。
        if "episode" in row and "U" in row:
            assert "return" in row, "SAC日志必须含实测return，不凭U伪造缺失记录"
            rows.append({"episode": row["episode"], "env_step": row["env_step"],
                         "update_step": row["updates"], "U": row["U"], "return": row["return"]})
    return rows


def collect():
    runs = []
    historical = REPORT / "远程结果原件20260921-0439/完成运行"
    paths = sorted(historical.glob("20260920T175318_*/训练结果.json"),
                   key=lambda p: read(p.parent/"配置快照.json")["train"]["target_entropy_ratio"])
    assert len(paths) == 3
    for idx, result in enumerate(paths):
        cfg = read(result.parent/"配置快照.json")
        runs.append(make_run(f"Horizon T-SAC · η={cfg['train']['target_entropy_ratio']:g}", "Horizon",
                    result.parent/"配置快照.json", result.parent/"运行清单.json", result, color=COLORS[idx]))
    labels = {"cnn_sac": "CNN-SAC", "mlp_sac": "MLP-SAC", "mlp_dqn": "MLP-DQN",
              "mlp_ppo": "MLP-PPO", "tsac_205": "T-SAC（局部205控制组）"}
    found = {}
    for result in (REPORT/"最终结果原件20260921-0601/学习运行清单").glob("*/基线训练结果.json"):
        found[read(result)["algorithm"]] = result
    assert set(found) == set(labels)
    for idx, (algo, label) in enumerate(labels.items()):
        result = found[algo]
        runs.append(make_run(label, "学习基线", result.parent/"配置快照.json",
                            result.parent/"运行清单.json", result, color=COLORS[idx]))
    plan_path = REPORT/"Spectrum修复继续训练计划.json"
    plan = read(plan_path)
    for seed in (42, 43):
        folder = REPORT/f"Spectrum对照300回合结果原件/CNN继续训练对照-seed{seed}"
        runs.append(make_run("CNN-SAC（继续训练对照）", "Spectrum配对", folder/"配置快照.json",
                            folder/"运行清单.json", folder/"训练结果.json", color=COLORS[0], transfer=True))
    r = make_run("Spectrum T-SAC（全局注意力残差）", "Spectrum配对",
                 RAW/"Spectrum-seed42-配置快照.json", RAW/"Spectrum-seed42-运行清单.json",
                 RAW/"Spectrum-seed42-训练结果.json", color=COLORS[1], transfer=True)
    r["status"] = "墙钟预算结束（271回合）"
    runs.append(r)
    old_path, new_path = RAW/"Spectrum-seed43-gpu4训练日志.txt", RAW/"gpu2-seed43恢复训练日志.txt"
    old, new = parse_log(old_path), parse_log(new_path)
    old_retained = [dict(e, segment=0) for e in old if e["episode"] <= 150]
    discarded = [e for e in old if e["episode"] > 150]
    assert len(old_retained) == 150 and old_retained[-1]["env_step"] == 21294
    assert old_retained[-1]["update_step"] == 20295
    assert new[0]["episode"] == 151
    new = [dict(e, segment=1) for e in new]
    config = plan["experiments"][3]["config"]
    resume_config_path = REPORT/"gpu2恢复配置.json"
    resume_config = read(resume_config_path)
    for group in ("data", "physics", "env", "model", "telemetry"):
        assert config[group] == resume_config[group]
    for key in config["train"]:
        if key not in ("episodes", "max_wall_seconds", "checkpoint_every"):
            assert config["train"][key] == resume_config["train"][key]
    progress = read(RAW/"gpu2-恢复进度.json")
    captured = read(RAW/"远程绘图数据取回清单.json")["captured_at_utc"]
    r = {"name": "Spectrum T-SAC（全局注意力残差）", "group": "Spectrum配对", "config": config,
         "manifest": {}, "algorithm_id": "spectrum_tsac", "run_ids": ["20260921T031515_931ec8ab244a",
                         Path(progress["run_dir"]).name], "rows": old_retained+new, "color": COLORS[1],
         "transfer": True, "status": f"阶段快照（至{new[-1]['episode']}回合）", "captured_at_utc": captured,
         "restore_episode": 150, "discarded_branch_rows": discarded, "resume_config": resume_config,
         "sources": [source(p) for p in (old_path, new_path, plan_path, resume_config_path,
                           RAW/"gpu2-恢复进度.json", RAW/"远程绘图数据取回清单.json")]}
    runs.append(r)
    for r in runs:
        rows = r["rows"]
        assert [e["episode"] for e in rows] == list(range(1, len(rows)+1))
        assert len({e["env_step"] for e in rows}) == len(rows)
        assert np.all(np.diff([e["env_step"] for e in rows]) > 0)
        assert np.all(np.diff([e["update_step"] for e in rows]) >= 0)
        assert all(np.isfinite(e["U"]) and 0 <= e["U"] <= 1 for e in rows)
        assert np.allclose([e["return"] for e in rows], np.array([e["U"] for e in rows])*100, rtol=1e-10, atol=1e-9)
    assert len(runs) == 12
    return runs


def architecture(r):
    m, algo = r["config"]["model"], r["algorithm_id"]
    enc = {"full_attention": "完整场景 Transformer", "cnn_local": "局部205 + 频谱CNN",
           "mlp_local": "局部205 + MLP", "legacy14": "局部205 Transformer（14 tokens）",
           "cnn_attention_residual": "CNN + 全局Transformer残差"}.get(m["encoder"], m["encoder"])
    lines = [enc]
    if algo == "mlp_ppo":
        lines += ["独立评分策略 + 状态价值 V", "MLP：2层 × 128"]
    elif algo == "mlp_dqn":
        lines += ["三分量可加 Q；ε-greedy", "MLP：2层 × 128"]
    else:
        actor = "条件策略" if m["actor"] == "conditional" else "独立评分策略"
        critic = "联合Q" if m["critic"] == "joint" else "可加Q"
        lines += [f"{actor} / 双{critic}", f"隐藏宽度 {m['d_model']}"]
        if m["encoder"] in ("cnn_local", "cnn_attention_residual"):
            lines += ["卷积通道32/64；卷积核5/3"]
        if "attention" in m["encoder"] or algo == "tsac_205":
            layers = m.get("context_layers", 1) if r["transfer"] and m["encoder"] == "cnn_attention_residual" else m["encoder_layers"]
            lines += [f"注意力 {m['attention_heads']}头 × {layers}层"]
        if m["encoder"] == "cnn_attention_residual":
            lines += [f"残差幅度系数 {m['context_residual_scale']:g}"]
    return lines


def parameters(r):
    t = r["config"]["train"]
    algo = r["algorithm_id"]
    result = [f"seed={t['seed']}    γ={number(t['gamma'])}"]
    if algo == "mlp_ppo":
        h = r["manifest"]["paper_baseline"]["agent_spec"]["hyperparameters"]
        result += [f"lrπ={lr(t['actor_lr'])}；lrV={lr(t['critic_lr'])}",
            f"GAE λ={h['gae_lambda']:g}；clip={h['clip_epsilon']:g}",
            f"每轮 {h['epochs']} epochs；minibatch={h['minibatch_size']}",
            f"熵系数={h['entropy_coefficient']:g}；价值系数={h['value_coefficient']:g}",
            "on-policy；优势归一化", "无Replay、SAC热身或温度α"]
    elif algo == "mlp_dqn":
        h = r["manifest"]["paper_baseline"]
        choices = h.get("unpublished_choices", {})
        result += [f"lrQ={lr(t['critic_lr'])}；τ={number(t['tau'])}",
            "ε：1 → 0.05 / 10000环境步", f"batch={t['batch_size']}；回放={t['replay_capacity']}",
            f"热身={t['warmup_steps']}环境步", "1次更新/步；无Actor/温度α"]
    else:
        result += [f"lrπ={lr(t['actor_lr'])}；lrQ={lr(t['critic_lr'])}",
            f"lrα={lr(t['alpha_lr'])}；τ={number(t['tau'])}", f"目标熵比例 η={t['target_entropy_ratio']:g}"]
        result += ["α继承CNN父模型，后续自动调节" if r["transfer"] else f"α初始={t['initial_alpha']:g}，自动调节"]
        result += [f"batch={t['batch_size']}；微批={t['microbatch_size']}",
            f"Replay={t['replay_capacity']}（回合均衡）", f"热身={t['warmup_steps']}环境步；1次更新/步"]
    result += [f"梯度裁剪阈值={t['gradient_clip_norm']:g}"]
    return result


def trailing(values):
    v = np.asarray(values, dtype=float)
    sums = np.r_[0.0, np.cumsum(v)]
    starts = np.maximum(np.arange(len(v))+1-SMOOTH, 0)
    return (sums[np.arange(len(v))+1]-sums[starts])/(np.arange(len(v))+1-starts)


def draw_curve(ax, r, metric="U", xkey="episode", raw=True, label=None):
    for index, seg in enumerate(sorted({e["segment"] for e in r["rows"]})):
        rows = [e for e in r["rows"] if e["segment"] == seg]
        x, y = [e[xkey] for e in rows], [e[metric] for e in rows]
        if raw:
            ax.plot(x, y, color=r["color"], alpha=.23, lw=.95, label="逐回合原始值" if index == 0 else None)
        ax.plot(x, trailing(y), color=r["color"], lw=2.3,
                label=(label or "后向10回合移动平均") if index == 0 else None)
    ax.grid(axis="y")
    ax.set_ylim(0, 1.02 if metric == "U" else 102)
    ax.set_ylabel("平均需求满足度 U" if metric == "U" else "累计回报 R")
    ax.set_xlabel(("新增训练回合（不含父模型100回合）" if r["transfer"] else "训练回合") if xkey == "episode" else "累计环境交互步（本次试验）")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    if xkey == "env_step":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
    ax.set_xlim(0, (300 if r["transfer"] else 100) if xkey == "episode" else r["rows"][-1]["env_step"]*1.025)
    if "restore_episode" in r:
        boundary = 150 if xkey == "episode" else 21294
        ax.axvline(boundary, color="#7b838a", ls=(0, (4, 4)), lw=1)
        ax.text(boundary, .06 if metric == "U" else 6, "  恢复点", color="#666f77", fontsize=9)


exports = []


def save(fig, name, runs, notes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.texts:
        bbox = text.get_window_extent(renderer=renderer)
        assert bbox.x0 >= -1 and bbox.y0 >= -1 and bbox.x1 <= fig.bbox.width+1 and bbox.y1 <= fig.bbox.height+1, (name, text.get_text())
    path = OUT/name
    fig.savefig(path, dpi=DPI, metadata={"Title": name.removesuffix(".png"),
        "Description": "真实逐回合训练记录；后向10回合移动平均；算法、实际超参、预算及谱系见图及JSON。"})
    plt.close(fig)
    exports.append({"file": name, "sha256": digest(path), "run_ids": [i for r in runs for i in r["run_ids"]],
                    "notes": notes, "dpi": DPI})


def individual(r, index):
    seed = r["config"]["train"]["seed"]
    fig = plt.figure(figsize=(15.2, 8.8), facecolor="white")
    title = f"{r['name']}  |  seed={seed}"
    fig.text(.055, .955, title, fontsize=21, fontweight="bold")
    last = r["rows"][-1]
    fig.text(.055, .91, f"训练收敛轨迹  ·  {r['status']}  ·  {len(r['rows'])}回合 / {last['env_step']:,}环境步 / {last['update_step']:,}次更新", fontsize=11.8)
    ax1 = fig.add_axes([.066, .565, .595, .285])
    ax2 = fig.add_axes([.066, .175, .595, .285])
    draw_curve(ax1, r)
    draw_curve(ax2, r, "return", "env_step")
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, frameon=False, fontsize=9)
    fig.text(.714, .85, "网络与实际生效参数", fontsize=13.5, fontweight="bold")
    info = architecture(r) + [""] + parameters(r) + ["", "统一物理：6000 W / 8组 / 100槽", "训练数据：70场景；需求倍率0.25"]
    if r["transfer"]:
        info += ["起点：同一个CNN100回合父模型", "父训练预算另计，非从零多seed"]
    if "restore_episode" in r:
        stamp = datetime.fromisoformat(r["captured_at_utc"]).astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
        info += [f"gpu4的1–150 + gpu2的151–{last['episode']}", "旧尝试151–160不纳入主线", "平滑不跨150回合恢复点", f"快照：{stamp}（北京时间）"]
    elif r["status"].startswith("墙钟"):
        info += ["单组3小时墙钟上限，完整回合收尾"]
    flattened = [line for item in info for line in (wrapped(item) if item else [""])]
    # 中英文混排按字符宽度换行，右栏底部留出来源区。
    step = min(.0265, .725/max(len(flattened), 1))
    y = .817
    for line in flattened:
        fig.text(.714, y, line, fontsize=10.2, va="top")
        y -= step
    fig.text(.055, .077, "浅线：逐回合原值；实线：后向10回合均值（起始不足10回合用现有样本）。R = 100 × U，为同一奖励定义的两种刻度。", fontsize=9.3)
    fig.text(.055, .045, "仅展示训练轨迹，不等于独立验证集性能或正式收敛证明。场景抽样与增强会造成波动；不跨算法、seed或初始化阶段混算。", fontsize=9.3)
    fig.text(.055, .019, "run_id："+" → ".join(r["run_ids"]), fontsize=8.2, color="#75808a")
    clean = r["name"].replace(" · ", "-").replace("=", "").replace("（", "-").replace("）", "").replace(" ", "")
    filename = f"{index:02d}-{clean}-seed{seed}-训练收敛曲线.png"
    r["png"] = filename
    save(fig, filename, [r], {"smoothing": "trailing10_per_segment", "raw_visible": True,
         "metric": ["train_U", "train_episode_return"], "status": r["status"]})


def summary(runs, group):
    if group == "Spectrum配对":
        fig = plt.figure(figsize=(15.2, 9.2), facecolor="white")
        fig.text(.055, .952, "Spectrum 修复试验：按续训 seed 配对的训练轨迹", fontsize=20, fontweight="bold")
        fig.text(.055, .912, "共同CNN100回合父权重；来源训练预算另计。每组曲线单独平滑，不对两个续训seed取平均。", fontsize=11)
        for idx, seed in enumerate((42, 43)):
            subset = [r for r in runs if r["config"]["train"]["seed"] == seed]
            ax = fig.add_axes([.07+idx*.48, .345, .40, .48])
            for r in subset:
                draw_curve(ax, r, raw=False, label=("CNN继续训练对照" if r["config"]["model"]["encoder"] == "cnn_local" else "Spectrum残差模型")+f" · {len(r['rows'])}回合")
            ax.set_title(f"续训 seed={seed}", fontsize=13.5, pad=14)
            ax.legend(loc="lower right", frameon=False, fontsize=10)
        lines = ["共同超参：lrπ=1e-5，lrQ=1e-4，lrα=1e-4；η=0.5；γ=1；τ=0.005；batch/微批=64/64；Replay=4000；热身1000步。",
                 "网络：两侧均独立评分Actor + 可加双Q、隐藏宽128；Spectrum额外使用全局4头×1层注意力与0.25有界残差。",
                 "seed42：CNN完成300回合，Spectrum达到3小时墙钟上限后结束于271回合；空白部分没有外推。",
                 "seed43：Spectrum为14:18绘图快照（完整记录至244回合）；150处恢复，旧分支151–160不重复计入；CNN已完成300回合。",
                 "曲线为后向10回合移动均值；原始值见各算法单图。这里是训练数据，公平性能比较应使用同预算检查点的独立validation。"]
        # 时间与回合均取保存的快照，避免重跑脚本后使用过时固定标注。
        snap = next(r for r in runs if "captured_at_utc" in r)
        stamp = datetime.fromisoformat(snap["captured_at_utc"]).astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
        lines[3] = f"seed43：Spectrum为{stamp}绘图快照（至{len(snap['rows'])}回合）；150处恢复，旧分支151–160不重复计入；CNN完成300回合。"
        for i, line in enumerate(lines):
            fig.text(.055, .245-i*.039, line, fontsize=10.1)
        save(fig, "15-Spectrum同源续训配对-训练收敛对比.png", runs, {"smoothing": "trailing10_per_segment", "seeds_not_pooled": True})
        return
    fig = plt.figure(figsize=(15.2, 8.8), facecolor="white")
    is_horizon = group == "Horizon"
    title = "Horizon T-SAC：目标熵比例 η 的训练曲线" if is_horizon else "论文学习基线：统一环境中的训练曲线"
    fig.text(.055, .95, title, fontsize=20, fontweight="bold")
    fig.text(.055, .91, "seed=42  ·  每组100回合 / 14281环境交互步  ·  70训练场景  ·  6000 W功率上限", fontsize=11.5)
    ax = fig.add_axes([.07, .35, .865, .48])
    for r in runs:
        draw_curve(ax, r, raw=False, label=r["name"])
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, ncol=1 if is_horizon else 2)
    if is_horizon:
        lines = ["仅η不同：0.2 / 0.5 / 0.8；完整场景Transformer（d=128，4头，2层），条件Actor + 联合双Q。",
                 "共同超参：lrπ=1e-5，lrQ=1e-4，lrα=1e-4；γ=1；τ=0.005；batch/微批=64/64；Replay=4000；热身1000步。",
                 "实线：后向10回合移动平均（无跨seed平均）；原始波动和累计回报见对应算法单图。",
                 "仅为训练曲线，不是验证集分数；η越大代表熵目标比例越高，不能单看训练曲线选出泛化最优模型。"]
        filename = "13-Horizon不同熵目标-训练收敛对比.png"
    else:
        sac = next(r for r in runs if r["algorithm_id"] == "cnn_sac")["config"]["train"]
        lines = [f"SAC族（CNN / MLP / 局部T-SAC）：lrπ={lr(sac['actor_lr'])}，lrQ={lr(sac['critic_lr'])}，lrα={lr(sac['alpha_lr'])}；η=0.5；τ=0.005；Replay=4000。",
                 "DQN：lrQ=1e-4；ε从1衰减至0.05（10000步）；τ=0.005；Replay=4000。",
                 "PPO：lrπ=lrV=3e-4；GAE λ=0.95；clip=0.2；epochs=10；minibatch=64；熵系数=0.01；无Replay。",
                 "共同：γ=1，隐藏宽128，局部205观察；SAC/DQN batch=64、热身1000步。方法级统一环境适配，非原论文4000回合数值复刻。",
                 "实线：后向10回合移动平均；优化次数因算法而异。Fixed / Greedy / Random没有学习过程，不编造收敛曲线。"]
        filename = "14-论文学习基线-训练收敛对比.png"
    for i, line in enumerate(lines):
        fig.text(.055, .253-i*.04, line, fontsize=10.2)
    save(fig, filename, runs, {"smoothing": "trailing10", "raw_visible": False, "training_only": True})


def main():
    runs = collect()
    for i, r in enumerate(runs, 1):
        individual(r, i)
    for group in ("Horizon", "学习基线", "Spectrum配对"):
        summary([r for r in runs if r["group"] == group], group)
    data = {"schema_version": "training_curve_export.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "第二阶段三η、五学习基线及四个同源续训流；运行中流为明确快照",
            "smoothing_window": SMOOTH, "smoothing": "causal_trailing_mean_min_periods1_per_segment", "runs": runs}
    (OUT/"训练曲线数据与配置.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    from PIL import Image
    for record in exports:
        with Image.open(OUT/record["file"]) as im:
            im.verify()
        with Image.open(OUT/record["file"]) as im:
            record["pixels"] = list(im.size)
    validation = {"files": exports, "run_count": len(runs), "png_count": len(exports),
        "checks": ["每回合编号连续且不重复", "环境步递增，更新步非递减", "U有限且属于[0,1]", "每回合R=100U",
            "配置与运行清单一致", "150回合恢复边界核实，丢失分支单列", "图面文字边界检查", "全部PNG可解码"],
        "record_counts": {r["png"]: len(r["rows"]) for r in runs},
        "limitations": ["不是独立验证收敛曲线", "不含loss/alpha：本次仅导出逐回合结果中的U及实测return",
                        "静态基线无训练曲线", "运行中seed43快照不冒充最终结果", "旧分支151–160仅保留原件不混入恢复主线"]}
    (OUT/"PNG导出与核验清单.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    doc = ["# 训练收敛曲线PNG导出说明", "", "已导出12张单运行曲线和3张分组对比，共15张PNG，240 dpi。原始历史文件保持不变；本脚本只读取已保存JSON/日志，未改变训练或评价。", "",
           "## 阅读方式", "", "单图上图为U对回合，下图为实测累计回报对环境步；奖励定义R=100U。浅线保留逐回合波动，粗线为后向10回合移动均值，开头不足10回合使用已有样本。分组图只画均值。训练场景有抽样和增强，因此这些图不能替代独立validation，也不能证明已收敛。", "",
           "每张单图标注算法、seed、网络、实际生效学习率、熵目标或PPO/DQN专属参数、batch、Replay、热身、实际回合/环境步/更新次数和run_id。PPO不标成使用SAC Replay、温度α或τ。", "",
           "## 续训边界", "", "Spectrum与CNN继续训练均从同一个CNN100回合父权重初始化，来源预算另计，不能说成从零独立多seed。Spectrum42在3小时预算后止于271回合；seed43图为保存时快照，保留旧运行1–150回合与新运行151起记录，旧尝试151–160不重复计入。平滑也在150处断开。", "",
           "Fixed、Greedy、Random为静态策略，没有训练收敛过程，因此不生成伪造的学习曲线。本次不读取test结果，不依据图改超参数。", "", "## 图片索引", ""]
    doc += [f"- [{record['file']}]({record['file']})" for record in exports]
    doc += ["", "## 数据与复现", "", "`训练曲线数据与配置.json`包含全部绘图点、配置、来源路径及SHA；`PNG导出与核验清单.json`包含PNG哈希、分辨率和验收结果。`绘图数据原件`保留远程关闭结果及日志快照。", "",
            "从本目录使用本地环境执行 `D:/graduate/PPO4090/.venv-horizon/Scripts/python.exe -X utf8 绘制训练收敛曲线.py` 可复现本批快照。脚本不会自动更新运行中数据；后续更新应另存新快照并保留本批证据。", ""]
    (OUT/"训练曲线导出说明.md").write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "png_count": len(exports), "runs": [(r["name"], r["config"]["train"]["seed"], len(r["rows"])) for r in runs]}, ensure_ascii=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        main()
