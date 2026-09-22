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


def parse_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if "episode" in e and "U" in e:
            rows.append({"episode": e["episode"], "env_step": e["env_step"],
                         "update_step": e["updates"], "U": e["U"], "return": e["return"]})
    return rows


def collect():
    data = read(OUT/"训练曲线数据与配置.json")
    runs = data["runs"]
    assert len(runs) == 12 and sum(len(r["rows"]) for r in runs) == 1971
    frozen = read(RAW/"冻结十一组训练曲线.json")
    assert runs[:11] == frozen["runs"], "其余11条训练流必须保持冻结"
    final = read(RAW/"gpu2完成状态初查原件.json")
    assert digest(RAW/"gpu2完成状态初查原件.json") == data["finalization"]["source"]["sha256"]
    result, manifest = final["run_files"]["训练结果.json"], final["run_files"]["运行清单.json"]
    r = runs[-1]
    assert manifest["status"] == "completed"
    assert r["manifest"] == manifest and r["resume_config"] == manifest["config"]
    assert manifest["parent_run_id"] == r["run_ids"][0]
    assert manifest["run_id"] == r["run_ids"][1] == result["run_id"]
    assert result["counters"] == {"episodes": 300, "env_step": 42632, "update_step": 41633}
    assert len(result["episodes"]) == 150
    assert len({e["episode_id"] for e in result["episodes"]}) == 150
    assert all(not e["truncated"] for e in result["episodes"])
    old = parse_log(RAW/"Spectrum-seed43-gpu4训练日志.txt")
    retained = [dict(e, segment=0) for e in old if e["episode"] <= 150]
    discarded = [e for e in old if e["episode"] > 150]
    assert len(retained) == 150 and [e["episode"] for e in discarded] == list(range(151, 161))
    assert retained[-1]["env_step"] == manifest["resume_env_step"] == 21294
    assert retained[-1]["update_step"] == manifest["resume_update_step"] == 20295
    actual = [{"episode": 151+i, "env_step": e["env_step"], "update_step": e["update_step"],
               "U": e["mean_satisfaction"], "return": e["episode_return"], "segment": 1}
              for i, e in enumerate(result["episodes"])]
    assert r["rows"] == retained+actual and r["discarded_branch_rows"] == discarded
    for r in runs:
        rows = r["rows"]
        assert [e["episode"] for e in rows] == list(range(1, len(rows)+1))
        assert np.all(np.diff([e["env_step"] for e in rows]) > 0)
        assert np.all(np.diff([e["update_step"] for e in rows]) >= 0)
        assert all(np.isfinite(e["U"]) and 0 <= e["U"] <= 1 for e in rows)
        assert np.allclose([e["return"] for e in rows], np.array([e["U"] for e in rows])*100, rtol=1e-10, atol=1e-9)
    assert r["rows"][-1]["env_step"] == 42632 and r["rows"][-1]["update_step"] == 41633
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
        info += [f"gpu4的1–150 + gpu2的151–{last['episode']}", "旧尝试151–160不纳入主线", "平滑不跨150回合恢复点", "最终训练结果：已完成300回合"]
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
                 "seed43：Spectrum已完成300回合；150处恢复，旧分支151–160不重复计入；CNN已完成300回合。",
                 "曲线为后向10回合移动均值；原始值见各算法单图。这里是训练数据，公平性能比较应使用同预算检查点的独立validation。"]
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
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser(description="从包内保存的真实训练记录复现最终版PNG；不联网、不训练。")
    parser.add_argument("--all", action="store_true", help="重绘全部15张；默认只重绘完成变化的12和15，并验证其余13张冻结哈希")
    args = parser.parse_args()
    runs = collect()
    if args.all:
        for i, r in enumerate(runs, 1):
            individual(r, i)
        for group in ("Horizon", "学习基线", "Spectrum配对"):
            summary([r for r in runs if r["group"] == group], group)
    else:
        individual(runs[-1], 12)
        summary([r for r in runs if r["group"] == "Spectrum配对"], "Spectrum配对")
    fixed = read(RAW/"冻结图片哈希.json")
    fixed_matches = {p["file"]: digest(OUT/p["file"]) == p["sha256"] for p in fixed["files"]}
    assert all(fixed_matches.values()), "13张固定PNG字节必须与已验收版一致"
    records = []
    for p in sorted(OUT.glob("*.png")):
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            pixels = list(im.size)
        records.append({"file": p.name, "sha256": digest(p), "dpi": DPI, "pixels": pixels,
                        "frozen_unchanged": p.name in fixed_matches})
    assert len(records) == 15
    validation = {"schema_version": "training_curve_final_validation.v1", "files": records,
        "run_count": 12, "png_count": 15, "record_count": 1971, "previous_record_count": 1915,
        "added_record_count": 56, "fixed_png_hash_matches": fixed_matches,
        "record_counts": {r["png"]: len(r["rows"]) for r in runs},
        "recovery": {"retained_old_episode_range": [1,150], "actual_new_episode_range": [151,300],
            "excluded_old_episode_range": [151,160], "cumulative_counters": runs[-1]["final_result_counters"],
            "new_result_episode_array_length": 150},
        "artifacts": {p.name: digest(p) for p in (OUT/"训练曲线数据与配置.json", OUT/"绘制训练收敛曲线.py", RAW/"gpu2完成状态初查原件.json", RAW/"Spectrum-seed43-gpu4训练日志.txt")},
        "checks": ["12条流共1971个真实训练点", "其余11流记录与配置完全冻结", "13张固定PNG哈希不变", "150恢复边界与谱系一致", "旧151–160单列排除", "新增150条与最终结果逐点精确相等", "编号连续无重复且环境步递增", "每回合R=100U", "分段后向10回合均值，不跨恢复边界", "重绘文字不超出画布", "15张PNG可解码"],
        "limitations": ["仅训练轨迹，不是独立验证收敛证明", "共同CNN父训练100回合另计", "seed42残差实际结束于271回合，未补齐或外推", "未使用终局validation或test分数作为训练点"]}
    (OUT/"PNG导出与核验清单.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "png_count": 15, "records": 1971, "frozen_pngs": 13, "final_counters": runs[-1]["final_result_counters"]}, ensure_ascii=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        main()
