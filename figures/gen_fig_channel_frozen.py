"""Generate a complete, paired fixed-checkpoint audit from recorded episodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from evidence_eval import suites
from evidence_stats import paired_fixed, holm


def interval(comparison):
    """Conservative 95% paired difference CI by a Bonferroni union bound.

    The two discordant cell proportions each have a 97.5% exact binomial CI.
    Subtraction covers their difference with probability at least 95%, including
    the zero-discordance case. Correlation between the cells is unrestricted.
    """
    n = comparison["episodes"]
    positive = binomtest(comparison["candidate_only"], n).proportion_ci(confidence_level=.975)
    negative = binomtest(comparison["reference_only"], n).proportion_ci(confidence_level=.975)
    return [positive.low - negative.high, positive.high - negative.low]


def load_audit(root):
    comparisons, counts, raw_p = {}, [], []
    for spec in suites("frozen"):
        name = spec["name"]
        data = {arm: json.loads((root / "frozen" / arm / f"{name}.json").read_text(encoding="utf-8"))
                for arm in ("channel", "reference")}
        if not all(d["passed"] and len(d["rows"]) == spec["episodes"] and d["suite"] == spec for d in data.values()):
            raise ValueError(f"Incomplete or mismatched evaluation: {name}")
        comparisons[name] = {}
        for metric, outcome in (("success", None), ("collision", 2), ("breach", 1)):
            rows = {arm: ([dict(r, outcome_code=int(r["success"])) for r in d["rows"]]
                          if outcome is None else d["rows"]) for arm, d in data.items()}
            test = paired_fixed(rows["channel"], rows["reference"], 1 if outcome is None else outcome)
            test["conservative_paired_95_interval"] = interval(test)
            comparisons[name][metric] = test
            raw_p.append(test["exact_mcnemar_two_sided_p"])
        for arm, d in data.items():
            s = d["summary"]
            counts.append(dict(condition=name, arm=arm, episodes=s["episodes"], successes=s["successes"],
                               collisions=s["collisions"], breaches=s["breaches"], captures=s["captures"],
                               timeout_denials=s["timeout_denials"], mean_return=s["mean_reward"],
                               checkpoint_sha256=d["policy"]["checkpoint_sha256"]))
    adjusted = iter(holm(raw_p))
    for values in comparisons.values():
        for value in values.values():
            value["holm_48_exploratory_p"] = next(adjusted)
    for arm in ("channel", "reference"):
        if len({r["checkpoint_sha256"] for r in counts if r["arm"] == arm}) != 1:
            raise ValueError("The frozen audit used changing checkpoints")
    return comparisons, counts


def figure(comparisons, output):
    labels = []
    for name in comparisons:
        if name == "nominal":
            label = "Nominal (3 vessels, agility 2.0)"
        elif name.startswith("team-"):
            label = f'{name.split("-")[1]} defenders'
        elif name.startswith("agility-"):
            label = f'Agility {name.split("-")[1]}'
        elif name.startswith("current-"):
            label = f'Current scale {name.split("-")[1]}x'
        else:
            label = f'Action delay {int(name.split("-")[1]) * .2:.1f} s'
        labels.append(label)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.5), sharey=True)
    y = np.arange(len(labels))
    for ax, metric, title, color in zip(axes, ("success", "collision", "breach"),
                                        ("Success (higher is better)", "Collision (lower is better)", "Breach (lower is better)"),
                                        ("#0072B2", "#D55E00", "#009E73")):
        estimate = np.asarray([100 * d[metric]["difference"] for d in comparisons.values()])
        bounds = np.asarray([[100 * x for x in d[metric]["conservative_paired_95_interval"]]
                             for d in comparisons.values()])
        error = np.stack((estimate - bounds[:, 0], bounds[:, 1] - estimate))
        ax.errorbar(estimate, y, xerr=error, fmt="o", color=color, markersize=3.5,
                    capsize=2, elinewidth=1., linewidth=0)
        ax.axvline(0., color="#666666", linewidth=.8, linestyle="--")
        for boundary in (.5, 6.5, 11.5, 13.5):
            ax.axhline(boundary, color="#dddddd", linewidth=.6)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Difference (percentage points)")
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Channel policy minus ARBoids: two frozen checkpoints", y=.995, fontsize=11)
    fig.text(.5, .012, "Nominal n = 2,048; each stress condition n = 256. Conservative paired 95% intervals.\n"
             "Intervals describe scenario variation conditional on these checkpoints; they are not training-seed uncertainty.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0., .07, 1., .96), w_pad=1.15)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"frozen-outcome-differences.{extension}", bbox_inches="tight")
    plt.close(fig)


def report(comparisons, counts, output):
    by_key = {(r["condition"], r["arm"]): r for r in counts}
    lines = ["# 冻结模型的独立测试证据", "",
             "在 2,048 个新的配对场景中，新模型成功 1,950 次，原 ARBoids 成功 1,871 次；"
             "碰撞分别为 62 次和 173 次，目标突破分别为 36 次和 4 次。"
             "这轮结果支持新模型减少碰撞、提高整体防守成功率，同时确认目标突破风险增加。", "",
             "范围：两个已冻结的模型，2D 水动力学环境，APF 攻击者。"
             "这些结果不分离结构、示范初始化和训练历史的贡献，也不替代正在运行的独立训练种子实验。", "",
             "| 测试条件 | 回合数 | 新模型成功 / 原模型成功 | 新模型碰撞 / 原模型碰撞 | 新模型突破 / 原模型突破 |",
             "|---|---:|---:|---:|---:|"]
    for condition in comparisons:
        a, b = by_key[(condition, "channel")], by_key[(condition, "reference")]
        lines.append(f'| {condition} | {a["episodes"]} | {a["successes"]} / {b["successes"]} | '
                     f'{a["collisions"]} / {b["collisions"]} | {a["breaches"]} / {b["breaches"]} |')
    lines += ["", "| 标准场景指标 | 新模型减原模型，百分点 | 保守配对 95% 区间 | 精确 McNemar p | 48 项探索比较 Holm p |",
              "|---|---:|---:|---:|---:|"]
    for metric in ("success", "collision", "breach"):
        r = comparisons["nominal"][metric]
        lo, hi = r["conservative_paired_95_interval"]
        lines.append(f'| {metric} | {100*r["difference"]:.3f} | [{100*lo:.3f}, {100*hi:.3f}] | '
                     f'{r["exact_mcnemar_two_sided_p"]:.4g} | {r["holm_48_exploratory_p"]:.4g} |')
    lines += ["", "两倍水流扰动下，新模型成功 218/256，原模型 229/256，点估计下降；"
              "应结合配对区间判断不确定性。8 艘防守艇时，尽管新模型优于原模型，"
              "其成功率仍只有 29/256，不能声称已具备可靠的大规模部署能力。", "",
              "每个策略在同一场景接收完全相同的初始状态；逐回合记录包含初始状态哈希和模型哈希。"
              "随机流独立于评估批量大小。成功包括捕获及到时阻止，分项计数见 CSV。"
              "回报统一为每步团队平均奖励的累计，不包含 reset 时的奖励。", "",
              "差值区间对两个不一致配对单元的发生概率分别采用 97.5% Clopper–Pearson 区间，"
              "再相减；由 Bonferroni 联合界得到至少 95% 的保守覆盖率，包含无不一致配对的情况。"
              "图中的区间不表示不同训练种子的方差。所有 16 个条件、3 个结局的探索检验均保留，"
              "并对 48 项检验统一给出 Holm 校正；未按显著性筛选场景。", "",
              "模型与完整统计信息位于 frozen-statistics.json；逐场原始数据位于 frozen/。"
              "图由 figures/gen_fig_channel_frozen.py 直接读取这些记录生成，提供矢量 PDF 和 300 dpi PNG。", ""]
    (output / "frozen-evidence.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "artifacts"
    output.mkdir(exist_ok=True)
    comparisons, counts = load_audit(args.root)
    record = dict(passed=True, comparisons=comparisons,
                  models={arm: next(r["checkpoint_sha256"] for r in counts if r["arm"] == arm)
                          for arm in ("channel", "reference")},
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "frozen-statistics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (output / "frozen-summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    figure(comparisons, output)
    report(comparisons, counts, output)
    print(json.dumps(dict(output=str(output.resolve()), nominal=comparisons["nominal"]), indent=2))


if __name__ == "__main__":
    main()
