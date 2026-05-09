from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_type = "mini_dev_ru_mod, w/o evidence"
path_to_csv = "artifacts/evaluation_results/mini_dev_ru_mod/results.csv"

df = pd.read_csv(path_to_csv)

if "Dyn-values" in df["type"].values:
    types_order = [
        "Baseline",
        "Dyn-examples & rerank-examples",
        "Dyn-values",
        "Dyn-examples & rerank-examples & dyn-values",
    ]
else:
    types_order = [
        "Baseline",
        "Dyn-examples",
        "Dyn-examples & rerank-examples",
        "Dyn-examples & rerank-examples & dyn-values",
    ]

models_order = [
    "nemotron-3-super-120b-a12b",
    "gpt-oss-120b",
    "nemotron-3-nano-30b-a3b",
    "gpt-oss-20b",
    "mistral-medium-3.5-128b",
    "gemma-3n-e4b-it",
    "Mistral-7B-Instruct-v0.3",
]
metrics = ["EX", "R-VES", "Soft-F1"]

out_dir = Path(".")
files = []

for metric in metrics:
    sub = (
        df[df["metric"] == metric]
        .pivot(index="model", columns="type", values="value_total")
        .reindex(models_order)[types_order]
    )
    x = np.arange(len(sub.index))
    width = 0.215
    fig, ax = plt.subplots(figsize=(17, 7))
    bars_by_type = {}
    # Define colors for the metrics plots to match the summary plot
    if "Dyn-values" in df["type"].values:
        metric_colors = {
            "Baseline": "tab:blue",
            "Dyn-examples & rerank-examples": "tab:orange",
            "Dyn-values": "tab:green",
            "Dyn-examples & rerank-examples & dyn-values": "tab:red",
        }
    else:
        metric_colors = {
            "Baseline": "tab:blue",
            "Dyn-examples": "tab:orange",
            "Dyn-examples & rerank-examples": "tab:green",
            "Dyn-examples & rerank-examples & dyn-values": "tab:red",
        }

    for i, t in enumerate(types_order):
        color = metric_colors.get(t, None)
        bars = ax.bar(x + (i - 1.5) * width, sub[t].values, width=width, label=t, color=color)
        bars_by_type[t] = bars

    baseline_vals = sub["Baseline"].values

    for t in types_order[1:]:
        bars = bars_by_type[t]
        vals = sub[t].values
        for i, (bar, val) in enumerate(zip(bars, vals)):
            delta = val - baseline_vals[i]
            label = f"{delta:+.1f}"
            text_color = "green" if delta >= 0 else "red"
            baseline_x = bars_by_type["Baseline"][i].get_x() + width / 2
            current_x = bar.get_x() + width / 2
            baseline_y = baseline_vals[i]
            current_y = val

            ax.text(
                current_x,
                current_y + 0.6,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    ax.set_title(f"{metric} by model and experiment type ({dataset_type})", fontsize=15)
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=25, ha="right")

    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
    type_short = {
        "Dyn-examples": "Dyn-examples Only",
        "Dyn-examples & rerank-examples": "Dyn-examples (reranked)",
        "Dyn-values": "Dyn-values",
        "Dyn-examples & rerank-examples & dyn-values": "Dyn-examples (reranked) & Dyn-values",
    }
    labels = [type_short.get(t, t) for t in types_order]
    ax.legend(labels, frameon=False, ncol=2, fontsize=9)

    fig.tight_layout()

    path = out_dir / f'bird_{metric.lower().replace("-", "_")}_value.png'
    fig.savefig(path, dpi=200, bbox_inches="tight")
    files.append(path)
    plt.close(fig)


rows = []
for metric in metrics:
    sub = df[df["metric"] == metric].pivot(index="model", columns="type", values="value_total")
    base = sub["Baseline"]
    for t in types_order[1:]:
        rows.append(
            {
                "metric": metric,
                "type": t,
                "avg_pct_gain": float(((sub[t] / base) - 1).mean() * 100),
            }
        )

summary = pd.DataFrame(rows)

type_short = {
    "Dyn-examples": "Dyn-examples Only",
    "Dyn-examples & rerank-examples": "Dyn-examples (reranked)",
    "Dyn-values": "Dyn-values",
    "Dyn-examples & rerank-examples & dyn-values": "Dyn-examples (reranked) & Dyn-values",
}

if "Dyn-values" in df["type"].values:
    metric_colors = {
        "Baseline": "tab:blue",
        "Dyn-examples & rerank-examples": "tab:orange",
        "Dyn-values": "tab:green",
        "Dyn-examples & rerank-examples & dyn-values": "tab:red",
    }
else:
    metric_colors = {
        "Baseline": "tab:blue",
        "Dyn-examples": "tab:orange",
        "Dyn-examples & rerank-examples": "tab:green",
        "Dyn-examples & rerank-examples & dyn-values": "tab:red",
    }

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, metric in zip(axes, metrics):
    sub = summary[summary["metric"] == metric].copy()
    sub["type_short"] = sub["type"].map(type_short)

    x = np.arange(len(sub))
    vals = sub["avg_pct_gain"].values
    colors = [metric_colors[t] for t in sub["type"]]

    bars = ax.bar(x, vals, color=colors)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.1 if val >= 0 else -0.1),
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
            fontweight="bold",
            color="green" if val >= 0 else "red",
        )

    ax.set_title(metric)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=None, top=ax.get_ylim()[1] * 1.1)

axes[0].set_ylabel("Average gain vs Baseline, %")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=metric_colors[t])
    for t in types_order
    if t in metric_colors and t != "Baseline"
]
fig.legend(
    handles,
    [type_short.get(t, t) for t in types_order if t in metric_colors and t != "Baseline"],
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)

fig.suptitle("Average relative gain vs Baseline", y=1.08, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.05)

path = out_dir / "bird_average_gain_vs_baseline.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
files.append(path)
plt.close(fig)

print("Saved files:")
for p in files:
    print(p)
