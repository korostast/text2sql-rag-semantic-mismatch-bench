import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

en_path = "artifacts/evaluation_results/no_evidence/results.csv"
ru_path = "artifacts/evaluation_results/mini_dev_ru_mod/results.csv"

df_en = pd.read_csv(en_path)
df_ru = pd.read_csv(ru_path)

df_en_base = df_en[df_en["type"] == "Baseline"][["model", "metric", "value_total"]]
df_ru_base = df_ru[df_ru["type"] == "Baseline"][["model", "metric", "value_total"]]

merged = pd.merge(df_en_base, df_ru_base, on=["model", "metric"], suffixes=("_en", "_ru"))

metrics = ["EX", "R-VES", "Soft-F1"]

fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)))
if len(metrics) == 1:
    axes = [axes]

models_order = [
    "nemotron-3-super-120b-a12b",
    "gpt-oss-120b",
    "nemotron-3-nano-30b-a3b",
    "gpt-oss-20b",
    "mistral-medium-3.5-128b",
    "gemma-3n-e4b-it",
    "Mistral-7B-Instruct-v0.3",
]

for ax, metric in zip(axes, metrics):
    data = merged[merged["metric"] == metric].copy()
    data["model"] = pd.Categorical(data["model"], categories=models_order, ordered=True)
    data = data.sort_values("model")

    x = np.arange(len(data))
    width = 0.35

    en_vals = data["value_total_en"].values
    ru_vals = data["value_total_ru"].values
    models_list = data["model"].tolist()

    rects1 = ax.bar(
        x - width / 2, en_vals, width, label="English (original)", color="tab:blue", alpha=0.8
    )
    rects2 = ax.bar(
        x + width / 2, ru_vals, width, label="Russian (ours)", color="tab:orange", alpha=0.8
    )

    ax.set_title(metric, fontsize=16)
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

plt.suptitle("BIRD mini_dev, English vs Russian\nBaseline, w/o evidence", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = "mini_dev_baseline_en_vs_ru.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Plot saved to '{out_path}'")

df_en_dyn = df_en[df_en["type"] == "Dyn-examples & rerank-examples & dyn-values"][
    ["model", "metric", "value_total"]
]
df_ru_dyn = df_ru[df_ru["type"] == "Dyn-examples & rerank-examples & dyn-values"][
    ["model", "metric", "value_total"]
]

merged_dyn = pd.merge(df_en_dyn, df_ru_dyn, on=["model", "metric"], suffixes=("_en", "_ru"))

fig2, axes2 = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)))
if len(metrics) == 1:
    axes2 = [axes2]

for ax, metric in zip(axes2, metrics):
    data = merged_dyn[merged_dyn["metric"] == metric].copy()
    data["model"] = pd.Categorical(data["model"], categories=models_order, ordered=True)
    data = data.sort_values("model")

    x = np.arange(len(data))
    width = 0.35

    en_vals = data["value_total_en"].values
    ru_vals = data["value_total_ru"].values
    models_list = data["model"].tolist()

    rects1 = ax.bar(
        x - width / 2, en_vals, width, label="English (original)", color="tab:blue", alpha=0.8
    )
    rects2 = ax.bar(
        x + width / 2, ru_vals, width, label="Russian (ours)", color="tab:orange", alpha=0.8
    )

    ax.set_title(metric, fontsize=16)
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

plt.suptitle(
    "BIRD mini_dev, English vs Russian\nDyn-examples (reranked) & Dyn-values, w/o evidence",
    fontsize=20,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path2 = "mini_dev_dyn_en_vs_ru.png"
plt.savefig(out_path2, dpi=200, bbox_inches="tight")
plt.close()

print(f"Plot saved to '{out_path2}'")
