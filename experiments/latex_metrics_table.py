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

type_short = {
    "Dyn-examples": "Dyn-examples Only",
    "Dyn-examples & rerank-examples": "Dyn-examples (reranked)",
    "Dyn-values": "Dyn-values",
    "Dyn-examples & rerank-examples & dyn-values": "Full",
}


def format_value(val, is_max=False):
    if is_max:
        return f"\\textbf{{{val:.1f}}}"
    else:
        return f"{val:.1f}"


def create_latex_table(metric):
    """Create a LaTeX table for a specific metric."""
    sub = (
        df[df["metric"] == metric]
        .pivot(index="model", columns="type", values="value_total")
        .reindex(models_order)[types_order]
    )

    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    caption = f"{metric}, {dataset_type}".replace("_", "\\_")
    latex_lines.append(f"\\caption{{{caption}}}")
    label = f"tab:{metric.lower()}".replace("_", "\\_")
    latex_lines.append(f"\\label{{{label}}}")

    num_cols = len(types_order) + 1
    col_spec = f"l{'c' * num_cols}"
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    header_labels = [type_short.get(t, t).replace("&", "\\&") for t in types_order]
    header = f"Model & {' & '.join(header_labels)}" + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    for model in models_order:
        if model not in sub.index:
            continue

        row_values = []
        model_escaped = model.replace("_", "\\_")
        row_values.append(model_escaped)

        row_vals = [sub.loc[model, t] for t in types_order]
        max_val = max(row_vals)

        for _, t in enumerate(types_order):
            val = sub.loc[model, t]
            is_max = val == max_val
            row_values.append(format_value(val, is_max))

        latex_lines.append(" & ".join(row_values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


for metric in metrics:
    print()
    print(create_latex_table(metric))
    print()
