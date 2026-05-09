# Experiment scripts

This folder contains utility scripts.

## json_to_jsonl.py

Converts JSON file to JSONL file needed for benchmark evaluation

Example:
```
python experiments/json_to_jsonl.py --input_file mini_dev/data/mini_dev_sqlite.json --output_file mini_dev/sqlite/mini_dev_sqlite.jsonl
```

## combine_predicts_with_gold.py

Combines ground truth data with predicted SQL queries by merging two JSON files.

Example:
```
python experiments/combine_predicts_with_gold.py --ground_truth_path mini_dev/data/mini_dev_sqlite.json --predicted_sql_path mini_dev/llm/exp_result/mini_dev/gpt-oss-120b_SQLite.json --output_path merged_output.json
```

## error_analysis.py

Performs error analysis on the annotated benchmark results and generates a visualization of error distributions.

Example:
```
python experiments/error_analysis.py
```

## compare_translation.py

Compares original BIRD mini_dev vs ours mini_dev_ru_mod translation results by creating bar charts showing performance differences between the two languages for different models and metrics (EX, R-VES, Soft-F1). Generates two plots: one for baseline and one for dynamic examples with reranking and dynamic values approaches.

Example:
```
python experiments/compare_translation.py
```

## plot_metrics.py

Creates visualizations of metrics by model and experiment type, generating bar charts showing metric values and average percentage gains compared to baseline. Produces individual plots for each metric and a summary plot showing average relative gains.

Example:
```
python experiments/plot_metrics.py
```


## latex_metrics_table.py

Generates LaTeX tables for metrics (EX, R-VES, Soft-F1) from evaluation results CSV files. Just for convenience. Highlights the maximum values in bold for easy comparison.

Example:
```
python experiments/latex_metrics_table.py
```
