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
python experiments/combine_predicts_with_gold.py --ground_truth_path mini_dev/data/mini_dev_sqlite.json --predicted_sql_path mini_dev/llm/exp_result/turbo_output_kg/gpt-oss-120b_SQLite.json --output_path merged_output.json
```
