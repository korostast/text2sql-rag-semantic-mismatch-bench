"""
Combine ground truth data with predicted SQL queries.

This script reads a ground truth JSON file and a predictions JSON file,
then merges them by adding the predicted SQL queries to the corresponding
ground truth entries. The predictions file should use string indices as keys.

Usage:
    python combine_predicts_with_gold.py --ground_truth_path <ground_truth_json> --predicted_sql_path <predictions_json> --output_path <output_json>
"""

import argparse
import json


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--predicted_sql_path", type=str, required=True, default="")
    args_parser.add_argument("--output_path", type=str, default="SQLite")
    args = args_parser.parse_args()
    import os

    print(os.getcwd())

    with open(args.ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(args.predicted_sql_path, encoding="utf-8") as f:
        predictions = json.load(f)

    merged_data = []
    for idx, item in enumerate(ground_truth):
        merged_item = item.copy()

        if str(idx) in predictions:
            merged_item["predicted_sql"] = predictions[str(idx)]
        else:
            merged_item["predicted_sql"] = None

        merged_data.append(merged_item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully merged {len(merged_data)} entries")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
