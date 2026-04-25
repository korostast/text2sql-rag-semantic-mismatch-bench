"""
Convert JSON file to JSONL format.

This script reads a JSON file containing an array of objects and converts it
to JSONL format (one JSON object per line). This format is used by mini_dev bench evaluation

Usage:
    python json_to_jsonl.py --input_file <input_json_file> --output_file <output_jsonl_file>
"""

import argparse
import json
import os
import sys


def json_to_jsonl(input_file: str, output_file: str) -> None:
    try:
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Error: Input file must contain a JSON array, got {type(data).__name__}")
            sys.exit(1)

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"{json.dumps(item, ensure_ascii=False)}\n")

        print(f"Successfully converted {len(data)} items from JSON to JSONL format")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")

    except FileNotFoundError:
        print(f'Error: Input file "{input_file}" not found')
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in input file "{input_file}": {e}')
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")

    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSONL file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f'Error: Input file "{args.input_file}" does not exist')
        sys.exit(1)

    json_to_jsonl(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
