#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ************************* #
# Replace with your predict sql json path. Ensure that filename is correct!
predicted_sql_path="${PROJECT_ROOT}/mini_dev/llm/exp_result/mini_dev/openai/gpt-oss-20b_dyn-examples_rerank-examples_SQLite.json"
# Path to sqlite databases
db_root_path="${PROJECT_ROOT}/data/mini_dev/MINIDEV_sqlite/dev_databases/"
# Extract the base filename without extension
base_name=$(basename "$predicted_sql_path" .json)
# Define the output log path
output_log_path="${SCRIPT_DIR}/eval_result/${base_name}.txt"
# ************************* #

num_cpus=16
meta_time_out=30.0
sql_dialect="SQLite"

case $sql_dialect in
  "SQLite")
    diff_json_path="${PROJECT_ROOT}/data/mini_dev/MINIDEV_sqlite/mini_dev_sqlite.json"
    ground_truth_path="${PROJECT_ROOT}/data/mini_dev/MINIDEV_sqlite/mini_dev_sqlite_gold.sql"
    ;;
  "PostgreSQL")
    diff_json_path="../postgresql/mini_dev_postgresql.json"
    ground_truth_path="../postgresql/mini_dev_postgresql_gold.sql"
    ;;
  "MySQL")
    diff_json_path="../mysql/mini_dev_mysql.json"
    ground_truth_path="../mysql/mini_dev_mysql_gold.sql"
    ;;
  *)
    echo "Invalid SQL dialect: $sql_dialect"
    exit 1
    ;;
esac

echo "Differential JSON Path: $diff_json_path"
echo "Ground Truth Path: $ground_truth_path"

echo "starting to compare with knowledge for ex, sql_dialect: ${sql_dialect}"
python3 -u "${SCRIPT_DIR}/evaluation_ex.py" --db_root_path "${db_root_path}" --predicted_sql_path "${predicted_sql_path}" \
--ground_truth_path "${ground_truth_path}" --num_cpus "${num_cpus}" --output_log_path "${output_log_path}" \
--diff_json_path "${diff_json_path}" --meta_time_out "${meta_time_out}" --sql_dialect "${sql_dialect}"

echo "starting to compare with knowledge for R-VES, sql_dialect: ${sql_dialect}"
python3 -u "${SCRIPT_DIR}/evaluation_ves.py" --db_root_path "${db_root_path}" --predicted_sql_path "${predicted_sql_path}" \
--ground_truth_path "${ground_truth_path}" --num_cpus "${num_cpus}" --output_log_path "${output_log_path}" \
--diff_json_path "${diff_json_path}" --meta_time_out "${meta_time_out}" --sql_dialect "${sql_dialect}"

echo "starting to compare with knowledge for soft-f1, sql_dialect: ${sql_dialect}"
python3 -u "${SCRIPT_DIR}/evaluation_f1.py" --db_root_path "${db_root_path}" --predicted_sql_path "${predicted_sql_path}" \
--ground_truth_path "${ground_truth_path}" --num_cpus "${num_cpus}" --output_log_path "${output_log_path}" \
--diff_json_path "${diff_json_path}" --meta_time_out "${meta_time_out}" --sql_dialect "${sql_dialect}"
