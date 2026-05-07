#!/bin/bash
# -------------------------------------------------------------
# NOTE: run from `mini_dev/llm` folder as `bash run/run_gpt.sh`
# -------------------------------------------------------------

eval_path='../data/mini_dev_sqlite.json' # _sqlite.json, _mysql.json, _postgresql.json
db_root_path='../data/dev_databases/'
use_knowledge='True'
mode='mini_dev' # dev, train, mini_dev
cot='False'

# Choose the SQL dialect to run, e.g. SQLite, MySQL, PostgreSQL
# PLEASE NOTE: You have to setup the database information in table_schema.py
# if you want to run the evaluation script using MySQL or PostgreSQL
sql_dialect='SQLite'

# Choose the output path for the generated SQL queries
data_kg_output_path='./exp_result/mini_dev/'

# Base LLM. Fill the arrays to run multiple experiments sequantially
llm_api_keys=('<apikey>')
llm_urls=('<url>')
llm_models=('openai/gpt-oss-120b')

# Choose the number of threads to run in parallel for each model above, 1 for single thread
num_threads=(4)

# Dynamic few-shot examples configuration
dynamic_examples_few_shot='True'
opensearch_url='http://localhost:9200'
embedder_url='<url>'
embedder_model='bge-m3'
embedder_api_key='<apikey>'
embedder_k_results=100

# Reranking configuration
rerank_dynamic_few_shots='True'
reranker_url='<url>'
reranker_model='bge-reranker'
reranker_api_key='<apikey>'
reranker_k_results=5
reranker_score_threshold=0.001

# Dynamic value few-shots configuration
dynamic_value_few_shots='True'
value_search_k_results=5
value_search_index_name='bird_dev_values'

if [ "$dynamic_examples_few_shot" = "True" ]; then
    echo "Indexing training data to OpenSearch..."
    python3 -u ./src/index_training_data.py \
        --train_path ../../data/train/train.json \
        --opensearch_url "${opensearch_url}" \
        --embedder_url "${embedder_url}" \
        --embedder_model "${embedder_model}" \
        --embedder_api_key "${embedder_api_key}"
fi

if [ "$dynamic_value_few_shots" = "True" ]; then
    echo "Indexing dev database values to OpenSearch..."
    python3 -u ./src/index_dev_values.py \
        --dev_tables_path ../data/dev_tables.json \
        --db_root_path "${db_root_path}" \
        --opensearch_url "${opensearch_url}" \
        --embedder_url "${embedder_url}" \
        --embedder_model "${embedder_model}" \
        --embedder_api_key "${embedder_api_key}" \
        --index_name "${value_search_index_name}"
fi

for i in "${!llm_models[@]}"; do
    llm_model="${llm_models[$i]}"
    llm_api_key="${llm_api_keys[$i]}"
    llm_url="${llm_urls[$i]}"
    current_num_threads="${num_threads[$i]}"

    echo "generate $llm_model batch, run in $current_num_threads threads, with knowledge: $use_knowledge, with chain of thought: $cot, with dynamic few-shot: $dynamic_examples_few_shot, with reranking: $rerank_dynamic_few_shots, with dynamic value few-shots: $dynamic_value_few_shots"
    python3 -u ./src/gpt_request.py --db_root_path "${db_root_path}" --llm_api_key "${llm_api_key}" --mode "${mode}" \
    --llm_model "${llm_model}" --eval_path "${eval_path}" --data_output_path "${data_kg_output_path}" --use_knowledge "${use_knowledge}" \
    --chain_of_thought "${cot}" --num_process "${current_num_threads}" --sql_dialect "${sql_dialect}" --llm_url "${llm_url}" \
    --dynamic_examples_few_shot "${dynamic_examples_few_shot}" --opensearch_url "${opensearch_url}" \
    --embedder_url "${embedder_url}" --embedder_model "${embedder_model}" --embedder_api_key "${embedder_api_key}" \
    --embedder_k_results "${embedder_k_results}" \
    --rerank_dynamic_few_shots "${rerank_dynamic_few_shots}" --reranker_url "${reranker_url}" \
    --reranker_model "${reranker_model}" --reranker_api_key "${reranker_api_key}" \
    --reranker_k_results "${reranker_k_results}" --reranker_score_threshold "${reranker_score_threshold}" \
    --dynamic_value_few_shots "${dynamic_value_few_shots}" --value_search_k_results "${value_search_k_results}" \
    --value_search_index_name "${value_search_index_name}"
done
