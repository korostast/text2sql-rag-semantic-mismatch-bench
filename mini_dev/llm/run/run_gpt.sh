#!/bin/bash
eval_path='../data/mini_dev_sqlite.json' # _sqlite.json, _mysql.json, _postgresql.json
db_root_path='../data/dev_databases/'
use_knowledge='True'
mode='mini_dev' # dev, train, mini_dev
cot='False'

YOUR_API_KEY='<apikey>'
BASE_URL='https://integrate.api.nvidia.com/v1'

# Choose the engine to run, e.g. gpt-4, gpt-4-32k, gpt-4-turbo, gpt-35-turbo, GPT35-turbo-instruct
engine='openai/gpt-oss-120b'

# Choose the number of threads to run in parallel, 1 for single thread
num_threads=3

# Choose the SQL dialect to run, e.g. SQLite, MySQL, PostgreSQL
# PLEASE NOTE: You have to setup the database information in table_schema.py
# if you want to run the evaluation script using MySQL or PostgreSQL
sql_dialect='SQLite'

# Choose the output path for the generated SQL queries
data_kg_output_path='./exp_result/turbo_output_kg/'

# Dynamic few-shot examples configuration
dynamic_examples_few_shot='True'
opensearch_url='http://localhost:9200'
embedder_url='https://integrate.api.nvidia.com/v1'
embedder_model='bge-m3'
embedder_api_key='<apikey>'

# Index training data if dynamic examples are enabled
if [ "$dynamic_examples_few_shot" = "True" ]; then
    echo "Indexing training data to OpenSearch..."
    python3 -u ./src/index_training_data.py \
        --train_path ../../data/train/train_sqlite.json \
        --opensearch_url ${opensearch_url} \
        --embedder_url ${embedder_url} \
        --embedder_model ${embedder_model} \
        --embedder_api_key ${embedder_api_key}
fi

echo "generate $engine batch, run in $num_threads threads, with knowledge: $use_knowledge, with chain of thought: $cot, with dynamic few-shot: $dynamic_examples_few_shot"
python3 -u ./src/gpt_request.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine} --eval_path ${eval_path} --data_output_path ${data_kg_output_path} --use_knowledge ${use_knowledge} \
--chain_of_thought ${cot} --num_process ${num_threads} --sql_dialect ${sql_dialect} --base_url ${BASE_URL} \
--dynamic_examples_few_shot ${dynamic_examples_few_shot} --opensearch_url ${opensearch_url} \
--embedder_url ${embedder_url} --embedder_model ${embedder_model} --embedder_api_key ${embedder_api_key}
