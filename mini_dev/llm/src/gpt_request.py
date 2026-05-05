import argparse
import concurrent.futures
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import APITimeoutError, OpenAI
from opensearch_search import search_similar_examples
from prompt import generate_combined_prompts_one, generate_value_verification_prompt
from reranker import rerank_search_results
from sql_parser import extract_column_value_pairs
from table_schema import generate_schema_prompt
from tqdm import tqdm
from value_search import search_similar_values_for_pairs


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def connect_gpt(engine, prompt, max_tokens, temperature, stop, client):
    """
    Function to connect to the GPT API and get the response.
    """
    MAX_API_RETRY = 10
    for _ in range(MAX_API_RETRY):
        try:

            if engine == "gpt-35-turbo-instruct":
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=120,
                )
                result = result.choices[0].text
            else:  # gpt-4-turbo, gpt-4, gpt-4-32k, gpt-35-turbo
                messages = [
                    {"role": "user", "content": prompt},
                ]
                result = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120,
                )
            break
        except APITimeoutError as e:
            result = f"error:{e}, {type(e)}"
            print(result)
            return result
        except Exception as e:
            result = f"error:{e}, {type(e)}"
            print(result)
            time.sleep(4)

    return result


def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = f"{db_root_path}{data['db_id']}/{data['db_id']}.sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(data["evidence"])

    return question_list, db_path_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Function to save the SQL results to a file.
    """
    sql_lst.sort(key=lambda x: x[1])
    result = {}
    for i, (sql, _) in enumerate(sql_lst):
        result[i] = sql

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)

    return result


def init_client(llm_api_key, llm_url):
    """
    Initialize the OpenAI client for a worker.
    """
    return OpenAI(
        api_key=llm_api_key,
        base_url=llm_url,
        timeout=120,
    )


def post_process_response(response, db_path):
    sql = response if isinstance(response, str) else response.choices[0].message.content
    sql = sql.replace("```sqlite", "").replace("```sql", "").replace("```", "").strip()
    db_id = db_path.split("/")[-1].split(".sqlite")[0]
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def worker_function(question_data):
    """
    Enhanced worker function with value verification
    """
    (
        prompt,
        llm_model,
        client,
        db_path,
        question,
        i,
        dynamic_value_few_shots,
        opensearch_url,
        value_search_index_name,
        embedder_url,
        embedder_model,
        embedder_api_key,
        value_search_k_results,
        sql_dialect,
        db_schema,
    ) = question_data
    response = connect_gpt(llm_model, prompt, 8192, 0, [], client)
    initial_sql = post_process_response(response, db_path)

    if dynamic_value_few_shots and db_schema:
        try:
            final_sql = correct_sql(
                initial_sql=initial_sql,
                question=question,
                db_path=db_path,
                db_schema=db_schema,
                llm_client=client,
                llm_model=llm_model,
                sql_dialect=sql_dialect,
                opensearch_url=opensearch_url,
                value_search_index_name=value_search_index_name,
                embedder_url=embedder_url,
                embedder_model=embedder_model,
                embedder_api_key=embedder_api_key,
                value_search_k_results=value_search_k_results,
            )
            return final_sql, i
        except Exception as e:
            print(f"Error in value verification for question {i}: {e}")
            return initial_sql, i
    else:
        return initial_sql, i


def correct_sql(
    initial_sql: str,
    question: str,
    db_path: str,
    db_schema: dict,
    llm_client: OpenAI,
    llm_model: str,
    sql_dialect: str,
    opensearch_url: str,
    value_search_index_name: str,
    embedder_url: str,
    embedder_model: str,
    embedder_api_key: str,
    value_search_k_results: int,
) -> str:
    """
    Verify and correct SQL using dynamic value few-shots with structured output
    """
    db_id = db_path.split("/")[-1].split(".sqlite")[0]

    try:
        column_value_pairs = extract_column_value_pairs(
            sql=initial_sql,
            db_schema=db_schema,
            llm_client=llm_client,
            llm_model=llm_model,
        )
    except Exception as e:
        print(f"Error extracting column-value pairs: {e}")
        return initial_sql

    if not column_value_pairs:
        return initial_sql

    for pair in column_value_pairs:
        pair["db_id"] = db_id

    try:
        similar_values = search_similar_values_for_pairs(
            column_value_pairs=column_value_pairs,
            opensearch_url=opensearch_url,
            index_name=value_search_index_name,
            embedder_url=embedder_url,
            embedder_model=embedder_model,
            embedder_api_key=embedder_api_key,
            k_results=value_search_k_results,
        )
    except Exception as e:
        print(f"Error searching for similar values: {e}")
        return initial_sql

    if not similar_values or not any(values for values in similar_values.values()):
        return initial_sql

    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    verification_prompt = generate_value_verification_prompt(
        original_sql=initial_sql,
        question=question,
        schema_prompt=schema_prompt,
        similar_values=similar_values,
    )

    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL verification expert. Review SQL queries and determine if they need correction based on similar database values. Check values in ALL filtration clauses (WHERE, HAVING, JOIN ON, CASE WHEN, IIF/IF, NULLIF, COALESCE, subqueries). Always return valid JSON.",
                    },
                    {"role": "user", "content": verification_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sql_verification_result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_be_corrected": {
                                    "type": "boolean",
                                    "description": "True if the SQL needs correction, False if it's already correct",
                                },
                                "corrected_sql": {
                                    "type": "string",
                                    "description": "The corrected SQL query (only filled when should_be_corrected is true, otherwise set empty string)",
                                },
                            },
                            "required": ["should_be_corrected", "corrected_sql"],
                        },
                    },
                },
                temperature=0,
                max_tokens=4096,
                timeout=120,
            )

            result = response.choices[0].message.content
            parsed = json.loads(result)

            should_be_corrected = parsed.get("should_be_corrected", False)
            corrected_sql = parsed.get("corrected_sql", "")

            if should_be_corrected and corrected_sql:
                print(f"SQL corrected for question: {question[:50]}...")
                print(f"- Was: {initial_sql}")
                print(f"- New: {corrected_sql}")
                return post_process_response(corrected_sql, db_path)
            else:
                print(f"SQL already correct, no correction needed for question: {question[:50]}...")
                return post_process_response(initial_sql, db_path)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed in SQL verification: {e}")
            if attempt == 2:
                print(f"Error in SQL verification: {e}")
                return post_process_response(initial_sql, db_path)


def construct_prompt_task(
    i,
    question_list,
    db_path_list,
    knowledge_list,
    sql_dialect,
    llm_model,
    client,
    dynamic_examples_few_shot,
    opensearch_url,
    embedder_url,
    embedder_model,
    embedder_api_key,
    embedder_k_results,
    rerank_dynamic_few_shots,
    reranker_url,
    reranker_model,
    reranker_api_key,
    reranker_k_results,
    reranker_score_threshold,
    dynamic_value_few_shots,
    value_search_index_name,
    value_search_k_results,
    db_schemas,
):
    few_shot_examples = None
    if dynamic_examples_few_shot:
        try:
            initial_results = search_similar_examples(
                question=question_list[i],
                k=embedder_k_results,
                opensearch_url=opensearch_url,
                embedder_url=embedder_url,
                embedder_model=embedder_model,
                embedder_api_key=embedder_api_key,
            )

            if rerank_dynamic_few_shots and initial_results:
                try:
                    few_shot_examples = rerank_search_results(
                        query=question_list[i],
                        search_results=initial_results,
                        k_ret=reranker_k_results,
                        score_threshold=reranker_score_threshold,
                        reranker_url=reranker_url,
                        reranker_model=reranker_model,
                        reranker_api_key=reranker_api_key,
                    )
                    if not few_shot_examples:
                        few_shot_examples = None
                except Exception as e:
                    print(f"Error reranking for question {i}: {e}")
                    few_shot_examples = (
                        initial_results[:reranker_k_results] if initial_results else None
                    )
            else:
                few_shot_examples = (
                    initial_results[:reranker_k_results] if initial_results else None
                )
        except Exception as e:
            print(f"Error searching for similar examples for question {i}: {e}")
            few_shot_examples = None

    return (
        generate_combined_prompts_one(
            db_path=db_path_list[i],
            question=question_list[i],
            sql_dialect=sql_dialect,
            knowledge=knowledge_list[i] if knowledge_list else None,
            few_shot_examples=few_shot_examples,
        ),
        llm_model,
        client,
        db_path_list[i],
        question_list[i],
        i,
        dynamic_value_few_shots,
        opensearch_url,
        value_search_index_name,
        embedder_url,
        embedder_model,
        embedder_api_key,
        value_search_k_results,
        sql_dialect,
        db_schemas.get(db_path_list[i].split("/")[-1].split(".sqlite")[0]) if db_schemas else None,
    )


def collect_response_from_gpt(
    db_path_list,
    question_list,
    llm_api_key,
    llm_model,
    llm_url,
    sql_dialect,
    num_threads=3,
    knowledge_list=None,
    dynamic_examples_few_shot=False,
    opensearch_url="http://localhost:9200",
    embedder_url="https://api.openai.com/v1",
    embedder_model="bge-m3",
    embedder_api_key="",
    embedder_k_results=5,
    rerank_dynamic_few_shots=False,
    reranker_url="https://api.openai.com/v1",
    reranker_model="bge-reranker-v2",
    reranker_api_key="",
    reranker_k_results=5,
    reranker_score_threshold=0.5,
    dynamic_value_few_shots=False,
    value_search_k_results=5,
    value_search_index_name="bird_dev_values",
    db_schemas=None,
):
    """
    Collect responses from GPT using multiple threads.
    """
    client = init_client(llm_api_key, llm_url)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(
                construct_prompt_task,
                i,
                question_list,
                db_path_list,
                knowledge_list,
                sql_dialect,
                llm_model,
                client,
                dynamic_examples_few_shot,
                opensearch_url,
                embedder_url,
                embedder_model,
                embedder_api_key,
                embedder_k_results,
                rerank_dynamic_few_shots,
                reranker_url,
                reranker_model,
                reranker_api_key,
                reranker_k_results,
                reranker_score_threshold,
                dynamic_value_few_shots,
                value_search_index_name,
                value_search_k_results,
                db_schemas,
            )
            for i in range(len(question_list))
        ]
        tasks = [
            future.result()
            for future in tqdm(futures, total=len(futures), desc="Constructing prompts")
        ]

    responses = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {
            executor.submit(worker_function, task): task for task in tasks
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks)):
            responses.append(future.result())

    return responses


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--test_path", type=str, default="")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, default="")
    args_parser.add_argument("--llm_api_key", type=str, required=True)
    args_parser.add_argument("--llm_url", type=str, required=True)
    args_parser.add_argument("--llm_model", type=str, required=True, default="code-davinci-002")
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("--num_processes", type=int, default=3)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--dynamic_examples_few_shot", type=str, default="False")
    args_parser.add_argument("--opensearch_url", type=str, default="http://localhost:9200")
    args_parser.add_argument("--embedder_url", type=str, default="https://api.openai.com/v1")
    args_parser.add_argument("--embedder_model", type=str, default="bge-m3")
    args_parser.add_argument("--embedder_api_key", type=str, default="")
    args_parser.add_argument("--embedder_k_results", type=int, default=5)
    args_parser.add_argument("--rerank_dynamic_few_shots", type=str, default="False")
    args_parser.add_argument("--reranker_url", type=str, default="https://api.openai.com/v1")
    args_parser.add_argument("--reranker_model", type=str, default="bge-reranker-v2")
    args_parser.add_argument("--reranker_api_key", type=str, default="")
    args_parser.add_argument("--reranker_k_results", type=int, default=5)
    args_parser.add_argument("--reranker_score_threshold", type=float, default=0.5)
    args_parser.add_argument("--dynamic_value_few_shots", type=str, default="False")
    args_parser.add_argument("--value_search_k_results", type=int, default=5)
    args_parser.add_argument("--value_search_index_name", type=str, default="bird_dev_values")
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path))

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    db_schemas = None
    if args.dynamic_value_few_shots == "True":
        try:
            with open(f"{args.db_root_path.rstrip('/')}/../dev_tables.json") as f:
                db_schemas_data = json.load(f)

            db_schemas = {}
            for schema in db_schemas_data:
                db_schemas[schema["db_id"]] = schema
        except Exception as e:
            print(f"Warning: Could not load database schemas: {e}")
            db_schemas = None

    if args.use_knowledge == "True":
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            args.llm_api_key,
            args.llm_model,
            args.llm_url,
            args.sql_dialect,
            args.num_processes,
            knowledge_list,
            args.dynamic_examples_few_shot == "True",
            args.opensearch_url,
            args.embedder_url,
            args.embedder_model,
            args.embedder_api_key,
            args.embedder_k_results,
            args.rerank_dynamic_few_shots == "True",
            args.reranker_url,
            args.reranker_model,
            args.reranker_api_key,
            args.reranker_k_results,
            args.reranker_score_threshold,
            args.dynamic_value_few_shots == "True",
            args.value_search_k_results,
            args.value_search_index_name,
            db_schemas,
        )
    else:
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            args.llm_api_key,
            args.llm_model,
            args.llm_url,
            args.sql_dialect,
            args.num_processes,
            None,
            args.dynamic_examples_few_shot == "True",
            args.opensearch_url,
            args.embedder_url,
            args.embedder_model,
            args.embedder_api_key,
            args.embedder_k_results,
            args.rerank_dynamic_few_shots == "True",
            args.reranker_url,
            args.reranker_model,
            args.reranker_api_key,
            args.reranker_k_results,
            args.reranker_score_threshold,
            args.dynamic_value_few_shots == "True",
            args.value_search_k_results,
            args.value_search_index_name,
            db_schemas,
        )

    flags = []
    if args.dynamic_examples_few_shot == "True":
        flags.append("dyn-examples")
    if args.rerank_dynamic_few_shots == "True":
        flags.append("rerank-examples")
    if args.dynamic_value_few_shots == "True":
        flags.append("dyn-values")

    flags_str = f"_{'_'.join(flags)}" if flags else ""
    cot_str = "_cot" if args.chain_of_thought == "True" else ""

    output_name = (
        args.data_output_path
        + "predict_"
        + args.mode
        + "_"
        + args.llm_model
        + cot_str
        + flags_str
        + "_"
        + args.sql_dialect
        + ".json"
    )
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}; Use dynamic few-shot: {}; Use reranking: {}; Use dynamic value few-shots: {}".format(
            args.llm_model,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
            args.dynamic_examples_few_shot,
            args.rerank_dynamic_few_shots,
            args.dynamic_value_few_shots,
        )
    )
