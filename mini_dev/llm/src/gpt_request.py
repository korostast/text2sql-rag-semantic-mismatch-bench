import argparse
import concurrent.futures
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from opensearch_search import search_similar_examples
from prompt import generate_combined_prompts_one
from reranker import rerank_search_results
from tqdm import tqdm


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def connect_gpt(engine, prompt, max_tokens, temperature, stop, client):
    """
    Function to connect to the GPT API and get the response.
    """
    MAX_API_RETRY = 10
    for i in range(MAX_API_RETRY):
        time.sleep(2)
        try:

            if engine == "gpt-35-turbo-instruct":
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
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
                    stop=stop,
                )
            break
        except Exception as e:
            result = f"error:{e}"
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
    )


def post_process_response(response, db_path):
    sql = response if isinstance(response, str) else response.choices[0].message.content
    db_id = db_path.split("/")[-1].split(".sqlite")[0]
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def worker_function(question_data):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the GPT response.
    """
    prompt, llm_model, client, db_path, question, i = question_data
    response = connect_gpt(llm_model, prompt, 4096, 0, [], client)
    sql = post_process_response(response, db_path)
    return sql, i


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
):
    """
    Collect responses from GPT using multiple threads.
    """
    client = init_client(llm_api_key, llm_url)

    tasks = []
    for i in tqdm(range(len(question_list)), desc="Constructing prompt for all the questions"):
        few_shot_examples = None
        if dynamic_examples_few_shot:
            try:
                # Retrieve initial results using embedder
                initial_results = search_similar_examples(
                    question=question_list[i],
                    k=embedder_k_results,
                    opensearch_url=opensearch_url,
                    embedder_url=embedder_url,
                    embedder_model=embedder_model,
                    embedder_api_key=embedder_api_key,
                )

                # Rerank if enabled
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
                    # Use top results from initial search
                    few_shot_examples = (
                        initial_results[:reranker_k_results] if initial_results else None
                    )
            except Exception as e:
                print(f"Error searching for similar examples for question {i}: {e}")
                few_shot_examples = None

        tasks.append(
            (
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
            )
        )

    responses = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {executor.submit(worker_function, task): task for task in tasks}
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
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path))

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

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
        )

    if args.chain_of_thought == "True":
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.llm_model
            + "_cot"
            + "_"
            + args.sql_dialect
            + ".json"
        )
    else:
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.llm_model
            + "_"
            + args.sql_dialect
            + ".json"
        )
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}; Use dynamic few-shot: {}; Use reranking: {}".format(
            args.llm_model,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
            args.dynamic_examples_few_shot,
            args.rerank_dynamic_few_shots,
        )
    )
