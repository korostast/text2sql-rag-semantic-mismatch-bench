from table_schema import generate_schema_prompt


def generate_comment_prompt(question, sql_dialect, knowledge=None):
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding External Knowledge" if knowledge else ""
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt(sql_dialect):
    return f"\nGenerate the {sql_dialect} for the above question after thinking step by step."


def generate_instruction_prompt(sql_dialect):
    return f"""
        \nIn your response, you do not need to mention your intermediate steps.
        Do not include any comments in your response.
        Do not need to start with the symbol ``` and ```sqlite
        You only need to return the result {sql_dialect} SQL code
        start from SELECT
        """


def generate_few_shot_examples(few_shot_examples: list[dict]) -> str:
    """
    Generate few-shot examples section for the prompt
    """
    if not few_shot_examples:
        return ""

    examples_text = "-- Here are some similar examples to help you understand the task:\n\n"

    for i, example in enumerate(few_shot_examples, 1):
        examples_text += f"-- Example {i}:\n"
        examples_text += f"-- Question: {example.get('question', '')}\n"

        evidence = example.get("evidence", "")
        if evidence:
            examples_text += f"-- Evidence: {evidence}\n"

        examples_text += f"-- SQL: {example.get('sql', '')}\n\n"

    return examples_text


def generate_combined_prompts_one(
    db_path: str,
    question: str,
    sql_dialect: str,
    knowledge: str | None = None,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """
    Generate combined prompt with optional few-shot examples
    """
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    few_shot_prompt = generate_few_shot_examples(few_shot_examples)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    instruction_prompt = generate_instruction_prompt(sql_dialect)

    if few_shot_prompt:
        combined_prompts = "\n\n".join(
            [schema_prompt, few_shot_prompt, comment_prompt, instruction_prompt]
        )
    else:
        combined_prompts = f"{schema_prompt}\n\n{comment_prompt}\n\n{instruction_prompt}"
    return combined_prompts


def generate_value_verification_prompt(
    original_sql: str,
    question: str,
    schema_prompt: str,
    similar_values: dict[str, list[dict]],
) -> str:
    """
    Generate a prompt for LLM to verify and correct SQL based on similar values
    """
    similar_values_section = ""
    if similar_values:
        similar_values_section = "-- Similar Values Found in Database\n"
        similar_values_section += "For each column-value pair in ALL filtration clauses (WHERE, HAVING, JOIN ON, CASE WHEN, IIF/IF, NULLIF, COALESCE, subqueries), here are the top 5 most similar values:\n\n"

        for key, values in similar_values.items():
            if values:
                similar_values_section += f"Column-Value Pair: {key}\n"
                similar_values_section += "Similar values:\n"
                for i, val_info in enumerate(values[:5], 1):
                    val = val_info.get("value", "")
                    score = val_info.get("score", 0.0)
                    similar_values_section += f"  {i}. '{val}' (similarity: {score:.3f})\n"
                similar_values_section += "\n"

    verification_prompt = f"""{schema_prompt}

-- Original Question
{question}

-- Initial SQL Query
{original_sql}

{similar_values_section}
-- Task
Review the initial SQL query and the similar values above.
Determine if any values in ANY filtration clause (WHERE, HAVING, JOIN ON, CASE WHEN, IIF/IF, NULLIF, COALESCE, subqueries) need correction or removal.

Return your response as a JSON object with the following structure:
{{
    "should_be_corrected": true/false,
    "corrected_sql": "SELECT ... (only if should_be_corrected is true)"
}}

Rules:
- Set "should_be_corrected" to true only if you find values that need correction or consider some filtration clauses are not necessary
- Set "should_be_corrected" to false if the initial SQL is already correct
- Only fill "corrected_sql" when "should_be_corrected" is true
- Leave "corrected_sql" as empty string or null when "should_be_corrected" is false
- Return ONLY the JSON object, no additional text or comments
- Do not change filtration operators
"""
    return verification_prompt
