from table_schema import generate_schema_prompt
from typing import List, Dict, Optional


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
    return f"\nGenerate the {sql_dialect} for the above question after thinking step by step: "


def generate_instruction_prompt(sql_dialect):
    return f"""
        \nIn your response, you do not need to mention your intermediate steps. 
        Do not include any comments in your response.
        Do not need to start with the symbol ```
        You only need to return the result {sql_dialect} SQL code
        start from SELECT
        """


def generate_few_shot_examples(few_shot_examples: List[Dict]) -> str:
    """
    Generate few-shot examples section for the prompt
    """
    if not few_shot_examples:
        return ""
    
    examples_text = "-- Here are some similar examples to help you understand the task:\n\n"
    
    for i, example in enumerate(few_shot_examples, 1):
        examples_text += f"-- Example {i}:\n"
        examples_text += f"-- Question: {example.get('question', '')}\n"
        
        evidence = example.get('evidence', '')
        if evidence:
            examples_text += f"-- Evidence: {evidence}\n"
        
        examples_text += f"-- SQL: {example.get('sql', '')}\n\n"
    
    return examples_text


def generate_combined_prompts_one(
    db_path: str,
    question: str,
    sql_dialect: str,
    knowledge: Optional[str] = None,
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """
    Generate combined prompt with optional few-shot examples
    """
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    few_shot_prompt = generate_few_shot_examples(few_shot_examples)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt(sql_dialect)
    instruction_prompt = generate_instruction_prompt(sql_dialect)

    if few_shot_prompt:
        combined_prompts = "\n\n".join(
            [schema_prompt, few_shot_prompt, comment_prompt, cot_prompt, instruction_prompt]
        )
    else:
        combined_prompts = "\n\n".join(
            [schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
        )
    return combined_prompts
