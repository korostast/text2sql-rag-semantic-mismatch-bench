from openai import OpenAI


def extract_column_value_pairs(
    sql: str,
    db_schema: dict,
    llm_client: OpenAI,
    llm_model: str,
) -> list[dict]:
    """
    Extract column-value pairs from SQL query using structured output
    """
    schema_context = _build_schema_context(db_schema)
    prompt = f"""You are a SQL parser. Extract all column-value pairs from the following SQL query.

Database Schema:
{schema_context}

SQL Query:
{sql}

Extract all column-value pairs from ALL filtration contexts including:
- WHERE clauses
- HAVING clauses
- JOIN ON conditions
- CASE WHEN expressions
- IIF/IF functions (SQLite: IIF, MySQL: IF)
- NULLIF functions
- COALESCE functions
- Subquery conditions (IN, =, EXISTS, etc.)
- AND/OR conditions within any clause

For each pair, identify:
- table_name: The table containing the column
- column_name: The column name
- value: The value being compared (as a string, without quotes)

Return only the JSON object with the extracted pairs."""

    for attempt in range(2):
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL parser that extracts column-value pairs from all filtration clauses (WHERE, HAVING, JOIN ON, CASE WHEN, IIF/IF, NULLIF, COALESCE, subqueries). Always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "column_value_pairs",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "column_value_pairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "table_name": {"type": "string"},
                                            "column_name": {"type": "string"},
                                            "value": {"type": "string"},
                                        },
                                        "required": ["table_name", "column_name", "value"],
                                    },
                                },
                            },
                        },
                        "required": ["column_value_pairs"],
                    },
                },
                temperature=0,
                max_tokens=8192,
                timeout=180,
            )

            result = response.choices[0].message.content
            parsed = eval(result)  # Parse JSON string to dict

            return parsed.get("column_value_pairs", [])
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to extract column-value pairs: {e}")
            if attempt == 1:
                return []


def _build_schema_context(db_schema: dict) -> str:
    """
    Build a human-readable schema context string
    """
    context_lines = []

    table_names = db_schema.get("table_names", [])
    column_names = db_schema.get("column_names", [])
    column_types = db_schema.get("column_types", [])

    for table_idx, table_name in enumerate(table_names):
        context_lines.append(f"\nTable: {table_name}")
        context_lines.append("  Columns:")
        for col_idx, col_info in enumerate(column_names):
            if col_info[0] == table_idx:
                col_name = col_info[1]
                col_type = column_types[col_idx] if col_idx < len(column_types) else "unknown"
                context_lines.append(f"    - {col_name} ({col_type})")

    return "\n".join(context_lines)
