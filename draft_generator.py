"""Database-specific draft generation followed by bounded schema repair."""

import json
import re
from typing import Dict, List

from llm_utils import call_llm
from sql_refinement import refine_sql


# ─── Your Implementation ───────────────────────────────────────────────────


def select_examples(question, evidence, train_examples, db_id, limit=3):
    """Rank supplied training examples by question/evidence word overlap."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "give", "in", "is", "it", "list", "me", "of", "on", "please",
        "refers", "show", "state", "that", "the", "their", "them", "these",
        "this", "to", "was", "were", "what", "which", "with",
    }

    def words(text):
        return set(re.findall(r"[a-z0-9_]+", text.lower())) - stop_words

    query_words = words(question + " " + (evidence or ""))
    ranked = []
    for index, example in enumerate(train_examples):
        if example.get("db_id", db_id) != db_id or not example.get("SQL"):
            continue
        example_words = words(
            example.get("question", "") + " " + (example.get("evidence") or "")
        )
        # Length normalization prevents long evidence strings dominating retrieval.
        denominator = (len(query_words) * len(example_words)) ** 0.5
        score = len(query_words & example_words) / denominator if denominator else 0
        ranked.append((score, index, example))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [example for _, _, example in ranked[:limit]]


def _generate_draft(
    question: str,
    evidence: str,
    schema: dict,
    train_examples: List[Dict],
    db_id: str,
) -> str:
    """
    Generate a SQL query from a natural language question.

    Args:
        question:       The natural language question.
        evidence:       Domain hints (e.g., "double type bond refers to
                        bond_type = '=';").
                        May be empty.
        schema:         Database schema dict (from schema.json) containing:
                        table_names_original, column_names_original,
                        column_types, primary_keys, foreign_keys,
                        column_examples.
                        How you format this for the LLM matters a lot.
        train_examples: A list of example dicts with keys: "question_id",
                        "db_id", "question", "evidence", "SQL",
                        "difficulty".
        db_id:          Database identifier.

    Returns:
        A valid SQLite query string.
    """

    # SQLite and output guidance for the chemical schema.
    sql_rules = """SQL correctness checklist:
- Use SQLite. Copy exact table and column names from the schema, and double-quote
  identifiers, including spaces and hyphens. Never replace spaces with underscores.
- Every alias.column must belong to the table assigned that alias. Join the table
  that owns a required column; do not invent columns or add unnecessary joins.
- Translate evidence formulas into SQLite: SUBTRACT(a,b) means (a-b),
  MULTIPLY(a,b) means (a*b), and DIVIDE(a,b) needs real-number division.
  For a year use CAST(strftime('%Y', date_column) AS INTEGER); for the current year
  use CAST(strftime('%Y', 'now') AS INTEGER). Do not use YEAR(), CURDATE(),
  TIMESTAMPDIFF(), or SUBTRACT() as SQLite functions.
- Return every requested quantity in the SELECT list. A column used for sorting
  is not automatically the requested output. Put separate requested counts in
  separate columns of one row, not in UNION rows.
- Count the entity asked for. When joins repeat an entity that should be counted
  once, count its DISTINCT identifier. Do not add DISTINCT to every calculation.
- For a percentage, apply the stated population filters to the denominator and
  count the matching subset in the numerator. Use 100.0 for real division.
  Only use ROUND when the current question explicitly requests rounding.
- Use the current question's evidence and filters; examples illustrate patterns
  but their constants, output columns, and rounding requirements do not carry over.
- Before returning SQL, check aliases, column ownership, output columns, joins,
  and filters. Return only the final SQL, without a written explanation."""
    # Keep the best measured prompt configuration for each database.
    if db_id != "toxicology":
        sql_rules = ""
    tables = schema.get("table_names_original", [])
    columns = schema.get("column_names_original", [])
    column_examples = schema.get("column_examples", [])
    def quote(name):
        return '"'  + name.replace('"', '""') + '"'

    schema_lines = []

    for table_index, table_name in enumerate(tables):
        schema_lines.append(f"Table: {quote(table_name)}")
        for column_index, (owner, column_name) in enumerate(columns):
            if owner == table_index:
                column_type = schema["column_types"][column_index]
                line = f"  Column: {quote(column_name)} ({column_type})"
                if db_id == "toxicology" and column_index < len(column_examples):
                    samples = column_examples[column_index][:3]
                    if samples:
                        line += f", sample values: {json.dumps(samples, ensure_ascii=False)}"
                schema_lines.append(line)

    schema_lines.append("Join relationships:")
    for source, target in schema["foreign_keys"]:
        source_table, source_column = columns[source]
        target_table, target_column = columns[target]
        schema_lines.append(
            f"  {quote(tables[source_table])}.{quote(source_column)} = "
            f"{quote(tables[target_table])}.{quote(target_column)}"
        )
    schema_text = "\n".join(schema_lines)

    examples_text = ""
    if db_id == "toxicology":
        selected = select_examples(question, evidence, train_examples, db_id)
        if selected:
            blocks = [
                f"Example {index}:\nQuestion: {example['question']}\n"
                f"Evidence: {example.get('evidence') or ''}\nSQL: {example['SQL']}"
                for index, example in enumerate(selected, start=1)
            ]
            examples_text = (
                "\n\nSQLite training examples:\n" + "\n\n".join(blocks)
                + "\n\nNow answer the following question using its own evidence."
            )

    prompt = f"""Given the following database schema and natural language query,
generate a SQL query that answers the question.

Schema:
{schema_text}{examples_text}

Question: {question}

Evidence: {evidence}
{sql_rules}
IMPORTANT: Return ONLY the SQL query. No explanations, no markdown."""

    sql_query = call_llm(prompt, max_tokens=512)
    sql_query = clean_sql(sql_query)
    return sql_query


def generate_sql(question, evidence, schema, train_examples, db_id):
    """Generate a draft, then apply schema-grounded bounded correction."""
    draft = _generate_draft(question, evidence, schema, train_examples, db_id)
    return refine_sql(
        draft, question, evidence, schema, train_examples, db_id,
        call_llm, clean_sql,
    )


# ─── Output parsing ─────────────────────────────────────────────────────────
# Turns the model's reply into one SQL statement. Adapt it if your prompts change the
# output format (e.g. reasoning before the SQL, or a "SQL:" prefix).
# If a helper module needs clean_sql, please move clean_sql into that helper and import it
# here: importing from starter in a helper that starter imports is a circular import.


def clean_sql(text):
    """Extract one SQL statement from LLM output: drop markdown fences and the prose around the SQL."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    # Last fenced block that actually contains SQL (models sometimes emit trailing empty fences).
    fences = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql_fences = [f for f in fences if re.search(r"(?i)\b(select|with)\b", f)]
    if sql_fences:
        text = sql_fences[-1]
    text = re.sub(r"```(?:sql|sqlite)?", "", text, flags=re.IGNORECASE).strip()
    if not re.match(r"(?is)^\s*(with|select)\b", text):
        m = re.search(r"(?is)\b(with|select)\b.*", text)
        if m:
            text = m.group(0)
    # Keep only the first statement; ignore semicolons inside quoted literals/identifiers.
    m = re.match(r"""(?s)((?:'[^']*'|"[^"]*"|`[^`]*`|[^;'"`])*)""", text)
    return (m.group(1) if m else text).strip()
