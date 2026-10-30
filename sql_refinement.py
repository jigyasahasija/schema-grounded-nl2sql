"""Schema-grounded SQL repair without database access.

Ideas adapted, with independently written code/prompts, from:
  DIN-SQL, Pourreza and Rafiei (2023), section 4.4: generation then correction.
  https://arxiv.org/abs/2304.11015
  DAIL-SQL, Gao et al. (2023/2024), section 3.3: select examples using
  both question and draft-query similarity. Here similarity is lexical/structural,
  not the paper's embedding method. https://arxiv.org/abs/2308.15363

Checks are static approximations, not execution or semantic correctness proofs.
"""

from dataclasses import dataclass
import re

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope


@dataclass
class SQLCheck:
    sql: str
    issues: list
    fixes: list


def _key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def schema_tables(schema):
    names = schema["table_names_original"]
    tables = {name.lower(): {} for name in names}
    for owner, column in schema["column_names_original"]:
        if owner >= 0:
            tables[names[owner].lower()][column.lower()] = column
    return tables


def _available_sources(scope):
    """Resolve local and correlated aliases, with inner aliases taking priority."""
    sources = {}
    current = scope
    while current is not None:
        for alias, source in current.sources.items():
            sources.setdefault(alias.lower(), source)
        current = current.parent
    return sources


def _parse(sql):
    statements = [item for item in sqlglot.parse(sql, read="sqlite") if item]
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Return exactly one SELECT/WITH query")
    return statements[0]


def _issues(tree, tables):
    issues = []
    for scope in traverse_scope(tree):
        sources = _available_sources(scope)
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table) and source.name.lower() not in tables:
                issues.append(f"Unknown table {source.name!r} for alias {alias!r}")
        for column in scope.columns:
            # SQLGlot also propagates some child-scope columns upward. Check
            # each reference in its own SELECT, where aliases can be resolved.
            if column.find_ancestor(exp.Select) is not scope.expression:
                continue
            if column.is_star:
                continue
            if column.table:
                source = sources.get(column.table.lower())
                if source is None:
                    issues.append(f"Undefined alias {column.table!r} in {column.sql()}")
                elif isinstance(source, exp.Table):
                    allowed = tables.get(source.name.lower(), {})
                    if column.name.lower() not in allowed:
                        owners = [name for name, cols in tables.items()
                                  if column.name.lower() in cols]
                        issues.append(
                            f"{column.sql()} is not a column of {source.name}; "
                            f"tables containing that exact name: {owners}"
                        )
            else:
                # Derived-table/CTE output columns and SELECT aliases need different
                # name resolution. Leave them alone rather than guess.
                aliases = {item.alias.lower() for item in scope.expression.expressions
                           if isinstance(item, exp.Alias)}
                if column.name.lower() in aliases or any(
                    isinstance(source, Scope) for source in sources.values()
                ):
                    continue
                if not any(
                    column.name.lower() in tables.get(source.name.lower(), {})
                    for source in sources.values() if isinstance(source, exp.Table)
                ):
                    issues.append(f"Unknown unqualified column {column.name!r}")
    for function in tree.find_all(exp.Func):
        if isinstance(function, exp.Anonymous):
            name = function.name.upper()
        else:
            name = function.sql_name().upper()
        if name in {"SUBTRACT", "MULTIPLY", "DIVIDE", "YEAR", "CURDATE",
                    "TIMESTAMPDIFF", "DATEDIFF", "DATE_FORMAT"}:
            issues.append(f"{name} is not a native SQLite function; translate it")
    return list(dict.fromkeys(issues))


def check_sql(sql, schema):
    """Repair unambiguous identifier spelling and elementary dialect mistakes.

    Only unknown names with a unique punctuation-insensitive match on the
    assigned table are renamed. No tables, predicates, aggregation, or result
    columns are added, removed, or deduplicated.
    """
    tables = schema_tables(schema)
    try:
        tree = _parse(sql)
    except (sqlglot.errors.SqlglotError, ValueError) as error:
        return SQLCheck(sql, [f"SQL parse error: {str(error)[:500]}"], [])

    fixes = []
    for scope in traverse_scope(tree):
        sources = _available_sources(scope)
        for column in scope.columns:
            if column.find_ancestor(exp.Select) is not scope.expression:
                continue
            if column.is_star or not column.table:
                continue
            source = sources.get(column.table.lower())
            if not isinstance(source, exp.Table):
                continue
            allowed = tables.get(source.name.lower(), {})
            if column.name.lower() in allowed:
                continue
            matches = [name for name in allowed.values()
                       if _key(name) == _key(column.name)]
            if len(matches) == 1:
                fixes.append(f"Exact identifier: {column.name} -> {matches[0]}")
                column.set("this", exp.to_identifier(matches[0], quoted=True))

    def translate(node):
        if isinstance(node, exp.Year):
            fixes.append("YEAR -> SQLite strftime")
            return exp.Cast(
                this=exp.Anonymous(this="strftime", expressions=[
                    exp.Literal.string("%Y"), node.this.copy()
                ]), to=exp.DataType.build("INTEGER"),
            )
        if isinstance(node, exp.Anonymous) and len(node.expressions) == 2:
            name = node.name.upper()
            left, right = [item.copy() for item in node.expressions]
            if name == "SUBTRACT":
                fixes.append("SUBTRACT -> subtraction")
                return exp.Paren(this=exp.Sub(this=left, expression=right))
            if name == "MULTIPLY":
                fixes.append("MULTIPLY -> multiplication")
                return exp.Paren(this=exp.Mul(this=left, expression=right))
            if name == "DIVIDE":
                fixes.append("DIVIDE -> real-number division")
                return exp.Paren(this=exp.Div(
                    this=exp.Cast(this=left, to=exp.DataType.build("REAL")),
                    expression=right,
                ))
        return node

    # Nested YEAR calls may be children of an arithmetic replacement, so make
    # a second pass. Once translated, these nodes do not match again.
    tree = tree.transform(translate)
    tree = tree.transform(translate)
    try:
        output = tree.sql(dialect="sqlite") if fixes else sql
        issues = _issues(_parse(output), tables)
    except (sqlglot.errors.SqlglotError, ValueError) as error:
        return SQLCheck(sql, [f"SQL analysis error: {str(error)[:500]}"], [])
    return SQLCheck(output, issues, fixes)


def _features(sql):
    """Approximate SQL structure without needing an embedding model."""
    try:
        tree = _parse(sql)
    except (sqlglot.errors.SqlglotError, ValueError):
        return set(), set()
    structure = {type(node).__name__ for node in tree.walk()
                 if isinstance(node, (exp.Func, exp.Join, exp.Group, exp.Order,
                                      exp.Limit, exp.Distinct, exp.Subquery,
                                      exp.Union, exp.Intersect, exp.Except,
                                      exp.Between, exp.Case))}
    return structure, {table.name.lower() for table in tree.find_all(exp.Table)}


def repair_examples(question, evidence, draft, examples, db_id, limit=3):
    """DAIL-inspired ranking by question words and draft SQL structure."""
    stop = {"the", "a", "an", "of", "to", "in", "for", "and", "is", "are",
            "what", "which", "please", "refers", "with", "their", "from"}

    def words(text):
        # Mask constants so a shared patient/molecule number does not dominate.
        text = re.sub(r"\b\w*\d\w*\b", " value ", text.lower())
        return set(re.findall(r"[a-z_]+", text)) - stop

    def overlap(left, right):
        return len(left & right) / max(1, len(left | right))

    target = words(question + " " + (evidence or ""))
    structure, tables = _features(draft)
    ranked = []
    for index, example in enumerate(examples):
        if example.get("db_id", db_id) != db_id or not example.get("SQL"):
            continue
        example_structure, example_tables = _features(example["SQL"])
        score = (
            0.6 * overlap(target, words(example.get("question", "") + " "
                                       + (example.get("evidence") or "")))
            + 0.25 * overlap(structure, example_structure)
            + 0.15 * overlap(tables, example_tables)
        )
        ranked.append((score, index, example))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [example for _, _, example in ranked[:limit]]


def _schema_text(schema):
    lines = []
    names = schema["table_names_original"]
    for owner, name in enumerate(names):
        columns = [f'"{column.replace(chr(34), chr(34) * 2)}"'
                   for table, column in schema["column_names_original"] if table == owner]
        lines.append(f'{name}: ' + ", ".join(columns))
    columns = schema["column_names_original"]
    for source, target in schema["foreign_keys"]:
        st, sc = columns[source]
        tt, tc = columns[target]
        lines.append(f'Join: {names[st]}."{sc}" = {names[tt]}."{tc}"')
    return "\n".join(lines)


def refine_sql(draft, question, evidence, schema, train_examples, db_id,
               call_model, clean_sql, max_repairs=2):
    """At most two repair calls; structurally clean drafts bypass the LLM reviewer."""
    best = check_sql(draft, schema)
    if not best.issues:
        return best.sql
    for _ in range(max_repairs):
        examples = repair_examples(question, evidence, best.sql, train_examples, db_id)
        example_text = "\n\n".join(
            f"Question: {item['question']}\nEvidence: {item.get('evidence') or ''}\n"
            f"SQL: {item['SQL']}" for item in examples
        )
        prompt = f"""Repair one SQLite query using the supplied schema and question.
The static checker found the problems below. It does not execute SQL.
Fix the diagnosed problems with the smallest necessary changes. Preserve valid
filters, aggregation, rounding, and output columns unless the question requires
a correction. Use the exact column names and their owning tables. Quote names
with spaces/hyphens. Formula hints must become SQLite arithmetic or strftime.
If a needed field belongs to a missing table, join that table using the listed keys.

Schema:
{_schema_text(schema)}

Relevant training examples (their values are not the current question's values):
{example_text}

Question: {question}
Evidence: {evidence or ''}
Draft SQL:
{best.sql}

Static diagnostics:
{chr(10).join('- ' + issue for issue in best.issues)}

Return ONLY the corrected SQL query. No explanation or markdown."""
        try:
            proposed = clean_sql(call_model(prompt, max_tokens=1024))
            candidate = check_sql(proposed, schema)
        except Exception:
            # A failed optional review must not lose an already generated draft.
            break
        if not candidate.issues:
            return candidate.sql
        if len(candidate.issues) < len(best.issues):
            best = candidate
        else:
            # Repeating an identical prompt at temperature zero would just
            # retrieve the same failed repair from the cache.
            break
    return best.sql
