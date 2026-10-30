"""Conservative output corrections with no additional model calls.

Experimental heuristics, not a paper replication or a correctness guarantee.
"""
import re
import sqlglot
from sqlglot import exp
from sql_refinement import check_sql


def correct_output(sql, question, evidence, schema):
    source = question + ' ' + (evidence or '')
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
        outputs = list(tree.selects)
        if len(outputs) != 1:
            return sql
        output = outputs[0]
        value = output.this if isinstance(output, exp.Alias) else output
        changed = False
        # Only direct entity-column counts; never COUNT(*), conditional counts,
        # expressions, subqueries or multiple measures.
        if (isinstance(value, exp.Count) and isinstance(value.this, exp.Column)
                and not value.this.is_star
                and re.search(r'\b(?:unique|distinct)\b', source, re.I)
                and not re.search(r'\b(?:not|non)[ -]+(?:unique|distinct)\b', source, re.I)):
            value.set('this', exp.Distinct(expressions=[value.this.copy()]))
            changed = True
        # Restrict removal to the outermost rounding of a percentage, with no
        # mention of rounding/precision/approximation anywhere in the input.
        if (isinstance(value, exp.Round)
                and re.search(r'\b(?:percent|percentage)\b|%', question, re.I)
                and not re.search(r'round|decimal|precision|nearest|approximat|significant', source, re.I)):
            value.replace(value.this.copy())
            changed = True
        if not changed:
            return sql
        checked = check_sql(tree.sql(dialect='sqlite'), schema)
        return sql if checked.issues else checked.sql
    except Exception:
        return sql


def generate_sql(question, evidence, schema, train_examples, db_id):
    from draft_generator import generate_sql as generate_draft
    baseline = generate_draft(question, evidence, schema, train_examples, db_id)
    return correct_output(baseline, question, evidence, schema)
