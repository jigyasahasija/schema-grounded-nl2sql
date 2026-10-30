# Schema-Grounded NL2SQL

Translate natural-language questions into SQLite queries using a locally served language model, schema-aware checks, and selective repair. Developed as a Georgia Tech CS 6400 database systems project.

The final approach combines database-specific prompting with bounded correction: generate a query, diagnose selected structural problems, and request a repair only when needed. Two narrow output rules address explicit unique counts and unrequested percentage rounding.

**Repository visibility:** Keep this repository private. The course assignment prohibits sharing code or prompts; obtain course-staff permission before making it public. Course datasets, reference SQL, reports, logs, caches, and submission archives are intentionally excluded.

## How it works

1. **Represent the schema.** Present tables, exact column names, types, and foreign-key relationships. The Toxicology configuration adds up to three sample values per column, three training examples selected by normalized word overlap, and a SQLite checklist. The Thrombosis configuration retains a simpler initial prompt.
2. **Generate a draft.** Call an OpenAI-compatible model server at temperature zero, with a 512-token output budget.
3. **Check and repair.** SQLGlot analyzes selected alias, table, and column issues. Deterministic transforms repair uniquely identifiable names and supported arithmetic/year expressions. Remaining issues trigger at most two 1,024-token repair calls. Examples for repair are ranked by masked word overlap (60%), SQL structure (25%), and table overlap (15%). Failed optional repairs preserve the best available query.
4. **Apply targeted corrections.** For a single direct-column count, an explicit uniqueness cue can add `DISTINCT`. For a single percentage output, outer `ROUND` is removed only when the input does not mention rounding, precision, or approximation.

Generation never executes SQL against a task database. The client does use a local SQLite **response cache**, which is separate from task data. Static checks and heuristics do not guarantee correct SQL or correct interpretation of a question.

## Validation results

Measured with `gemma4:e4b` on PACE ICE using the course evaluator's SQLite result-set comparison:

| Approach | Toxicology (40 questions) | Thrombosis Prediction (39 questions) |
|---|---:|---:|
| Provided starter | 12/40 (30.0%) | 9/39 (23.1%) |
| Schema-aware selective repair | 27/40 (67.5%) | 28/39 (71.8%) |
| Final targeted corrections | **28/40 (70.0%)** | **29/39 (74.4%)** |

Final combined validation accuracy: **57/79 (72.2%)**. The targeted corrections added one correct answer per database without observed validation regressions. One Thrombosis execution error remained.

These are validation results, not hidden-test or leaderboard scores. Repeated tuning on small validation sets can overestimate generalization. A retrieval-based alternative improved a small training holdout but regressed on full validation; broad answer-plan rewrites also regressed. The final approach therefore retains bounded, targeted corrections. Response caching affects measured runtime, so no cold-inference speed claim is made.

## Repository contents

```text
solution.py                 Public generate_sql interface and output corrections
draft_generator.py         Database-specific prompting and repair orchestration
sql_refinement.py           SQLGlot checks, deterministic repair and retrieval
llm_utils.py                Model client and response cache (course-derived)
run.py                      JSON-to-SQL command-line entry point
requirements.txt            Runtime dependencies
.env.example                Example model configuration; no credentials
examples/request.json       Synthetic request, not course data
tests/                     Offline tests for repair and output corrections
```

## Setup and usage

Use Python 3.11 or newer. In a terminal at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Provide a running OpenAI-compatible chat-completions endpoint. The example configuration uses Ollama at `http://127.0.0.1:11434/v1` and the project's evaluated model, `gemma4:e4b`. Configure the endpoint and model in `.env` before importing the solution. If Ollama is installed and this model is available in your environment:

```bash
ollama pull gemma4:e4b
ollama serve
```

If the Ollama app is already serving, do not start a second server. In another terminal with the virtual environment activated:

```bash
python run.py examples/request.json
```

The command prints generated SQL; it does not execute it. The synthetic example demonstrates the API and is not an accuracy benchmark. A different model or serving configuration can produce different results. The helper sends `reasoning_effort: none` for Gemma model names, so a replacement endpoint must support the corresponding request format.

### Python interface

```python
from solution import generate_sql

sql = generate_sql(
    question=question,
    evidence=evidence,
    schema=schema,
    train_examples=examples,
    db_id="toxicology",  # or "thrombosis_prediction"; other IDs use the simpler prompt
)
```

The schema contains `table_names_original`, `column_names_original` as `[table_index, name]` pairs, `column_types`, `foreign_keys` as column-index pairs, and optional `primary_keys` and `column_examples`. A table index of `-1` denotes the wildcard placeholder. Training examples contain `question`, `SQL`, and optionally `evidence` and `db_id`. Supply only examples authorized for use as demonstrations.

The `.env` configuration is read by `llm_utils.py`; process environment variables take precedence. Requests at temperature zero may be cached in `output/llm_cache.sqlite`. Keep that cache private because prompts and responses may contain input data. `LLM_CACHE` can override its path.

## Tests

```bash
python -m unittest discover -s tests -v
```

These offline tests cover selected identifier/scope checks, bounded repair, failure fallback, and output corrections. They do not need a model server and do not establish model accuracy.

The course evaluator and datasets are not redistributed, so this repository alone cannot reproduce the validation table. Authorized dataset holders can integrate `generate_sql` with their evaluator. Before executing generated SQL in another application, enforce read-only access and execution timeouts separately.

## Research and attribution

- [DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction](https://arxiv.org/abs/2304.11015) motivated a separate correction stage.
- [DAIL-SQL: Text-to-SQL Empowered by Large Language Models](https://arxiv.org/abs/2308.15363) motivated retrieval using both the question and a draft query. This implementation uses lexical and AST features, not the paper's embedding-based retrieval.
- [SQLGlot](https://sqlglot.com/sqlglot.html) supplies parsing, expression traversal, and scope analysis.
- `llm_utils.py` is adapted from the course-provided client helper; the default model was changed to the evaluated Gemma model. The project originated from the course starter interface. No license to redistribute course material is asserted, and no blanket open-source license is added.

OpenAI Codex assisted with implementation, debugging, paper research, experiment analysis, and documentation. The measured results originate from saved project evaluation runs. This repository contains the final approach, not full reproductions of the cited research systems.
