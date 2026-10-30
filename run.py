"""Generate SQL from a JSON request; does not execute the generated query."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path, help='JSON with question, schema and optional evidence/examples/db_id')
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    from solution import generate_sql
    print(generate_sql(request['question'], request.get('evidence', ''),
                       request['schema'], request.get('train_examples', []),
                       request.get('db_id', 'demo')))


if __name__ == '__main__':
    main()
