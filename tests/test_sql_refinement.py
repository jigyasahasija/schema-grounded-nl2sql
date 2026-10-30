"""Focused tests for scope-aware repair and bounded model calls, without a server."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sql_refinement import check_sql, refine_sql, repair_examples


SCHEMA = {
    "table_names_original": ["Patient", "Lab"],
    "column_names_original": [[-1, "*"], [0, "ID"], [0, "Birthday"],
                              [1, "ID"], [1, "Lab Value"], [1, "Result"]],
    "foreign_keys": [[3, 1]],
}


class SQLRefinementTests(unittest.TestCase):
    def test_valid_nested_and_correlated_sql_is_unchanged(self):
        queries = [
            'SELECT p.ID FROM Patient p WHERE EXISTS '
            '(SELECT 1 FROM Lab l WHERE l.ID=p.ID AND l.Result>2)',
            'SELECT COUNT(*) FROM Lab WHERE ID=(SELECT ID FROM Patient '
            "WHERE Birthday='2000-01-01')",
            'WITH x AS (SELECT ID, Result FROM Lab) SELECT Result FROM x',
            'SELECT AVG(total) FROM (SELECT COUNT(*) AS total FROM Lab) x',
        ]
        for query in queries:
            with self.subTest(query=query):
                result = check_sql(query, SCHEMA)
                self.assertEqual(result.issues, [])
                self.assertEqual(result.sql, query)

    def test_unknown_name_is_repaired_only_on_its_actual_table(self):
        result = check_sql('SELECT l.Lab_Value FROM Lab l', SCHEMA)
        self.assertEqual(result.issues, [])
        self.assertIn('l."Lab Value"', result.sql)
        wrong_table = check_sql('SELECT p.Lab_Value FROM Patient p', SCHEMA)
        self.assertTrue(wrong_table.issues)
        self.assertEqual(wrong_table.fixes, [])

    def test_literals_are_preserved_during_identifier_repair(self):
        result = check_sql("SELECT l.Lab_Value FROM Lab l WHERE l.Result='Lab_Value'", SCHEMA)
        self.assertIn("'Lab_Value'", result.sql)

    def test_sqlite_year_and_arithmetic_are_translated(self):
        result = check_sql('SELECT SUBTRACT(YEAR(CURRENT_TIMESTAMP), '
                           'YEAR(p.Birthday)) FROM Patient p', SCHEMA)
        self.assertEqual(result.issues, [])
        self.assertIn('STRFTIME', result.sql)
        self.assertNotIn('SUBTRACT(', result.sql)
        self.assertNotIn('YEAR(', result.sql)

    def test_unknown_alias_and_unqualified_column_are_detected(self):
        self.assertTrue(check_sql('SELECT missing.Result FROM Lab l', SCHEMA).issues)
        self.assertTrue(check_sql('SELECT nonexistent FROM Lab', SCHEMA).issues)

    def test_clean_draft_never_invokes_model(self):
        def forbidden(*args, **kwargs):
            self.fail('A clean draft must bypass model repair')
        query = 'SELECT ID FROM Patient'
        self.assertEqual(refine_sql(query, 'IDs?', '', SCHEMA, [], 'test',
                                    forbidden, str.strip), query)

    def test_invalid_draft_is_corrected_with_one_model_call(self):
        calls = []
        def model(prompt, **kwargs):
            calls.append(prompt)
            return 'SELECT l.Result FROM Lab l'
        output = refine_sql('SELECT l.Result FROM Patient l', 'Results?', '',
                            SCHEMA, [], 'test', model, str.strip)
        self.assertEqual(output, 'SELECT l.Result FROM Lab l')
        self.assertEqual(len(calls), 1)
        self.assertIn('Static diagnostics:', calls[0])

    def test_failed_optional_review_retains_draft(self):
        query = 'SELECT p.Result FROM Patient p'
        def unavailable(*args, **kwargs):
            raise TimeoutError('test')
        self.assertEqual(refine_sql(query, 'Results?', '', SCHEMA, [], 'test',
                                    unavailable, str.strip), query)

    def test_repair_examples_come_only_from_supplied_training(self):
        examples = [
            {'db_id': 'test', 'question': 'Count patients', 'evidence': '',
             'SQL': 'SELECT COUNT(*) FROM Patient'},
            {'db_id': 'other', 'question': 'Count patients',
             'SQL': 'SELECT COUNT(*) FROM Patient'},
        ]
        selected = repair_examples('Count patients', '',
                                   'SELECT COUNT(*) FROM Patient', examples, 'test')
        self.assertEqual(selected, examples[:1])


if __name__ == '__main__':
    unittest.main()
