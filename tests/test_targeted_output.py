from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import correct_output
S={'table_names_original':['items'], 'column_names_original':[[-1,'*'],[0,'id']], 'column_types':['text','number'], 'foreign_keys':[]}

class TargetedTests(unittest.TestCase):
    def test_explicit_unique_count(self):
        self.assertIn('DISTINCT', correct_output('SELECT COUNT(id) FROM items','How many?', 'Count unique ones',S))
    def test_ordinary_count_unchanged(self):
        sql='SELECT COUNT(id) FROM items'
        self.assertEqual(correct_output(sql,'How many?', '',S),sql)
    def test_star_not_guessed(self):
        sql='SELECT COUNT(*) FROM items'
        self.assertEqual(correct_output(sql,'How many unique?', '',S),sql)
    def test_percentage_round_removed(self):
        self.assertNotIn('ROUND',correct_output('SELECT ROUND(100.0/3, 5) AS pct','What percentage?', '',S))
    def test_evidence_precision_preserved(self):
        sql='SELECT ROUND(100.0/3, 5) FROM items'
        self.assertEqual(correct_output(sql,'What percentage?', 'Round to five decimals',S),sql)
    def test_nonpercentage_unchanged(self):
        sql='SELECT ROUND(AVG(id), 2) FROM items'
        self.assertEqual(correct_output(sql,'What average?', '',S),sql)
