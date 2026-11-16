"""Negative handoff contracts; no model or network requests."""
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation_adapter import convert_predictions
from src.run_experiment import STRATEGIES


class EvaluationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = self.root / 'labels.csv'
        self.labels.write_text('sample_id,pair_id,gold_label\na,p,CLEAR\nb,p,NEEDS_CLARIFICATION\n')
        self.rows = [dict(sample_id=sid, strategy=strategy, run_id='offline',
                          status='success', raw_text='CLEAR')
                     for strategy in STRATEGIES for sid in ('a', 'b')]

    def convert(self, rows):
        path = self.root / 'predictions.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return convert_predictions(path, self.labels)

    def test_duplicate_prediction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.convert(self.rows + [self.rows[0]])

    def test_missing_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.convert(self.rows[:-2])

    def test_mixed_runs_are_rejected(self):
        self.rows[0]['run_id'] = 'another-run'
        with self.assertRaisesRegex(ValueError, 'one run'):
            self.convert(self.rows)

    def test_unknown_status_and_legacy_output_are_rejected(self):
        self.rows[0]['status'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'status'):
            self.convert(self.rows)
        self.rows[0]['status'] = 'success'
        del self.rows[0]['raw_text']
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            self.convert(self.rows)
