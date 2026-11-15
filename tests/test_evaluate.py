These tests pin the rules that every number in the report has to be
reproducible against: strip-only label matching, INVALID versus
REQUEST_FAILED, the six metric denominators, and the four manual-check
samples printed on page 6 of the group handbook.
"""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluate import (
    SAMPLES,
    VALID_LABELS,
    calculate_f1,
    calculate_metrics,
    export_scored_csv,
    parse_prediction,
    run_parser_tests,
    score_samples,
    validate_samples,
)
from src.evaluation_adapter import convert_predictions
from src.run_experiment import STRATEGIES


def record(sample_id, pair_id, gold_label, raw_response, status='SUCCESS'):
    return dict(sample_id=sample_id, pair_id=pair_id, gold_label=gold_label,
                raw_response=raw_response, request_status=status)


class ParsePredictionTests(unittest.TestCase):
    def test_exact_labels_are_accepted(self):
        for label in sorted(VALID_LABELS):
            self.assertEqual(parse_prediction(label, 'SUCCESS'), label)

    def test_only_surrounding_whitespace_is_stripped(self):
        self.assertEqual(parse_prediction('  CLEAR\n', 'SUCCESS'), 'CLEAR')
        self.assertEqual(parse_prediction('\t NEEDS_CLARIFICATION  ', 'SUCCESS'),
                         'NEEDS_CLARIFICATION')

    def test_surrounding_explanation_is_invalid(self):
        for raw in ['The answer is CLEAR',
                    'CLEAR.',
                    'CLEAR because the bowl is on the table',
                    'NEEDS_CLARIFICATION (the object is unclear)',
                    'Answer: NEEDS_CLARIFICATION']:
            self.assertEqual(parse_prediction(raw, 'SUCCESS'), 'INVALID', raw)

    def test_a_label_inside_other_text_is_invalid(self):
        # Substring matching would silently accept these; exact match must not.
        for raw in ['CLEARLY', 'NOT CLEAR', 'CLEAR CLEAR', 'clear',
                    'needs_clarification', 'NEEDS  CLARIFICATION',
                    'CLEAR\nNEEDS_CLARIFICATION']:
            self.assertEqual(parse_prediction(raw, 'SUCCESS'), 'INVALID', raw)

    def test_empty_and_non_string_are_invalid(self):
        for raw in ['', '   ', '\n\t', None, 1]:
            self.assertEqual(parse_prediction(raw, 'SUCCESS'), 'INVALID', raw)

    def test_failed_request_outranks_the_response_text(self):
        for status in ['error', 'interrupted', 'REQUEST_FAILED', 'timeout']:
            self.assertEqual(parse_prediction('CLEAR', status), 'REQUEST_FAILED', status)

    def test_success_status_spellings_are_normalised(self):
        for status in ['SUCCESS', 'success', ' Success ', 'OK', 'ok',
                       'SUCCEEDED', 'completed']:
            self.assertEqual(parse_prediction('CLEAR', status), 'CLEAR', status)

    def test_builtin_parser_self_test_passes(self):
        run_parser_tests()


class ValidateSamplesTests(unittest.TestCase):
    def test_balanced_pairs_pass(self):
        validate_samples([record('a', 'p1', 'CLEAR', 'CLEAR'),
                          record('b', 'p1', 'NEEDS_CLARIFICATION', 'NEEDS_CLARIFICATION')])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_samples([])

    def test_missing_field_is_rejected(self):
        broken = record('a', 'p1', 'CLEAR', 'CLEAR')
        del broken['request_status']
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            validate_samples([broken])

    def test_duplicate_sample_id_is_rejected(self):
        rows = [record('a', 'p1', 'CLEAR', 'CLEAR'),
                record('a', 'p1', 'NEEDS_CLARIFICATION', 'NEEDS_CLARIFICATION')]
        with self.assertRaisesRegex(ValueError, 'Duplicate sample_id'):
            validate_samples(rows)

    def test_unknown_gold_label_is_rejected(self):
        rows = [record('a', 'p1', 'MAYBE', 'CLEAR'),
                record('b', 'p1', 'NEEDS_CLARIFICATION', 'NEEDS_CLARIFICATION')]
        with self.assertRaisesRegex(ValueError, 'gold_label'):
            validate_samples(rows)

    def test_single_sample_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            validate_samples([record('a', 'p1', 'CLEAR', 'CLEAR')])

    def test_pair_with_two_identical_labels_is_rejected(self):
        rows = [record('a', 'p1', 'CLEAR', 'CLEAR'),
                record('b', 'p1', 'CLEAR', 'CLEAR')]
        with self.assertRaisesRegex(ValueError, 'one CLEAR'):
            validate_samples(rows)

    def test_three_sample_pair_is_rejected(self):
        rows = [record('a', 'p1', 'CLEAR', 'CLEAR'),
                record('b', 'p1', 'NEEDS_CLARIFICATION', 'NEEDS_CLARIFICATION'),
                record('c', 'p1', 'CLEAR', 'CLEAR')]
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            validate_samples(rows)


class HandbookManualCheckTests(unittest.TestCase):
    """The four samples on handbook page 6, with the hand-computed values."""

    def setUp(self):
        self.scored = score_samples(SAMPLES)
        self.metrics = calculate_metrics(self.scored)

    def test_sample_structure(self):
        self.assertEqual(len(self.scored), 4)
        self.assertEqual(self.metrics['total_samples'], 4)
        self.assertEqual(self.metrics['total_pairs'], 2)

    def test_each_prediction_follows_the_handbook(self):
        self.assertEqual([row['prediction'] for row in self.scored],
                         ['CLEAR', 'NEEDS_CLARIFICATION',
                          'NEEDS_CLARIFICATION', 'INVALID'])

    def test_six_metrics_match_the_handwritten_values(self):
        expected = {
            'accuracy': 0.50,
            'f1_clear': 2 / 3,
            'f1_needs_clarification': 0.50,
            'macro_f1': (2 / 3 + 0.50) / 2,
            'missed_ambiguity_rate': 0.50,
            'unnecessary_clarification_rate': 0.50,
            'pair_level_correctness': 0.50,
            'valid_output_rate': 0.75,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(self.metrics[name], value, places=12, msg=name)

    def test_error_counts(self):
        self.assertEqual(self.metrics['invalid_count'], 1)
        self.assertEqual(self.metrics['request_failed_count'], 0)

    def test_baselines_are_fifty_percent_on_the_balanced_set(self):
        self.assertAlmostEqual(self.metrics['always_clear_baseline_accuracy'], 0.5)
        self.assertAlmostEqual(self.metrics['always_clarify_baseline_accuracy'], 0.5)


class MetricRuleTests(unittest.TestCase):
    def metrics(self, rows):
        return calculate_metrics(score_samples(rows))

    def test_invalid_answer_counts_as_wrong_and_as_missed(self):
        rows = [record('a', 'p1', 'CLEAR', 'CLEAR'),
                record('b', 'p1', 'NEEDS_CLARIFICATION',
                       'The answer is NEEDS_CLARIFICATION')]
        metrics = self.metrics(rows)
        self.assertAlmostEqual(metrics['accuracy'], 0.5)
        self.assertAlmostEqual(metrics['missed_ambiguity_rate'], 1.0)
        self.assertAlmostEqual(metrics['unnecessary_clarification_rate'], 0.0)
        self.assertAlmostEqual(metrics['valid_output_rate'], 0.5)
        self.assertAlmostEqual(metrics['pair_level_correctness'], 0.0)
        self.assertEqual(metrics['invalid_count'], 1)

    def test_request_failure_is_wrong_but_never_a_clarification(self):
        rows = [record('a', 'p1', 'CLEAR', None, 'error'),
                record('b', 'p1', 'NEEDS_CLARIFICATION', 'NEEDS_CLARIFICATION')]
        metrics = self.metrics(rows)
        self.assertAlmostEqual(metrics['accuracy'], 0.5)
        self.assertAlmostEqual(metrics['unnecessary_clarification_rate'], 0.0)
        self.assertAlmostEqual(metrics['missed_ambiguity_rate'], 0.0)
        self.assertAlmostEqual(metrics['valid_output_rate'], 0.5)
        self.assertEqual(metrics['request_failed_count'], 1)
        self.assertEqual(metrics['invalid_count'], 0)

    def test_unnecessary_clarification_needs_an_explicit_label(self):
        explicit = self.metrics([record('a', 'p1', 'CLEAR', 'NEEDS_CLARIFICATION'),
                                 record('b', 'p1', 'NEEDS_CLARIFICATION',
                                        'NEEDS_CLARIFICATION')])
        self.assertAlmostEqual(explicit['unnecessary_clarification_rate'], 1.0)
        self.assertAlmostEqual(explicit['missed_ambiguity_rate'], 0.0)
        self.assertAlmostEqual(explicit['accuracy'], 0.5)

        invalid = self.metrics([record('a', 'p1', 'CLEAR', ''),
                                record('b', 'p1', 'NEEDS_CLARIFICATION',
                                       'NEEDS_CLARIFICATION')])
        self.assertAlmostEqual(invalid['unnecessary_clarification_rate'], 0.0)
        self.assertAlmostEqual(invalid['valid_output_rate'], 0.5)

    def test_pair_level_correctness_needs_both_samples(self):
        rows = [record('a', 'p1', 'CLEAR', 'CLEAR'),
                record('b', 'p1', 'NEEDS_CLARIFICATION', 'NEEDS_CLARIFICATION'),
                record('c', 'p2', 'CLEAR', 'CLEAR'),
                record('d', 'p2', 'NEEDS_CLARIFICATION', 'CLEAR')]
        metrics = self.metrics(rows)
        self.assertAlmostEqual(metrics['accuracy'], 0.75)
        self.assertAlmostEqual(metrics['pair_level_correctness'], 0.5)

    def test_denominator_covers_every_selected_sample(self):
        rows = []
        for index in range(3):
            rows.append(record(f'c{index}', f'p{index}', 'CLEAR', 'CLEAR'))
            rows.append(record(f'n{index}', f'p{index}', 'NEEDS_CLARIFICATION',
                               None, 'error'))
        metrics = self.metrics(rows)
        self.assertEqual(metrics['total_samples'], 6)
        self.assertEqual(metrics['total_pairs'], 3)
        self.assertAlmostEqual(metrics['accuracy'], 0.5)
        self.assertAlmostEqual(metrics['valid_output_rate'], 0.5)

    def test_macro_f1_averages_both_classes(self):
        rows = [record('a', 'p1', 'CLEAR', 'CLEAR'),
                record('b', 'p1', 'NEEDS_CLARIFICATION', 'CLEAR')]
        metrics = self.metrics(rows)
        self.assertAlmostEqual(metrics['f1_clear'], 2 / 3)
        self.assertAlmostEqual(metrics['f1_needs_clarification'], 0.0)
        self.assertAlmostEqual(metrics['macro_f1'], 1 / 3)

    def test_empty_class_denominator_is_zero_not_an_error(self):
        only_clear = [dict(gold_label='CLEAR', prediction='CLEAR')]
        self.assertEqual(calculate_f1(only_clear, 'NEEDS_CLARIFICATION'), 0.0)
        with self.assertRaises(ValueError):
            calculate_metrics([])


class ExportTests(unittest.TestCase):
    def test_scored_csv_keeps_the_handoff_columns(self):
        scored = score_samples(SAMPLES)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'scored.csv'
            export_scored_csv(scored, path)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(list(rows[0].keys()),
                         ['sample_id', 'pair_id', 'gold_label', 'raw_response',
                          'request_status', 'prediction', 'valid_output', 'correct'])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[3]['prediction'], 'INVALID')
        self.assertEqual(rows[3]['valid_output'], 'False')
        self.assertEqual(rows[3]['correct'], 'False')


class AdapterHandoffTests(unittest.TestCase):
    """CAO's adapter output has to land in evaluate.py without manual edits."""

    def convert(self, statuses, raw_text='CLEAR'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / 'labels.csv'
            labels.write_text('sample_id,pair_id,gold_label\n'
                              'c,p1,CLEAR\nn,p1,NEEDS_CLARIFICATION\n',
                              encoding='utf-8')
            rows = [dict(sample_id=sid, strategy=strategy, run_id='offline',
                         status=status, raw_text=raw_text)
                    for strategy in STRATEGIES
                    for sid, status in (('c', statuses[0]), ('n', statuses[1]))]
            predictions = root / 'predictions.jsonl'
            predictions.write_text(''.join(json.dumps(r) + '\n' for r in rows),
                                   encoding='utf-8')
            return convert_predictions(predictions, labels)

    def test_adapter_output_is_scorable_as_is(self):
        converted = self.convert(('success', 'success'))
        self.assertEqual(set(converted), set(STRATEGIES))
        for group in converted.values():
            metrics = calculate_metrics(score_samples(group))
            self.assertEqual(metrics['total_samples'], 2)
            self.assertEqual(metrics['total_pairs'], 1)
            self.assertAlmostEqual(metrics['accuracy'], 0.5)

    def test_invalid_output_is_not_a_request_failure(self):
        converted = self.convert(('success', 'invalid_output'), raw_text='')
        for group in converted.values():
            metrics = calculate_metrics(score_samples(group))
            self.assertEqual(metrics['invalid_count'], 2)
            self.assertEqual(metrics['request_failed_count'], 0)

    def test_error_and_interrupted_become_request_failures(self):
        for status in ('error', 'interrupted'):
            converted = self.convert((status, status))
            for group in converted.values():
                metrics = calculate_metrics(score_samples(group))
                self.assertEqual(metrics['request_failed_count'], 2)
                self.assertEqual(metrics['invalid_count'], 0)
                self.assertAlmostEqual(metrics['accuracy'], 0.0)


if __name__ == '__main__':
    unittest.main()
