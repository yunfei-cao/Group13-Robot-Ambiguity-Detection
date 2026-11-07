# ============================================================
# Group 13 - WANG HONGFENG
# evaluate.py
# Purpose: validate model outputs and calculate evaluation metrics
# This version uses simulated data first. It does not need pandas.
# ============================================================

from collections import defaultdict
import csv
from pathlib import Path


# ------------------------------------------------------------
# 1. Labels and status definitions
# ------------------------------------------------------------

VALID_LABELS = {"CLEAR", "NEEDS_CLARIFICATION"}
POSITIVE_LABEL = "NEEDS_CLARIFICATION"
NEGATIVE_LABEL = "CLEAR"

# The API/client may use a different success word.
# When CAO hands over the real output, check the actual status values
# and add them here only if they mean that a model response was returned.
SUCCESS_STATUSES = {"SUCCESS", "SUCCEEDED", "OK", "COMPLETED"}


# ------------------------------------------------------------
# 2. Four manual-check examples from the group handbook
# ------------------------------------------------------------
# raw_response is the model's original text.
# request_status tells us whether the API request succeeded.

SAMPLES = [
    {
        "sample_id": "S001",
        "pair_id": "P1",
        "gold_label": "CLEAR",
        "raw_response": "CLEAR",
        "request_status": "SUCCESS",
    },
    {
        "sample_id": "S002",
        "pair_id": "P1",
        "gold_label": "NEEDS_CLARIFICATION",
        "raw_response": "NEEDS_CLARIFICATION",
        "request_status": "SUCCESS",
    },
    {
        "sample_id": "S003",
        "pair_id": "P2",
        "gold_label": "CLEAR",
        "raw_response": "NEEDS_CLARIFICATION",
        "request_status": "SUCCESS",
    },
    {
        "sample_id": "S004",
        "pair_id": "P2",
        "gold_label": "NEEDS_CLARIFICATION",
        "raw_response": "INVALID",
        "request_status": "SUCCESS",
    },
]


# ------------------------------------------------------------
# 3. Convert a raw model response into a prediction label
# ------------------------------------------------------------

def parse_prediction(raw_response, request_status):
    """Return a valid label, INVALID, or REQUEST_FAILED.

    Rules:
    1. If the API request did not succeed, return REQUEST_FAILED.
    2. If it succeeded, remove surrounding whitespace only.
    3. Accept exactly CLEAR or NEEDS_CLARIFICATION.
    4. Everything else is INVALID.

    Important: do not use lower(), upper(), or search for a label
    inside a sentence. The handbook requires exact matching after strip().
    """

    status = str(request_status).strip().upper()

    if status not in SUCCESS_STATUSES:
        return "REQUEST_FAILED"

    if not isinstance(raw_response, str):
        return "INVALID"

    cleaned = raw_response.strip()

    if cleaned in VALID_LABELS:
        return cleaned

    return "INVALID"


# ------------------------------------------------------------
# 4. Check the input data before scoring
# ------------------------------------------------------------

def validate_samples(samples):
    """Check IDs, gold labels, and paired-sample structure."""

    if not samples:
        raise ValueError("There are no samples to evaluate.")

    required_fields = {
        "sample_id",
        "pair_id",
        "gold_label",
        "raw_response",
        "request_status",
    }

    seen_sample_ids = set()
    pairs = defaultdict(list)

    for row_number, item in enumerate(samples, start=1):
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            raise ValueError(
                f"Row {row_number} is missing fields: {sorted(missing_fields)}"
            )

        sample_id = item["sample_id"]
        pair_id = item["pair_id"]
        gold_label = item["gold_label"]

        if not sample_id:
            raise ValueError(f"Row {row_number} has an empty sample_id.")

        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id found: {sample_id}")
        seen_sample_ids.add(sample_id)

        if not pair_id:
            raise ValueError(f"Row {row_number} has an empty pair_id.")

        if gold_label not in VALID_LABELS:
            raise ValueError(
                f"Unknown gold_label in sample {sample_id}: {gold_label}"
            )

        pairs[pair_id].append(item)

    # Every AmbiK pair should contain one CLEAR and one
    # NEEDS_CLARIFICATION sample.
    for pair_id, pair_items in pairs.items():
        if len(pair_items) != 2:
            raise ValueError(
                f"Pair {pair_id} has {len(pair_items)} samples; expected 2."
            )

        labels_in_pair = {item["gold_label"] for item in pair_items}
        if labels_in_pair != VALID_LABELS:
            raise ValueError(
                f"Pair {pair_id} must contain one CLEAR and one "
                "NEEDS_CLARIFICATION sample."
            )


# ------------------------------------------------------------
# 5. Score each sample
# ------------------------------------------------------------

def score_samples(samples):
    """Add prediction, valid_output, and correct fields to each row."""

    validate_samples(samples)

    scored = []

    for item in samples:
        prediction = parse_prediction(
            item["raw_response"],
            item["request_status"],
        )

        scored_item = dict(item)
        scored_item["prediction"] = prediction
        scored_item["valid_output"] = prediction in VALID_LABELS
        scored_item["correct"] = prediction == item["gold_label"]
        scored.append(scored_item)

    return scored


# ------------------------------------------------------------
# 6. Calculate F1 for one class
# ------------------------------------------------------------

def calculate_f1(scored_samples, label):
    tp = 0
    fp = 0
    fn = 0

    for item in scored_samples:
        gold = item["gold_label"]
        prediction = item["prediction"]

        if gold == label and prediction == label:
            tp += 1
        elif gold != label and prediction == label:
            fp += 1
        elif gold == label and prediction != label:
            # INVALID and REQUEST_FAILED count as FN for the true class.
            fn += 1

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 0.0

    return 2 * tp / denominator


# ------------------------------------------------------------
# 7. Calculate all six project metrics and useful error counts
# ------------------------------------------------------------

def calculate_metrics(scored_samples):
    total = len(scored_samples)

    if total == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset.")

    # Accuracy: invalid outputs and request failures are incorrect.
    correct_count = sum(1 for item in scored_samples if item["correct"])
    accuracy = correct_count / total

    # Macro-F1: average of the two class-specific F1 scores.
    f1_clear = calculate_f1(scored_samples, NEGATIVE_LABEL)
    f1_needs = calculate_f1(scored_samples, POSITIVE_LABEL)
    macro_f1 = (f1_clear + f1_needs) / 2

    # Missed-ambiguity rate: invalid and failed ambiguous samples count as missed.
    ambiguous_samples = [
        item for item in scored_samples
        if item["gold_label"] == POSITIVE_LABEL
    ]
    missed_count = sum(
        1 for item in ambiguous_samples
        if item["prediction"] != POSITIVE_LABEL
    )
    missed_rate = missed_count / len(ambiguous_samples)

    # Unnecessary-clarification rate: only NEEDS_CLARIFICATION on a clear
    # sample counts. INVALID and REQUEST_FAILED are reported separately.
    clear_samples = [
        item for item in scored_samples
        if item["gold_label"] == NEGATIVE_LABEL
    ]
    unnecessary_count = sum(
        1 for item in clear_samples
        if item["prediction"] == POSITIVE_LABEL
    )
    unnecessary_rate = unnecessary_count / len(clear_samples)

    # Pair-level correctness: both samples in a pair must be correct.
    pair_results = defaultdict(list)
    for item in scored_samples:
        pair_results[item["pair_id"]].append(item["correct"])

    correct_pairs = sum(
        1 for pair_correctness in pair_results.values()
        if len(pair_correctness) == 2 and all(pair_correctness)
    )
    pair_accuracy = correct_pairs / len(pair_results)

    # Valid-output rate: only the two exact labels after strip() are valid.
    valid_count = sum(
        1 for item in scored_samples
        if item["valid_output"]
    )
    valid_rate = valid_count / total

    # Additional diagnostic counts. These are not extra project metrics.
    invalid_count = sum(
        1 for item in scored_samples
        if item["prediction"] == "INVALID"
    )
    failed_count = sum(
        1 for item in scored_samples
        if item["prediction"] == "REQUEST_FAILED"
    )

    # Baselines: always predict one of the two labels.
    always_clear_accuracy = sum(
        1 for item in scored_samples
        if item["gold_label"] == NEGATIVE_LABEL
    ) / total

    always_clarify_accuracy = sum(
        1 for item in scored_samples
        if item["gold_label"] == POSITIVE_LABEL
    ) / total

    return {
        "total_samples": total,
        "total_pairs": len(pair_results),
        "accuracy": accuracy,
        "f1_clear": f1_clear,
        "f1_needs_clarification": f1_needs,
        "macro_f1": macro_f1,
        "missed_ambiguity_rate": missed_rate,
        "unnecessary_clarification_rate": unnecessary_rate,
        "pair_level_correctness": pair_accuracy,
        "valid_output_rate": valid_rate,
        "invalid_count": invalid_count,
        "request_failed_count": failed_count,
        "always_clear_baseline_accuracy": always_clear_accuracy,
        "always_clarify_baseline_accuracy": always_clarify_accuracy,
    }


# ------------------------------------------------------------
# 8. Export sample-level scoring results to CSV
# ------------------------------------------------------------

def export_scored_csv(scored_samples, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "pair_id",
        "gold_label",
        "raw_response",
        "request_status",
        "prediction",
        "valid_output",
        "correct",
    ]

    with output_file.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in scored_samples:
            writer.writerow({key: item.get(key) for key in fieldnames})


# ------------------------------------------------------------
# 9. Test output parsing separately
# ------------------------------------------------------------

def run_parser_tests():
    test_cases = [
        ("Exact CLEAR", "CLEAR", "SUCCESS", "CLEAR"),
        (
            "Exact NEEDS_CLARIFICATION",
            "NEEDS_CLARIFICATION",
            "SUCCESS",
            "NEEDS_CLARIFICATION",
        ),
        ("Leading/trailing spaces", " CLEAR ", "SUCCESS", "CLEAR"),
        ("Extra explanation", "The answer is CLEAR", "SUCCESS", "INVALID"),
        ("Empty answer", "", "SUCCESS", "INVALID"),
        ("Missing response", None, "SUCCESS", "INVALID"),
        ("API request failed", None, "REQUEST_FAILED", "REQUEST_FAILED"),
        ("Lowercase label", "clear", "SUCCESS", "INVALID"),
    ]

    print("\nPARSER TESTS")
    print("-" * 60)

    all_passed = True

    for test_name, raw_response, status, expected in test_cases:
        actual = parse_prediction(raw_response, status)
        passed = actual == expected
        all_passed = all_passed and passed
        result_text = "PASS" if passed else "FAIL"
        print(f"{result_text}: {test_name} -> {actual} (expected: {expected})")

    print("-" * 60)

    if not all_passed:
        raise AssertionError("At least one parser test failed.")

    print("All parser tests passed.")


# ------------------------------------------------------------
# 10. Print the evaluation report
# ------------------------------------------------------------

def print_report(metrics):
    print("\n" + "=" * 58)
    print("GROUP 13 - WANG: EVALUATION RESULTS")
    print("=" * 58)
    print(f"Number of samples: {metrics['total_samples']}")
    print(f"Number of pairs: {metrics['total_pairs']}")
    print("-" * 58)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"CLEAR F1: {metrics['f1_clear']:.2%}")
    print(
        "NEEDS_CLARIFICATION F1: "
        f"{metrics['f1_needs_clarification']:.2%}"
    )
    print(f"Macro-F1: {metrics['macro_f1']:.2%}")
    print(
        "Missed-ambiguity rate: "
        f"{metrics['missed_ambiguity_rate']:.2%}"
    )
    print(
        "Unnecessary-clarification rate: "
        f"{metrics['unnecessary_clarification_rate']:.2%}"
    )
    print(
        "Pair-level correctness: "
        f"{metrics['pair_level_correctness']:.2%}"
    )
    print(f"Valid-output rate: {metrics['valid_output_rate']:.2%}")
    print("-" * 58)
    print(f"INVALID responses: {metrics['invalid_count']}")
    print(f"REQUEST_FAILED requests: {metrics['request_failed_count']}")
    print(
        "Always-CLEAR baseline accuracy: "
        f"{metrics['always_clear_baseline_accuracy']:.2%}"
    )
    print(
        "Always-NEEDS baseline accuracy: "
        f"{metrics['always_clarify_baseline_accuracy']:.2%}"
    )
    print("=" * 58)


# ------------------------------------------------------------
# 11. Main program
# ------------------------------------------------------------

def main():
    # A. Test the output-format rules first.
    run_parser_tests()

    # B. Score the four manual-check examples.
    scored_samples = score_samples(SAMPLES)

    # C. Calculate the six metrics.
    metrics = calculate_metrics(scored_samples)

    # D. Print the results.
    print_report(metrics)

    # E. Export the sample-level result for inspection.
    export_scored_csv(
        scored_samples,
        "results/scored_predictions_simulation.csv",
    )
    print(
        "\nScored sample details saved to: "
        "results/scored_predictions_simulation.csv"
    )

    # F. Check the four expected manual-calculation results.
    expected = {
        "accuracy": 0.50,
        "f1_clear": 2 / 3,
        "f1_needs_clarification": 0.50,
        "macro_f1": (2 / 3 + 0.50) / 2,
        "missed_ambiguity_rate": 0.50,
        "unnecessary_clarification_rate": 0.50,
        "pair_level_correctness": 0.50,
        "valid_output_rate": 0.75,
    }

    for metric_name, expected_value in expected.items():
        actual_value = metrics[metric_name]
        if abs(actual_value - expected_value) > 1e-9:
            raise AssertionError(
                f"{metric_name} mismatch: got {actual_value}, "
                f"expected {expected_value}"
            )

    print("\nManual-check example: PASSED")
    print("All evaluation checks completed successfully.")


if __name__ == "__main__":
    main()
