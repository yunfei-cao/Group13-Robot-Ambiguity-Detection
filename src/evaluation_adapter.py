"""Offline evaluation handoff. Gold labels are joined only after inference."""
import csv
import json
from pathlib import Path
from .evaluate import validate_samples
from .run_experiment import STRATEGIES


def convert_predictions(predictions_path, labels_path):
    predictions = [json.loads(line) for line in Path(predictions_path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    with Path(labels_path).open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream, strict=True))
    by_id = {r["sample_id"]: r for r in labels}
    if len(by_id) != len(labels):
        raise ValueError("Duplicate label IDs")
    groups = {s: [] for s in STRATEGIES}
    seen, runs = set(), set()
    for row in predictions:
        sid, strategy = row["sample_id"], row["strategy"]
        if strategy not in groups or (sid, strategy) in seen or sid not in by_id:
            raise ValueError("Unknown or duplicate prediction identity")
        seen.add((sid, strategy))
        runs.add(row["run_id"])
        status = row["status"]
        if status not in {"success", "invalid_output", "error", "interrupted"}:
            raise ValueError("Unknown prediction status")
        if "raw_text" not in row:
            raise ValueError("Legacy predictions lack raw_text; use an explicit attempt-journal migration")
        meta = by_id[sid]
        groups[strategy].append(dict(sample_id=sid, pair_id=meta["pair_id"],
            gold_label=meta["gold_label"], raw_response=row["raw_text"],
            request_status="SUCCESS" if status in {"success", "invalid_output"} else "REQUEST_FAILED"))
    if len(runs) != 1:
        raise ValueError("Expected exactly one run")
    identities = [{r["sample_id"] for r in rows} for rows in groups.values()]
    if any(ids != identities[0] for ids in identities):
        raise ValueError("Strategy coverage mismatch")
    for rows in groups.values():
        validate_samples(rows)  # Reject partial pairs rather than invent pair scores.
    return groups
