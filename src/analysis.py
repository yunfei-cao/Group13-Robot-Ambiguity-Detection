import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = (
    ROOT / "results/analysis/v03-development-evaluation-20261113/scored_predictions.csv"
)
DEFAULT_LABELS = ROOT / "data/processed/dev_labels.csv"
OUT_DIR = ROOT / "results/analysis/recomputed"
FIG_DIR = ROOT / "figures"

POSITIVE = "NEEDS_CLARIFICATION"


def class_f1(gold, pred, label):
    """F1 = 2TP / (2TP + FP + FN)；分母为 0 时返回 0。"""
    tp = ((gold == label) & (pred == label)).sum()
    fp = ((gold != label) & (pred == label)).sum()
    fn = ((gold == label) & (pred != label)).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def metrics_for(gold, pred):
    """一组预测的五项指标（有效输出率另算）。分母口径按手册第 6 页。"""
    n = len(gold)
    correct = (gold == pred).sum()
    yes, no = (gold == POSITIVE), (gold == "CLEAR")
    return {
        "n": n,
        "correct": int(correct),
        "accuracy": correct / n,
        "macro_f1": (class_f1(gold, pred, "CLEAR") + class_f1(gold, pred, POSITIVE)) / 2,
        "miss_rate": (yes & (pred != POSITIVE)).sum() / yes.sum(),
        "unnecessary_rate": (no & (pred == POSITIVE)).sum() / no.sum(),
    }


def build_overall(df):
    rows = []
    for strategy, g in df.groupby("strategy"):
        m = metrics_for(g["gold_label"], g["prediction"])
        m["strategy"] = strategy
        m["valid_output_rate"] = g["valid_output"].astype(bool).mean()
        pairs = g.groupby("pair_id")["correct"].agg(["sum", "size"])
        m["pair_accuracy"] = (pairs["sum"] == pairs["size"]).mean()
        m["n_pairs"] = len(pairs)
        rows.append(m)
    cols = [
        "strategy", "n", "correct", "accuracy", "macro_f1",
        "miss_rate", "unnecessary_rate", "pair_accuracy",
        "n_pairs", "valid_output_rate",
    ]
    return pd.DataFrame(rows)[cols].sort_values("accuracy", ascending=False)


def build_by_category(df):
    rows = []
    for (strategy, category), g in df.groupby(["strategy", "category"]):
        if g["gold_label"].nunique() < 2:
            continue
        rows.append(
            {"strategy": strategy, "category": category,
             **metrics_for(g["gold_label"], g["prediction"])}
        )
    return pd.DataFrame(rows)


def build_pairs(df):
    pairs = df.groupby(["strategy", "pair_id"], as_index=False).agg(
        category=("category", "first"),
        n=("correct", "size"),
        n_correct=("correct", "sum"),
    )
    pairs["both_correct"] = pairs["n_correct"] == pairs["n"]
    return pairs


def plot_overview(overall):
    labels = list(overall["strategy"])
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([i - width / 2 for i in x], overall["accuracy"], width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], overall["pair_accuracy"], width, label="Pair accuracy")
    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="grey")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Instruction-level vs pair-level performance (n=170 per strategy)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_overview.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores, encoding="utf-8-sig")
    labels = pd.read_csv(args.labels, encoding="utf-8-sig")[["sample_id", "category"]]

    df = scores.merge(labels, on="sample_id", how="left", validate="many_to_one")
    missing = df["category"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no category, filled with 'unknown'")
        df["category"] = df["category"].fillna("unknown")

    overall = build_overall(df)
    by_category = build_by_category(df)
    pairs = build_pairs(df)

    overall.round(6).to_csv(OUT_DIR / "metrics_overall.csv", index=False)
    by_category.round(6).to_csv(OUT_DIR / "metrics_by_category.csv", index=False)
    pairs.to_csv(OUT_DIR / "pair_results.csv", index=False)
    plot_overview(overall)

    show = overall.copy()
    for c in ["accuracy", "macro_f1", "miss_rate", "unnecessary_rate", "pair_accuracy"]:
        show[c] = (show[c] * 100).round(4).astype(str) + "%"
    print(show.to_string(index=False))
    print(f"\noutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
