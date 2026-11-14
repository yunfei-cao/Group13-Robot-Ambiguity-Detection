from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data" / "processed" / "scored_predictions.csv"
RESULTS = ROOT / "results" / "analysis"
FIGURES = ROOT / "figures"


def build_overall(df):
    return df.groupby("strategy", as_index=False).agg(
        n=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
        valid_output_rate=("valid_output", "mean"),
    )


def build_by_category(df):
    return df.groupby(["strategy", "category"], as_index=False).agg(
        n=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    )


def build_pairs(df):
    pairs = df.groupby(["strategy", "pair_id"], as_index=False).agg(
        n=("correct", "size"),
        n_correct=("correct", "sum"),
    )
    pairs["both_correct"] = pairs["n_correct"] == pairs["n"]
    return pairs


def plot_overview(overall, pairs):
    pair_acc = pairs.groupby("strategy")["both_correct"].mean()
    labels = list(overall["strategy"])
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([i - width / 2 for i in x], overall["accuracy"], width, label="Accuracy")
    ax.bar(
        [i + width / 2 for i in x],
        [pair_acc[s] for s in labels],
        width,
        label="Pair accuracy",
    )
    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="grey")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Instruction-level vs pair-level performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "metrics_overview.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SCORES)
    overall = build_overall(df)
    by_category = build_by_category(df)
    pairs = build_pairs(df)

    overall.to_csv(RESULTS / "metrics_overall.csv", index=False)
    by_category.to_csv(RESULTS / "metrics_by_category.csv", index=False)
    pairs.to_csv(RESULTS / "pair_results.csv", index=False)

    plot_overview(overall, pairs)

    print(overall.to_string(index=False))
    print()
    print("pair accuracy:")
    print(pairs.groupby("strategy")["both_correct"].mean().round(4).to_string())
    print(f"\nfigures -> {FIGURES / 'metrics_overview.png'}")


if __name__ == "__main__":
    main()
