"""
import_generation_scores.py

Merges DeepSeek's judging output (JSON, produced by pasting the
judge_batch_*.md files from evaluate_generation.py into DeepSeek Chat)
with the local, non-LLM generation diagnostics, and prints/saves the
final route-broken-down generation evaluation.

Workflow:
1. python evaluate_generation.py --dataset functional_requirements_eval_set_merged.csv
   -> writes generation_diagnostics.csv and judge_export_gen/judge_batch_01.md, ...
2. Paste each judge_batch_XX.md into DeepSeek Chat. Save each JSON reply
   to its own file (e.g. gen_scores_01.json, gen_scores_02.json, ...).
3. python import_generation_scores.py --scores gen_scores_01.json gen_scores_02.json ...

Output:
generation_eval_final.csv   — per-question diagnostics + judge scores
printed overall + per-route breakdown
"""

import argparse
import json

import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Unicode BiDi isolates so Arabic displays correctly in terminals
RLI = "\u2067"
PDI = "\u2069"


def rtl(text: str) -> str:
    return f"{RLI}{text}{PDI}"


def pct(x):
    """Format a fraction as a percentage."""
    return f"{x * 100:.1f}%" if pd.notna(x) else "N/A"


def load_scores(paths: list[str]) -> list[dict]:
    all_scores = []

    for p in paths:
        with open(p, encoding="utf-8") as f:
            raw = f.read().strip()

        # Accept ```json fenced output from DeepSeek
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        if not isinstance(data, list):
            raise ValueError(
                f"{p} should contain a JSON array of per-question objects, "
                f"got {type(data)}"
            )

        all_scores.extend(data)

    return all_scores


def faithfulness_from_verdicts(row) -> float | None:
    verdicts = row.get("faithfulness_verdicts")

    if not verdicts:
        return None

    return sum(verdicts) / len(verdicts)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(score_paths: list[str], diagnostics_path: str, out_path: str):

    scores = load_scores(score_paths)

    scores_df = pd.DataFrame(scores)

    if "id" not in scores_df.columns:
        raise ValueError(
            "Judge JSON is missing an 'id' field per question — can't merge."
        )

    scores_df["faithfulness"] = scores_df.apply(
        faithfulness_from_verdicts,
        axis=1,
    )

    diag_df = pd.read_csv(diagnostics_path)

    diag_df["id"] = diag_df["id"].astype(str)
    scores_df["id"] = scores_df["id"].astype(str)

    merged = diag_df.merge(
        scores_df[
            [
                "id",
                "faithfulness",
                "answer_relevancy",
                "answer_correctness",
            ]
        ],
        on="id",
        how="left",
    )

    missing = merged[merged["answer_correctness"].isna()]
    if len(missing):
        print(
            f"[WARNING] {len(missing)} question(s) have no judge score "
            f"— ids: {missing['id'].tolist()}"
        )

    dupes = scores_df["id"][scores_df["id"].duplicated()].tolist()
    if dupes:
        print(f"[WARNING] Duplicate ids in judge scores (kept first): {dupes}")

    merged.to_csv(out_path, index=False)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    # ------------------------------------------------------------------
    # OVERALL
    # ------------------------------------------------------------------

    print("\n===== OVERALL =====")

    overall = merged[metric_cols].mean(numeric_only=True)

    print(f"{'Metric':<24}{'Value':>10}")
    print("-" * 36)
    print(f"{'Faithfulness':<24}{pct(overall['faithfulness']):>10}")
    print(f"{'Answer Relevancy':<24}{pct(overall['answer_relevancy']):>10}")
    print(f"{'Answer Correctness':<24}{pct(overall['answer_correctness']):>10}")
    print(f"{'Error Rate':<24}{pct(merged['is_error'].mean()):>10}")

    # ------------------------------------------------------------------
    # BY ROUTE
    # ------------------------------------------------------------------

    print("\n===== BY ROUTE =====")

    route_means = merged.groupby("route")[metric_cols].mean(numeric_only=True)

    print(
        f"{'Route':<12}"
        f"{'Faithfulness':>16}"
        f"{'Relevancy':>16}"
        f"{'Correctness':>16}"
    )

    print("-" * 60)

    for route, row in route_means.iterrows():
        print(
            f"{route:<12}"
            f"{pct(row['faithfulness']):>16}"
            f"{pct(row['answer_relevancy']):>16}"
            f"{pct(row['answer_correctness']):>16}"
        )

    # ------------------------------------------------------------------
    # BY REQUIREMENT CATEGORY
    # ------------------------------------------------------------------

    print("\n===== BY REQUIREMENT CATEGORY (Top 10) =====")

    top_cats = merged["requirement_category"].value_counts().head(10)

    cat_groups = (
        merged[merged["requirement_category"].isin(top_cats.index)]
        .groupby("requirement_category")
    )

    cat_means = cat_groups[metric_cols].mean()
    cat_counts = cat_groups.size()

    CAT_WIDTH = 45

    print(
        f"{'Requirement Category':<{CAT_WIDTH}}"
        f"{'Count':>8}"
        f"{'Faithful':>14}"
        f"{'Relevant':>14}"
        f"{'Correct':>14}"
    )

    print("-" * (CAT_WIDTH + 50))

    for cat in top_cats.index:
        print(
            f"{rtl(cat):<{CAT_WIDTH}}"
            f"{cat_counts[cat]:>8}"
            f"{pct(cat_means.loc[cat, 'faithfulness']):>14}"
            f"{pct(cat_means.loc[cat, 'answer_relevancy']):>14}"
            f"{pct(cat_means.loc[cat, 'answer_correctness']):>14}"
        )

    # ------------------------------------------------------------------
    # WORST 5 QUESTIONS
    # ------------------------------------------------------------------

    print("\n===== WORST 5 BY ANSWER CORRECTNESS =====")

    worst = (
        merged.sort_values("answer_correctness")
        .head(5)
    )

    for _, row in worst.iterrows():
        print("-" * 80)
        print(f"ID            : {row['id']}")
        print(f"Route         : {row['route']}")
        print(f"Correctness   : {pct(row['answer_correctness'])}")
        print(f"Faithfulness  : {pct(row['faithfulness'])}")
        print(f"Question      : {row['question']}")

    print("-" * 80)

    print(f"\nSaved merged results to {out_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="One or more JSON files with DeepSeek's scores",
    )

    parser.add_argument(
        "--diagnostics",
        default="generation_diagnostics.csv",
    )

    parser.add_argument(
        "--out",
        default="generation_eval_final.csv",
    )

    args = parser.parse_args()

    main(
        args.scores,
        args.diagnostics,
        args.out,
    )