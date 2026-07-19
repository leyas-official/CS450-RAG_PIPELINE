"""
import_judge_scores.py

Merges DeepSeek's judging output (JSON, produced by pasting the
judge_batch_*.md files from evaluate_retriever.py into DeepSeek Chat)
with the local, non-LLM retrieval diagnostics, and prints/saves the
final route-broken-down retrieval evaluation.

Workflow:
    1. python evaluate_retriever.py --dataset functional_requirements_eval_set_merged.csv
       -> writes retriever_diagnostics.csv and judge_export/judge_batch_01.md, _02.md, ...
    2. Paste each judge_batch_XX.md into DeepSeek Chat. It should reply with
       a JSON array (see the instructions embedded at the top of each file).
       Save each reply as its own .json file (e.g. scores_01.json, scores_02.json),
       or combine them into one file — either works, this script accepts
       multiple --scores files and concatenates them.
    3. python import_judge_scores.py --scores scores_01.json scores_02.json ...

Output:
    retriever_eval_final.csv   — per-question diagnostics + judge scores
    printed overall + per-route breakdown
"""

import argparse
import json

import pandas as pd

# Must match HIT_AT_K_CUTOFFS in evaluate_retriever.py — these are the
# cutoffs retriever_diagnostics.csv has article_hit@k / actor_hit@k columns
# for. Kept in sync manually since the two scripts run as separate processes.
HIT_AT_K_CUTOFFS = [1, 3, 5, 10, 25]


def load_scores(paths: list[str]) -> list[dict]:
    all_scores = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            raw = f.read().strip()
        # Tolerate DeepSeek wrapping the array in ```json fences.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"{p} should contain a JSON array of per-question objects, got {type(data)}")
        all_scores.extend(data)
    return all_scores


def precision_from_verdicts(row) -> float | None:
    verdicts = row.get("precision_verdicts")
    if not verdicts:
        return None
    return sum(verdicts) / len(verdicts)


def main(score_paths: list[str], diagnostics_path: str, out_path: str):
    scores = load_scores(score_paths)
    scores_df = pd.DataFrame(scores)
    if "id" not in scores_df.columns:
        raise ValueError("Judge JSON is missing an 'id' field per question — can't merge.")
    scores_df["context_precision"] = scores_df.apply(precision_from_verdicts, axis=1)

    diag_df = pd.read_csv(diagnostics_path)
    diag_df["id"] = diag_df["id"].astype(str)
    scores_df["id"] = scores_df["id"].astype(str)

    merged = diag_df.merge(
        scores_df[["id", "context_precision", "context_recall", "entity_recall"]],
        on="id", how="left",
    )

    missing = merged[merged["context_recall"].isna()]
    if len(missing):
        print(f"[WARNING] {len(missing)} question(s) have no judge score — "
              f"ids: {missing['id'].tolist()}")

    dupes = scores_df["id"][scores_df["id"].duplicated()].tolist()
    if dupes:
        print(f"[WARNING] Duplicate ids in judge scores (kept first): {dupes}")

    merged.to_csv(out_path, index=False)

    metric_cols = ["context_precision", "context_recall", "entity_recall"]

    print("\n===== OVERALL =====")
    print(merged[metric_cols].mean(numeric_only=True))

    print("\n===== BY ROUTE (article / actor / else) =====")
    print(merged.groupby("route")[metric_cols].mean(numeric_only=True))

    print("\n===== BY REQUIREMENT CATEGORY (top 10 by count) =====")
    top_cats = merged["requirement_category"].value_counts().head(10).index
    print(merged[merged["requirement_category"].isin(top_cats)]
          .groupby("requirement_category")[metric_cols].mean(numeric_only=True))

    # --- Hit@k breakdown ---
    # retriever_diagnostics.csv (from the current evaluate_retriever.py) has
    # article_hit@k / actor_hit@k columns per cutoff instead of the old
    # single article_ref_hit / actor_ref_hit columns. This shows, for each
    # cutoff, what fraction of applicable questions found the right article
    # (on the "article" route) or actor (on the "actor" route) within the
    # top k retrieved chunks. .dropna() excludes questions with no
    # ground-truth reference to check, so they don't skew the average.
    article_hit_cols = [f"article_hit@{k}" for k in HIT_AT_K_CUTOFFS if f"article_hit@{k}" in merged.columns]
    actor_hit_cols = [f"actor_hit@{k}" for k in HIT_AT_K_CUTOFFS if f"actor_hit@{k}" in merged.columns]

    if article_hit_cols:
        print("\n===== article_hit@k rate (article-route questions) =====")
        article_route = merged[merged["route"] == "article"]
        for col in article_hit_cols:
            rate = article_route[col].dropna()
            print(f"  {col}: {rate.mean():.1%}" if len(rate) else f"  {col}: n/a (no applicable rows)")

    if actor_hit_cols:
        print("\n===== actor_hit@k rate (actor-route questions) =====")
        actor_route = merged[merged["route"] == "actor"]
        for col in actor_hit_cols:
            rate = actor_route[col].dropna()
            print(f"  {col}: {rate.mean():.1%}" if len(rate) else f"  {col}: n/a (no applicable rows)")

    if not article_hit_cols and not actor_hit_cols:
        # Backward-compat fallback, in case an older-format diagnostics CSV
        # (with the original single-column article_ref_hit/actor_ref_hit) is
        # passed in instead.
        if "article_ref_hit" in merged.columns and merged["article_ref_hit"].notna().any():
            print("\narticle_ref_hit rate (article-route questions):",
                  merged.loc[merged["route"] == "article", "article_ref_hit"].mean())
        if "actor_ref_hit" in merged.columns and merged["actor_ref_hit"].notna().any():
            print("actor_ref_hit rate (actor-route questions):",
                  merged.loc[merged["route"] == "actor", "actor_ref_hit"].mean())

    print(f"\nSaved merged results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True,
                         help="One or more JSON files with DeepSeek's scores")
    parser.add_argument("--diagnostics", default="retriever_diagnostics.csv")
    parser.add_argument("--out", default="retriever_eval_final.csv")
    args = parser.parse_args()
    main(args.scores, args.diagnostics, args.out)