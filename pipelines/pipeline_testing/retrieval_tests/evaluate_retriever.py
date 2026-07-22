"""
evaluate_retriever.py

Runs ONLY the retriever (never calls the generation LLM) over your eval
CSV, computes cheap non-LLM diagnostics, and exports "judge batches" —
Markdown files you paste into DeepSeek Chat (or any chat LLM) to score
context precision / recall / entity recall. No local judge model needed.

Reads the eval set from a CSV with columns:
    id, route, source_id, actor_reference, article_reference,
    requirement_category, question, expected_answer

Produces:
    retriever_diagnostics.csv        — per-question, non-LLM stats
    judge_export/judge_batch_XX.md   — paste each into DeepSeek Chat

After grading, see import_judge_scores.py to merge DeepSeek's scores
back in and get the final route-by-route breakdown.

Usage:
    python evaluate_retriever.py --dataset ../eval_test_set.csv
"""

import argparse
import os
import sys
import asyncio
from pathlib import Path

import pandas as pd

_here = Path(__file__).resolve()
_rag_pipeline_dir = None
for _parent in _here.parents:
    if (_parent / "RAG_pipeline.py").is_file():
        _rag_pipeline_dir = _parent
        break

if _rag_pipeline_dir is None:
    raise RuntimeError(
        "Could not find RAG_pipeline.py by walking up from this script's location — "
        "check the folder structure hasn't moved."
    )

sys.path.insert(0, str(_rag_pipeline_dir))          # lets bare `pipeline_core`, `pipeline_utils` import
sys.path.insert(0, str(_rag_pipeline_dir.parent))   # lets `pipelines.RAG_PIPELINE`, `pipelines.pipeline_core` import

from pipelines.RAG_pipeline import Pipeline


def load_eval_questions(path: str) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"question", "expected_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{path} is empty — add at least one eval question.")
    return df.to_dict(orient="records")


def _digits(s) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit()) if s else ""


def article_ref_hit_at_k(nodes, article_reference, k: int) -> bool | None:
    """Best-effort, non-LLM check: does the ground-truth article number show
    up in the TOP-k retrieved chunks (by rank order)?

    NOTE: this is a hit-rate / recall-style check ("is the right info in
    there somewhere within the top k?"), NOT precision. True Precision@k —
    "of the top k chunks, what fraction are relevant?" — needs a relevance
    judgment for every single chunk, which only the DeepSeek judge step
    (import_judge_scores.py, using precision_verdicts) can give you. This
    function just tells you HOW FAR DOWN the ranked list you have to look
    before the right article shows up, which is exactly the "is it buried
    past position 25?" question.
    """
    target = _digits(article_reference)
    if not target:
        return None
    for n in nodes[:k]:
        parsed = n.metadata.get("article_numbers") or []
        if any(_digits(p) == target for p in parsed):
            return True
        if target in n.text:
            return True
    return False


def actor_ref_hit_at_k(nodes, actor_reference, k: int) -> bool | None:
    """Same idea as article_ref_hit_at_k, but for the ground-truth actor(s),
    checked only within the TOP-k retrieved chunks."""
    if not actor_reference or not isinstance(actor_reference, str):
        return None
    candidates = [a.strip() for a in actor_reference.split("/") if a.strip()]
    if not candidates:
        return None
    for n in nodes[:k]:
        actors_meta = " ".join(str(a) for a in (n.metadata.get("actors") or []))
        blob = n.text + " " + actors_meta
        if any(c in blob for c in candidates):
            return True
    return False


# Cutoffs to report Hit@k for. Chosen to span "does the reranker put it right
# at the top" (k=1) through "is it anywhere in the full retrieved set" (k=25,
# matching the retrieve() top_k cap in RAG_PIPELINE.py).
HIT_AT_K_CUTOFFS = [1, 3, 5, 10, 25]


JUDGE_INSTRUCTIONS = """# Retrieval Evaluation — Judge Instructions

You are grading a document retrieval system for an Arabic legal/regulatory RAG pipeline.
For each question below you are given:
- The question
- A reference answer (assume it is correct — the ground truth)
- A numbered list of chunks the retriever actually returned, in rank order

For EACH question, return:
1. "precision_verdicts": one 0 or 1 per retrieved chunk, SAME ORDER as shown below —
   1 if that chunk contains information actually used in the reference answer, 0 if it's irrelevant/noise.
2. "context_recall": float 0.0-1.0 — what fraction of the information in the reference answer
   is actually present somewhere among the retrieved chunks? (1.0 = everything needed is there.)
3. "entity_recall": float 0.0-1.0 — of the key entities in the reference answer (مادة/article
   numbers, actor/authority names, deadlines, amounts), what fraction appear somewhere among
   the retrieved chunks?

Return ONLY a JSON array, one object per question, no prose outside it, like:
[
  {"id": 1, "precision_verdicts": [1,0,1,1,0], "context_recall": 0.8, "entity_recall": 1.0},
  {"id": 2, "precision_verdicts": [0,1,0], "context_recall": 0.3, "entity_recall": 0.5}
]
"""


def format_question_block(row: dict, nodes, top_n: int) -> str:
    lines = [
        f"## Question (id={row.get('id')}, route={row.get('route')}, "
        f"article_reference={row.get('article_reference')}, actor_reference={row.get('actor_reference')})",
        "",
        f"**Question:** {row['question']}",
        "",
        "**Reference answer (ground truth):**",
        str(row.get("expected_answer", "")),
        "",
        "**Retrieved chunks (rank order):**",
    ]
    for i, n in enumerate(nodes[:top_n], start=1):
        lines.append(f"{i}. {n.text}")
    lines.append("\n---\n")
    return "\n".join(lines)


def main(dataset_path: str, batch_size: int, top_n_contexts: int, out_dir: str):
    pipeline = Pipeline()
    asyncio.run(pipeline.on_startup())  # loads/indexes docs — needs Ollama + ./data reachable

    rows = load_eval_questions(dataset_path)

    diagnostics = []
    blocks = []
    for row in rows:
        nodes, _ = pipeline.retrieve(row["question"])

        row_diagnostics = {
            "id": row.get("id"),
            "route": row.get("route"),
            "requirement_category": row.get("requirement_category"),
            "article_reference": row.get("article_reference"),
            "actor_reference": row.get("actor_reference"),
            "question": row.get("question"),
            "expected_answer": row.get("expected_answer"),
            "num_retrieved": len(nodes),
        }

        # One column per cutoff, e.g. article_hit@1, article_hit@3, ...
        # Reading across a row tells you exactly where in the ranking the
        # right chunk shows up: e.g. False, False, True, True, True means
        # it first appears somewhere between rank 4 and rank 5.
        for k in HIT_AT_K_CUTOFFS:
            row_diagnostics[f"article_hit@{k}"] = article_ref_hit_at_k(
                nodes, row.get("article_reference"), k
            )
            row_diagnostics[f"actor_hit@{k}"] = actor_ref_hit_at_k(
                nodes, row.get("actor_reference"), k
            )

        diagnostics.append(row_diagnostics)
        blocks.append(format_question_block(row, nodes, top_n_contexts))

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv("retriever_diagnostics.csv", index=False)
    print(f"Wrote retriever_diagnostics.csv ({len(diagnostics)} questions)")

    # --- Aggregate Hit@k summary ---
    # For each cutoff k, what fraction of questions (that HAD a ground-truth
    # article/actor to check against) found it within the top k? This is the
    # "curve" that answers: is the retriever's problem that it never finds
    # the right chunk (flat/low across all k), or that it finds it but buries
    # it below where the reranker/top_k cutoff would surface it to the LLM
    # (low at k=1/3, high at k=25)?
    print("\n=== Hit@k summary (fraction of applicable questions with a hit) ===")
    print(f"{'k':>4} | {'article_hit@k':>14} | {'actor_hit@k':>12}")
    for k in HIT_AT_K_CUTOFFS:
        article_col = diagnostics_df[f"article_hit@{k}"]
        actor_col = diagnostics_df[f"actor_hit@{k}"]

        # dropna() excludes questions with no ground-truth article/actor to
        # check (those came back as None, not False) so they don't drag the
        # average down artificially.
        article_applicable = article_col.dropna()
        actor_applicable = actor_col.dropna()

        article_rate = article_applicable.mean() if len(article_applicable) else float("nan")
        actor_rate = actor_applicable.mean() if len(actor_applicable) else float("nan")

        print(f"{k:>4} | {article_rate:>13.1%} | {actor_rate:>11.1%}")
    print("=" * 40)

    os.makedirs(out_dir, exist_ok=True)
    for batch_idx, start in enumerate(range(0, len(blocks), batch_size), start=1):
        batch_blocks = blocks[start:start + batch_size]
        content = JUDGE_INSTRUCTIONS + "\n---\n\n" + "\n".join(batch_blocks)
        out_path = os.path.join(out_dir, f"judge_batch_{batch_idx:02d}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote {out_path} ({len(batch_blocks)} questions)")

    print(f"\nNext: paste each {out_dir}/judge_batch_XX.md into DeepSeek Chat, "
          f"save its JSON reply, then run import_judge_scores.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../eval_test_set.csv")
    parser.add_argument("--batch-size", type=int, default=5, help="Questions per judge file")
    parser.add_argument("--top-n-contexts", type=int, default=20, help="Max retrieved chunks shown to the judge per question")
    parser.add_argument("--out-dir", default="judge_export")
    args = parser.parse_args()
    main(args.dataset, args.batch_size, args.top_n_contexts, args.out_dir)