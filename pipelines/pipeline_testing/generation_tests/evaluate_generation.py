"""
evaluate_generation.py

Runs the FULL pipeline (retrieval + generation, via pipeline.generate())
over your eval CSV, computes cheap non-LLM diagnostics, and exports
"judge batches" — Markdown files you paste into DeepSeek Chat to score
faithfulness, answer relevancy, and answer correctness. No paid API,
same workflow as evaluate_retriever.py.

Reads the eval set from a CSV with columns:
    id, route, source_id, actor_reference, article_reference,
    requirement_category, question, expected_answer

Produces:
    generation_diagnostics.csv           — per-question, non-LLM stats
    judge_export_gen/judge_batch_XX.md   — paste each into DeepSeek Chat

After grading, see import_generation_scores.py to merge DeepSeek's
scores back in and get the final route-by-route breakdown.

Usage:
    python evaluate_generation.py --dataset ../functional_requirements_eval_set_merged.csv
"""

import argparse
import asyncio
import os
import sys
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

ERROR_PREFIX = "خطأ أثناء المعالجة"


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


def article_ref_in_text(text: str, article_reference) -> bool | None:
    target = _digits(article_reference)
    if not target:
        return None
    return target in text


def actor_ref_in_text(text: str, actor_reference) -> bool | None:
    if not actor_reference or not isinstance(actor_reference, str):
        return None
    candidates = [a.strip() for a in actor_reference.split("/") if a.strip()]
    if not candidates:
        return None
    return any(c in text for c in candidates)


JUDGE_INSTRUCTIONS = """# Generation Evaluation — Judge Instructions

You are grading the FINAL ANSWER produced by a RAG system for an Arabic legal/regulatory
pipeline (NOT the retrieval step — the answer generation). For each question below you
are given:
- The question
- The retrieved context chunks the system had available when generating (rank order)
- The reference answer (assume it is correct — the ground truth)
- The system's generated answer

For EACH question, return:
1. "faithfulness_verdicts": break the GENERATED ANSWER into its individual factual
   claims/statements, and for each one say whether it is supported by the retrieved
   context, as an array of 0/1 (one entry per claim you identify — the array length
   will vary per question, that's fine). A claim not backed by any retrieved chunk is
   a hallucination and gets 0.
2. "answer_relevancy": float 0.0-1.0 — does the answer actually address what was asked?
   Penalize evasive, generic, or off-topic answers, even if factually fine.
3. "answer_correctness": float 0.0-1.0 — how well does the generated answer match the
   reference answer on the essential facts (correct article/مادة number, correct actor,
   correct condition or action described)? 1.0 = fully correct and complete, 0.0 = wrong
   or contradicts the reference. If the system returned an error message instead of an
   answer, score all three metrics 0.0.

Return ONLY a JSON array, one object per question, no prose outside it, like:
[
  {"id": 1, "faithfulness_verdicts": [1,1,0], "answer_relevancy": 0.9, "answer_correctness": 0.8},
  {"id": 2, "faithfulness_verdicts": [1], "answer_relevancy": 0.4, "answer_correctness": 0.2}
]
"""


def format_question_block(row: dict, contexts: list[str], answer: str, top_n: int) -> str:
    lines = [
        f"## Question (id={row.get('id')}, route={row.get('route')}, "
        f"article_reference={row.get('article_reference')}, actor_reference={row.get('actor_reference')})",
        "",
        f"**Question:** {row['question']}",
        "",
        "**Retrieved context (rank order, what the system had available):**",
    ]
    for i, ctx in enumerate(contexts[:top_n], start=1):
        lines.append(f"{i}. {ctx}")
    lines += [
        "",
        "**Reference answer (ground truth):**",
        str(row.get("expected_answer", "")),
        "",
        "**Generated answer (system output — grade this):**",
        answer,
        "\n---\n",
    ]
    return "\n".join(lines)


def main(dataset_path: str, batch_size: int, top_n_contexts: int, out_dir: str):
    pipeline = Pipeline()
    asyncio.run(pipeline.on_startup())  # loads/indexes docs — needs Ollama + ./data reachable

    rows = load_eval_questions(dataset_path)

    diagnostics = []
    blocks = []
    for row in rows:
        answer, contexts = pipeline.generate(row["question"])
        is_error = answer.strip().startswith(ERROR_PREFIX)

        diagnostics.append({
            "id": row.get("id"),
            "route": row.get("route"),
            "requirement_category": row.get("requirement_category"),
            "article_reference": row.get("article_reference"),
            "actor_reference": row.get("actor_reference"),
            "question": row.get("question"),
            "expected_answer": row.get("expected_answer"),
            "generated_answer": answer,
            "is_error": is_error,
            "answer_word_count": len(answer.split()),
            "article_ref_in_response": article_ref_in_text(answer, row.get("article_reference")),
            "actor_ref_in_response": actor_ref_in_text(answer, row.get("actor_reference")),
        })
        blocks.append(format_question_block(row, contexts, answer, top_n_contexts))

    pd.DataFrame(diagnostics).to_csv("generation_diagnostics.csv", index=False)
    print(f"Wrote generation_diagnostics.csv ({len(diagnostics)} questions)")

    n_errors = sum(d["is_error"] for d in diagnostics)
    if n_errors:
        print(f"[WARNING] {n_errors} question(s) errored out during generation — "
              f"check generation_diagnostics.csv for details before judging.")

    os.makedirs(out_dir, exist_ok=True)
    for batch_idx, start in enumerate(range(0, len(blocks), batch_size), start=1):
        batch_blocks = blocks[start:start + batch_size]
        content = JUDGE_INSTRUCTIONS + "\n---\n\n" + "\n".join(batch_blocks)
        out_path = os.path.join(out_dir, f"judge_batch_{batch_idx:02d}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote {out_path} ({len(batch_blocks)} questions)")

    print(f"\nNext: paste each {out_dir}/judge_batch_XX.md into DeepSeek Chat, "
          f"save its JSON reply, then run import_generation_scores.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../functional_requirements_eval_set_merged.csv")
    parser.add_argument("--batch-size", type=int, default=5, help="Questions per judge file")
    parser.add_argument("--top-n-contexts", type=int, default=20,
                         help="Max retrieved chunks shown to the judge per question (keep small — files now also include the answer)")
    parser.add_argument("--out-dir", default="judge_export_gen")
    args = parser.parse_args()
    main(args.dataset, args.batch_size, args.top_n_contexts, args.out_dir)
