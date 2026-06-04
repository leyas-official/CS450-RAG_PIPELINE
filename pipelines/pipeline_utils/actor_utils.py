"""
Actor extraction utilities used during pipeline startup.
Responsible for running the LLM-based actor extraction pass over all chunks.
"""

from typing import List
from pipelines.pipeline_core import Chunk
from pipelines.pipeline_core.prompts import ACTOR_EXTRACTION_PROMPT


def extract_actors_from_chunks(chunks: List[Chunk], llm) -> None:
    print(f"Extracting actors from {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            response = llm.complete(
                ACTOR_EXTRACTION_PROMPT.format(text=chunk.text)
            )
            raw = response.text.strip()

            if "لا يوجد" in raw:
                chunk.actors = []
            else:
                chunk.actors = [
                    line.strip()
                    for line in raw.splitlines()
                    if line.strip()
                ]

            print(f"  [{i + 1}/{len(chunks)}] {chunk.chunk_id} → actors: {chunk.actors}")

        except Exception as e:
            print(f"  [ERROR] {chunk.chunk_id}: {e}")
            chunk.actors = []

    print("Actor extraction complete.")
