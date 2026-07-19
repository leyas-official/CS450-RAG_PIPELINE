"""
pipeline_utils/utils.py — actor_matches, rewritten to use embedding
similarity instead of substring matching.

Reuses Settings.embed_model, which is already configured once in
Pipeline.on_startup() (OllamaEmbedding, per your .env's
LLAMAINDEX_EMBEDDING_MODEL_NAME). No new model to load, no signature
change needed at call sites in RAG_PIPELINE.py.
"""

import numpy as np
from llama_index.core import Settings

# In-process cache: actor string -> embedding vector. Actor phrases repeat
# heavily across chunks (the same committee/entity is named in dozens of
# articles), and actor_matches() gets called once per node in retrieve()'s
# filtering step — without this cache, the same phrase gets re-embedded via
# an Ollama HTTP call every single time it's checked against every node, on
# every query. This cache makes each unique phrase cost one embedding call
# for the lifetime of the process, not per (node, query) pair.
_actor_embedding_cache: dict[str, list[float]] = {}


def _get_actor_embedding(actor: str) -> list[float]:
    if actor not in _actor_embedding_cache:
        _actor_embedding_cache[actor] = Settings.embed_model.get_text_embedding(actor)
    return _actor_embedding_cache[actor]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def actor_matches(chunk_actors: list, query_actors: list, threshold: float = 0.85) -> bool:
    """
    Embedding-similarity match instead of substring match. Two actor
    phrases match if their cosine similarity clears `threshold`, using
    whichever embedding model Pipeline.on_startup() already configured
    on Settings.embed_model.

    threshold is a starting guess (0.85), not a validated number — tune it
    against your actor-route eval questions (see note below).
    """
    if not chunk_actors or not query_actors:
        return False

    query_embeds = [_get_actor_embedding(qa) for qa in query_actors]
    chunk_embeds = [_get_actor_embedding(ca) for ca in chunk_actors]

    for qe in query_embeds:
        for ce in chunk_embeds:
            if _cosine_similarity(qe, ce) >= threshold:
                return True
    return False