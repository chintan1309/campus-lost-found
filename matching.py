"""
matching.py — Cosine similarity engine for item matching.

Given a newly posted item, this module queries all items of the *opposite*
status (lost vs found) from the database, computes cosine similarity between
the new item's embedding and each candidate's embedding, and returns the top-N
matches sorted by descending similarity.
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import get_items_by_status


def find_top_matches(item_embedding: list, item_status: str, top_n: int = 5) -> list:
    """
    Find the top-N most similar items of the opposite status.

    Args:
        item_embedding: 1280-dim embedding of the newly uploaded item (list of floats).
        item_status:    'lost' or 'found' — the status of the NEW item.
        top_n:          Maximum number of matches to return (default 5).

    Returns:
        A list of dicts, each containing:
            - 'item':       the sqlite3.Row object for the matched item
            - 'score':      cosine similarity score (0.0 – 1.0)
            - 'percentage': human-readable percentage string e.g. "87.4%"
        Sorted by score descending.
    """
    # Determine which status pool to search in
    opposite_status = 'found' if item_status == 'lost' else 'lost'
    candidates = get_items_by_status(opposite_status)

    if not candidates:
        return []

    # Build matrix of candidate embeddings (skip items without embeddings)
    valid_candidates = []
    candidate_embeddings = []
    for row in candidates:
        if row['embedding']:
            emb = json.loads(row['embedding'])
            if emb:
                valid_candidates.append(row)
                candidate_embeddings.append(emb)

    if not valid_candidates:
        return []

    # Compute cosine similarity: query (1×1280) vs candidates (N×1280)
    query_vec = np.array(item_embedding, dtype=np.float32).reshape(1, -1)
    cand_matrix = np.array(candidate_embeddings, dtype=np.float32)

    scores = cosine_similarity(query_vec, cand_matrix)[0]  # shape: (N,)

    # Pair each candidate with its score and sort descending
    scored = sorted(
        zip(valid_candidates, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-N as list of dicts
    results = []
    for row, score in scored[:top_n]:
        results.append({
            'item': row,
            'score': round(score, 4),
            'percentage': f"{score * 100:.1f}%"
        })

    return results
