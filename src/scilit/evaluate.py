from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retrieval import HybridRetriever


def evaluate_retrieval(retriever: HybridRetriever, cases_path: str | Path, top_k: int = 5) -> dict[str, Any]:
    cases = [json.loads(line) for line in Path(cases_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("evaluation file must contain at least one JSONL case")
    recall, reciprocal_rank = [], []
    for case in cases:
        relevant = set(case["relevant_source_ids"])
        if not relevant:
            raise ValueError("each evaluation case needs at least one relevant_source_id")
        results = retriever.retrieve(case["query"], top_k=top_k)
        ranks = [rank for rank, result in enumerate(results, start=1) if result["source_id"] in relevant]
        recall.append(len(ranks) / len(relevant))
        reciprocal_rank.append(1.0 / ranks[0] if ranks else 0.0)
    return {"case_count": len(cases), "k": top_k,
            "recall_at_k": sum(recall) / len(recall), "mean_reciprocal_rank": sum(reciprocal_rank) / len(reciprocal_rank),
            "metric_scope": "retrieval only; this does not evaluate answer factuality or scientific validity"}

