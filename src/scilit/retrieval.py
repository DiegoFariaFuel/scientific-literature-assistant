from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .corpus import DEFAULT_EMBEDDER

TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridRetriever:
    def __init__(self, index_directory: str | Path, reranker_model: str | None = None) -> None:
        root = Path(index_directory)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.embed_model_name = self.manifest.get("embedding_model", DEFAULT_EMBEDDER)
        with (root / "chunks.jsonl").open(encoding="utf-8") as stream:
            self.chunks = [json.loads(line) for line in stream if line.strip()]
        self.embeddings = np.load(root / "embeddings.npy", allow_pickle=False)
        if len(self.chunks) != len(self.embeddings) or not self.chunks:
            raise ValueError("index metadata and embedding files are inconsistent or empty")
        self.embedder = SentenceTransformer(self.embed_model_name, trust_remote_code=False)
        self.bm25 = BM25Okapi([_tokens(chunk["text"]) for chunk in self.chunks])
        self.reranker = None
        if reranker_model:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model, trust_remote_code=False)

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 30) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        pool = min(max(candidate_pool, top_k), len(self.chunks))
        query_vector = self.embedder.encode([f"query: {query}"], normalize_embeddings=True,
                                            convert_to_numpy=True)[0]
        dense_scores = self.embeddings @ query_vector
        dense_rank = np.argsort(-dense_scores)[:pool]
        lexical_scores = np.asarray(self.bm25.get_scores(_tokens(query)))
        lexical_rank = np.argsort(-lexical_scores)[:pool]

        # Reciprocal-rank fusion puts lexical and semantic rankings on a comparable scale.
        fused: dict[int, float] = {}
        for ranking in (dense_rank, lexical_rank):
            for rank, index in enumerate(ranking, start=1):
                fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (60 + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:pool]
        if self.reranker is not None:
            pairs = [[query, self.chunks[index]["text"]] for index in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            candidates = [candidates[i] for i in np.argsort(-np.asarray(scores))]
        return [{**self.chunks[index], "retrieval_score": float(fused[index])} for index in candidates[:top_k]]

