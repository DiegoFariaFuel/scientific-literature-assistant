from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .answer import answer_question
from .corpus import DEFAULT_EMBEDDER, build_index
from .evaluate import evaluate_retrieval
from .retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="scilit", description="Local bilingual scientific literature assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="index researcher-provided PDF files")
    ingest.add_argument("--pdf-dir", default="data/papers")
    ingest.add_argument("--index-dir", default="index")
    ingest.add_argument("--embedding-model", default=os.environ.get("SCILIT_EMBED_MODEL", DEFAULT_EMBEDDER))
    ask = commands.add_parser("ask", help="answer a question with cited local evidence")
    ask.add_argument("question")
    ask.add_argument("--index-dir", default="index")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--reranker", default=os.environ.get("SCILIT_RERANK_MODEL"))
    ask.add_argument("--llm", default=os.environ.get("OLLAMA_MODEL"))
    evaluate = commands.add_parser("evaluate", help="measure retrieval recall and reciprocal rank")
    evaluate.add_argument("--cases", required=True, help="JSONL with query and relevant_source_ids")
    evaluate.add_argument("--index-dir", default="index")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--reranker", default=os.environ.get("SCILIT_RERANK_MODEL"))
    args = parser.parse_args()

    if args.command == "ingest":
        count = build_index(args.pdf_dir, args.index_dir, args.embedding_model)
        print(json.dumps({"indexed_chunks": count, "index_directory": str(Path(args.index_dir))}, indent=2))
    elif args.command == "ask":
        result = answer_question(HybridRetriever(args.index_dir, args.reranker), args.question,
                                 top_k=args.top_k, ollama_model=args.llm)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "evaluate":
        metrics = evaluate_retrieval(HybridRetriever(args.index_dir, args.reranker), args.cases, args.top_k)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

