from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
DEFAULT_EMBEDDER = "intfloat/multilingual-e5-small"


def _chunks(text: str, words_per_chunk: int = 180, overlap: int = 35) -> list[str]:
    words = text.split()
    result: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        value = " ".join(words[start:end]).strip()
        if len(value) >= 80:
            result.append(value)
        if end == len(words):
            break
        start = end - overlap
    return result


def _read_pdf(path: Path) -> list[dict[str, Any]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    sidecar = path.with_suffix(".json")
    extra = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    with pymupdf.open(path) as pdf:
        pdf_title = (pdf.metadata or {}).get("title") or ""
        first_pages = "\n".join(pdf[i].get_text(sort=True) for i in range(min(2, len(pdf))))
        title = str(extra.get("title") or pdf_title or path.stem)
        doi_match = DOI_PATTERN.search(first_pages)
        doi = str(extra.get("doi") or (doi_match.group(0).rstrip(".,;)") if doi_match else ""))
        year = extra.get("year")
        chunks: list[dict[str, Any]] = []
        for page_number, page in enumerate(pdf, start=1):
            for chunk_number, text in enumerate(_chunks(page.get_text(sort=True))):
                chunks.append({
                    "chunk_id": f"{digest}:p{page_number}:c{chunk_number}",
                    "source_id": digest,
                    "title": title,
                    "doi": doi,
                    "year": year,
                    "page": page_number,
                    "text": text,
                })
    return chunks


def build_index(pdf_directory: str | Path, index_directory: str | Path,
                embed_model: str = DEFAULT_EMBEDDER) -> int:
    root = Path(pdf_directory)
    files = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    if not files:
        raise ValueError(f"no PDF files found under {root}")
    chunks = [chunk for path in files for chunk in _read_pdf(path)]
    if not chunks:
        raise ValueError("the PDFs contain no extractable text; OCR scanned documents before indexing")

    model = SentenceTransformer(embed_model, trust_remote_code=False)
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    output = Path(index_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    manifest = {"embedding_model": embed_model, "chunk_count": len(chunks),
                "paper_count": len({chunk['source_id'] for chunk in chunks}),
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return len(chunks)

