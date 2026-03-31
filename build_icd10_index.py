import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from app.services.chroma_service import ChromaVectorStore
from app.utils.embeddings import get_embedding


def _pick(row: dict[str, Any], *keys: str) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to ICD-10 CSV file",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "rag_data" / "icd10_faiss"),
        help="Output folder where the FAISS index + metadata will be stored",
    )
    parser.add_argument(
        "--batch-id",
        default="icd10_fy2026",
        help="Collection/batch id to store embeddings under",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="How many rows to embed before flushing to disk",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = ChromaVectorStore(persist_directory=str(out_dir))

    embedded = 0
    failed = 0

    chunk_ids: list[str] = []
    chunks: list[str] = []
    embeddings: list[np.ndarray] = []
    metadatas: list[dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            code = _pick(row, "CODE", "code")
            short_desc = _pick(
                row,
                "SHORT DESCRIPTION (VALID ICD-10 FY2026)",
                "SHORT DESCRIPTION",
                "short description",
                "short_description",
            )
            long_desc = _pick(
                row,
                "LONG DESCRIPTION (VALID ICD-10 FY2026)",
                "LONG DESCRIPTION",
                "long description",
                "long_description",
            )
            nf_excl = _pick(row, "NF EXCL", "nf excl", "NF_EXCL")

            if not code:
                failed += 1
                continue

            text = " ".join([p for p in [code, short_desc, long_desc] if p]).strip()
            if not text:
                failed += 1
                continue

            chunk_id = f"icd10_{code}"

            try:
                emb = get_embedding(text)
            except Exception as e:
                print(f"Failed embedding code={code}: {e}")
                failed += 1
                continue

            metadata: dict[str, Any] = {
                "source_type": "icd10_csv",
                "code": code,
                "short_description": short_desc,
                "long_description": long_desc,
                "nf_excl": nf_excl,
                "row_index": i,
                "source_file": str(csv_path),
            }

            chunk_ids.append(chunk_id)
            chunks.append(text)
            embeddings.append(emb)
            metadatas.append(metadata)
            embedded += 1

            if embedded % 25 == 0:
                print(f"Embedded {embedded} rows...")

            if len(chunk_ids) >= args.flush_every:
                store.add_chunks(
                    batch_id=args.batch_id,
                    chunk_ids=chunk_ids,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                chunk_ids.clear()
                chunks.clear()
                embeddings.clear()
                metadatas.clear()

    if chunk_ids:
        store.add_chunks(
            batch_id=args.batch_id,
            chunk_ids=chunk_ids,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print("=" * 60)
    print("ICD-10 indexing complete")
    print(f"Input CSV: {csv_path}")
    print(f"Output dir: {out_dir}")
    print(f"Batch id: {args.batch_id}")
    print(f"Embedded rows: {embedded}")
    print(f"Failed rows: {failed}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
