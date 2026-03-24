"""Build Chroma index from Hugging Face dataset using Ollama embeddings."""

from __future__ import annotations

import shutil
import httpx
from datasets import load_dataset
from tqdm import tqdm

from rag.chroma_client import persistent_chroma_client
from rag.config import Settings, load_settings
from rag.ollama_api import embed_text


def clean_output(text: str) -> str:
    for token in ("<|im_end|>", "<|im_end|>", "<|im_end|>", "<|im_end|"):
        text = text.replace(token, "")
    return text.strip()


def build_document(instruction: str, output: str) -> str:
    q = instruction.strip()
    a = clean_output(output)
    return f"Question: {q}\nAnswer: {a}"


def build_index(
    settings: Settings,
    *,
    limit: int | None = None,
    http_client: httpx.Client | None = None,
) -> int:
    """
    Load dataset, embed each row with Ollama, persist to Chroma.
    Returns number of rows indexed.
    """
    ds = load_dataset(settings.dataset_name, split=settings.dataset_split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    chroma_path = settings.chroma_path
    chroma_path.parent.mkdir(parents=True, exist_ok=True)

    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    client_chroma = persistent_chroma_client(chroma_path)
    collection = client_chroma.create_collection(
        name=settings.collection_name,
        metadata={
            "embed_model": settings.embed_model,
            "dataset": settings.dataset_name,
            "split": settings.dataset_split,
        },
    )

    own_client = http_client is None
    http = http_client or httpx.Client()

    try:
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for i in tqdm(range(len(ds)), desc="Embedding", unit="row"):
            row = ds[i]
            instruction = row["instruction"]
            output = row["output"]
            doc = build_document(instruction, output)
            emb = embed_text(http, settings.ollama_host, settings.embed_model, doc)

            ids.append(str(i))
            embeddings.append(emb)
            documents.append(doc)
            instr_preview = instruction.strip()
            if len(instr_preview) > 400:
                instr_preview = instr_preview[:397] + "..."
            metadatas.append({"row_id": str(i), "instruction_preview": instr_preview})

        if not ids:
            return 0

        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
    finally:
        if own_client:
            http.close()

    return len(ids)


def main() -> None:
    settings = load_settings()
    n = build_index(settings)
    print(f"Indexed {n} documents into {settings.chroma_path}")


if __name__ == "__main__":
    main()
