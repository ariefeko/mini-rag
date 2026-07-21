"""Helper untuk menampilkan informasi debug pada pipeline Mini-RAG."""

import os
from collections.abc import Sequence
from typing import Any, Callable


DEBUG_RAG = os.getenv("RAG_DEBUG", "true").lower() in {"1", "true", "yes", "on"}
DEBUG_VECTOR_PREVIEW = 10
DEBUG_CHUNK_LIMIT = 3


def debug_chunks_and_vectors(
    chunks: Sequence[Any],
    embeddings: Any,
    limit: int = DEBUG_CHUNK_LIMIT,
) -> None:
    """Tampilkan isi chunk dan preview embedding."""
    if not DEBUG_RAG:
        return

    print("\n=== DEBUG: CHUNKS & EMBEDDINGS ===")
    for index, chunk in enumerate(chunks[:limit], start=1):
        vector = embeddings.embed_query(chunk.page_content)

        print(f"\nChunk #{index}")
        print(f"Content   : {chunk.page_content}")
        print(f"Metadata  : {chunk.metadata}")
        print(f"Dimension : {len(vector)}")
        print(f"Vector    : {vector[:DEBUG_VECTOR_PREVIEW]}")

    if len(chunks) > limit:
        print(f"\n... {len(chunks) - limit} chunk lain tidak ditampilkan")

    print("==================================\n")


def debug_similarity_search(
    vectorstore: Any,
    question: str,
    prepare_question: Callable[[str], dict],
    k: int = 4,
) -> None:
    """Tampilkan query interpretasi, chunk terdekat, dan distance score."""
    if not DEBUG_RAG:
        return

    prepared = prepare_question(question)
    search_query = prepared["search_query"]
    results = vectorstore.similarity_search_with_score(search_query, k=k)

    print("\n=== DEBUG: VECTOR SEARCH ===")
    print(f"Original query : {question}")
    print(f"Search query   : {search_query}")
    print("Catatan        : score Chroma umumnya berupa distance; makin kecil makin dekat.")

    for index, (document, score) in enumerate(results, start=1):
        print(f"\nResult #{index}")
        print(f"Distance : {score:.6f}")
        print(f"Content  : {document.page_content}")
        print(f"Metadata : {document.metadata}")

    print("============================\n")