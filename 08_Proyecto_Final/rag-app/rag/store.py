import chromadb

from rag.chunk import Chunk


def get_collection(path: str = "chroma", name: str = "camaras"):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=name, configuration={"hnsw": {"space": "cosine"}}
    )


def add_chunks(collection, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    collection.upsert(
        ids=[f"{c.source}::{c.index}" for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": c.source, "title": c.doc_title, "index": c.index} for c in chunks
        ],
    )


def search(collection, query_vector: list[float], top_k: int = 3) -> list[dict]:
    res = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return [
        {
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "score": 1 - res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]