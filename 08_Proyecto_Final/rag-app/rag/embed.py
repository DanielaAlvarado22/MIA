from rag.client import client

MODEL = "gemini-embedding-001"


def embed_text(text: str, model: str = MODEL) -> list[float]:
    result = client.models.embed_content(model=model, contents=text)
    return list(result.embeddings[0].values)


def embed_texts(texts: list[str], model: str = MODEL, batch_size: int = 20) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(model=model, contents=batch)
        vectors.extend(list(e.values) for e in result.embeddings)
    return vectors