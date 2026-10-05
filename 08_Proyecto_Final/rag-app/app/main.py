from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.chunk import chunk_corpus
from rag.data import Corpus, document_from_text
from rag.embed import embed_text, embed_texts
from rag.store import add_chunks, get_collection, search
import httpx
from google.genai import errors

from rag.generate import NO_EVIDENCE, generate_answer

ABSTAIN_MESSAGE = (
    "No tengo evidencia suficiente en los documentos para responder. "
    "Este sistema solo responde sobre cámaras y fotografía con base en los "
    "documentos cargados."
)

ROOT = Path(__file__).resolve().parent.parent
MIN_SCORE = 0.65
DEFAULT_TOP_K = 4
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
ALLOWED_EXTENSIONS = {".md", ".txt"}

app = FastAPI(title="RAG de cámaras")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)
collection = get_collection(path=str(ROOT / "chroma"))


class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class Citation(BaseModel):
    id: str
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        results = []
        if collection.count() > 0:
            results = search(collection, embed_text(question), req.top_k)
        relevant = [r for r in results if r["score"] >= MIN_SCORE]
        answer = generate_answer(question, relevant) if relevant else NO_EVIDENCE
    except (errors.APIError, httpx.TimeoutException) as e:
        if getattr(e, "code", None) == 429:
            detail = "Se agotó la cuota de Google AI para hoy. Intenta más tarde."
        else:
            detail = "El servicio de Google AI no está disponible. Intenta de nuevo en unos segundos."
        raise HTTPException(status_code=502, detail=detail)

    if answer == NO_EVIDENCE:
        return QueryResponse(answer=ABSTAIN_MESSAGE, citations=[], abstained=True)

    return QueryResponse(
        answer=answer,
        citations=[Citation(**r) for r in relevant],
        abstained=False,
    )

@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...)):
    docs, skipped = [], []
    for f in files:
        name = f.filename or ""
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            skipped.append(name)
            continue
        text = f.file.read().decode("utf-8", errors="ignore").strip()
        if not text:
            skipped.append(name)
            continue
        docs.append(document_from_text(name, text))

    if not docs:
        raise HTTPException(
            status_code=400, detail="No se recibió ningún archivo .md o .txt con contenido."
        )

    chunks = chunk_corpus(Corpus(documents=docs), CHUNK_SIZE, CHUNK_OVERLAP)
    add_chunks(collection, chunks, embed_texts([c.text for c in chunks]))
    return {"documents": len(docs), "chunks": len(chunks), "skipped": skipped}