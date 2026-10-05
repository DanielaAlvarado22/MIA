import logging
import time

import httpx
from google.genai import errors, types

from rag.client import client

logger = logging.getLogger("uvicorn.error")

# Orden de preferencia: si un modelo no responde, se prueba el siguiente.
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
TOTAL_BUDGET_S = 100  # tras este tiempo total no se prueban más modelos
SKIP_MODEL_CODES = {404, 429}  # modelo no disponible o cuota agotada: pasar al siguiente
NO_EVIDENCE = "NO_EVIDENCE"

SYSTEM = f"""Eres un asistente que responde preguntas sobre cámaras y fotografía \
usando EXCLUSIVAMENTE la evidencia numerada que se te proporciona.

Reglas:
1. Responde siempre en español, de forma clara y concisa.
2. Usa solo información que aparezca en la evidencia. No uses conocimiento propio \
ni inventes datos, cifras o nombres.
3. Cita la evidencia con su número entre corchetes, por ejemplo [1] o [2][3], \
al final de cada afirmación que apoyes en ella.
4. Ignora los fragmentos que no tengan relación con la pregunta (listas de enlaces, \
secciones "Véase también", etc.).
5. Si la evidencia no contiene la respuesta a la pregunta, responde exactamente \
{NO_EVIDENCE} y nada más."""


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return NO_EVIDENCE

    evidence = "\n\n".join(
        f"[{i}] (fuente: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )
    prompt = f"Evidencia:\n\n{evidence}\n\nPregunta: {question}"

    deadline = time.monotonic() + TOTAL_BUDGET_S
    last_error: Exception | None = None
    for model in MODELS:
        if last_error is not None and time.monotonic() > deadline:
            raise last_error
        try:
            response = _call_model(model, prompt)
            logger.info("Respuesta generada con %s", model)
            break
        except (errors.ServerError, httpx.TimeoutException) as e:
            last_error = e
        except errors.ClientError as e:
            if e.code not in SKIP_MODEL_CODES:
                raise
            last_error = e
        logger.warning(
            "Modelo %s no disponible (%s); probando el siguiente.",
            model,
            getattr(last_error, "code", "timeout"),
        )
    else:
        raise last_error

    return (response.text or "").strip() or NO_EVIDENCE


def _call_model(model: str, prompt: str):
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0),
    )
