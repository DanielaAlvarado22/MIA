# RAG de cámaras y fotografía

Sistema de **generación aumentada por recuperación (RAG)** que responde preguntas
sobre cámaras y fotografía usando solo la evidencia de un conjunto de documentos,
con citas `[n]`, y que **se abstiene** cuando los documentos no tienen la respuesta.

```
incrustar -> indexar -> recuperar top-k -> generar respuesta anclada
```

## Arquitectura

```
Usuario
  └── Streamlit (puerto 8501)
        └── HTTP JSON
              └── FastAPI (puerto 8000)
                    ├── Google AI  -> embeddings (gemini-embedding-001)
                    ├── ChromaDB   -> persistencia en disco y k-NN (coseno)
                    └── Google AI  -> generación de la respuesta (Gemini)
```

| Capa | Tecnología | Rol |
|---|---|---|
| UI | Streamlit | Cargar documentos, preguntar y ver respuesta, citas, scores |
| API | FastAPI | `/health`, `/ingest`, `/query` (documentada en `/docs`) |
| Índice | ChromaDB | Guarda chunks + vectores en `chroma/` y devuelve los más similares |
| Modelos | Google AI | Embeddings de documentos y preguntas, y generación con Gemini |

Streamlit **no** habla con Chroma ni con Google AI: todo pasa por la API.

## Requisitos

- Python 3.9 o superior (desarrollado con 3.9.6; aparecen avisos de versión, no afectan).
- Una clave de Google AI Studio (gratis): <https://aistudio.google.com/apikey>.

## Instalación

Desde la carpeta `rag-app/`:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env
```

Abre `.env` y pega tu clave (el archivo está en `.gitignore`, nunca se sube):

```
GOOGLE_API_KEY=tu_clave_aqui
```

## Ejecución

Necesitas **dos terminales**, ambas en `rag-app/` y con el venv activo.

**Terminal 1 — API:**

```bash
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva: <http://localhost:8000/docs>

**Terminal 2 — Interfaz:**

```bash
streamlit run ui/streamlit_app.py
```

Se abre en <http://localhost:8501>.

### Indexar el corpus (solo la primera vez)

El índice vive en disco (`chroma/`) y sobrevive a reinicios; solo hay que indexar
para agregar o actualizar documentos. Con la API corriendo, desde la UI
(pestaña **Documentos**, seleccionar los `.md` de `data/` y pulsar *Indexar*) o con:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@data/01_sensor_de_imagen.md" \
  -F "files=@data/02_camara_fotografica.md" \
  -F "files=@data/03_objetivo_fotografico.md" \
  -F "files=@data/04_exposicion_fotografica.md" \
  -F "files=@data/05_historia_de_la_fotografia.md"
```

Respuesta esperada: `{"documents": 5, "chunks": 61, "skipped": []}`.
Volver a indexar los mismos archivos no duplica nada (los chunks se sobrescriben).

## Probar una pregunta

Con la UI (pestaña **Preguntar**) o por la API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿qué diferencia hay entre un sensor CCD y uno CMOS?"}'
```

Casos de prueba sugeridos:

| Pregunta | Resultado esperado |
|---|---|
| ¿qué diferencia hay entre un sensor CCD y uno CMOS? | Respuesta en español con citas `[1]`, `[2]`, `abstained: false` |
| ¿qué es la distancia focal? | Respuesta con citas |
| ¿cuánto cuesta una Sony A7IV en México? | `abstained: true` (no está en el corpus) |
| ¿cómo se prepara una paella? | `abstained: true` (tema ajeno) |
| *(pregunta vacía)* | HTTP 400 con mensaje claro |

## API

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado de la API y número de chunks en Chroma: `{"status": "ok", "chunks": 61}` |
| `POST /ingest` | Recibe archivos `.md` / `.txt` (multipart, campo `files`), los parte en chunks, calcula embeddings y los guarda. Responde `{documents, chunks, skipped}` |
| `POST /query` | Cuerpo `{"question": "...", "top_k": 4}` (`top_k` opcional). Responde `{answer, citations, abstained}` |

Cada elemento de `citations` trae `id` (`archivo::posición`), `source`, `text` y `score`
(similitud coseno). Las citas `[n]` de la respuesta corresponden a la posición `n` de
esa lista (el primer chunk es `[1]`).

Errores: pregunta vacía -> `400`; Google AI no disponible, sin cuota o sin respuesta a
tiempo -> `502` con un mensaje explicativo. Una pregunta fuera de dominio **no** es un
error: devuelve `abstained: true`.

## Cómo funciona

**Corpus.** 5 artículos de Wikipedia en español sobre cámaras (sensor de imagen, cámara
fotográfica, objetivo fotográfico, exposición e historia de la fotografía), en `data/`.
Son unas 14,300 palabras en total. Cada archivo indica su fuente en el encabezado
(texto de Wikipedia, licencia CC BY-SA).

**Chunking** (`rag/chunk.py`). Ventanas de **300 palabras con 60 de solape**. El solape
evita que una idea quede cortada justo en la frontera entre dos chunks: lo que queda a
caballo aparece completo en al menos uno. El corpus produce **61 chunks**.

**Embeddings** (`rag/embed.py`). `gemini-embedding-001` de Google AI, el **mismo** modelo
para chunks y preguntas (si se mezclaran modelos, la búsqueda no tendría sentido). Los
chunks se envían en lotes de 20.

**Índice** (`rag/store.py`). ChromaDB persistente en `chroma/`, con distancia coseno
(`score = 1 - distancia`). Los vectores los calcula Google AI; Chroma solo los guarda y
busca los `top_k` más cercanos. Cada chunk guarda `source`, `title` e `index`, y su id es
`archivo::posición`, así que reindexar un archivo lo actualiza en vez de duplicarlo.

**Generación** (`rag/generate.py`). Se le entregan a Gemini los chunks numerados
`[1]`, `[2]`... y una instrucción de responder solo con esa evidencia, en español y
citando `[n]`. Si la evidencia no cubre la pregunta, debe contestar exactamente
`NO_EVIDENCE`. La respuesta nunca se arma concatenando chunks a mano.

### Regla de abstención (dos capas)

1. **Umbral de similitud, `MIN_SCORE = 0.65`.** Si ningún chunk recuperado alcanza ese
   score, la API se abstiene sin llamar a Gemini. Se calibró midiendo preguntas reales:
   las del corpus sacan de **0.72 a 0.76** y las ajenas de **0.53 a 0.58**
   (Sony A7IV: 0.584; paella: 0.530), así que 0.65 queda en medio con margen.
2. **El propio modelo.** Solo los chunks con score >= 0.65 llegan a Gemini. Si aun así la
   evidencia no contiene la respuesta (por ejemplo, una pregunta *de cámaras* pero sobre
   algo que no está en los documentos), Gemini responde `NO_EVIDENCE` y la API devuelve
   `abstained: true`. El umbral por sí solo no basta para esos casos: una pregunta de
   cámaras sobre un dato ausente se parece al corpus más que una pregunta de cocina.

Al abstenerse, `answer` dice explícitamente que no hay evidencia suficiente; no se
rellena con conocimiento propio del modelo.

### Modelos de generación y respaldo

La cuota gratuita de Gemini es de unas 20 peticiones de generación por día **por modelo**,
y los modelos a veces responden 503 por alta demanda. Por eso `rag/generate.py` prueba una
lista en orden y pasa al siguiente si uno responde 429 (cuota), 404 (no disponible), 5xx
(saturado) o no contesta en 40 segundos (y deja de probar modelos tras 100 s en total):

```python
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
```

En la terminal de uvicorn se ve qué modelo respondió. Los nombres de modelo cambian con
el tiempo; para ver los disponibles con tu clave: `client.models.list()`.

## Configuración

| Qué | Dónde | Valor |
|---|---|---|
| Umbral de abstención | `app/main.py` (`MIN_SCORE`) | `0.65` |
| Chunks por defecto | `app/main.py` (`DEFAULT_TOP_K`) | `4` |
| Tamaño / solape del chunking | `app/main.py` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) | `300` / `60` |
| Mensaje de abstención | `app/main.py` (`ABSTAIN_MESSAGE`) | texto libre |
| Modelo de embeddings | `rag/embed.py` (`MODEL`) | `gemini-embedding-001` |
| Modelos de generación | `rag/generate.py` (`MODELS`) | ver arriba |
| Tiempo máximo por petición | `rag/client.py` (`REQUEST_TIMEOUT_MS`) | `40000` |
| Presupuesto total de la cadena de modelos | `rag/generate.py` (`TOTAL_BUDGET_S`) | `100` s |
| Tema de la UI | `.streamlit/config.toml` | oscuro, acento ámbar |

Si cambias el modelo de embeddings, borra `chroma/` y reindexa: los vectores de modelos
distintos no son comparables.

## Estructura

```
rag-app/
  README.md
  requirements.txt
  .env.example          # GOOGLE_API_KEY= (vacío)
  .gitignore            # .env, venv/, chroma/
  .streamlit/config.toml
  data/                 # corpus (5 .md)
  chroma/               # índice persistente (se genera solo, no se versiona)
  app/
    main.py             # FastAPI: /health, /ingest, /query
  rag/
    client.py           # cliente de Google AI (clave desde .env)
    data.py             # Document / Corpus
    chunk.py            # partición en chunks con solape
    embed.py            # embeddings (Google AI)
    store.py            # ChromaDB: alta y consulta top-k
    generate.py         # Gemini: respuesta anclada, NO_EVIDENCE, respaldo de modelos
  ui/
    streamlit_app.py    # carga, preguntas, citas con score, historial
```

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| La UI dice "API desconectada" | uvicorn no está corriendo en el puerto 8000. Levántalo (terminal 1). |
| `KeyError: 'GOOGLE_API_KEY'` al arrancar | Falta el archivo `.env` o la clave está vacía. |
| `502` "Se agotó la cuota..." | Se acabó la cuota diaria de **todos** los modelos de la lista; se renueva solo. |
| `502` "El servicio de Google AI no está disponible" | Saturación temporal; reintenta en unos segundos. |
| `chunks: 0` en `/health` | El índice está vacío: indexa los documentos (sección *Indexar el corpus*). |
| `ModuleNotFoundError: rag` | Corre uvicorn desde la carpeta `rag-app/`, no desde dentro de `app/`. |
| `404 ... model ... not found` | El nombre del modelo ya no existe; actualiza la lista con `client.models.list()`. |

## Notas

- No hay claves en el repositorio: `.env` y `chroma/` están en `.gitignore`;
  `.env.example` solo trae el nombre de la variable.
- El índice no se sube: se reconstruye indexando `data/` (unos segundos).
- Los textos del corpus provienen de Wikipedia en español (CC BY-SA 4.0).
