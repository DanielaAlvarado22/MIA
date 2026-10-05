import base64
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://localhost:8000"
ASSETS = Path(__file__).parent / "assets"


def data_uri(name: str) -> str:
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"

EXAMPLES = [
    "¿qué diferencia hay entre un sensor CCD y uno CMOS?",
    "¿qué es la distancia focal?",
    "¿cuánto cuesta una Sony A7IV en México?",
]

st.set_page_config(page_title="RAG de cámaras", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #1F2933; --paper: #F6F1E7; --teal: #0F6B6B;
        --deep: #14414A; --film: #E07A2F; --line: #D9CFB8;
    }
    .block-container { padding-top: 1.6rem; max-width: 1100px; }

    .hero {
        display: flex; align-items: center; gap: 1.2rem;
        background: linear-gradient(135deg, var(--teal) 0%, var(--deep) 100%);
        color: #FFFFFF; padding: 1.5rem 1.8rem; border-radius: 16px;
        margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(20, 65, 74, 0.25);
    }
    .hero h1 {
        font-family: Georgia, "Times New Roman", serif; font-weight: 700;
        font-size: 2.1rem; margin: 0; padding: 0; color: #FFFFFF;
    }
    .hero p { margin: 0.3rem 0 0 0; color: #CFE3E3; font-size: 0.98rem; }
    .hero-camera {
        flex: none; width: 128px; height: auto;
        filter: drop-shadow(0 6px 10px rgba(0, 0, 0, 0.35));
    }

    h2, h3 { font-family: Georgia, "Times New Roman", serif; color: var(--deep); }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background-image: url(__STRIP__), url(__STRIP__);
        background-repeat: repeat-y, repeat-y;
        background-position: left 4px top 0, right 4px top 0;
        background-size: 34px auto, 34px auto;
    }
    [data-testid="stSidebarUserContent"] { padding-left: 2.4rem; padding-right: 2.4rem; }

    [data-testid="stExpander"] {
        background: #FFFDF8; border: 1px solid var(--line); border-radius: 10px;
    }
    .st-key-answer-card {
        border-left: 5px solid var(--film) !important; background: #FFFDF8; border-radius: 10px;
    }
    .stButton > button { border-radius: 999px; border: 1px solid var(--teal); }
    .stButton > button[kind="primary"] { color: #FFFFFF; font-weight: 600; }
    .stProgress > div > div > div > div { background-color: var(--film); }
    </style>
    """.replace("__STRIP__", data_uri("filmstrip.png")),
    unsafe_allow_html=True,
)

st.session_state.setdefault("question", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("flash", None)


def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def error_detail(resp) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text


def set_example(text: str) -> None:
    st.session_state["question"] = text


def run_query(question: str, top_k: int) -> None:
    try:
        with st.spinner("Buscando evidencia y generando la respuesta..."):
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=170,
            )
    except requests.RequestException:
        st.error("No se pudo conectar con la API.")
        return
    if not resp.ok:
        st.error(error_detail(resp))
        return
    data = resp.json()
    st.session_state["result"] = {"question": question, **data}
    st.session_state["history"].insert(
        0, {"question": question, "abstained": data["abstained"]}
    )
    st.rerun()


health = get_health()

with st.sidebar:
    st.header("Estado del sistema")
    if health is None:
        st.error("API desconectada. ¿Está corriendo uvicorn en el puerto 8000?")
    else:
        st.markdown("**:green[● API activa]**")
        st.metric("Chunks indexados", health["chunks"])

    st.divider()
    st.subheader("Recuperación")
    top_k = st.slider("Chunks a recuperar (top-k)", min_value=1, max_value=8, value=4)

    st.divider()
    st.subheader("Historial de la sesión")
    if not st.session_state["history"]:
        st.caption("Aún no has hecho preguntas.")
    else:
        for item in st.session_state["history"][:8]:
            mark = "sin evidencia" if item["abstained"] else "respondida"
            st.caption(f"{item['question']}  \n_{mark}_")
        if st.button("Borrar historial"):
            st.session_state["history"] = []
            st.rerun()

st.markdown(
    """
    <div class="hero">
      <img class="hero-camera" src="__CAMERA__" alt="cámara">
      <div>
        <h1>RAG de cámaras y fotografía</h1>
        <p>Respuestas ancladas en documentos: embeddings de Google AI, índice en ChromaDB
        y generación con Gemini, siempre con citas.</p>
      </div>
    </div>
    """.replace("__CAMERA__", data_uri("camera.png")),
    unsafe_allow_html=True,
)

tab_ask, tab_docs = st.tabs(["Preguntar", "Documentos"])

with tab_ask:
    st.caption("Prueba con un ejemplo o escribe tu propia pregunta:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        col.button(example, on_click=set_example, args=(example,), use_container_width=True)

    question = st.text_input(
        "Tu pregunta", key="question", placeholder="Ej. ¿qué es el triángulo de exposición?"
    )
    if st.button("Preguntar", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta.")
        elif health is not None and health["chunks"] == 0:
            st.warning("El índice está vacío. Carga documentos en la pestaña Documentos.")
        else:
            run_query(question, top_k)

    result = st.session_state["result"]
    if result:
        st.divider()
        st.markdown(f"**Pregunta:** {result['question']}")
        if result["abstained"]:
            st.warning(result["answer"])
        else:
            with st.container(border=True, key="answer-card"):
                st.markdown(result["answer"])
            citations = result["citations"]
            best = max(c["score"] for c in citations)
            m1, m2 = st.columns(2)
            m1.metric("Fuentes usadas", len(citations))
            m2.metric("Mejor similitud", f"{best:.2f}")

            st.subheader("Fuentes")
            for i, c in enumerate(citations, start=1):
                with st.expander(f"[{i}]  {c['source']}  ·  similitud {c['score']:.3f}"):
                    st.progress(min(max(c["score"], 0.0), 1.0))
                    st.caption(f"id del chunk: {c['id']}")
                    st.write(c["text"])

with tab_docs:
    st.subheader("Cargar documentos")
    st.caption(
        "Los documentos indexados se guardan en disco: solo necesitas indexar para "
        "agregar o actualizar archivos."
    )
    flash = st.session_state["flash"]
    if flash:
        kind, text = flash
        (st.success if kind == "success" else st.warning)(text)
        st.session_state["flash"] = None

    uploaded = st.file_uploader(
        "Archivos .md o .txt", type=["md", "txt"], accept_multiple_files=True
    )
    if st.button("Indexar documentos", type="primary"):
        if not uploaded:
            st.warning("Selecciona al menos un archivo.")
        else:
            files = [("files", (f.name, f.getvalue())) for f in uploaded]
            try:
                with st.spinner("Indexando (puede tardar unos segundos)..."):
                    resp = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
            except requests.RequestException:
                st.error("No se pudo conectar con la API.")
            else:
                if resp.ok:
                    data = resp.json()
                    msg = f"Se indexaron {data['documents']} documentos ({data['chunks']} chunks)."
                    kind = "success"
                    if data["skipped"]:
                        msg += f" Se omitieron: {', '.join(data['skipped'])}."
                        kind = "warning"
                    st.session_state["flash"] = (kind, msg)
                    st.rerun()
                else:
                    st.error(error_detail(resp))
