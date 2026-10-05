import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Tiempo máximo por petición (ms): bajo carga un modelo puede tardar ~30 s en responder bien,
# pero un modelo saturado tarda >90 s en devolver su 503.
REQUEST_TIMEOUT_MS = 40_000

client = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"],
    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
)
