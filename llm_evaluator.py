"""
llm_evaluator.py
----------------
Evaluador de cohesión semántica para segmentos de texto.
Utiliza la API de Cerebras (modelo gpt-oss-120b) con prompt optimizado.
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise EnvironmentError("CEREBRAS_API_KEY no encontrada en el archivo .env")

CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0
TIMEOUT = 60
MAX_TOKENS = 100000          # alto para evitar truncamiento

# Caché de evaluaciones
_eval_cache: dict[tuple, float] = {}
_cache_hits = 0
_cache_misses = 0


# ---------------------------------------------------------------------------
# Llamada a la API de Cerebras
# ---------------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                CEREBRAS_API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else 10
                print(f"\n  [LLM] Rate limit. Esperando {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError(f"Respuesta sin choices: {data}")

            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", "").strip()

            if not content:
                raise ValueError("Respuesta vacía del LLM")

            return content

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"\n  [LLM] Error de red intento {attempt+1}: {e}")
        except (KeyError, ValueError, IndexError) as e:
            last_error = e
            print(f"\n  [LLM] Error de parsing intento {attempt+1}: {e}")
        except Exception as e:
            last_error = e
            print(f"\n  [LLM] Error inesperado intento {attempt+1}: {e}")

        wait = RETRY_DELAY_BASE * (2 ** attempt)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait)

    raise RuntimeError(f"Fallo en llamada a Cerebras tras {MAX_RETRIES} intentos: {last_error}")


# ---------------------------------------------------------------------------
# Construcción del prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a ruthless text segmentation algorithm. Your job is to analyze the following text and determine if it is homogeneous or if it contains a thematic shift.

Rules:
- Score 10.0 ONLY if the text is 100% homogeneous (e.g., all sentences describe the exact same physical structure, OR all sentences describe the exact same step in a process).
- Score 2.0 if the text transitions between sub-topics (e.g., from author A to author B, from historical context to physical measurement, or from one century to another).

Text to analyze:
{texto}

Put your reasoning inside <think></think> tags. Output your final decision exactly as: [SCORE: X.X]"""


def _build_prompt(text_content: str) -> str:
    return PROMPT_TEMPLATE.format(texto=text_content)


# ---------------------------------------------------------------------------
# Extracción del score numérico
# ---------------------------------------------------------------------------
def _parse_score(response: str) -> float:
    numbers = re.findall(r'\b(10|[1-9])\b', response)
    if numbers:
        return float(numbers[0])
    digits = re.findall(r'\d', response)
    if digits:
        return float(min(10, max(1, int(digits[0]))))
    return 5.0


# ---------------------------------------------------------------------------
# Evaluación de cohesión de un segmento
# ---------------------------------------------------------------------------
def evaluate_segment(sentences: list[str], start: int, end: int) -> float:
    global _cache_hits, _cache_misses

    if start >= end:
        return 0.0

    text_content = " ".join(sentences[start:end])
    key = (start, end, hash(text_content))

    if key in _eval_cache:
        _cache_hits += 1
        return _eval_cache[key]

    _cache_misses += 1
    prompt = _build_prompt(text_content)

    print(f"  [LLM] Evaluando ({start:02d} -> {end:02d})...", end=" ", flush=True)

    try:
        response = _call_llm(prompt)
        score = _parse_score(response)
        print(f"Score: {score}")
    except RuntimeError as e:
        print(f"FALLO: {e} -> Score: 5.0")
        score = 5.0

    _eval_cache[key] = score
    return score


# ---------------------------------------------------------------------------
# Gestión de caché
# ---------------------------------------------------------------------------
def reset_cache():
    global _eval_cache, _cache_hits, _cache_misses
    _eval_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def get_cache_stats() -> dict:
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total) if total > 0 else 0.0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": hit_rate,
    }