"""
llm_evaluator.py
----------------
Evaluador de cohesión semántica para segmentos de texto.
Utiliza la API de Cerebras (modelo gpt-oss-120b).

Arquitectura de selección de modo (Suave vs Estricto):

  Evaluación Inicial (UNA VEZ por ejecución, select_evaluation_mode):
    Se evalúa el texto completo con PROMPT_SOFT para obtener una
    puntuación de calidad/coherencia global (0-10).

  Condicional de Selección:
    - Escenario A (score_global >= GLOBAL_SCORE_THRESHOLD, default 8.0):
      el texto es coherente y estructuralmente sólido. Se usa
      EXCLUSIVAMENTE el Prompt Estricto (PROMPT_STRICT) para evaluar
      tanto el texto completo como cada subsegmento, buscando límites
      finos con precisión.
    - Escenario B (score_global < GLOBAL_SCORE_THRESHOLD):
      el texto presenta inconsistencias / es heterogéneo a priori.
      Se usa EXCLUSIVAMENTE el Prompt Suave (PROMPT_SOFT), más tolerante
      a digresiones menores, para no sobre-segmentar.

  A diferencia del esquema anterior (Fase1 + Fase2 condicional con
  min(suave, estricto) por cada segmento), el modo se decide UNA SOLA
  VEZ al inicio y se aplica de forma CONSISTENTE durante toda la
  ejecución. Esto evita que el prompt estricto (que por diseño tiende a
  encontrar "un sub-tema" en casi cualquier texto) penalice en cascada
  bloques que ya son coherentes al nivel de granularidad correcto.
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
MODEL            = "gpt-oss-120b"
MAX_RETRIES      = 3
RETRY_DELAY_BASE = 2.0
TIMEOUT          = 60
MAX_TOKENS       = 100000

# Umbral de la evaluación inicial (Escenario A vs B).
# score_global >= GLOBAL_SCORE_THRESHOLD → modo "strict"
# score_global <  GLOBAL_SCORE_THRESHOLD → modo "soft"
GLOBAL_SCORE_THRESHOLD = 8.0

# Caché de evaluaciones: clave → score final
_eval_cache: dict[tuple, float] = {}
_cache_hits   = 0
_cache_misses = 0

# Modo activo para la ejecución actual ("strict" | "soft" | None)
# y score de la evaluación inicial que lo determinó.
_active_mode: str | None = None
_global_score: float | None = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Prompt Suave: enfocado en flexibilidad, permite inferencias ligeras
# cuando la estructura no es obvia.
PROMPT_SOFT = """Analyze the following text and rate its broader thematic coherence from 1 to 10.

- 1-3: The text abruptly mixes completely different major categories (e.g., History of a discovery mixed with detailed chemical equations, or astronomy mixed with biology).
- 4-6: The text has a general theme but includes noticeable structural shifts (e.g., finishing a historical intro and starting a technical deep-dive).
- 7-9: The text is mostly about one overarching topic, even if it progresses naturally through related sub-points or a timeline.
- 10: The text is perfectly cohesive and belongs to a single logical block.

Text:
{texto}

Put your reasoning inside <think></think> tags. Output your final score exactly as: [SCORE: X.X]"""


# Prompt Estricto: Diferencia entre flujo narrativo y saltos de sub-procesos.
PROMPT_STRICT = """You are a text segmentation algorithm. Find if there is a sub-topic boundary in the text.

A sub-topic boundary means the text changes its NATURE: e.g., from describing a historical discovery process to describing a physical structure, or from one scientific mechanism to a completely different mechanism.

A continuous chronological narrative that mentions many different scientists, years, or experiments is NOT a boundary — it is ONE narrative thread (e.g., "the history of how X was discovered"). Do NOT mark a cut just because a new person or date appears.

Examples of NO boundary (single narrative thread):
- "Aristotle proposed X. In the 18th century, Priestley discovered Y. Later, Calvin traced Z." (all history of photosynthesis discovery)
→ Score: 9.5

Examples of YES boundary (true sub-topic change):
- "The history of photosynthesis discovery. The structure of chloroplasts." (history → anatomy)
→ Score: 2.0
- "The light-dependent reactions of photosynthesis. The Calvin cycle." (two different biochemical mechanisms)
→ Score: 2.0

Rules:
- Score 9.0-10.0 if the whole text is ONE continuous thread with zero sub-topic change.
- Score 1.0-4.0 if there is a clear change in the NATURE of the content (true sub-topic boundary).
- Score 5.0-8.0 ONLY if you are genuinely unsure.

Now analyze this text.

Text:
{texto}

Put your reasoning inside <think></think> tags. Output your final score exactly as: [SCORE: X.X]"""
# ---------------------------------------------------------------------------
# Llamada a la API de Cerebras
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Llama a la API de Cerebras con reintentos y manejo de rate limit."""
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
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
                timeout=TIMEOUT,
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

            content = data["choices"][0].get("message", {}).get("content", "").strip()
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

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_BASE * (2 ** attempt))

    raise RuntimeError(f"Fallo en llamada a Cerebras tras {MAX_RETRIES} intentos: {last_error}")


# ---------------------------------------------------------------------------
# Parsing de scores
# ---------------------------------------------------------------------------

def _parse_score(response: str) -> float:
    """
    Extrae el score numérico de la respuesta del LLM.
    Busca primero el formato [SCORE: X.X], luego números sueltos.
    """
    match = re.search(r'\[SCORE:\s*([0-9]*\.?[0-9]+)\]', response, re.IGNORECASE)
    if match:
        return float(match.group(1))

    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    numbers = re.findall(r'\b(10(?:\.0)?|[1-9](?:\.[0-9])?)\b', clean)
    if numbers:
        return float(numbers[-1])

    digits = re.findall(r'\d', clean)
    if digits:
        return float(min(10, max(1, int(digits[-1]))))

    return 5.0


def _make_cache_key(sentences: list[str], start: int, end: int) -> tuple:
    text = " ".join(sentences[start:end])
    return (start, end, hash(text))


# ---------------------------------------------------------------------------
# Selección de modo (Evaluación Inicial)
# ---------------------------------------------------------------------------

def select_evaluation_mode(sentences: list[str]) -> tuple[str, float]:
    """
    Evaluación Inicial: analiza el texto completo con PROMPT_SOFT para
    obtener una puntuación de calidad/coherencia global (0-10) y decide
    qué prompt usar de forma EXCLUSIVA durante el resto de la ejecución:

      score_global >= GLOBAL_SCORE_THRESHOLD → modo "strict"
      score_global <  GLOBAL_SCORE_THRESHOLD → modo "soft"

    El modo queda activo (_active_mode) hasta el próximo reset_cache().
    Esta llamada cuenta como 1 invocación al LLM (no usa caché, ya que
    es una evaluación de calidad separada de las evaluaciones de
    segmentos individuales que hace evaluate_segment).
    """
    global _active_mode, _global_score

    n = len(sentences)
    text_content = " ".join(sentences[0:n])

    print(f"  [LLM] Evaluación inicial de calidad (00 -> {n:02d})...", end=" ", flush=True)
    try:
        response = _call_llm(PROMPT_SOFT.format(texto=text_content))
        score = _parse_score(response)
    except RuntimeError:
        score = 5.0

    mode = "strict" if score >= GLOBAL_SCORE_THRESHOLD else "soft"
    print(f"score_global={score:.1f} (umbral={GLOBAL_SCORE_THRESHOLD}) → modo='{mode}'")

    _active_mode  = mode
    _global_score = score
    return mode, score


# ---------------------------------------------------------------------------
# Evaluación de segmentos (modo único, decidido por select_evaluation_mode)
# ---------------------------------------------------------------------------

def evaluate_segment(sentences: list[str], start: int, end: int) -> float:
    """
    Evalúa la cohesión de un segmento usando EXCLUSIVAMENTE el prompt
    correspondiente al modo activo (_active_mode), fijado por
    select_evaluation_mode() para toda la ejecución.

    Si no se llamó select_evaluation_mode() previamente, se usa "soft"
    por defecto (comportamiento conservador).

    Usa caché para evitar llamadas repetidas al LLM.
    Devuelve 5.0 si la llamada falla.
    """
    global _cache_hits, _cache_misses

    if start >= end:
        return 0.0

    key = _make_cache_key(sentences, start, end)
    if key in _eval_cache:
        _cache_hits += 1
        return _eval_cache[key]

    _cache_misses += 1
    text_content = " ".join(sentences[start:end])

    mode = _active_mode or "soft"
    prompt_template = PROMPT_STRICT if mode == "strict" else PROMPT_SOFT

    print(f"  [LLM] Evaluando ({start:02d} -> {end:02d}) [{mode}]...", end=" ", flush=True)
    try:
        response = _call_llm(prompt_template.format(texto=text_content))
        score = _parse_score(response)
        print(f"score={score:.1f}")
    except RuntimeError as e:
        print(f"FALLO ({e}) → 5.0")
        score = 5.0

    _eval_cache[key] = score
    return score


# ---------------------------------------------------------------------------
# Funciones de compatibilidad (sin uso activo en el pipeline principal)
# ---------------------------------------------------------------------------

def suggest_move(sentences: list[str], cuts: list[int], cut_index: int) -> str:
    """Stub de compatibilidad."""
    return "MANTENER"


def judge_boundary(sentences: list[str], cut: int, window: int = 3) -> bool:
    """Stub de compatibilidad."""
    return True


# ---------------------------------------------------------------------------
# Gestión de caché
# ---------------------------------------------------------------------------

def reset_cache() -> None:
    global _eval_cache, _cache_hits, _cache_misses, _active_mode, _global_score
    _eval_cache.clear()
    _cache_hits   = 0
    _cache_misses = 0
    _active_mode  = None
    _global_score = None


def get_cache_stats() -> dict:
    total    = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total) if total > 0 else 0.0
    return {
        "hits":            _cache_hits,
        "misses":          _cache_misses,
        "hit_rate":        hit_rate,
        "llm_calls_saved": _cache_hits,
        "evaluation_mode": _active_mode,
        "global_score":    _global_score,
    }


# ---------------------------------------------------------------------------
# Prueba standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("PRUEBA LLM — Selección de modo + evaluación (Cerebras)")
    print(f"GLOBAL_SCORE_THRESHOLD = {GLOBAL_SCORE_THRESHOLD}")
    print("=" * 65)

    print("\nTest de conexión...")
    try:
        resp = _call_llm("Reply with exactly: [SCORE: 7.0]")
        print(f"  OK → '{resp[:60]}'")
    except Exception as e:
        print(f"  FALLO: {e}")
        exit(1)

    COHESIVE = [
        "La fotosíntesis es el proceso por el cual las plantas convierten luz solar en energía química.",
        "Las plantas capturan la luz mediante la clorofila, un pigmento verde presente en los cloroplastos.",
        "Durante la fotosíntesis se producen glucosa y oxígeno a partir de dióxido de carbono y agua.",
        "La glucosa obtenida sirve como fuente de energía para el crecimiento y desarrollo de la planta.",
        "Sin fotosíntesis, la cadena alimentaria y la producción de oxígeno en la Tierra serían imposibles.",
    ]

    MIXED = [
        "La fotosíntesis convierte la luz solar en energía química almacenada en glucosa.",
        "El Imperio romano dominó gran parte de Europa, el norte de África y el Medio Oriente.",
        "La Vía Láctea es una galaxia espiral barrada que contiene entre 200 y 400 mil millones de estrellas.",
        "Las plantas usan la clorofila para absorber la energía lumínica necesaria para la fotosíntesis.",
        "Roma fundó instituciones que influyeron profundamente en el derecho y la cultura occidentales.",
    ]

    for label, sample in [("COHESIVO", COHESIVE), ("MEZCLADO", MIXED)]:
        reset_cache()
        print(f"\n--- {label} ---")
        mode, gscore = select_evaluation_mode(sample)
        score = evaluate_segment(sample, 0, len(sample))
        print(f"  modo='{mode}' | score_global={gscore:.1f} | score_segmento={score:.1f}")