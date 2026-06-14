"""
llm_evaluator.py
----------------
Evaluador de cohesión semántica para segmentos de texto.
Utiliza la API de Cerebras (modelo gpt-oss-120b) con estrategia de
evaluación en dos fases:

  Fase 1 — Prompt suave (PROMPT_SOFT):
    Evalúa la cohesión general del segmento en escala 1-10.
    Es permisivo: detecta los casos claramente mezclados (score bajo)
    y los claramente cohesivos (score alto).

  Fase 2 — Prompt estricto (PROMPT_STRICT), solo si Fase 1 da score alto:
    Se aplica ÚNICAMENTE cuando el segmento parece cohesivo (score >= STRICT_THRESHOLD).
    Busca activamente subtemas ocultos, transiciones sutiles o cambios de
    énfasis que el prompt suave podría haber ignorado.
    Si el prompt estricto detecta un cambio, devuelve un score más bajo.

  El score final es el mínimo de ambas fases cuando se aplican las dos,
  o el score de la Fase 1 si no se supera el umbral.

Parámetro clave:
  STRICT_THRESHOLD = 7.0
    Si el score suave >= 7.0, se activa la Fase 2 (prompt estricto).
    Ajustar hacia arriba si el estricto penaliza demasiado textos válidos.
    Ajustar hacia abajo si queremos más agresividad en la detección.
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
MAX_TOKENS       = 1000

# Umbral para activar el prompt estricto (Fase 2).
# Si score_suave >= STRICT_THRESHOLD → se lanza Fase 2.
STRICT_THRESHOLD = 7.0

# Caché de evaluaciones: clave → score final
_eval_cache: dict[tuple, float] = {}
_cache_hits   = 0
_cache_misses = 0
_strict_activations = 0   # cuántas veces se activó la Fase 2 (para stats)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Fase 1: evaluación general, permisiva.
# Identifica claramente los extremos (muy mezclado vs muy cohesivo).
PROMPT_SOFT = """Analyze the following text and rate its thematic coherence from 1 to 10.

- 1-3: The text mixes completely unrelated topics (e.g., biology + Roman history + astronomy in the same paragraph).
- 4-6: The text has a general theme but includes noticeable digressions or shifts in focus.
- 7-9: The text is mostly about one topic, with minor transitions or elaborations.
- 10: The text is perfectly homogeneous — all sentences describe the exact same concept or process without any shift.

Text:
{texto}

Put your reasoning inside <think></think> tags. Output your final score exactly as: [SCORE: X.X]"""


# Fase 2: evaluación estricta, activada solo cuando Fase 1 da score alto.
# Busca activamente subtemas ocultos que el prompt suave pudo ignorar.
PROMPT_STRICT = """You are a ruthless text segmentation algorithm. Your job is to find reasons to CUT this text.

The text has already been classified as generally cohesive. Your task is to look harder:
- Does the text TRANSITION between different sub-topics? (e.g., from a general theory to a specific experiment, from one historical period to another, from one author to another)
- Does the focus SHIFT even slightly? (e.g., from macroscopic biology to molecular chemistry)
- Are there sentences that would more naturally belong to a DIFFERENT section?

Rules:
- Score 9.0-10.0 ONLY if the text is 100% homogeneous with zero thematic shift.
- Score 5.0-8.0 if there is a subtle but detectable shift in focus or sub-topic.
- Score 1.0-4.0 if there is a clear transition between different sub-topics.

Text:
{texto}

Put your reasoning inside <think></think> tags. Output your final decision exactly as: [SCORE: X.X]"""


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
            data    = response.json()

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
    # Formato preferido: [SCORE: X.X]
    match = re.search(r'\[SCORE:\s*([0-9]*\.?[0-9]+)\]', response, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: último número en la respuesta (sin el bloque <think>)
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
# Evaluación principal en dos fases
# ---------------------------------------------------------------------------

def evaluate_segment(sentences: list[str], start: int, end: int) -> float:
    """
    Evalúa la cohesión de un segmento en dos fases:

      Fase 1 (siempre): PROMPT_SOFT → score_suave
      Fase 2 (condicional): PROMPT_STRICT si score_suave >= STRICT_THRESHOLD

    Score final:
      - Si solo Fase 1: score_suave
      - Si ambas fases: min(score_suave, score_estricto)
        → el más conservador gana; si el estricto detecta subtemas
          ocultos, el score baja aunque el suave fuera alto.

    Usa caché para evitar llamadas repetidas al LLM.
    Devuelve 5.0 si ambas fases fallan.
    """
    global _cache_hits, _cache_misses, _strict_activations

    if start >= end:
        return 0.0

    key = _make_cache_key(sentences, start, end)
    if key in _eval_cache:
        _cache_hits += 1
        return _eval_cache[key]

    _cache_misses += 1
    text_content = " ".join(sentences[start:end])

    # --- Fase 1: prompt suave ---
    print(f"  [LLM] Evaluando ({start:02d} -> {end:02d}) Fase1...", end=" ", flush=True)
    try:
        response_soft = _call_llm(PROMPT_SOFT.format(texto=text_content))
        score_soft    = _parse_score(response_soft)
        print(f"score_suave={score_soft:.1f}", end="", flush=True)
    except RuntimeError as e:
        print(f"FALLO Fase1 → 5.0")
        _eval_cache[key] = 5.0
        return 5.0

    # --- Fase 2: prompt estricto (solo si score_suave es alto) ---
    if score_soft >= STRICT_THRESHOLD:
        _strict_activations += 1
        print(f" | Fase2...", end=" ", flush=True)
        try:
            response_strict = _call_llm(PROMPT_STRICT.format(texto=text_content))
            score_strict    = _parse_score(response_strict)
            print(f"score_estricto={score_strict:.1f}", end="", flush=True)
            # El score final es el más conservador de los dos
            score_final = min(score_soft, score_strict)
        except RuntimeError as e:
            print(f"FALLO Fase2 → usando score_suave", end="", flush=True)
            score_final = score_soft
    else:
        score_final = score_soft

    print(f" → final={score_final:.1f}")

    _eval_cache[key] = score_final
    return score_final


# ---------------------------------------------------------------------------
# Funciones de compatibilidad (usadas por dp_segmentation.py)
# ---------------------------------------------------------------------------

def suggest_move(sentences: list[str], cuts: list[int], cut_index: int) -> str:
    """Stub de compatibilidad. Sin uso activo en el pipeline principal."""
    return "MANTENER"


def judge_boundary(sentences: list[str], cut: int, window: int = 3) -> bool:
    """Stub de compatibilidad. Sin uso activo en el pipeline principal."""
    return True


# ---------------------------------------------------------------------------
# Gestión de caché
# ---------------------------------------------------------------------------

def reset_cache() -> None:
    global _eval_cache, _cache_hits, _cache_misses, _strict_activations
    _eval_cache.clear()
    _cache_hits          = 0
    _cache_misses        = 0
    _strict_activations  = 0


def get_cache_stats() -> dict:
    total    = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total) if total > 0 else 0.0
    return {
        "hits":               _cache_hits,
        "misses":             _cache_misses,
        "hit_rate":           hit_rate,
        "strict_activations": _strict_activations,
        "llm_calls_saved":    _cache_hits,
    }


# ---------------------------------------------------------------------------
# Prueba standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    print("=" * 65)
    print("PRUEBA LLM — Evaluación en dos fases (Cerebras)")
    print(f"STRICT_THRESHOLD = {STRICT_THRESHOLD}")
    print("=" * 65)

    # Test de conexión
    print("\nTest de conexión...")
    try:
        resp = _call_llm("Reply with exactly: [SCORE: 7.0]")
        print(f"  OK → '{resp[:60]}'")
    except Exception as e:
        print(f"  FALLO: {e}")
        exit(1)

    # Textos de prueba
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

    SUBTLE = [
        "A Sachs se debe la formulación de la ecuación básica de la fotosíntesis: 6 CO2 + 6 H2O → C6H12O6 + 6 O2.",
        "Andreas Franz Wilhelm Schimper daría el nombre de cloroplastos a los cuerpos coloreados de Sachs.",
        "En el último tercio del siglo XIX se sucederían los esfuerzos por establecer las propiedades físico-químicas de las clorofilas.",
        "En 1905, Frederick Frost Blackman midió la velocidad a la que se produce la fotosíntesis en diferentes condiciones.",
        "Los cloroplastos están delimitados por una envoltura formada por dos membranas llamadas envueltas.",
    ]

    print("\n[1/3] Segmento COHESIVO (fotosíntesis pura):")
    s1 = evaluate_segment(COHESIVE, 0, len(COHESIVE))
    print(f"      Score final: {s1:.1f}/10  (esperado: 7-10)")

    print("\n[2/3] Segmento MEZCLADO (fotosíntesis + Roma + astronomía):")
    s2 = evaluate_segment(MIXED, 0, len(MIXED))
    print(f"      Score final: {s2:.1f}/10  (esperado: 1-4)")

    print("\n[3/3] Segmento SUTIL (historia de la fotosíntesis — subtemas cercanos):")
    s3 = evaluate_segment(SUBTLE, 0, len(SUBTLE))
    print(f"      Score final: {s3:.1f}/10  (esperado: 5-8)")

    print(f"\n{'─'*65}")
    stats = get_cache_stats()
    print(f"Llamadas LLM reales    : {stats['misses']}")
    print(f"  de las cuales Fase 2 : {stats['strict_activations']}")
    print(f"Cache hits             : {stats['hits']}")
    diff = s1 - s2
    print(f"\nDiferencia cohesivo - mezclado: {diff:.1f}")
    if diff >= 3:
        print("✓ El sistema distingue bien.")
    else:
        print("⚠ Diferencia baja. Revisar STRICT_THRESHOLD o prompts.")