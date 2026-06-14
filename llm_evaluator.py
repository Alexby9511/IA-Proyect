"""
llm_evaluator.py
----------------
Evalúa la cohesión semántica de segmentos de texto usando
un LLM local con Ollama (modelo deepseek-local:1.5b).

Nota sobre la arquitectura (ver simulated_annealing.py y pipeline.py):
  El LLM YA NO se usa dentro del bucle de Simulated Annealing. Se usa
  ÚNICAMENTE al final del pipeline, evaluando K segmentos por cada K
  candidato. suggest_move y judge_boundary quedan disponibles pero sin
  uso en el flujo principal.

Requisitos:
    pip install requests

Uso como módulo:
    from llm_evaluator import evaluate_segment, suggest_move, judge_boundary
"""

import json
import os
import re
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "deepseek-local:1.5b"
MAX_RETRIES = 2
RETRY_DELAY = 3.0

# Cache global
_eval_cache: dict[tuple, float] = {}
_cache_hits   = 0
_cache_misses = 0

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# EVAL_PROMPT mejorado con anclas de referencia para mejor calibración:
# - 1-2: texto que mezcla temas completamente distintos sin relación
# - 9-10: texto que trata un único tema de principio a fin
# Esto reduce el sesgo hacia puntuaciones altas en modelos pequeños.
EVAL_PROMPT = """En una escala del 1 al 10, ¿qué tan cohesionado temáticamente es este texto?
1 = trata varios temas sin relación. 10 = trata un único tema de principio a fin.
Responde SOLO con el número.

Texto: {texto}

Puntuación:"""

SUGGEST_PROMPT = """Fragmento A: {seg_izq}
Fragmento B: {seg_der}

El corte entre A y B, ¿está bien colocado?
Responde SOLO con una palabra: IZQUIERDA, DERECHA o MANTENER.
Respuesta:"""

BOUNDARY_PROMPT = """Fragmento 1: {seg_izq}
Fragmento 2: {seg_der}

¿Tratan estos dos fragmentos temas distintos?
Responde SOLO con SI o NO.
Respuesta:"""


# ---------------------------------------------------------------------------
# Llamada al LLM local
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, timeout: int = 60) -> str:
    """
    Llama a Ollama local via /api/generate.
    num_predict=5 — solo necesitamos 1-2 tokens para el número.
    Sin stop tokens para no cortar la respuesta antes del número.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 15,  # suficiente para capturar el número + breve explicación
        }
    }
    t0 = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            data    = response.json()
            content = data.get("response", "").strip()
            if not content:
                raise ValueError("Respuesta vacía del LLM")
            elapsed = time.time() - t0
            if elapsed > 5:
                print(f"[!] LLM tardó {elapsed:.1f}s", end=" ", flush=True)
            return content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [LLM] Error intento {attempt+1}: {e}. Reintentando...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"LLM falló tras {MAX_RETRIES} intentos: {e}")


# ---------------------------------------------------------------------------
# Parsing de respuestas
# ---------------------------------------------------------------------------

def _parse_score(response: str) -> float:
    """
    Extrae el primer número 1-10 de la respuesta.
    Si no encuentra ninguno, devuelve 5.0 (neutro).
    """
    numbers = re.findall(r'\b(10|[1-9])\b', response)
    if numbers:
        return float(numbers[0])
    digits = re.findall(r'\d', response)
    if digits:
        return float(min(10, max(1, int(digits[0]))))
    return 5.0


def _parse_bool(response: str) -> bool:
    """Parsea SI/NO. Por defecto True (conservador)."""
    upper = response.upper()
    if upper.startswith("NO") or upper.startswith("N "):
        return False
    return True


def _make_cache_key(sentences: list[str], start: int, end: int) -> tuple:
    text = " ".join(sentences[start:end])
    return (start, end, hash(text))


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def evaluate_segment(
    sentences: list[str],
    start: int,
    end: int,
    verbose: bool = False,
) -> float:
    """
    Evalúa la cohesión semántica de un segmento (1-10).
    Usa cache para evitar llamadas repetidas.
    Devuelve 5.0 si la API falla.

    En el pipeline actual se llama UNA VEZ por segmento de la partición
    final de cada K candidato (no dentro del bucle de SA).
    """
    global _cache_hits, _cache_misses

    if start >= end:
        return 0.0

    key = _make_cache_key(sentences, start, end)
    if key in _eval_cache:
        _cache_hits += 1
        return _eval_cache[key]

    _cache_misses += 1

    texto = " ".join(sentences[start:end])
    words = texto.split()
    if len(words) > 300:
        texto = " ".join(words[:300]) + "..."

    prompt = EVAL_PROMPT.format(texto=texto)

    n_words = len(texto.split())
    print(f"  [LLM] [{start}:{end}] ({end-start} seg, {n_words} words)...", end=" ", flush=True)

    try:
        response = _call_llm(prompt)
    except RuntimeError:
        print("FALLO → 5.0")
        return 5.0

    score = _parse_score(response)
    print(f"→ '{response[:20].strip()}' → {score:.1f}")

    if verbose:
        print(f"  [LLM detalle] [{start}:{end}] → '{response}' → {score:.1f}")

    _eval_cache[key] = score
    return score


def evaluate_partition(
    sentences: list[str],
    cuts: list[int],
    verbose: bool = False,
) -> tuple[float, list[float]]:
    """Evalúa una partición completa."""
    n          = len(sentences)
    boundaries = [0] + cuts + [n]
    scores     = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        score = evaluate_segment(sentences, start, end, verbose=verbose)
        scores.append(score)
        if verbose:
            print(f"  Segmento {i+1} [{start}:{end}]: {score:.1f}/10")
    return sum(scores), scores


def suggest_move(
    sentences: list[str],
    cuts: list[int],
    cut_index: int,
) -> str:
    """
    Sugiere si un corte debería moverse (IZQUIERDA, DERECHA, MANTENER).
    NOTA: sin uso en el pipeline principal. Se conserva como variante experimental.
    """
    n             = len(sentences)
    boundaries    = [0] + cuts + [n]
    cut_pos       = cuts[cut_index]
    seg_izq_start = boundaries[cut_index]
    seg_der_end   = boundaries[cut_index + 2]
    MAX_CTX = 3
    seg_izq = sentences[max(seg_izq_start, cut_pos - MAX_CTX) : cut_pos]
    seg_der = sentences[cut_pos : min(seg_der_end, cut_pos + MAX_CTX)]
    prompt = SUGGEST_PROMPT.format(
        seg_izq=" ".join(seg_izq),
        seg_der=" ".join(seg_der),
    )
    try:
        response = _call_llm(prompt)
    except RuntimeError:
        return "MANTENER"
    upper = response.upper()
    if "IZQUIERDA" in upper:
        return "IZQUIERDA"
    elif "DERECHA" in upper:
        return "DERECHA"
    return "MANTENER"


def judge_boundary(
    sentences: list[str],
    cut: int,
    window: int = 3,
) -> bool:
    """
    Decide si una posición es una frontera temática real (True=sí, False=no).
    NOTA: sin uso en el pipeline principal. Se conserva como filtro adicional opcional.
    """
    n = len(sentences)
    seg_izq = sentences[max(0, cut - window) : cut]
    seg_der = sentences[cut : min(n, cut + window)]
    if not seg_izq or not seg_der:
        return True
    seg_izq_text = " ".join(seg_izq[-2:])
    seg_der_text = " ".join(seg_der[:2])
    prompt = BOUNDARY_PROMPT.format(seg_izq=seg_izq_text, seg_der=seg_der_text)
    print(f"  [LLM] judge_boundary corte={cut}...", end=" ", flush=True)
    try:
        response = _call_llm(prompt)
    except RuntimeError:
        print("FALLO → True")
        return True
    result = _parse_bool(response)
    print(f"→ '{response[:15]}' → {'SI' if result else 'NO'}")
    return result


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------

def get_cache_stats() -> dict:
    total = _cache_hits + _cache_misses
    return {
        "hits":            _cache_hits,
        "misses":          _cache_misses,
        "total_calls":     total,
        "hit_rate":        _cache_hits / total if total > 0 else 0.0,
        "llm_calls_saved": _cache_hits,
    }


def reset_cache() -> None:
    global _eval_cache, _cache_hits, _cache_misses
    _eval_cache.clear()
    _cache_hits   = 0
    _cache_misses = 0


# ---------------------------------------------------------------------------
# Prueba standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA LLM LOCAL")
    print("=" * 60)

    print("\nTest 1: conexión básica...")
    try:
        resp = _call_llm("Responde SOLO con el número 7.", timeout=30)
        print(f"  Respuesta raw: '{resp}'")
        print(f"  Score parseado: {_parse_score(resp)}")
    except Exception as e:
        print(f"  FALLO: {e}")
        print("  Verifica que Ollama está corriendo: ollama serve")
        exit(1)

    print("\nTest 2: evaluate_segment (cohesivo)...")
    cohesive = [
        "La fotosíntesis convierte luz solar en energía química.",
        "Las plantas usan clorofila para capturar la luz.",
        "El proceso produce oxígeno y glucosa como subproductos.",
        "Sin fotosíntesis, la vida vegetal en la Tierra no sería posible.",
    ]
    score_coh = evaluate_segment(cohesive, 0, len(cohesive), verbose=False)
    print(f"  Score cohesivo (esperado ~7-10): {score_coh}")

    print("\nTest 3: evaluate_segment (mezclado)...")
    mixed = [
        "La fotosíntesis convierte luz solar en energía química.",
        "El Imperio romano dominó el Mediterráneo durante siglos.",
        "La Vía Láctea es una galaxia espiral barrada.",
        "Las plantas producen oxígeno mediante la clorofila.",
    ]
    score_mix = evaluate_segment(mixed, 0, len(mixed), verbose=False)
    print(f"  Score mezclado (esperado ~1-4): {score_mix}")

    print(f"\n{'─'*60}")
    if score_coh > score_mix + 2:
        print(f"✓ El LLM distingue bien: cohesivo={score_coh} vs mezclado={score_mix}")
    else:
        print(f"⚠ El LLM NO distingue bien: cohesivo={score_coh} vs mezclado={score_mix}")
        print("  Considera bajar llm_weight en pipeline.py o documentar la limitación.")

    stats = get_cache_stats()
    print(f"\nLlamadas LLM: {stats['misses']} | Cache hits: {stats['hits']}")
