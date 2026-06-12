"""
llm_evaluator.py (Ollama local - endpoint generate)
---------------------------------------------------
Usa el endpoint /api/generate de Ollama, más rápido y compatible
con cualquier modelo, incluso si solo soporta "completion".

Requisitos:
    pip install requests

Uso:
    from llm_evaluator import evaluate_segment, suggest_move, judge_boundary
"""

import json
import re
import time
from pathlib import Path
import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
OLLAMA_URL   = "http://localhost:11434/api/generate"   # endpoint de generación
MODEL        = "deepseek-local:1.5b"
MAX_RETRIES  = 2
RETRY_DELAY  = 5.0

# Cache de evaluaciones
_eval_cache: dict[tuple, float] = {}
_cache_hits   = 0
_cache_misses = 0

# ---------------------------------------------------------------------------
# Prompts (versiones acortadas para máxima velocidad)
# ---------------------------------------------------------------------------
EVAL_PROMPT = """Evalúa la cohesión temática de este texto (1-10).
Responde solo con el número.

Texto: {texto}
Número:"""

SUGGEST_PROMPT = """Dados dos fragmentos consecutivos:
IZQUIERDA: {seg_izq}
DERECHA: {seg_der}

¿El corte entre ellos está bien colocado?
Responde SOLO con: IZQUIERDA, DERECHA o MANTENER.
Respuesta:"""

BOUNDARY_PROMPT = """¿Hay cambio de tema entre estas dos oraciones?
Oración 1: {seg_izq}
Oración 2: {seg_der}
Responde SOLO con SI o NO.
Respuesta:"""

# ---------------------------------------------------------------------------
# Llamada al LLM local
# ---------------------------------------------------------------------------
def _call_llm(prompt: str, timeout: int = 120) -> str:
    """
    Llama a Ollama usando el endpoint /api/generate.
    Retorna el texto de la respuesta o lanza RuntimeError si falla.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 10          # suficiente para SI/NO o un número
        }
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "").strip()
            if not content:
                raise ValueError("Respuesta vacía del LLM")
            return content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [LLM] Error en intento {attempt+1}: {e}. Reintentando...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"LLM local falló tras {MAX_RETRIES} intentos: {e}")

# ---------------------------------------------------------------------------
# Parsing de puntuaciones
# ---------------------------------------------------------------------------
def _parse_score(response: str) -> float:
    """Extrae el primer número entero del 1 al 10 de la respuesta."""
    numbers = re.findall(r'\b([1-9]|10)\b', response)
    return float(numbers[-1]) if numbers else 5.0

def _make_cache_key(sentences: list[str], start: int, end: int) -> tuple:
    text = " ".join(sentences[start:end])
    return (start, end, hash(text))

# ---------------------------------------------------------------------------
# Evaluación de cohesión de segmentos
# ---------------------------------------------------------------------------
def evaluate_segment(
    sentences: list[str],
    start: int,
    end: int,
    verbose: bool = False,
) -> float:
    """
    Evalúa la cohesión semántica de un segmento.
    Retorna puntuación 1-10 (o 5.0 si hay error).
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
    prompt = EVAL_PROMPT.format(texto=texto)

    try:
        response = _call_llm(prompt)
    except RuntimeError:
        return 5.0

    if verbose:
        print(f"\n  [LLM respuesta] oraciones[{start}:{end}]:")
        print(f"  {response}")

    score = _parse_score(response)
    _eval_cache[key] = score
    return score

def evaluate_partition(
    sentences: list[str],
    cuts: list[int],
    verbose: bool = False,
) -> tuple[float, list[float]]:
    """Evalúa una partición completa, devolviendo (score_total, [score_seg1, ...])."""
    n = len(sentences)
    boundaries = [0] + cuts + [n]
    scores = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        score = evaluate_segment(sentences, start, end, verbose=verbose)
        scores.append(score)
        if verbose:
            print(f"  Segmento {i+1} [{start}:{end}]: {score:.1f}/10")
    return sum(scores), scores

# ---------------------------------------------------------------------------
# Movimiento guiado para Simulated Annealing
# ---------------------------------------------------------------------------
def suggest_move(
    sentences: list[str],
    cuts: list[int],
    cut_index: int,
) -> str:
    """Sugiere IZQUIERDA, DERECHA o MANTENER para un corte."""
    n = len(sentences)
    boundaries = [0] + cuts + [n]
    cut_pos = cuts[cut_index]
    seg_izq_start = boundaries[cut_index]
    seg_der_end = boundaries[cut_index + 2]

    MAX_CTX = 4  # reducido para velocidad
    seg_izq = sentences[max(seg_izq_start, cut_pos - MAX_CTX): cut_pos]
    seg_der = sentences[cut_pos: min(seg_der_end, cut_pos + MAX_CTX)]

    prompt = SUGGEST_PROMPT.format(
        seg_izq=" ".join(seg_izq),
        seg_der=" ".join(seg_der)
    )
    try:
        response = _call_llm(prompt)
    except RuntimeError:
        return "MANTENER"

    resp_upper = response.upper()
    if "IZQUIERDA" in resp_upper:
        return "IZQUIERDA"
    elif "DERECHA" in resp_upper:
        return "DERECHA"
    return "MANTENER"

# ---------------------------------------------------------------------------
# Validación de fronteras para segmentación adaptativa
# ---------------------------------------------------------------------------
def judge_boundary(
    sentences: list[str],
    cut: int,
    window: int = 1,          # solo una oración por lado
) -> bool:
    """
    Decide si un corte es una frontera temática real.
    Compara las dos oraciones adyacentes al corte.
    Retorna True si hay cambio de tema.
    """
    n = len(sentences)
    seg_izq = sentences[max(0, cut - window): cut]
    seg_der = sentences[cut: min(n, cut + window)]

    if not seg_izq or not seg_der:
        return True  # en los bordes, asumir frontera

    prompt = BOUNDARY_PROMPT.format(
        seg_izq=seg_izq[0] if seg_izq else "",
        seg_der=seg_der[0] if seg_der else ""
    )

    try:
        response = _call_llm(prompt, timeout=120)
    except RuntimeError:
        return True  # conservador: si falla, mantener el corte

    return "SI" in response.upper()

# ---------------------------------------------------------------------------
# Estadísticas de cache
# ---------------------------------------------------------------------------
def get_cache_stats() -> dict:
    total = _cache_hits + _cache_misses
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total_calls": total,
        "hit_rate": _cache_hits / total if total > 0 else 0.0,
        "llm_calls_saved": _cache_hits,
    }

def reset_cache() -> None:
    global _eval_cache, _cache_hits, _cache_misses
    _eval_cache.clear()
    _cache_hits = 0
    _cache_misses = 0

# ---------------------------------------------------------------------------
# Prueba standalone (rápida)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA LLM LOCAL (deepseek-local:1.5b)")
    print("=" * 60)

    # Prueba de conexión simple
    try:
        test_resp = _call_llm("Responde solo: Hola", timeout=30)
        print(f"Test conexión: {test_resp}")
    except Exception as e:
        print(f"Fallo conexión: {e}")
        exit(1)

    # Cargar dataset
    for dataset_file in ["dataset.json", "dataset_synthetic.json"]:
        path = Path(dataset_file)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Dataset cargado: {dataset_file}")
            break
    else:
        print("No se encontró dataset")
        exit(1)

    instance = dataset[0]
    sentences = instance["sentences"]
    K = instance["K"]
    gt_cuts = instance["ground_truth_cuts"]
    print(f"\nInstancia: {instance['title']}")
    print(f"Oraciones: {len(sentences)} | K={K} | GT cuts: {gt_cuts}")

    # Probar judge_boundary con los cortes reales
    print("\nValidación de cortes reales (window=1):")
    for cut in gt_cuts:
        result = judge_boundary(sentences, cut, window=1)
        print(f"  Corte {cut}: {'✓' if result else '✗'}")