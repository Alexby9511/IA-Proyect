"""
llm_evaluator.py
----------------
Fase 3 del sistema de segmentación óptima de contenido.

Evalúa la cohesión semántica de segmentos de texto usando
el LLM de Cerebras. Incluye:
  - Cache de evaluaciones para minimizar llamadas a la API
  - Evaluación de segmentos individuales (usada por el SA)
  - Sugerencia de movimiento guiado por LLM (mejora del SA)
  - Validación de fronteras para segmentación adaptativa (judge_boundary)

Requisitos:
    pip install cerebras-cloud-sdk python-dotenv

Uso como módulo:
    from llm_evaluator import evaluate_segment, suggest_move, judge_boundary

Uso standalone (prueba):
    python llm_evaluator.py
"""

import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise EnvironmentError(
        "No se encontró CEREBRAS_API_KEY en el archivo .env\n"
        "Asegúrate de tener un archivo .env con:\n"
        "CEREBRAS_API_KEY=tu_api_key"
    )

MODEL       = "gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # segundos entre reintentos

# Cache global: (inicio, fin, texto_hash) → puntuación
_eval_cache: dict[tuple, float] = {}
_cache_hits   = 0
_cache_misses = 0

# Cliente Cerebras (inicializado una vez)
_client = Cerebras(api_key=CEREBRAS_API_KEY)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVAL_PROMPT = """Analiza el siguiente fragmento de texto y evalúa qué tan cohesionado es temáticamente, es decir, qué tan bien todas sus oraciones tratan un mismo tema central.

Primero explica brevemente tu razonamiento (2-3 oraciones).
Luego, en la última línea, escribe SOLO un número entero del 1 al 10.

Texto:
{texto}"""

SUGGEST_PROMPT = """Tienes un texto dividido en segmentos. Analiza los dos segmentos adyacentes que se muestran y decide si el corte entre ellos está bien colocado o debería moverse.

Segmento izquierdo (termina en el corte actual):
{seg_izq}

Segmento derecho (empieza en el corte actual):
{seg_der}

Responde SOLO con una de estas tres opciones:
- IZQUIERDA (el corte debería moverse una posición hacia la izquierda)
- DERECHA (el corte debería moverse una posición hacia la derecha)
- MANTENER (el corte está bien donde está)

Respuesta:"""

BOUNDARY_PROMPT = """Analiza los dos fragmentos de texto que se muestran a continuación y decide si entre ellos hay un cambio de tema real.

Fragmento IZQUIERDO (final del segmento anterior):
{seg_izq}

Fragmento DERECHO (inicio del siguiente segmento):
{seg_der}

Responde SOLO con una de estas dos opciones:
- SI (hay un cambio de tema claro entre los dos fragmentos)
- NO (los dos fragmentos tratan el mismo tema o son muy similares)

Respuesta:"""


# ---------------------------------------------------------------------------
# Llamada al LLM con manejo de errores
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """
    Llama a la API de Cerebras con reintentos automáticos.
    Maneja respuestas None y errores de red.
    Retorna el texto de la respuesta o lanza RuntimeError si falla.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Respuesta vacía del LLM (content=None)")
            return content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [LLM] Error en intento {attempt+1}: {e}. Reintentando...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"LLM falló tras {MAX_RETRIES} intentos: {e}")


# ---------------------------------------------------------------------------
# Funciones de parsing
# ---------------------------------------------------------------------------

def _parse_score(response: str) -> float:
    """
    Extrae la puntuación numérica del 1 al 10 de la respuesta del LLM.
    Toma el último número entero que aparece en la respuesta.
    """
    numbers = re.findall(r'\b([1-9]|10)\b', response)
    if not numbers:
        return 5.0
    return float(numbers[-1])


def _make_cache_key(sentences: list[str], start: int, end: int) -> tuple:
    """Genera una clave única para el cache basada en el contenido del segmento."""
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
    Evalúa la cohesión semántica de un segmento.

    Parámetros:
        sentences : lista completa de oraciones del texto
        start     : índice de inicio del segmento (inclusivo)
        end       : índice de fin del segmento (exclusivo)
        verbose   : si True, muestra la respuesta completa del LLM

    Retorna:
        Puntuación de cohesión entre 1.0 y 10.0
        Devuelve 5.0 si la API falla (valor neutro, no detiene el SA)
    """
    global _cache_hits, _cache_misses

    if start >= end:
        return 0.0

    # Verificar cache
    key = _make_cache_key(sentences, start, end)
    if key in _eval_cache:
        _cache_hits += 1
        return _eval_cache[key]

    _cache_misses += 1

    texto  = " ".join(sentences[start:end])
    prompt = EVAL_PROMPT.format(texto=texto)

    try:
        response = _call_llm(prompt)
    except RuntimeError:
        # Si la API falla, devolver valor neutro sin detener el SA
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
    """
    Evalúa una partición completa sumando la cohesión de todos sus segmentos.

    Retorna:
        (score_total, [score_seg1, score_seg2, ...])
    """
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
    Pide al LLM que sugiera si un corte específico debería moverse.
    Usada por el SA para el movimiento guiado.

    Retorna:
        "IZQUIERDA", "DERECHA", o "MANTENER"
    """
    n             = len(sentences)
    boundaries    = [0] + cuts + [n]
    cut_pos       = cuts[cut_index]
    seg_izq_start = boundaries[cut_index]
    seg_der_end   = boundaries[cut_index + 2]

    MAX_CTX = 6
    seg_izq = sentences[max(seg_izq_start, cut_pos - MAX_CTX) : cut_pos]
    seg_der = sentences[cut_pos : min(seg_der_end, cut_pos + MAX_CTX)]

    prompt   = SUGGEST_PROMPT.format(
        seg_izq=" ".join(seg_izq),
        seg_der=" ".join(seg_der),
    )

    try:
        response = _call_llm(prompt)
    except RuntimeError:
        return "MANTENER"

    response_upper = response.upper()
    if "IZQUIERDA" in response_upper:
        return "IZQUIERDA"
    elif "DERECHA" in response_upper:
        return "DERECHA"
    return "MANTENER"


def judge_boundary(
    sentences: list[str],
    cut: int,
    window: int = 3,
) -> bool:
    """
    Decide si una posición de corte representa una frontera temática real.
    Usada por la segmentación adaptativa (adaptive_segmentation.py).

    Compara las `window` oraciones antes y después del corte.
    Devuelve True si el LLM detecta un cambio de tema claro.

    Parámetros:
        sentences : lista completa de oraciones
        cut       : posición del corte candidato
        window    : número de oraciones de contexto a cada lado

    Retorna:
        True  → hay cambio de tema, mantener el corte
        False → mismo tema, descartar el corte
        True  → si la API falla (conservador: no descartar por error)
    """
    n = len(sentences)

    seg_izq = sentences[max(0, cut - window) : cut]
    seg_der = sentences[cut : min(n, cut + window)]

    if not seg_izq or not seg_der:
        return True  # en los bordes, mantener el corte

    prompt = BOUNDARY_PROMPT.format(
        seg_izq=" ".join(seg_izq),
        seg_der=" ".join(seg_der),
    )

    try:
        response = _call_llm(prompt)
    except RuntimeError:
        return True  # si falla, asumir frontera válida

    return "SI" in response.upper()


def get_cache_stats() -> dict:
    """Retorna estadísticas del cache para análisis experimental."""
    total = _cache_hits + _cache_misses
    return {
        "hits":             _cache_hits,
        "misses":           _cache_misses,
        "total_calls":      total,
        "hit_rate":         _cache_hits / total if total > 0 else 0.0,
        "llm_calls_saved":  _cache_hits,
    }


def reset_cache() -> None:
    """Limpia el cache (útil entre instancias del experimento)."""
    global _eval_cache, _cache_hits, _cache_misses
    _eval_cache.clear()
    _cache_hits   = 0
    _cache_misses = 0


# ---------------------------------------------------------------------------
# Prueba standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA FASE 3 — LLM EVALUADOR (Cerebras)")
    print("=" * 60)

    for dataset_file in ["dataset.json", "dataset_synthetic.json"]:
        path = Path(dataset_file)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Dataset cargado: {dataset_file} ({len(dataset)} instancias)")
            break
    else:
        print("ERROR: No se encontró ningún dataset.")
        exit(1)

    instance  = dataset[0]
    sentences = instance["sentences"]
    K         = instance["K"]
    gt_cuts   = instance["ground_truth_cuts"]
    n         = instance["n_sentences"]

    print(f"\nInstancia : {instance['title']}")
    print(f"Oraciones : {n}  |  K={K}  |  GT cuts: {gt_cuts}")

    # Prueba 1: partición con ground truth
    print(f"\n{'─'*60}")
    print("Prueba 1: Partición con cortes reales (ground truth)")
    print(f"{'─'*60}")
    total_gt, scores_gt = evaluate_partition(sentences, gt_cuts, verbose=True)
    print(f"\nScore total (GT)   : {total_gt:.1f}")
    print(f"Score promedio     : {total_gt/K:.2f}/10")

    # Prueba 2: partición equidistante mala
    print(f"\n{'─'*60}")
    print("Prueba 2: Partición equidistante (mala)")
    print(f"{'─'*60}")
    step     = n // K
    bad_cuts = [step * i for i in range(1, K)]
    print(f"Cortes             : {bad_cuts}")
    total_bad, scores_bad = evaluate_partition(sentences, bad_cuts, verbose=True)
    print(f"\nScore total (malo) : {total_bad:.1f}")
    print(f"Score promedio     : {total_bad/K:.2f}/10")

    # Prueba 3: judge_boundary en el primer corte real
    print(f"\n{'─'*60}")
    print("Prueba 3: judge_boundary en cortes reales")
    print(f"{'─'*60}")
    for cut in gt_cuts:
        result = judge_boundary(sentences, cut, window=3)
        print(f"  Corte {cut}: {'✓ frontera válida' if result else '✗ descartado'}")

    # Estadísticas
    print(f"\n{'─'*60}")
    stats = get_cache_stats()
    print("Estadísticas del cache:")
    print(f"  Llamadas al LLM : {stats['misses']}")
    print(f"  Cache hits      : {stats['hits']}")
    print(f"  Tasa de acierto : {stats['hit_rate']:.0%}")