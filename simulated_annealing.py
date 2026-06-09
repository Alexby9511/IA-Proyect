"""
simulated_annealing.py
----------------------
Fase 2 del sistema de segmentación óptima de contenido.

Dado un texto dividido en oraciones, un número de segmentos K,
y un estado inicial de cortes (de la Fase 1), busca la partición
que maximiza la cohesión semántica total evaluada por el LLM (Fase 3).

Algoritmo: Simulated Annealing con dos tipos de movimiento:
  - Movimiento aleatorio (80%): desplaza un corte ±1 posición
  - Movimiento guiado (20%):    el LLM sugiere la dirección del desplazamiento

Optimizaciones:
  - Cache de evaluaciones LLM (heredado de llm_evaluator.py)
  - Solo evalúa los segmentos que cambian en cada iteración (1-2 llamadas/iter)
  - Iteraciones calculadas automáticamente según el tamaño del problema

Fórmula de iteraciones:
  max_iter = (N - 1) * (K - 1) * ITER_FACTOR
  donde N = número de oraciones, K = número de segmentos
  Ejemplo: N=88, K=7 → 87 * 6 * 5 = 2610 iteraciones

Requisitos:
    pip install sentence-transformers cerebras-cloud-sdk python-dotenv

Uso como módulo:
    from simulated_annealing import run_sa
    result = run_sa(sentences, K, initial_cuts, model_path)

Uso standalone:
    python simulated_annealing.py
"""

import json
import math
import random
import sys
import time
from pathlib import Path

from embeddings import get_ranked_candidates, get_initial_cuts, MODEL_A, MODEL_B
from llm_evaluator import evaluate_segment, suggest_move, get_cache_stats, reset_cache

# ---------------------------------------------------------------------------
# Parámetros del SA
# ---------------------------------------------------------------------------

TEMP_INITIAL   = 1.0
COOLING_RATE   = 0.995
ITER_FACTOR    = 5      # multiplicador para calcular iteraciones automáticamente
LLM_GUIDE_PROB = 0.0
MIN_SEGMENT_SIZE = 3   # por debajo de esto se penaliza el score
PENALTY_PER_MISSING = 0.5  # penalización por cada oración que falta para llegar al mínimo   # probabilidad de movimiento guiado por LLM


# ---------------------------------------------------------------------------
# Cálculo automático de iteraciones
# ---------------------------------------------------------------------------

def compute_max_iter(n_sentences: int, K: int) -> int:
    """
    Calcula el número de iteraciones en base al tamaño del problema.

    Fórmula: (N-1) * (K-1) * ITER_FACTOR
      - (N-1) = posiciones posibles de corte
      - (K-1) = cortes a colocar
      - ITER_FACTOR = factor de exploración (default 5)

    Ejemplos:
      N=24,  K=3 →  23 *  2 * 5 =   230 iteraciones
      N=88,  K=7 →  87 *  6 * 5 = 2,610 iteraciones
      N=89,  K=7 →  88 *  6 * 5 = 2,640 iteraciones
    """
    return (n_sentences - 1) * (K - 1) * ITER_FACTOR


# ---------------------------------------------------------------------------
# Barra de progreso
# ---------------------------------------------------------------------------

def print_progress(
    iteration: int,
    max_iter: int,
    current_score: float,
    best_score: float,
    temperature: float,
    n_improvements: int,
) -> None:
    """Imprime una barra de progreso en la misma línea."""
    pct      = iteration / max_iter
    bar_len  = 30
    filled   = int(bar_len * pct)
    bar      = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r  [{bar}] {pct:>5.1%} | "
        f"iter={iteration}/{max_iter} | "
        f"score={current_score:.2f} | "
        f"best={best_score:.2f} | "
        f"T={temperature:.3f} | "
        f"mejoras={n_improvements}",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Funciones auxiliares del SA
# ---------------------------------------------------------------------------

def size_penalty(segment_size: int) -> float:
    """
    Penalización suave por segmentos pequeños.

    Si el segmento tiene menos de MIN_SEGMENT_SIZE oraciones,
    se descuenta PENALTY_PER_MISSING por cada oración que falta.

    Ejemplos con MIN=3, PENALTY=0.5:
        tamaño=3 → penalización=0.0  (sin penalización)
        tamaño=2 → penalización=0.5  (falta 1)
        tamaño=1 → penalización=1.0  (faltan 2)
    """
    if segment_size >= MIN_SEGMENT_SIZE:
        return 0.0
    return (MIN_SEGMENT_SIZE - segment_size) * PENALTY_PER_MISSING


def compute_partition_score(
    sentences: list[str],
    cuts: list[int],
    changed_segments: list[int] | None = None,
    score_cache: dict | None = None,
) -> tuple[float, list[float]]:
    """
    Calcula el score total de una partición.
    Si se especifica changed_segments, solo recalcula esos segmentos
    y reutiliza los demás del score_cache → minimiza llamadas al LLM.
    """
    n          = len(sentences)
    boundaries = [0] + cuts + [n]
    K          = len(boundaries) - 1
    scores     = [0.0] * K

    for i in range(K):
        if (changed_segments is not None
                and i not in changed_segments
                and score_cache
                and i in score_cache):
            scores[i] = score_cache[i]
        else:
            seg_size   = boundaries[i + 1] - boundaries[i]
            llm_score  = evaluate_segment(sentences, boundaries[i], boundaries[i + 1])
            scores[i]  = llm_score - size_penalty(seg_size)

    return sum(scores), scores


def get_changed_segments(
    old_cuts: list[int],
    new_cuts: list[int],
    n: int,
) -> list[int]:
    """
    Identifica qué segmentos cambiaron entre dos estados de cortes.
    Un corte movido en posición i afecta los segmentos i e i+1.
    """
    changed        = set()
    old_boundaries = [0] + old_cuts + [n]
    new_boundaries = [0] + new_cuts + [n]

    for i, (ob, nb) in enumerate(zip(old_boundaries, new_boundaries)):
        if ob != nb:
            if i > 0:
                changed.add(i - 1)
            changed.add(i)

    return list(changed)


def random_move(cuts: list[int], n: int) -> tuple[list[int], int] | None:
    """
    Movimiento aleatorio: desplaza un corte elegido al azar ±1 posición.
    Retorna (nuevo_estado, índice_del_corte) o None si no hay movimiento válido.
    """
    if not cuts:
        return None

    for _ in range(10):
        idx       = random.randrange(len(cuts))
        direction = random.choice([-1, 1])
        new_pos   = cuts[idx] + direction

        if new_pos < 1 or new_pos >= n:
            continue
        if new_pos in cuts:
            continue

        new_cuts      = cuts[:]
        new_cuts[idx] = new_pos
        new_cuts.sort()
        return new_cuts, idx

    return None


def guided_move(
    sentences: list[str],
    cuts: list[int],
    n: int,
) -> tuple[list[int], int] | None:
    """
    Movimiento guiado: el LLM sugiere la dirección para un corte elegido al azar.
    Retorna (nuevo_estado, índice_del_corte) o None si el LLM dice MANTENER.
    """
    if not cuts:
        return None

    idx        = random.randrange(len(cuts))
    suggestion = suggest_move(sentences, cuts, cut_index=idx)

    if suggestion == "MANTENER":
        return None

    direction = -1 if suggestion == "IZQUIERDA" else 1
    new_pos   = cuts[idx] + direction

    if new_pos < 1 or new_pos >= n or new_pos in cuts:
        return None

    new_cuts      = cuts[:]
    new_cuts[idx] = new_pos
    new_cuts.sort()
    return new_cuts, idx


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------

def run_sa(
    sentences: list[str],
    K: int,
    initial_cuts: list[int],
    temp_initial:   float = TEMP_INITIAL,
    cooling_rate:   float = COOLING_RATE,
    iter_factor:    int   = ITER_FACTOR,
    llm_guide_prob: float = LLM_GUIDE_PROB,
    verbose:        bool  = True,
) -> dict:
    """
    Ejecuta el Simulated Annealing para encontrar la mejor partición en K segmentos.

    Parámetros:
        sentences       : lista de oraciones del texto
        K               : número de segmentos
        initial_cuts    : cortes iniciales (de la Fase 1)
        temp_initial    : temperatura inicial
        cooling_rate    : factor de enfriamiento por iteración
        iter_factor     : multiplicador para calcular iteraciones automáticamente
        llm_guide_prob  : probabilidad de movimiento guiado por LLM
        verbose         : si True, muestra barra de progreso

    Retorna dict con:
        best_cuts, best_score, best_scores,
        initial_cuts, initial_score,
        n_iter, n_llm_guided, n_improvements, n_accepted_worse,
        history, cache_stats
    """
    n        = len(sentences)
    max_iter = compute_max_iter(n, K)

    current     = initial_cuts[:]
    temperature = temp_initial

    # Evaluar estado inicial
    current_score, current_scores = compute_partition_score(sentences, current)
    score_cache = {i: s for i, s in enumerate(current_scores)}

    best          = current[:]
    best_score    = current_score
    best_scores   = current_scores[:]
    initial_score = current_score

    n_llm_guided     = 0
    n_improvements   = 0
    n_accepted_worse = 0
    history          = [(0, current_score)]

    if verbose:
        print(f"  Iteraciones calculadas: {max_iter}  "
              f"(N={n}, K={K}, factor={iter_factor})")
        print(f"  Score inicial: {current_score:.2f} | Cortes: {current}")
        print()

    for iteration in range(1, max_iter + 1):

        # Mostrar progreso cada 10 iteraciones
        if verbose and iteration % 10 == 0:
            print_progress(iteration, max_iter, current_score, best_score,
                           temperature, n_improvements)

        # 1. Elegir tipo de movimiento
        use_guided = random.random() < llm_guide_prob
        if use_guided:
            result = guided_move(sentences, current, n)
            if result is not None:
                n_llm_guided += 1
        else:
            result = random_move(current, n)

        if result is None:
            temperature *= cooling_rate
            continue

        new_cuts, _ = result

        # 2. Calcular score solo de segmentos afectados
        changed = get_changed_segments(current, new_cuts, n)
        new_score, new_scores = compute_partition_score(
            sentences, new_cuts,
            changed_segments=changed,
            score_cache=score_cache,
        )

        # 3. Decisión de aceptación
        delta = new_score - current_score

        if delta > 0:
            accept = True
            n_improvements += 1
        else:
            prob   = math.exp(delta / temperature) if temperature > 1e-10 else 0.0
            accept = random.random() < prob
            if accept:
                n_accepted_worse += 1

        if accept:
            current        = new_cuts
            current_score  = new_score
            current_scores = new_scores
            score_cache    = {i: s for i, s in enumerate(current_scores)}

            if current_score > best_score:
                best        = current[:]
                best_score  = current_score
                best_scores = current_scores[:]

        # 4. Historial cada 50 iteraciones
        if iteration % 50 == 0:
            history.append((iteration, best_score))

        # 5. Enfriar
        temperature *= cooling_rate

    if verbose:
        # Imprimir progreso final en 100%
        print_progress(max_iter, max_iter, current_score, best_score,
                       temperature, n_improvements)
        print()  # salto de línea tras la barra

    return {
        "best_cuts":        best,
        "best_score":       best_score,
        "best_scores":      best_scores,
        "initial_cuts":     initial_cuts,
        "initial_score":    initial_score,
        "n_iter":           max_iter,
        "n_llm_guided":     n_llm_guided,
        "n_improvements":   n_improvements,
        "n_accepted_worse": n_accepted_worse,
        "history":          history,
        "cache_stats":      get_cache_stats(),
    }


# ---------------------------------------------------------------------------
# Prueba standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("PRUEBA FASE 2 — SIMULATED ANNEALING")
    print("=" * 65)

    for fname in ["dataset.json", "dataset_synthetic.json"]:
        p = Path(fname)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Dataset: {fname} ({len(dataset)} instancias)")
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

    # Fase 1
    print(f"\n{'─'*65}")
    print("Fase 1: calculando cortes iniciales...")
    ranked       = get_ranked_candidates(sentences, MODEL_B)
    initial_cuts = get_initial_cuts(ranked, K)
    print(f"Cortes iniciales: {initial_cuts}")

    # Fase 2
    print(f"\n{'─'*65}")
    print("Fase 2: ejecutando Simulated Annealing...")
    reset_cache()
    t_start = time.time()

    result  = run_sa(sentences=sentences, K=K, initial_cuts=initial_cuts)
    elapsed = time.time() - t_start

    # Resultados
    print(f"\n{'─'*65}")
    print("RESULTADOS")
    print(f"{'─'*65}")
    print(f"Cortes iniciales : {result['initial_cuts']}  (score={result['initial_score']:.2f})")
    print(f"Mejor partición  : {result['best_cuts']}  (score={result['best_score']:.2f})")
    print(f"Ground truth     : {gt_cuts}")
    print(f"Scores por seg.  : {[f'{s:.1f}' for s in result['best_scores']]}")
    print(f"Mejora obtenida  : {result['best_score'] - result['initial_score']:+.2f}")
    print(f"Tiempo total     : {elapsed:.1f}s")

    print(f"\n{'─'*65}")
    print("ESTADÍSTICAS SA")
    print(f"{'─'*65}")
    print(f"  Iteraciones totales  : {result['n_iter']}")
    print(f"  Mejoras aceptadas    : {result['n_improvements']}")
    print(f"  Peores aceptadas     : {result['n_accepted_worse']}")
    print(f"  Movimientos LLM      : {result['n_llm_guided']}")

    stats = result["cache_stats"]
    print(f"\n  Llamadas LLM reales  : {stats['misses']}")
    print(f"  Cache hits           : {stats['hits']}")
    print(f"  Tasa de cache        : {stats['hit_rate']:.0%}")
    print(f"  Llamadas ahorradas   : {stats['llm_calls_saved']}")

    print(f"\n{'─'*65}")
    best_set = set(result["best_cuts"])
    gt_set   = set(gt_cuts)
    exactos  = len(best_set & gt_set)
    print(f"Cortes exactos vs GT : {exactos}/{len(gt_set)}")