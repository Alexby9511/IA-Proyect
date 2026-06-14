"""
simulated_annealing.py
----------------------
Fase de optimización: Simulated Annealing para refinar los cortes
de una segmentación, usando ÚNICAMENTE embeddings como función de coste.

Arquitectura (LLM-al-final):
  - El SA optimiza score_partition_embeddings (de adaptive_segmentation.py),
    la MISMA métrica que se usa para comparar entre K y combinar con el LLM
    en pipeline.py. Esto evita que el SA converja a particiones degeneradas
    (ej. segmentos de tamaño 1 con cohesión artificial de 1.0) que parecían
    buenas bajo una métrica distinta a la final.
  - El LLM se usa SOLO al final del pipeline (ver pipeline.py), evaluando
    una vez cada segmento de la partición final de cada K candidato.
  - guided_move (movimiento sugerido por LLM) queda definido pero sin uso
    en el bucle principal (LLM_GUIDE_PROB=0.0). Se conserva como variante
    experimental.

Requisitos:
    pip install numpy
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np

from embeddings import compute_embeddings, get_ranked_candidates, get_initial_cuts, MODEL_A, MODEL_B
from adaptive_segmentation import score_partition_embeddings, compute_segment_cohesions
from llm_evaluator import suggest_move

# ---------------------------------------------------------------------------
# Parámetros del SA
# ---------------------------------------------------------------------------
TEMP_INITIAL     = 1.0
COOLING_RATE     = 0.995
ITER_FACTOR      = 5
MIN_SEGMENT_SIZE = 3

# LLM_GUIDE_PROB = 0.0: el LLM no participa en el bucle de SA.
LLM_GUIDE_PROB = 0.0


def compute_max_iter(n_sentences: int, K: int, cap: int = 2000) -> int:
    """
    Calcula el número de iteraciones del SA.
    min((N-1) * (K-1) * ITER_FACTOR, cap)
    Como el SA no llama al LLM, cap puede ser alto sin impacto relevante.
    """
    raw = (n_sentences - 1) * (K - 1) * ITER_FACTOR
    return min(raw, cap)


def print_progress(iteration, max_iter, current_score, best_score, temperature, n_improvements):
    pct = iteration / max_iter
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:>5.1%} | iter={iteration}/{max_iter} | "
          f"score={current_score:.3f} | best={best_score:.3f} | "
          f"T={temperature:.3f} | mejoras={n_improvements}",
          end="", flush=True)


def random_move(cuts: list[int], n: int) -> tuple[list[int], int] | None:
    """Desplaza un corte al azar ±1 posición."""
    if not cuts:
        return None
    for _ in range(10):
        idx = random.randrange(len(cuts))
        direction = random.choice([-1, 1])
        new_pos = cuts[idx] + direction
        if new_pos < 1 or new_pos >= n:
            continue
        if new_pos in cuts:
            continue
        new_cuts = cuts[:]
        new_cuts[idx] = new_pos
        new_cuts.sort()
        return new_cuts, idx
    return None


def guided_move(sentences: list[str], cuts: list[int], n: int) -> tuple[list[int], int] | None:
    """
    Movimiento guiado por el LLM. SIN USO en el flujo actual (LLM_GUIDE_PROB=0.0).
    Se conserva como variante experimental.
    """
    if not cuts:
        return None
    idx = random.randrange(len(cuts))
    suggestion = suggest_move(sentences, cuts, cut_index=idx)
    if suggestion == "MANTENER":
        return None
    direction = -1 if suggestion == "IZQUIERDA" else 1
    new_pos = cuts[idx] + direction
    if new_pos < 1 or new_pos >= n or new_pos in cuts:
        return None
    new_cuts = cuts[:]
    new_cuts[idx] = new_pos
    new_cuts.sort()
    return new_cuts, idx


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------

def run_sa(
    sentences: list[str],
    embeddings: np.ndarray,
    K: int,
    initial_cuts: list[int],
    min_seg: int = MIN_SEGMENT_SIZE,
    temp_initial: float = TEMP_INITIAL,
    cooling_rate: float = COOLING_RATE,
    iter_factor: int = ITER_FACTOR,
    max_iter_cap: int = 2000,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta el Simulated Annealing usando score_partition_embeddings
    como función de coste en TODO el bucle (aceptación, best_score,
    history). El LLM NO se usa aquí.

    Al usar la misma métrica que pipeline.py para comparar entre K y
    combinar con llm_score, se garantiza que el SA no converge a
    particiones degeneradas (segmentos trivialmente cohesionados de
    tamaño 1) que serían penalizadas después por score_partition_embeddings.

    Parámetros:
        sentences    : lista de oraciones (por compatibilidad y guided_move)
        embeddings   : embeddings precalculados
        K            : número de segmentos
        initial_cuts : cortes iniciales (de la Fase 1)
        min_seg      : tamaño mínimo de segmento (para la penalización)
        max_iter_cap : tope de iteraciones

    Retorna dict con:
        best_cuts, best_score, best_scores (cohesión por segmento),
        initial_cuts, initial_score,
        n_iter, n_improvements, n_accepted_worse, history
    """
    n        = len(sentences)
    max_iter = compute_max_iter(n, K, cap=max_iter_cap)

    current     = initial_cuts[:]
    temperature = temp_initial

    # Función de coste unificada: score_partition_embeddings
    current_score = score_partition_embeddings(embeddings, current, n, min_seg)

    best          = current[:]
    best_score    = current_score
    initial_score = current_score

    n_improvements   = 0
    n_accepted_worse = 0
    history          = [(0, current_score)]

    if verbose:
        print(f"  Iteraciones calculadas: {max_iter} (N={n}, K={K}, cap={max_iter_cap})")
        print(f"  Score inicial (emb, 0-1): {current_score:.3f} | Cortes: {current}")
        print()

    for iteration in range(1, max_iter + 1):
        if verbose and iteration % 200 == 0:
            print_progress(iteration, max_iter, current_score, best_score,
                           temperature, n_improvements)

        result = random_move(current, n)

        if result is None:
            temperature *= cooling_rate
            continue

        new_cuts, _ = result

        # Misma métrica que la final: score_partition_embeddings
        new_score = score_partition_embeddings(embeddings, new_cuts, n, min_seg)

        delta = new_score - current_score
        if delta > 0:
            accept = True
            n_improvements += 1
        else:
            prob = math.exp(delta / temperature) if temperature > 1e-10 else 0.0
            accept = random.random() < prob
            if accept:
                n_accepted_worse += 1

        if accept:
            current       = new_cuts
            current_score = new_score
            if current_score > best_score:
                best       = current[:]
                best_score = current_score

        if iteration % 100 == 0:
            history.append((iteration, best_score))

        temperature *= cooling_rate

    if verbose:
        print_progress(max_iter, max_iter, current_score, best_score,
                       temperature, n_improvements)
        print()

    # Cohesión por segmento del mejor estado (para reporting)
    best_scores = compute_segment_cohesions(embeddings, best, n)

    return {
        "best_cuts":        best,
        "best_score":       best_score,
        "best_scores":      best_scores,
        "initial_cuts":     initial_cuts,
        "initial_score":    initial_score,
        "n_iter":           max_iter,
        "n_improvements":   n_improvements,
        "n_accepted_worse": n_accepted_worse,
        "history":          history,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("PRUEBA — SIMULATED ANNEALING SOBRE EMBEDDINGS (sin LLM)")
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

    instance = dataset[0]
    sentences = instance["sentences"]
    K = instance["K"]
    gt_cuts = instance["ground_truth_cuts"]
    n = instance["n_sentences"]
    print(f"\nInstancia : {instance['title']}")
    print(f"Oraciones : {n}  |  K={K}  |  GT cuts: {gt_cuts}")

    print(f"\n{'─'*65}")
    print("Fase 1: calculando embeddings y cortes iniciales...")
    embeddings = compute_embeddings(sentences, MODEL_B)
    ranked = get_ranked_candidates(sentences, MODEL_B)
    initial_cuts = get_initial_cuts(ranked, K)
    print(f"Cortes iniciales: {initial_cuts}")

    print(f"\n{'─'*65}")
    print("Fase 2: ejecutando SA sobre embeddings...")
    t_start = time.time()
    result = run_sa(sentences=sentences, embeddings=embeddings, K=K, initial_cuts=initial_cuts)
    elapsed = time.time() - t_start

    print(f"\n{'─'*65}")
    print("RESULTADOS")
    print(f"{'─'*65}")
    print(f"Cortes iniciales : {result['initial_cuts']}  (score={result['initial_score']:.3f})")
    print(f"Mejor partición  : {result['best_cuts']}  (score={result['best_score']:.3f})")
    print(f"Ground truth     : {gt_cuts}")
    print(f"Scores por seg.  : {[f'{s:.3f}' for s in result['best_scores']]}")
    print(f"Mejora obtenida  : {result['best_score'] - result['initial_score']:+.3f}")
    print(f"Tiempo total     : {elapsed:.3f}s")
    print(f"Iteraciones      : {result['n_iter']}")
    print(f"Mejoras aceptadas: {result['n_improvements']}")
    print(f"Peores aceptadas : {result['n_accepted_worse']}")
