"""
pipeline.py
-----------
Pipeline integrado de segmentación óptima.

Flujo del LLM (importante para el informe — "Rol del LLM en el sistema"):
  1. Para cada K candidato (2..max_k), se ejecuta Simulated Annealing
     usando ÚNICAMENTE el score de embeddings (rápido, sin red).
  2. Sobre la partición resultante de ese K, el LLM se llama UNA VEZ POR
     SEGMENTO (K llamadas) para evaluar su cohesión semántica (1-10).
  3. emb_score y llm_score (ambos normalizados a [0,1]) se combinan con
     llm_weight para obtener el score combinado de ese K.
  4. Se elige el K con mejor score combinado.

  Total de llamadas al LLM ≈ sum(K) para K=2..max_possible_k — del orden
  de decenas, no de miles.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from embeddings import (
    compute_embeddings,
    compute_similarity_profile,
    find_local_valleys,
    get_ranked_candidates,
    get_initial_cuts,
    MODEL_B,
)
from adaptive_segmentation import score_partition_embeddings, segment_cohesion
from simulated_annealing import run_sa
from llm_evaluator import evaluate_segment, reset_cache, get_cache_stats


def run_pipeline(
    sentences: List[str],
    model_path: str = MODEL_B,
    min_seg: int = 3,
    max_k: int = 8,
    llm_weight: float = 0.4,
    max_iter_cap: int = 2000,
    use_sa: bool = True,
) -> dict[str, Any]:
    n = len(sentences)
    print(f"\n[Pipeline] Iniciando con {n} oraciones, max_k={max_k}, use_sa={use_sa}")

    if n < 2:
        print("[Pipeline] Texto demasiado corto, devolviendo segmento único")
        return {
            "best_cuts": [], "best_K": 1, "segments": [sentences],
            "n_sentences": n, "cohesion_mean": 1.0, "seg_cohesions": [1.0],
            "combined_score": 0.0, "llm_score": 0.0, "emb_score": 1.0,
            "total_llm_calls": 0, "cache_stats": get_cache_stats(), "scores_por_K": {},
        }

    # 1. Embeddings
    print(f"[Pipeline] Paso 1/4: Calculando embeddings...")
    embeddings   = compute_embeddings(sentences, model_path)
    similarities = compute_similarity_profile(embeddings)
    print(f"[Pipeline] Embeddings calculados. Perfil de similitud: {len(similarities)} valores")

    # 2. Valles
    print(f"[Pipeline] Paso 2/4: Detectando valles locales...")
    valleys = find_local_valleys(similarities)
    print(f"[Pipeline] Valles encontrados: {len(valleys)}")

    if not valleys:
        print("[Pipeline] Sin valles detectados, devolviendo segmento único")
        return {
            "best_cuts": [], "best_K": 1, "segments": [sentences],
            "n_sentences": n, "cohesion_mean": segment_cohesion(embeddings, 0, n),
            "seg_cohesions": [segment_cohesion(embeddings, 0, n)],
            "combined_score": 1.0, "llm_score": 0.0, "emb_score": 1.0,
            "total_llm_calls": 0, "cache_stats": get_cache_stats(), "scores_por_K": {},
        }

    # 3. Probar cada K
    max_possible_k = min(max_k, n // min_seg, len(valleys) + 1)
    print(f"[Pipeline] Paso 3/4: Probando K de 2 a {max_possible_k}...")

    best_combined = -np.inf
    best_result   = None
    scores_por_K  = {}

    for K in range(2, max_possible_k + 1):
        print(f"\n[Pipeline]   → K={K}:")

        selected_valleys = valleys[:K - 1]
        initial_cuts     = sorted([int(idx) + 1 for idx, _ in selected_valleys])
        print(f"[Pipeline]     Cortes iniciales: {initial_cuts}")

        if use_sa:
            print(f"[Pipeline]     Ejecutando SA sobre embeddings "
                  f"(rápido, sin LLM, max_iter_cap={max_iter_cap})...")
            sa_result = run_sa(
                sentences,
                embeddings,
                K=K,
                initial_cuts=initial_cuts,
                min_seg=min_seg,
                max_iter_cap=max_iter_cap,
                verbose=True,
            )
            cuts = sa_result["best_cuts"]
            print(f"[Pipeline]     SA terminado. Cortes: {cuts} | "
                  f"score_emb_SA={sa_result['best_score']:.3f} | "
                  f"iter={sa_result['n_iter']} | mejoras={sa_result['n_improvements']}")
        else:
            cuts = initial_cuts
            print(f"[Pipeline]     SA desactivado, usando cortes iniciales: {cuts}")

        # Evaluación final con el LLM: una llamada por segmento de esta partición
        print(f"[Pipeline]     Evaluando partición final con LLM ({K} llamadas)...")
        reset_cache()
        boundaries = [0] + cuts + [n]
        seg_llm_scores = []
        for i in range(K):
            score = evaluate_segment(sentences, boundaries[i], boundaries[i + 1])
            seg_llm_scores.append(score)
        llm_score = (sum(seg_llm_scores) / K) / 10.0

        cache_stats     = get_cache_stats()
        total_llm_calls = cache_stats["misses"]

        emb_score = score_partition_embeddings(embeddings, cuts, n, min_seg=min_seg)
        combined  = (1 - llm_weight) * emb_score + llm_weight * llm_score

        print(f"[Pipeline]     emb_score={emb_score:.3f} | llm_score={llm_score:.3f} | combined={combined:.3f}")

        scores_por_K[K] = {
            "cuts": cuts, "emb_score": emb_score,
            "llm_score": llm_score, "combined": combined,
            "llm_calls": total_llm_calls,
        }

        if combined > best_combined:
            best_combined = combined
            best_result   = {
                "best_cuts": cuts, "best_K": K,
                "combined_score": combined,
                "emb_score": emb_score, "llm_score": llm_score,
                "total_llm_calls": total_llm_calls,
                "cache_stats": cache_stats,
            }
            print(f"[Pipeline]     ★ Nuevo mejor K={K} con combined={combined:.3f}")

    if best_result is None:
        print("[Pipeline] Ningún K válido encontrado, devolviendo segmento único")
        return {
            "best_cuts": [], "best_K": 1, "segments": [sentences],
            "n_sentences": n, "cohesion_mean": segment_cohesion(embeddings, 0, n),
            "seg_cohesions": [segment_cohesion(embeddings, 0, n)],
            "combined_score": 1.0, "llm_score": 0.0, "emb_score": 1.0,
            "total_llm_calls": 0, "cache_stats": get_cache_stats(), "scores_por_K": scores_por_K,
        }

    # 4. Construir resultado final
    print(f"\n[Pipeline] Paso 4/4: Construyendo resultado final (K={best_result['best_K']})...")
    boundaries    = [0] + best_result["best_cuts"] + [n]
    K_best        = best_result["best_K"]
    segments      = [sentences[boundaries[i]:boundaries[i + 1]] for i in range(K_best)]
    cohs          = [segment_cohesion(embeddings, boundaries[i], boundaries[i + 1]) for i in range(K_best)]
    cohesion_mean = float(np.mean(cohs)) if cohs else 0.0

    best_result.update({
        "segments": segments, "n_sentences": n,
        "cohesion_mean": cohesion_mean, "seg_cohesions": cohs,
        "scores_por_K": scores_por_K,
    })

    print(f"[Pipeline] ✓ Completado. K={K_best} | cohesion_media={cohesion_mean:.3f} | "
          f"LLM_calls(K elegido)={best_result['total_llm_calls']}")
    return best_result