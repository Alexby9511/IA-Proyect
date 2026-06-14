"""
pipeline.py
-----------
Pipeline de segmentación basado en DP con Grafo Esparcido y poda por
embeddings (ver dp_segmentation.py para el detalle del costo de cada
arista y las podas aplicadas).
"""

from typing import Any, List
import numpy as np

from embeddings import compute_embeddings, MODEL_B
from dp_segmentation import run_dp_segmentation
from llm_evaluator import reset_cache, get_cache_stats
from adaptive_segmentation import segment_cohesion


def run_pipeline(
    sentences: List[str],
    model_path: str = MODEL_B,
    min_seg: int = 3,
    lambda_penalty: float = 3.0,
    max_seg_len: int | None = None,
    min_cohesion: float = 0.5,
    embedding_weight: float = 0.3,
    **kwargs,
) -> dict[str, Any]:
    n = len(sentences)
    print(f"\n[Pipeline] Preparando modelo matemático...")

    if n < 2:
        return {
            "best_cuts": [], "best_K": 1, "segments": [sentences],
            "n_sentences": n, "cohesion_mean": 1.0, "seg_cohesions": [1.0],
            "total_llm_calls": 0, "cache_stats": get_cache_stats(),
            "dp_stats": {},
        }

    # 1. Representaciones base
    embeddings = compute_embeddings(sentences, model_path)
    reset_cache()

    # 2. Ejecutar motor de Programación Dinámica con poda por embeddings
    cuts, dp_stats = run_dp_segmentation(
        sentences=sentences,
        embeddings=embeddings,
        min_seg=min_seg,
        lambda_penalty=lambda_penalty,
        max_seg_len=max_seg_len,
        min_cohesion=min_cohesion,
        embedding_weight=embedding_weight,
    )

    # 3. Formatear salida
    K = len(cuts) + 1
    boundaries = [0] + cuts + [n]
    segments = [sentences[boundaries[i]:boundaries[i + 1]] for i in range(K)]

    cohs = [segment_cohesion(embeddings, boundaries[i], boundaries[i + 1]) for i in range(K)]
    cohesion_mean = float(np.mean(cohs)) if cohs else 0.0

    cache_stats     = get_cache_stats()
    total_llm_calls = cache_stats["misses"]

    print(f"\n[Pipeline] ¡Camino óptimo encontrado! Cortes finales: {cuts}")

    return {
        "best_cuts": cuts,
        "best_K": K,
        "segments": segments,
        "n_sentences": n,
        "cohesion_mean": cohesion_mean,
        "seg_cohesions": cohs,
        "total_llm_calls": total_llm_calls,
        "cache_stats": cache_stats,
        "dp_stats": dp_stats,
    }