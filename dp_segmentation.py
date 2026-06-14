"""
dp_segmentation.py
------------------
Segmentación Multiescala Adaptativa con Validación de Cortes por LLM.

Cambio: los puntos de corte candidatos ya NO se buscan exhaustivamente en
cada nivel de recursión (lo cual recalculaba similitudes coseno una y otra
vez sobre el mismo texto). En su lugar, se precomputan UNA SOLA VEZ los
"valles" del perfil de similitud (find_local_valleys, de embeddings.py:
mínimos locales ordenados por profundidad descendente) para todo el texto,
y tanto get_subtheme_cuts como hybrid_segmentation filtran esa lista al
rango que les corresponde. Esto va "directo" a los cortes sugeridos por los
embeddings en vez de hacer una búsqueda lineal del mínimo en cada llamada.

Si ningún valle global cae dentro de un sub-rango (puede pasar en
subdivisiones muy pequeñas tras varios niveles de recursión),
hybrid_segmentation recurre a una búsqueda local de respaldo, para no
perder cortes válidos.
"""

from __future__ import annotations
import numpy as np
from embeddings import cosine_similarity, compute_similarity_profile, find_local_valleys
from llm_evaluator import evaluate_segment


def validate_cut(sentences: list[str], cut_idx: int) -> bool:
    """Valida si un corte propuesto por embeddings realmente separa subtemas."""
    # Creamos una ventana de contexto de 10 oraciones centradas en el corte
    start = max(0, cut_idx - 5)
    end = min(len(sentences), cut_idx + 5)

    # Pasamos la lista original y los índices absolutos.
    # Esto mantiene la coherencia de la caché y arregla el log de consola.
    score = evaluate_segment(sentences, start, end)

    # Si el LLM puntúa bajo, el corte marca un salto semántico relevante
    return score < 5.0


def get_subtheme_cuts(
    sentences: list[str],
    start: int,
    end: int,
    min_seg: int,
    valley_cuts: list[tuple[int, float]],
) -> list[int]:
    """
    Identifica y valida cortes de subtema dentro de [start, end) usando los
    valles globales precomputados por embeddings (valley_cuts), filtrados
    al rango y ordenados por posición de izquierda a derecha para aplicar
    la distancia mínima entre cortes (min_seg).
    """
    candidates_in_range = [
        pos for pos, _depth in valley_cuts
        if start + min_seg <= pos <= end - min_seg
    ]
    candidates_in_range.sort()  # de izquierda a derecha, para la distancia mínima

    filtered_cuts = []
    last_cut = start - min_seg
    for pos in candidates_in_range:
        if pos - last_cut >= min_seg:
            if validate_cut(sentences, pos):
                filtered_cuts.append(pos)
                last_cut = pos

    return filtered_cuts


def hybrid_segmentation(
    sentences: list[str],
    embeddings: np.ndarray,
    start: int,
    end: int,
    min_seg: int,
    score_threshold: float,
    stats: dict,
    valley_cuts: list[tuple[int, float]],
) -> list[int]:
    """Segmentación recursiva para bloques detectados como mezclados."""
    stats["edges_evaluated_llm"] += 1
    score = evaluate_segment(sentences, start, end)

    if score >= score_threshold or (end - start) < (min_seg * 2):
        return []

    # 1. Ir directo a los valles ya detectados globalmente por embeddings,
    #    en orden de profundidad descendente: el primero que caiga dentro
    #    de este rango es el corte más prominente disponible aquí.
    best_cut = -1
    for pos, _depth in valley_cuts:
        if start + min_seg <= pos <= end - min_seg:
            best_cut = pos
            break

    # 2. Fallback: ningún valle global cae en este sub-rango (común tras
    #    varios niveles de recursión sobre bloques pequeños). Búsqueda
    #    local como respaldo, igual que antes.
    if best_cut == -1:
        min_sim = float('inf')
        for i in range(start + min_seg, end - min_seg + 1):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < min_sim:
                min_sim = sim
                best_cut = i

    if best_cut == -1:
        return []

    cuts_left = hybrid_segmentation(sentences, embeddings, start, best_cut, min_seg, score_threshold, stats, valley_cuts)
    cuts_right = hybrid_segmentation(sentences, embeddings, best_cut, end, min_seg, score_threshold, stats, valley_cuts)

    return sorted(list(set(cuts_left + [best_cut] + cuts_right)))


def run_dp_segmentation(
    sentences: list[str],
    embeddings: np.ndarray,
    min_seg: int = 3,
    **kwargs
) -> tuple[list[int], dict]:
    """Función principal orquestadora."""
    n = len(sentences)
    score_threshold = kwargs.get("score_threshold", 9.0)

    stats = {
        "edges_evaluated_llm": 0,
        "edges_total": 0,
        "edges_pruned_length": 0,
        "edges_pruned_cohesion": 0
    }

    # Precomputar UNA SOLA VEZ el perfil de similitud y los valles globales
    similarities = compute_similarity_profile(embeddings)
    raw_valleys = find_local_valleys(similarities)  # [(idx, profundidad), ...] desc.
    # idx en `similarities` corresponde al corte en la posición idx+1
    valley_cuts = [(idx + 1, depth) for idx, depth in raw_valleys]

    print(f"  [Multiescala] Valles candidatos (embeddings, por profundidad): "
          f"{[pos for pos, _ in valley_cuts]}")

    # Evaluación inicial estricta
    score = evaluate_segment(sentences, 0, n)
    stats["edges_evaluated_llm"] += 1

    if score >= score_threshold:
        print(f"  [Multiescala] Bloque cohesivo ({n} oraciones). Aplicando refinamiento fino...")
        final_cuts = get_subtheme_cuts(sentences, 0, n, min_seg, valley_cuts)
    else:
        print(f"  [Multiescala] Bloque mezclado. Segmentación recursiva guiada por valles...")
        final_cuts = hybrid_segmentation(sentences, embeddings, 0, n, min_seg, score_threshold, stats, valley_cuts)

    stats["edges_total"] = stats["edges_evaluated_llm"]

    return sorted(list(set(final_cuts))), stats