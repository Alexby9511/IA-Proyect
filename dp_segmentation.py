"""
dp_segmentation.py
------------------
Segmentación Multiescala Adaptativa con búsqueda binaria guiada por LLM.

Arquitectura:
  1. Se evalúa el segmento completo con el LLM (dos fases en llm_evaluator).
  2. Si score >= score_threshold → segmento cohesivo → NO recursionar.
     El segmento se acepta tal cual, sin buscar subcortes.
  3. Si score < score_threshold → segmento mezclado → recursionar:
     a. Se busca el valle más profundo (embeddings) dentro del rango.
     b. Se divide el segmento en dos mitades por ese valle.
     c. Se repite el proceso en cada mitad de forma independiente.

Ventajas de esta búsqueda binaria guiada:
  - Los segmentos cohesivos NO generan llamadas adicionales al LLM.
  - Solo los segmentos mezclados se subdividen, reduciendo llamadas totales.
  - El umbral score_threshold controla la agresividad de la segmentación.

Integración con llm_evaluator (dos fases):
  - Fase 1 (prompt suave): detecta casos claros (muy mezclado / muy cohesivo).
  - Fase 2 (prompt estricto): se activa solo si Fase 1 da score alto,
    buscando subtemas ocultos. El score final es min(suave, estricto).
  - Si score_final >= score_threshold → cohesivo → parar recursión.
  - Si score_final < score_threshold → mezclado → buscar corte y dividir.
"""

from __future__ import annotations
import numpy as np
from embeddings import cosine_similarity, compute_similarity_profile, find_local_valleys
from llm_evaluator import evaluate_segment, select_evaluation_mode, GLOBAL_SCORE_THRESHOLD


def _find_best_valley_in_range(
    valley_cuts: list[tuple[int, float]],
    embeddings: np.ndarray,
    start: int,
    end: int,
    min_seg: int,
) -> int:
    """
    Encuentra el mejor punto de corte dentro de [start, end).

    Estrategia:
      1. Buscar en los valles globales precomputados (más eficiente).
      2. Si ningún valle global cae en el rango (común en sub-rangos
         pequeños tras varios niveles de recursión), hacer búsqueda local
         del mínimo de similitud coseno como fallback.

    Retorna la posición del mejor corte, o -1 si no es posible cortar
    respetando min_seg.
    """
    # 1. Valles globales ordenados por profundidad (descendente)
    for pos, _depth in valley_cuts:
        if start + min_seg <= pos <= end - min_seg:
            return pos

    # 2. Fallback: mínimo local de similitud coseno en el rango
    best_cut = -1
    min_sim  = float('inf')
    for i in range(start + min_seg, end - min_seg + 1):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < min_sim:
            min_sim  = sim
            best_cut = i

    return best_cut


def _recursive_segment(
    sentences: list[str],
    embeddings: np.ndarray,
    start: int,
    end: int,
    min_seg: int,
    score_threshold: float,
    valley_cuts: list[tuple[int, float]],
    stats: dict,
    depth: int = 0,
) -> list[int]:
    """
    Segmentación recursiva (búsqueda binaria guiada por LLM).

    Para cada bloque [start, end):
      1. Evaluar cohesión con LLM (dos fases).
      2. Si score >= score_threshold → cohesivo → devolver [] (sin cortes).
      3. Si score < score_threshold → mezclado:
         a. Encontrar el mejor valle en el rango.
         b. Dividir en [start, cut) y [cut, end).
         c. Recursionar en cada mitad.

    Parámetros:
      depth : nivel de recursión (para logging con indentación)
    """
    indent = "  " * (depth + 1)

    # Condición de parada: bloque demasiado pequeño para dividir
    if (end - start) < (min_seg * 2):
        print(f"{indent}[Recursión] Bloque [{start},{end}) demasiado pequeño → parar")
        return []

    stats["edges_evaluated_llm"] += 1
    score = evaluate_segment(sentences, start, end)

    if score >= score_threshold:
        # Segmento cohesivo: NO recursionar, NO buscar subcortes
        print(f"{indent}[Recursión] [{start},{end}) cohesivo (score={score:.1f}) → aceptar sin cortes")
        return []

    # Segmento mezclado: buscar el mejor corte y dividir
    print(f"{indent}[Recursión] [{start},{end}) mezclado (score={score:.1f}) → buscando corte...")
    best_cut = _find_best_valley_in_range(valley_cuts, embeddings, start, end, min_seg)

    if best_cut == -1:
        print(f"{indent}[Recursión] No se encontró corte válido en [{start},{end}) → parar")
        return []

    print(f"{indent}[Recursión] Corte en posición {best_cut} → dividiendo [{start},{best_cut}) y [{best_cut},{end})")

    cuts_left  = _recursive_segment(sentences, embeddings, start,    best_cut, min_seg, score_threshold, valley_cuts, stats, depth + 1)
    cuts_right = _recursive_segment(sentences, embeddings, best_cut, end,      min_seg, score_threshold, valley_cuts, stats, depth + 1)

    return sorted(set(cuts_left + [best_cut] + cuts_right))


def run_dp_segmentation(
  sentences: list[str],
  embeddings: np.ndarray,
  min_seg: int = 3,
  **kwargs,
) -> tuple[list[int], dict]:
  """
  Función principal del motor de segmentación.

  Flujo:
    0. Evaluación Inicial: select_evaluation_mode() puntúa la calidad/
     coherencia global del texto (0-10) y fija el modo de prompt
     ("strict" si score_global >= GLOBAL_SCORE_THRESHOLD, "soft" en
     caso contrario) para TODA la ejecución.
    1. Precomputar perfil de similitud y valles globales (una sola vez).
    2. Evaluar el texto completo con el LLM, usando el modo ya fijado.
    3. Si cohesivo (score >= score_threshold) → devolver sin cortes.
     Si mezclado → iniciar recursión con búsqueda binaria guiada,
     donde TODOS los subsegmentos se evalúan con el mismo modo.

  Parámetros kwargs:
    score_threshold : umbral de cohesión para parar la recursión
            (default 7.0). Es independiente de
            GLOBAL_SCORE_THRESHOLD (que decide el modo de
            prompt); score_threshold decide cuándo dejar de
            dividir un bloque.

  Retorna:
    (cortes_finales, estadísticas)
  """
  n               = len(sentences)
  score_threshold = kwargs.get("score_threshold", 7.0)

  stats = {
    "edges_evaluated_llm":   0,
    "edges_total":           0,
    "edges_pruned_length":   0,
    "edges_pruned_cohesion": 0,
    "evaluation_mode":       None,
    "global_score":          None,
  }

  # 0. Evaluación Inicial: decide el modo de prompt para toda la ejecución
  mode, global_score = select_evaluation_mode(sentences)
  stats["evaluation_mode"] = mode
  stats["global_score"]    = global_score
  stats["edges_evaluated_llm"] += 1
  print(f"  [DP] Escenario {'A (estricto)' if mode == 'strict' else 'B (suave)'}: "
      f"score_global={global_score:.1f} (umbral={GLOBAL_SCORE_THRESHOLD}) → modo='{mode}'")

  # 1. Precomputar valles globales UNA SOLA VEZ
  similarities = compute_similarity_profile(embeddings)
  raw_valleys  = find_local_valleys(similarities)
  valley_cuts  = [(idx + 1, depth) for idx, depth in raw_valleys]

  print(f"  [DP] Valles precomputados (por profundidad): "
      f"{[pos for pos, _ in valley_cuts]}")
  print(f"  [DP] score_threshold={score_threshold} | min_seg={min_seg}")

  # 2. Evaluación inicial del texto completo, con el modo ya fijado
  stats["edges_evaluated_llm"] += 1
  score_total = evaluate_segment(sentences, 0, n)

  if score_total >= score_threshold:
    print(f"  [DP] Texto completo cohesivo (score={score_total:.1f}) → sin cortes")
    stats["edges_total"] = stats["edges_evaluated_llm"]
    return [], stats

  # 3. Texto mezclado → recursión con búsqueda binaria, modo consistente
  print(f"  [DP] Texto mezclado (score={score_total:.1f}) → iniciando segmentación recursiva...")
  final_cuts = _recursive_segment(
    sentences       = sentences,
    embeddings      = embeddings,
    start           = 0,
    end             = n,
    min_seg         = min_seg,
    score_threshold = score_threshold,
    valley_cuts     = valley_cuts,
    stats           = stats,
    depth           = 0,
  )

  stats["edges_total"] = stats["edges_evaluated_llm"]

  print(f"  [DP] Segmentación completada. Cortes: {final_cuts} "
      f"| LLM calls: {stats['edges_evaluated_llm']} | modo='{mode}'")

  return sorted(set(final_cuts)), stats