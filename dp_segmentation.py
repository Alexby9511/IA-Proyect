"""
dp_segmentation.py
------------------
Segmentación Óptima mediante Grafo Esparcido (Sparse DAG) y Ecuación de Bellman.
Incluye la "Poda de Solapamiento Temático" para eliminar llamadas redundantes al LLM.
"""

from __future__ import annotations
import numpy as np

from embeddings import cosine_similarity, compute_similarity_profile, find_local_valleys
from llm_evaluator import evaluate_segment, select_evaluation_mode

def _fast_segment_cohesion(embeddings: np.ndarray, start: int, end: int) -> float:
    """Calcula rápidamente la cohesión base usando el centroide de los embeddings."""
    seg = embeddings[start:end]
    if len(seg) < 2:
        return 1.0
    centroid = seg.mean(axis=0)
    sims = [cosine_similarity(emb, centroid) for emb in seg]
    return float(np.mean(sims))

def run_dp_segmentation(
    sentences: list[str],
    embeddings: np.ndarray,
    min_seg: int = 3,
    lambda_penalty: float = 3.0,
    max_seg_len: int | None = None,
    min_cohesion: float = 0.5,
    **kwargs
) -> tuple[list[int], dict]:
    n = len(sentences)
    
    if max_seg_len is None:
        max_seg_len = max(min_seg * 5, n // 2)

    stats = {
        "edges_evaluated_llm": 0,
        "edges_total": 0,
        "edges_pruned_length": 0,
        "edges_pruned_cohesion": 0,
        "edges_pruned_math": 0,
        "edges_pruned_theme": 0,
        "edges_pruned_overlap": 0,  # NUEVA MÉTRICA: Poda inteligente
        "evaluation_mode": None,
        "global_score": None
    }

    # 1. Modo de evaluación global
    mode, global_score = select_evaluation_mode(sentences)
    stats["evaluation_mode"] = mode
    stats["global_score"] = global_score

    # 2. Generar Nodos
    similarities = compute_similarity_profile(embeddings)
    raw_valleys = find_local_valleys(similarities)
    valid_cuts = {idx + 1 for idx, depth in raw_valleys}
    nodes = sorted(list({0, n} | valid_cuts))
    
    print(f"  [DP] Construyendo Sparse DAG. N={n}, Nodos={len(nodes)}")

    # 3. Inicializar DP
    dp = {node: float('inf') for node in nodes}
    parent = {node: -1 for node in nodes}
    dp[0] = 0.0

    # Memoria para la Poda de Solapamiento Temático. Guarda tuplas: (inicio_u, fin_cohesivo_w, muro_v)
    thematic_walls = []

    # 4. Construcción y Relajación
    for i in range(len(nodes)):
        u = nodes[i]
        if dp[u] == float('inf'):
            continue
            
        last_cohesive_v = u # Rastrea hasta dónde llegó el tema actual con éxito

        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            length = v - u
            
            # Poda de Longitud
            if length > max_seg_len:
                stats["edges_pruned_length"] += (len(nodes) - j)
                break 
                
            if length < min_seg and v != n:
                stats["edges_pruned_length"] += 1
                continue

            # Poda Matemática (Dijkstra/A*)
            if dp[u] + lambda_penalty >= dp[v]:
                stats["edges_pruned_math"] += 1
                continue

            # Poda Rápida por Embeddings
            coh = _fast_segment_cohesion(embeddings, u, v)
            if coh < min_cohesion:
                stats["edges_pruned_cohesion"] += 1
                continue

            # ==========================================================
            # NUEVA OPTIMIZACIÓN: PODA DE SOLAPAMIENTO TEMÁTICO
            # ==========================================================
            hit_wall = False
            for (wall_u, wall_w, wall_bad_v) in thematic_walls:
                # Si nuestro nodo de inicio 'u' está dentro del bloque cohesivo conocido
                # Y nuestro destino 'v' cruza o alcanza el muro donde se rompe el tema...
                if wall_u <= u < wall_w and v >= wall_bad_v:
                    hit_wall = True
                    break
            
            if hit_wall:
                # Sabemos matemáticamente que será una mezcla de temas.
                stats["edges_pruned_overlap"] += (len(nodes) - j)
                break # Rompemos porque cualquier 'v' posterior también cruzará el muro
            # ==========================================================

            stats["edges_total"] += 1
            stats["edges_evaluated_llm"] += 1
            
            # Oráculo LLM
            score = evaluate_segment(sentences, u, v)

            # --- Umbral de rechazo estricto ---
            # Si score < 7.0, el coste es prohibitivo (100.0)
            # El algoritmo preferirá dividir el texto antes que aceptar segmentos poco cohesivos.
            if score < 7.0:
                cost = 100.0
            else:
                cost = (10.0 - score) + lambda_penalty
            
            # Ecuación de Bellman
            if dp[u] + cost < dp[v]:
                dp[v] = dp[u] + cost
                parent[v] = u

            # Registro de cohesión y muros temáticos
            if score >= 7.0:
                last_cohesive_v = v # El tema se mantiene fuerte hasta aquí
                
            elif score <= 3.5:
                # Hemos chocado contra un muro de cambio de tema.
                if last_cohesive_v > u:
                    # Guardamos la regla: "Cualquier bloque que empiece entre 'u' y 'last_cohesive_v' 
                    # se estrellará inevitablemente si intenta llegar hasta 'v'".
                    thematic_walls.append((u, last_cohesive_v, v))
                
                stats["edges_pruned_theme"] += (len(nodes) - j - 1)
                break

    # 5. Backtracking
    if dp[n] == float('inf'):
        print("  [DP] ADVERTENCIA: Grafo desconectado. Usando fallback.")
        return list(range(max_seg_len, n, max_seg_len)), stats

    cuts = []
    curr = n
    while curr > 0:
        p = parent[curr]
        if p > 0:
            cuts.append(p)
        curr = p

    print(f"  [DP] Optimizaciones: {stats['edges_pruned_math']} matemáticas, "
          f"{stats['edges_pruned_theme']} rupturas, "
          f"{stats['edges_pruned_overlap']} solapamientos evitados.")
          
    return sorted(cuts), stats