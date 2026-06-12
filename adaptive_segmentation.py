"""
adaptive_segmentation.py
-----------------------
Segmentación óptima de contenido con detección de valles,
refinamiento local y scoring que combina embeddings + LLM.
El LLM siempre se utiliza (modelo local Ollama deepseek-local:1.5b).

Uso:
  python adaptive_segmentation.py --dataset dataset.json --index 0
  python adaptive_segmentation.py --text-file texto.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from scipy.signal import argrelextrema

from embeddings import MODEL_B, compute_embeddings, compute_similarity_profile, cosine_similarity

# ---------------------------------------------------------------------------
# Tokenización
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    pattern = r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÜ])"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if len(s.split()) >= 5]


# ---------------------------------------------------------------------------
# Detección de valles (mínimos locales)
# ---------------------------------------------------------------------------
def find_local_valleys(similarities: List[float]) -> List[Tuple[int, float]]:
    """Mínimos locales con profundidad. Ordenados por profundidad descendente."""
    if len(similarities) < 3:
        return []
    sim_arr = np.array(similarities)
    local_min_idx = argrelextrema(sim_arr, np.less)[0]
    valleys = []
    for idx in local_min_idx:
        if idx > 0 and idx < len(sim_arr) - 1:
            depth = max(sim_arr[idx - 1], sim_arr[idx + 1]) - sim_arr[idx]
            valleys.append((idx, depth))
    valleys.sort(key=lambda x: x[1], reverse=True)
    return valleys


# ---------------------------------------------------------------------------
# Cohesión de un segmento (embeddings)
# ---------------------------------------------------------------------------
def segment_cohesion(embeddings: np.ndarray, start: int, end: int) -> float:
    seg = embeddings[start:end]
    if len(seg) < 2:
        return 1.0
    centroid = seg.mean(axis=0)
    sims = [cosine_similarity(emb, centroid) for emb in seg]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Refinamiento local de un corte
# ---------------------------------------------------------------------------
def refine_cut_local(
    embeddings: np.ndarray,
    left_boundary: int,
    right_boundary: int,
    initial_cut: int,
    max_shift: int = 2,
) -> int:
    best_cut = initial_cut
    best_coh = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        candidate = initial_cut + shift
        if candidate <= left_boundary + 1 or candidate >= right_boundary - 1:
            continue
        left_coh = segment_cohesion(embeddings, left_boundary, candidate)
        right_coh = segment_cohesion(embeddings, candidate, right_boundary)
        avg = (left_coh + right_coh) / 2.0
        if avg > best_coh:
            best_coh = avg
            best_cut = candidate
    return best_cut


# ---------------------------------------------------------------------------
# Puntuación de una partición (embeddings)
# ---------------------------------------------------------------------------
def score_partition_embeddings(
    embeddings: np.ndarray,
    cuts: List[int],
    n: int,
    min_seg: int = 3,
) -> Tuple[float, float]:
    boundaries = [0] + sorted(cuts) + [n]
    K = len(boundaries) - 1
    cohs = []
    centroids = []
    for i in range(K):
        start, end = boundaries[i], boundaries[i + 1]
        coh = segment_cohesion(embeddings, start, end)
        cohs.append(coh)
        centroids.append(embeddings[start:end].mean(axis=0))
    mean_coh = np.mean(cohs)
    # Separación entre segmentos adyacentes (1 - similitud)
    seps = []
    for i in range(K - 1):
        sim = cosine_similarity(centroids[i], centroids[i + 1])
        seps.append(1.0 - sim)
    mean_sep = np.mean(seps) if seps else 0.0
    # Penalización por segmentos cortos
    penalty = 0.0
    for i in range(K):
        size = boundaries[i + 1] - boundaries[i]
        if size < min_seg:
            penalty += (min_seg - size) * 0.2
    score = mean_coh + 0.5 * mean_sep - penalty
    return score, mean_coh


# ---------------------------------------------------------------------------
# Puntuación usando el LLM (evalúa cada segmento, retorna promedio normalizado)
# ---------------------------------------------------------------------------
def score_partition_with_llm(
    sentences: List[str],
    cuts: List[int],
    llm_eval_func,
) -> Tuple[float, int]:
    n = len(sentences)
    boundaries = [0] + sorted(cuts) + [n]
    scores = []
    calls = 0
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        try:
            score = llm_eval_func(sentences, start, end)
            scores.append(score / 10.0)   # normalizar 1-10 -> 0-1
            calls += 1
        except Exception:
            # Si falla, asignar puntuación neutra
            scores.append(0.5)
    avg_llm = np.mean(scores) if scores else 0.5
    return avg_llm, calls


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def segment_sentences(
    sentences: List[str],
    model_path: str = MODEL_B,
    min_seg: int = 3,
    max_k: int = 8,
    llm_weight: float = 0.4,   # peso del LLM en la decisión final
) -> dict[str, Any]:
    n = len(sentences)
    if n < 2:
        return {
            "cuts": [], "segments": [sentences],
            "K": 1, "cohesion_mean": 1.0, "llm_calls": 0, "n_sentences": n
        }

    embeddings = compute_embeddings(sentences, model_path)
    similarities = compute_similarity_profile(embeddings)

    # 1. Valles profundos
    valleys = find_local_valleys(similarities)
    if not valleys:
        return {
            "cuts": [], "segments": [sentences],
            "K": 1, "cohesion_mean": segment_cohesion(embeddings, 0, n),
            "llm_calls": 0, "n_sentences": n
        }

    # 2. Probar distintos K y generar candidatos
    max_possible_k = min(max_k, n // min_seg, len(valleys) + 1)
    candidates = []   # (cuts, emb_score, coh_mean)

    for K in range(2, max_possible_k + 1):
        selected = valleys[:K - 1]
        initial_cuts = sorted([int(idx) + 1 for idx, _ in selected])

        # Refinamiento local con contexto
        refined = []
        prev_boundary = 0
        for i, cut in enumerate(initial_cuts):
            next_boundary = n if i == len(initial_cuts) - 1 else initial_cuts[i + 1]
            best_cut = refine_cut_local(embeddings, prev_boundary, next_boundary, cut, max_shift=2)
            refined.append(best_cut)
            prev_boundary = best_cut

        emb_score, coh_mean = score_partition_embeddings(embeddings, refined, n, min_seg)
        candidates.append((refined, emb_score, coh_mean))

    if not candidates:
        return {"cuts": [], "segments": [sentences], "K": 1,
                "cohesion_mean": 1.0, "llm_calls": 0, "n_sentences": n}

    # 3. Evaluar con LLM todos los candidatos (se podría limitar a los mejores N)
    #    Para mantener calidad, evaluamos todos los que sean factibles.
    try:
        from llm_evaluator import evaluate_segment, reset_cache, get_cache_stats
        llm_eval = evaluate_segment
    except ImportError:
        print("ERROR: llm_evaluator.py no encontrado. El LLM es obligatorio.")
        sys.exit(1)

    best_final_score = -np.inf
    best_cuts = []
    best_cohs = []
    total_llm_calls = 0

    for cuts, emb_score, coh in candidates:
        llm_avg, calls = score_partition_with_llm(sentences, cuts, llm_eval)
        total_llm_calls += calls
        # Combinar puntuaciones (embeddings y LLM)
        combined = (1 - llm_weight) * emb_score + llm_weight * llm_avg
        if combined > best_final_score:
            best_final_score = combined
            best_cuts = cuts
            best_cohs = [segment_cohesion(embeddings, 0, cuts[0])] if cuts else []  # placeholder
            # Recalcular cohesiones reales después
    # Recalcular cohesiones finales para el mejor candidato
    boundaries = [0] + best_cuts + [n]
    final_cohs = [segment_cohesion(embeddings, boundaries[i], boundaries[i + 1])
                  for i in range(len(boundaries) - 1)]
    cohesion_mean = float(np.mean(final_cohs)) if final_cohs else 0.0

    return {
        "cuts": best_cuts,
        "segments": [sentences[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)],
        "K": len(best_cuts) + 1 if best_cuts else 1,
        "cohesion_mean": cohesion_mean,
        "seg_cohesions": final_cohs,
        "llm_calls": total_llm_calls,
        "n_sentences": n,
        "combined_score": best_final_score,
    }


# ---------------------------------------------------------------------------
# F1 con tolerancia
# ---------------------------------------------------------------------------
def compute_f1(predicted, ground_truth, tolerance=1):
    if not ground_truth and not predicted:
        return 1.0, 1.0, 1.0
    if not ground_truth or not predicted:
        return 0.0, 0.0, 0.0
    gt_matched = set()
    pred_matched = set()
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            if abs(p - g) <= tolerance and j not in gt_matched:
                gt_matched.add(j)
                pred_matched.add(i)
                break
    tp = len(pred_matched)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return f1, precision, recall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_instance(dataset_path: Path, index: int) -> dict:
    with dataset_path.open(encoding="utf-8") as f:
        dataset = json.load(f)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} fuera de rango.")
    return dataset[index]


def main():
    parser = argparse.ArgumentParser(description="Segmentación óptima con LLM local obligatorio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path, help="Ruta a dataset.json")
    group.add_argument("--text-file", type=Path, help="Ruta a un .txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default=MODEL_B)
    parser.add_argument("--min-seg", type=int, default=3, help="Tamaño mínimo de segmento")
    parser.add_argument("--max-k", type=int, default=8, help="Número máximo de segmentos a probar")
    parser.add_argument("--llm-weight", type=float, default=0.4, help="Peso del LLM en la decisión (0-1)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.dataset:
        instance = load_instance(args.dataset, args.index)
        sentences = instance["sentences"]
        gt_cuts = instance.get("ground_truth_cuts", [])
        title = instance.get("title", f"instancia_{args.index}")
    else:
        text = args.text_file.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        gt_cuts = []
        title = args.text_file.name

    result = segment_sentences(
        sentences,
        model_path=args.model,
        min_seg=args.min_seg,
        max_k=args.max_k,
        llm_weight=args.llm_weight,
    )

    print("=" * 62)
    print("SEGMENTACION OPTIMA (embeddings + LLM)")
    print("=" * 62)
    print(f"Instancia        : {title}")
    print(f"Oraciones        : {result['n_sentences']}")
    print(f"K detectado      : {result['K']}")
    print(f"Cortes           : {result['cuts']}")
    print(f"Cohesion media   : {result['cohesion_mean']:.4f}")
    print(f"Cohesion x seg.  : {[f'{c:.3f}' for c in result['seg_cohesions']]}")
    print(f"Llamadas LLM     : {result['llm_calls']}")

    if gt_cuts:
        f1, p, r = compute_f1(result["cuts"], gt_cuts, tolerance=1)
        print(f"\nGround truth     : {gt_cuts}")
        print(f"K real           : {len(gt_cuts) + 1}")
        print(f"F1 (tol=±1)      : {f1:.3f} | P: {p:.3f} | R: {r:.3f}")
        exact_f1, ep, er = compute_f1(result["cuts"], gt_cuts, tolerance=0)
        print(f"F1 (exacto)      : {exact_f1:.3f} | P: {ep:.3f} | R: {er:.3f}")

    print(f"\n{'─'*62}")
    print("SEGMENTOS ENCONTRADOS")
    print(f"{'─'*62}")
    for i, seg in enumerate(result["segments"]):
        coh = result["seg_cohesions"][i]
        print(f"\nSegmento {i+1} ({len(seg)} oraciones | cohesion={coh:.3f})")
        for sent in seg:
            print(f"  • {sent[:90]}")

    if args.output:
        payload = {
            "title": title,
            "cuts": result["cuts"],
            "K": result["K"],
            "cohesion_mean": result["cohesion_mean"],
            "llm_calls": result["llm_calls"],
            "n_sentences": result["n_sentences"],
        }
        if gt_cuts:
            f1, p, r = compute_f1(result["cuts"], gt_cuts, tolerance=1)
            payload.update({"f1": f1, "precision": p, "recall": r, "gt_cuts": gt_cuts})
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultado guardado en: {args.output}")


if __name__ == "__main__":
    main()