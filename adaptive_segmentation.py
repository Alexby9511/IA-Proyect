"""
adaptive_segmentation.py
-----------------------
Funciones reutilizables para segmentación de texto:
tokenización, cohesión de segmentos, scoring de embeddings,
métrica F1 y CLI simple que invoca el pipeline principal.

Uso:
  python adaptive_segmentation.py --dataset dataset.json --index 0
  python adaptive_segmentation.py --text-file texto.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List

import numpy as np
from embeddings import MODEL_B, compute_embeddings, cosine_similarity

# ---------------------------------------------------------------------------
# Tokenización de oraciones
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    pattern = r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÜ])"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if len(s.split()) >= 5]

# ---------------------------------------------------------------------------
# Cohesión de un segmento
# ---------------------------------------------------------------------------
def segment_cohesion(embeddings: np.ndarray, start: int, end: int) -> float:
    seg = embeddings[start:end]
    if len(seg) < 2:
        return 1.0
    centroid = seg.mean(axis=0)
    sims = [cosine_similarity(emb, centroid) for emb in seg]
    return float(np.mean(sims))

# ---------------------------------------------------------------------------
# Cohesión por segmento de una partición (función compartida)
# ---------------------------------------------------------------------------
def compute_segment_cohesions(embeddings: np.ndarray, cuts: list[int], n: int) -> list[float]:
    """
    Calcula la cohesión (0-1) de cada segmento de una partición.

    Esta función se comparte entre score_partition_embeddings (aquí) y
    compute_partition_score_embeddings (simulated_annealing.py), que la
    usa como función de coste dentro del bucle de Simulated Annealing
    (sin llamar al LLM), evitando duplicar la lógica de cohesión.
    """
    boundaries = [0] + sorted(cuts) + [n]
    K = len(boundaries) - 1
    return [segment_cohesion(embeddings, boundaries[i], boundaries[i + 1]) for i in range(K)]

# ---------------------------------------------------------------------------
# Scoring de embeddings normalizado a [0,1]
# ---------------------------------------------------------------------------
def score_partition_embeddings(
    embeddings: np.ndarray,
    cuts: list[int],
    n: int,
    min_seg: int = 3,
) -> float:
    """
    Evalúa una partición basada únicamente en embeddings.
    Combina cohesión media, separación entre segmentos (1 - similitud de
    centroides adyacentes) y penalización por segmentos cortos.
    El resultado se normaliza dividiendo por un factor teórico máximo (1.5)
    para llevarlo al rango [0,1].
    """
    boundaries = [0] + sorted(cuts) + [n]
    K = len(boundaries) - 1
    cohs = compute_segment_cohesions(embeddings, cuts, n)
    centroids = [embeddings[boundaries[i]:boundaries[i + 1]].mean(axis=0) for i in range(K)]
    mean_coh = np.mean(cohs)
    # Separación: 1 - similitud coseno entre centroides adyacentes
    seps = []
    for i in range(K - 1):
        sim = cosine_similarity(centroids[i], centroids[i + 1])
        seps.append(1.0 - sim)
    mean_sep = np.mean(seps) if seps else 0.0
    # Penalización por segmentos muy cortos
    penalty = 0.0
    for i in range(K):
        size = boundaries[i + 1] - boundaries[i]
        if size < min_seg:
            penalty += (min_seg - size) * 0.2
    raw = mean_coh + 0.5 * mean_sep - penalty
    # Normalización: el máximo teórico (sin penalización) es 1.0 + 0.5*1.0 = 1.5
    normalized = raw / 1.5
    # Asegurar que quede en [0,1]
    return max(0.0, min(1.0, normalized))

# ---------------------------------------------------------------------------
# F1 con tolerancia
# ---------------------------------------------------------------------------
def compute_f1(
    predicted: list[int],
    ground_truth: list[int],
    tolerance: int = 1,
) -> tuple[float, float, float]:
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
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

# ---------------------------------------------------------------------------
# CLI (delega en pipeline.py)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Segmentación óptima con LLM local")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path, help="Ruta a dataset.json")
    group.add_argument("--text-file", type=Path, help="Ruta a un .txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default=MODEL_B)
    parser.add_argument("--min-seg", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--llm-weight", type=float, default=0.4)
    parser.add_argument("--max-iter-cap", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Importar aquí para no obligar en import del módulo
    from pipeline import run_pipeline

    if args.dataset:
        with args.dataset.open(encoding="utf-8") as f:
            dataset = json.load(f)
        instance = dataset[args.index]
        sentences = instance["sentences"]
        gt_cuts = instance.get("ground_truth_cuts", [])
        title = instance.get("title", f"instancia_{args.index}")
    else:
        text = args.text_file.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        gt_cuts = []
        title = args.text_file.name

    result = run_pipeline(
        sentences,
        model_path=args.model,
        min_seg=args.min_seg,
        max_k=args.max_k,
        llm_weight=args.llm_weight,
        max_iter_cap=args.max_iter_cap,
    )

    print("=" * 62)
    print("SEGMENTACION OPTIMA (pipeline integrado)")
    print("=" * 62)
    print(f"Instancia        : {title}")
    print(f"Oraciones        : {result['n_sentences']}")
    print(f"K detectado      : {result['best_K']}")
    print(f"Cortes           : {result['best_cuts']}")
    print(f"Cohesion media   : {result['cohesion_mean']:.4f}")
    print(f"Cohesion x seg.  : {[f'{c:.3f}' for c in result['seg_cohesions']]}")
    print(f"Llamadas LLM     : {result['total_llm_calls']}")
    print(f"Score combinado  : {result['combined_score']:.4f}")

    if gt_cuts:
        f1, p, r = compute_f1(result["best_cuts"], gt_cuts, tolerance=1)
        print(f"\nGround truth     : {gt_cuts}")
        print(f"K real           : {len(gt_cuts) + 1}")
        print(f"F1 (tol=±1)      : {f1:.3f} | P: {p:.3f} | R: {r:.3f}")
        exact_f1, ep, er = compute_f1(result["best_cuts"], gt_cuts, tolerance=0)
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
            "cuts": result["best_cuts"],
            "K": result["best_K"],
            "cohesion_mean": result["cohesion_mean"],
            "llm_calls": result["total_llm_calls"],
            "n_sentences": result["n_sentences"],
        }
        if gt_cuts:
            f1, p, r = compute_f1(result["best_cuts"], gt_cuts, tolerance=1)
            payload.update({"f1": f1, "precision": p, "recall": r, "gt_cuts": gt_cuts})
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultado guardado en: {args.output}")

if __name__ == "__main__":
    main()