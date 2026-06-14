"""
adaptive_segmentation.py
-----------------------
Funciones reutilizables para segmentación de texto:
tokenización, cohesión de segmentos, scoring de embeddings,
métrica F1 y CLI que invoca el pipeline avanzado.

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
# Cohesión de un segmento (basada en embeddings)
# ---------------------------------------------------------------------------
def segment_cohesion(embeddings: np.ndarray, start: int, end: int) -> float:
    seg = embeddings[start:end]
    if len(seg) < 2:
        return 1.0
    centroid = seg.mean(axis=0)
    sims = [cosine_similarity(emb, centroid) for emb in seg]
    return float(np.mean(sims))

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
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Segmentación óptima de contenido (Pipeline Avanzado)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path, help="Ruta a dataset.json")
    group.add_argument("--text-file", type=Path, help="Ruta a un .txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default=MODEL_B)
    parser.add_argument("--min-seg", type=int, default=3, help="Tamaño mínimo de segmento")
    parser.add_argument("--lambda-penalty", type=float, default=3.0,
                        help="Penalización estructural por segmento (favorece menos cortes)")
    parser.add_argument("--max-seg-len", type=int, default=None,
                        help="Longitud máxima de un segmento candidato (por defecto: max(min_seg*4, n//2))")
    parser.add_argument("--min-cohesion", type=float, default=0.5,
                        help="Cohesión mínima de embeddings para evaluar una arista con el LLM")
    parser.add_argument("--embedding-weight", type=float, default=0.3,
                        help="Peso de la cohesión de embeddings al combinar con el costo del LLM (0-1)")
    parser.add_argument("--output", type=Path, help="Guardar resultado en JSON")
    args = parser.parse_args()

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
        lambda_penalty=args.lambda_penalty,
        max_seg_len=args.max_seg_len,
        min_cohesion=args.min_cohesion,
        embedding_weight=args.embedding_weight,
    )

    print("=" * 62)
    print("SEGMENTACION OPTIMA (Pipeline Avanzado)")
    print("=" * 62)
    print(f"Instancia        : {title}")
    print(f"Oraciones        : {result['n_sentences']}")
    print(f"K detectado      : {result['best_K']}")
    print(f"Cortes           : {result['best_cuts']}")
    print(f"Cohesion media   : {result['cohesion_mean']:.4f}")
    print(f"Cohesion x seg.  : {[f'{c:.3f}' for c in result['seg_cohesions']]}")
    print(f"Llamadas LLM     : {result['total_llm_calls']}")

    dp_stats = result.get("dp_stats", {})
    cache_stats = result.get("cache_stats", {})
    if dp_stats:
        print(f"\nAristas totales      : {dp_stats['edges_total']}")
        print(f"  podadas (longitud) : {dp_stats['edges_pruned_length']}")
        print(f"  podadas (cohesión) : {dp_stats['edges_pruned_cohesion']}")
        print(f"  evaluadas con LLM  : {dp_stats['edges_evaluated_llm']}")
    if cache_stats:
        print(f"  respuestas truncadas (<think> sin cerrar): "
              f"{cache_stats.get('truncated', 0)} / {dp_stats.get('edges_evaluated_llm', 0)}")

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
            "dp_stats": dp_stats,
        }
        if gt_cuts:
            f1, p, r = compute_f1(result["best_cuts"], gt_cuts, tolerance=1)
            payload.update({"f1": f1, "precision": p, "recall": r, "gt_cuts": gt_cuts})
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultado guardado en: {args.output}")

if __name__ == "__main__":
    main()