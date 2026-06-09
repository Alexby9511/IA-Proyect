"""
adaptive_segmentation.py
-----------------------
Segmentacion optima de contenido basada en embeddings locales,
umbral adaptativo con GMM y validacion puntual con LLM
en todas las fronteras.

Mejoras implementadas:
  - bic_delta adaptativo segun longitud del texto
  - Suavizado del perfil de similitud antes del GMM
  - Fallback a minimos locales cuando GMM no encuentra dos modos
  - Post-fusion de segmentos adyacentes similares (reduce K)
  - F1 con tolerancia de ±1 posicion
  - Cohesion corregida (centroide por segmento, no global)

Uso:
  python adaptive_segmentation.py --dataset dataset.json --index 0
  python adaptive_segmentation.py --text-file texto.txt

Requisitos:
  pip install sentence-transformers scikit-learn python-dotenv cerebras-cloud-sdk
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from embeddings import MODEL_A, MODEL_B, compute_embeddings, compute_similarity_profile, cosine_similarity

# ---------------------------------------------------------------------------
# Tokenizacion de oraciones (sin dependencias externas)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Divide texto en oraciones usando regex. Sin dependencias externas."""
    text = re.sub(r"\s+", " ", text.strip())
    pattern = r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÜ])"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if len(s.split()) >= 5]


# ---------------------------------------------------------------------------
# Suavizado del perfil de similitud
# ---------------------------------------------------------------------------

def smooth_profile(similarities: list[float], window: int = 3) -> list[float]:
    """
    Aplica una media movil al perfil de similitud para reducir ruido local.
    Reduce falsos positivos causados por oraciones de transicion individuales.
    """
    if len(similarities) < window:
        return similarities
    kernel = np.ones(window) / window
    smoothed = np.convolve(similarities, kernel, mode="same")
    # Preservar bordes sin suavizar (evitar artefactos)
    smoothed[0] = similarities[0]
    smoothed[-1] = similarities[-1]
    return smoothed.tolist()


# ---------------------------------------------------------------------------
# Umbral adaptativo con GMM
# ---------------------------------------------------------------------------

def _gaussian_intersection(
    w1: float, mu1: float, var1: float,
    w2: float, mu2: float, var2: float,
) -> float:
    """Punto de interseccion entre dos gaussianas ponderadas."""
    a = 1.0 / (2.0 * var2) - 1.0 / (2.0 * var1)
    b = mu1 / var1 - mu2 / var2
    c = (mu2 ** 2) / (2.0 * var2) - (mu1 ** 2) / (2.0 * var1)
    c += math.log((w2 * math.sqrt(var1)) / (w1 * math.sqrt(var2) + 1e-12) + 1e-12)

    if abs(a) < 1e-12:
        return -c / b if abs(b) > 1e-12 else (mu1 + mu2) / 2.0

    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        return (mu1 + mu2) / 2.0

    sqrt_disc = math.sqrt(disc)
    r1 = (-b + sqrt_disc) / (2.0 * a)
    r2 = (-b - sqrt_disc) / (2.0 * a)
    lo, hi = sorted([mu1, mu2])
    mid = (lo + hi) / 2.0
    for r in (r1, r2):
        if lo <= r <= hi:
            return r
    return r1 if abs(r1 - mid) < abs(r2 - mid) else r2


def _local_minima_threshold(similarities: list[float], percentile: float = 20.0) -> float:
    """
    Fallback cuando GMM no encuentra dos modos.
    Usa el percentil bajo de las similitudes como umbral.
    Mas agresivo que percentil 1 para capturar mas cortes candidatos.
    """
    return float(np.percentile(similarities, percentile))


def compute_adaptive_threshold(
    similarities: list[float],
    n_sentences: int,
) -> dict[str, Any]:
    """
    Calcula umbral T usando GMM con bic_delta adaptativo.

    bic_delta se escala con el numero de oraciones:
      - Textos cortos (<20):  delta pequeno (5) → mas sensible
      - Textos medios (20-60): delta medio (10)
      - Textos largos (>60):  delta grande (15) → mas estricto

    Si GMM no encuentra dos modos, usa percentil 20 como fallback
    en vez de percentil 1 (mas agresivo, captura mas candidatos).
    """
    if len(similarities) < 2:
        return {"threshold": 0.0, "mode": "empty", "bic_1": None, "bic_2": None}

    # bic_delta adaptativo
    if n_sentences < 20:
        bic_delta = 5.0
    elif n_sentences < 60:
        bic_delta = 10.0
    else:
        bic_delta = 15.0

    X = np.array(similarities, dtype=float).reshape(-1, 1)

    # Ejecutar GMM varias veces y tomar mediana del umbral (mas estable)
    thresholds_gmm2 = []
    bic1_vals, bic2_vals = [], []

    for seed in range(5):
        gmm1 = GaussianMixture(n_components=1, random_state=seed)
        gmm2 = GaussianMixture(n_components=2, random_state=seed)
        gmm1.fit(X)
        gmm2.fit(X)

        bic1_vals.append(gmm1.bic(X))
        bic2_vals.append(gmm2.bic(X))

        if gmm2.bic(X) < gmm1.bic(X) - bic_delta:
            means = gmm2.means_.ravel()
            covs  = gmm2.covariances_.ravel()
            weights = gmm2.weights_.ravel()
            change_idx = int(np.argmin(means))
            same_idx   = 1 - change_idx
            t = _gaussian_intersection(
                weights[same_idx], means[same_idx], covs[same_idx],
                weights[change_idx], means[change_idx], covs[change_idx],
            )
            thresholds_gmm2.append(t)

    bic1 = float(np.mean(bic1_vals))
    bic2 = float(np.mean(bic2_vals))

    if thresholds_gmm2:
        threshold = float(np.median(thresholds_gmm2))
        return {"threshold": threshold, "mode": "gmm2", "bic_1": bic1, "bic_2": bic2}

    # Fallback: percentil 20 (mas agresivo que el anterior percentil 1)
    threshold = _local_minima_threshold(similarities, percentile=20.0)
    return {"threshold": threshold, "mode": "gmm1_fallback", "bic_1": bic1, "bic_2": bic2}


# ---------------------------------------------------------------------------
# Cohesion interna de segmentos (corregida)
# ---------------------------------------------------------------------------

def segment_cohesion(embeddings: np.ndarray, start: int, end: int) -> float:
    """
    Cohesion interna de un segmento: similitud coseno promedio
    entre cada oracion y el centroide DEL SEGMENTO (no global).
    Rango: 0.0 (incoherente) a 1.0 (perfectamente cohesionado).
    """
    block = embeddings[start:end]
    if len(block) == 0:
        return 0.0
    if len(block) == 1:
        return 1.0  # un solo elemento es trivialmente cohesionado
    centroid = block.mean(axis=0)
    sims = [cosine_similarity(emb, centroid) for emb in block]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Post-fusion de segmentos adyacentes similares
# ---------------------------------------------------------------------------

def merge_similar_segments(
    cuts: list[int],
    embeddings: np.ndarray,
    n: int,
    merge_threshold: float = 0.90,
) -> list[int]:
    """
    Fusiona pares de segmentos adyacentes cuyo centroide es muy similar.
    Reduce K sin sacrificar cohesion — ataca directamente el objetivo
    de K minimo.

    merge_threshold: similitud minima entre centroides para fusionar.
    Valor alto (0.90) = solo fusiona si son casi identicos.
    """
    if not cuts:
        return cuts

    boundaries = [0] + sorted(cuts) + [n]
    changed = True

    while changed and len(boundaries) > 2:
        changed = False
        new_boundaries = [boundaries[0]]

        i = 0
        while i < len(boundaries) - 1:
            start = boundaries[i]
            end   = boundaries[i + 1]

            if i + 1 < len(boundaries) - 1:
                next_end = boundaries[i + 2]
                c1 = embeddings[start:end].mean(axis=0)
                c2 = embeddings[end:next_end].mean(axis=0)
                sim = cosine_similarity(c1, c2)

                if sim >= merge_threshold:
                    # Fusionar: saltar el corte entre i e i+1
                    new_boundaries.append(next_end)
                    i += 2
                    changed = True
                    continue

            new_boundaries.append(end)
            i += 1

        boundaries = new_boundaries

    return sorted([b for b in boundaries if 0 < b < n])


# ---------------------------------------------------------------------------
# Construccion de segmentos
# ---------------------------------------------------------------------------

def build_segments(sentences: list[str], cuts: list[int]) -> list[list[str]]:
    n = len(sentences)
    boundaries = [0] + sorted(cuts) + [n]
    return [sentences[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# Evaluacion F1 con tolerancia
# ---------------------------------------------------------------------------

def compute_f1(
    predicted: list[int],
    ground_truth: list[int],
    tolerance: int = 1,
) -> tuple[float, float, float]:
    """
    Calcula F1, precision y recall con tolerancia de ±tolerance posiciones.
    Un corte predicho cuenta como correcto si hay un corte real
    a distancia <= tolerance.

    tolerance=1 significa que un corte en posicion 9 cuando el real
    es 8 cuenta como correcto — razonable dado el ruido del tokenizador.
    """
    if not ground_truth and not predicted:
        return 1.0, 1.0, 1.0
    if not ground_truth or not predicted:
        return 0.0, 0.0, 0.0

    gt_matched   = set()
    pred_matched = set()

    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            if abs(p - g) <= tolerance and j not in gt_matched:
                gt_matched.add(j)
                pred_matched.add(i)
                break

    tp        = len(pred_matched)
    precision = tp / len(predicted)    if predicted    else 0.0
    recall    = tp / len(ground_truth) if ground_truth else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return f1, precision, recall


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def segment_sentences(
    sentences: list[str],
    model_path: str = MODEL_A,
    merge_threshold: float = 0.90,
    smooth_window: int = 3,
) -> dict[str, Any]:
    """
    Pipeline completo de segmentacion adaptativa.

    1. Calcular embeddings
    2. Calcular perfil de similitud y suavizarlo
    3. GMM con bic_delta adaptativo → umbral T
    4. Detectar cortes candidatos donde similitud < T
    5. Validar cada corte candidato con LLM (judge_boundary)
    6. Post-fusion de segmentos adyacentes similares
    7. Calcular cohesion final por segmento

    Retorna dict con cortes, segmentos, metricas y estadisticas.
    """
    n = len(sentences)

    if n < 2:
        return {
            "cuts": [], "segments": [sentences],
            "threshold": 0.0, "mode": "short",
            "cohesion_mean": 1.0, "llm_calls": 0,
            "n_sentences": n,
        }

    # 1. Embeddings
    embeddings = compute_embeddings(sentences, model_path)

    # 2. Perfil de similitud + suavizado
    raw_similarities  = compute_similarity_profile(embeddings)
    similarities       = smooth_profile(raw_similarities, window=smooth_window)

    # 3. Umbral adaptativo
    gmm_info  = compute_adaptive_threshold(similarities, n_sentences=n)
    threshold = gmm_info["threshold"]

    # 4. Cortes candidatos
    candidates = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    # 5. Validacion con LLM
    llm_calls = 0
    if candidates:
        from llm_evaluator import judge_boundary
        validated = []
        for cut in candidates:
            llm_calls += 1
            if judge_boundary(sentences, cut, window=3):
                validated.append(cut)
        cuts = validated
    else:
        cuts = []

    # 6. Post-fusion de segmentos similares (reduce K)
    cuts = merge_similar_segments(cuts, embeddings, n, merge_threshold=merge_threshold)

    # 7. Cohesion final por segmento
    boundaries = [0] + cuts + [n]
    seg_cohesions = [
        segment_cohesion(embeddings, boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]
    cohesion_mean = float(np.mean(seg_cohesions)) if seg_cohesions else 0.0

    return {
        "cuts":          cuts,
        "segments":      build_segments(sentences, cuts),
        "threshold":     threshold,
        "mode":          gmm_info["mode"],
        "bic_1":         gmm_info["bic_1"],
        "bic_2":         gmm_info["bic_2"],
        "cohesion_mean": cohesion_mean,
        "seg_cohesions": seg_cohesions,
        "llm_calls":     llm_calls,
        "n_sentences":   n,
        "n_candidates":  len(candidates),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_instance(dataset_path: Path, index: int) -> dict:
    with dataset_path.open(encoding="utf-8") as f:
        dataset = json.load(f)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} fuera de rango (dataset tiene {len(dataset)} instancias).")
    return dataset[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentacion adaptativa con embeddings + GMM + LLM puntual",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset",   type=Path, help="Ruta a dataset.json")
    group.add_argument("--text-file", type=Path, help="Ruta a un .txt")

    parser.add_argument("--index",  type=int,   default=0,      help="Indice de instancia en el dataset")
    parser.add_argument("--model",  type=str,   default=MODEL_B, help="Ruta al modelo de embeddings")
    parser.add_argument("--merge-threshold", type=float, default=0.90,
                        help="Umbral de similitud para fusionar segmentos adyacentes (default: 0.90)")
    parser.add_argument("--smooth-window",   type=int,   default=3,
                        help="Tamano de ventana para suavizado del perfil (default: 3)")
    parser.add_argument("--output", type=Path, help="Ruta para guardar resultado en JSON")

    args = parser.parse_args()

    # Cargar instancia
    if args.dataset:
        instance  = load_instance(args.dataset, args.index)
        sentences = instance["sentences"]
        gt_cuts   = instance.get("ground_truth_cuts", [])
        title     = instance.get("title", f"instancia_{args.index}")
    else:
        text      = args.text_file.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        gt_cuts   = []
        title     = args.text_file.name

    # Ejecutar pipeline
    result = segment_sentences(
        sentences,
        model_path       = args.model,
        merge_threshold  = args.merge_threshold,
        smooth_window    = args.smooth_window,
    )

    # Mostrar resultados
    print("=" * 62)
    print("SEGMENTACION ADAPTATIVA")
    print("=" * 62)
    print(f"Instancia        : {title}")
    print(f"Oraciones        : {result['n_sentences']}")
    print(f"Modo GMM         : {result['mode']}")
    print(f"Umbral T         : {result['threshold']:.4f}")
    print(f"Candidatos LLM   : {result['n_candidates']}")
    print(f"Llamadas LLM     : {result['llm_calls']}")
    print(f"Cortes finales   : {result['cuts']}")
    print(f"K encontrado     : {len(result['cuts']) + 1}")
    print(f"Cohesion media   : {result['cohesion_mean']:.4f}")
    print(f"Cohesion x seg.  : {[f'{c:.3f}' for c in result['seg_cohesions']]}")

    if gt_cuts:
        f1, p, r = compute_f1(result["cuts"], gt_cuts, tolerance=1)
        print(f"\nGround truth     : {gt_cuts}")
        print(f"K real           : {len(gt_cuts) + 1}")
        print(f"F1 (tol=±1)      : {f1:.3f} | P: {p:.3f} | R: {r:.3f}")
        exact_f1, ep, er = compute_f1(result["cuts"], gt_cuts, tolerance=0)
        print(f"F1 (exacto)      : {exact_f1:.3f} | P: {ep:.3f} | R: {er:.3f}")

    # Mostrar segmentos
    print(f"\n{'─'*62}")
    print("SEGMENTOS ENCONTRADOS")
    print(f"{'─'*62}")
    for i, seg in enumerate(result["segments"]):
        coh = result["seg_cohesions"][i]
        print(f"\nSegmento {i+1} ({len(seg)} oraciones | cohesion={coh:.3f})")
        for sent in seg:
            print(f"  • {sent[:90]}")

    # Guardar JSON si se pide
    if args.output:
        payload = {
            "title":         title,
            "cuts":          result["cuts"],
            "K":             len(result["cuts"]) + 1,
            "threshold":     result["threshold"],
            "mode":          result["mode"],
            "cohesion_mean": result["cohesion_mean"],
            "llm_calls":     result["llm_calls"],
            "n_sentences":   result["n_sentences"],
        }
        if gt_cuts:
            f1, p, r = compute_f1(result["cuts"], gt_cuts, tolerance=1)
            payload.update({"f1": f1, "precision": p, "recall": r, "gt_cuts": gt_cuts})
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nResultado guardado en: {args.output}")


if __name__ == "__main__":
    main()