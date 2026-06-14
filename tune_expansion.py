"""
tune_expansion.py
-----------------
Encuentra el valor óptimo de expansion para la Fase 1.
Prueba varios valores y mide el recall promedio sobre el dataset.

Uso:
    python tune_expansion.py
"""

import json
from pathlib import Path
from embeddings import get_ranked_candidates, MODEL_A, MODEL_B


def get_cut_candidates(
    sentences: list[str],
    K: int,
    model_path: str,
    expansion: int = 2,
) -> list[int]:
    """
    Wrapper de compatibilidad: devuelve los top (K-1)*expansion candidatos
    del ranking de get_ranked_candidates.
    """
    ranked = get_ranked_candidates(sentences, model_path)
    M = min((K - 1) * expansion, len(ranked))
    return sorted([pos for pos, _ in ranked[:M]])


def evaluate_expansion(
    dataset: list[dict],
    model_path: str,
    model_label: str,
    expansions: list[int],
) -> None:
    print(f"\n{'─'*60}")
    print(f"Modelo: {model_label}")
    print(f"{'─'*60}")
    print(f"{'Expansion':>10} | {'Recall %':>8} | {'Candidatos':>12} | {'% espacio':>10}")
    print(f"{'─'*10}-+-{'─'*8}-+-{'─'*12}-+-{'─'*10}")

    for expansion in expansions:
        recalls, n_cands, spaces = [], [], []

        for instance in dataset:
            sentences = instance["sentences"]
            K         = instance["K"]
            gt_cuts   = set(instance["ground_truth_cuts"])
            n         = instance["n_sentences"]

            candidates = get_cut_candidates(sentences, K, model_path, expansion)
            cand_set   = set(candidates)

            recall = len(gt_cuts & cand_set) / len(gt_cuts) if gt_cuts else 0.0
            recalls.append(recall)
            n_cands.append(len(candidates))
            spaces.append(len(candidates) / (n - 1) if n > 1 else 1.0)

        print(f"{expansion:>10} | {sum(recalls)/len(recalls)*100:>7.1f}% | "
              f"{sum(n_cands)/len(n_cands):>12.1f} | "
              f"{sum(spaces)/len(spaces)*100:>9.1f}%")


if __name__ == "__main__":
    dataset_path = Path("dataset.json")
    if not dataset_path.exists():
        print("ERROR: dataset.json no encontrado.")
        exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 60)
    print("BÚSQUEDA DE EXPANSION ÓPTIMO — FASE 1")
    print(f"Dataset: {len(dataset)} instancias")
    print("=" * 60)

    expansions = [2, 3, 4, 5, 6, 8, 10]

    for label, path in [("Modelo A (MiniLM)", MODEL_A), ("Modelo B (mpnet)", MODEL_B)]:
        if Path(path).exists():
            evaluate_expansion(dataset, path, label, expansions)
        else:
            print(f"  [!] {label} no encontrado, se omite.")

    print(f"\n{'─'*60}")
    print("Recall ≥ 80% es el objetivo mínimo para la Fase 1.")
