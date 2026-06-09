"""
tune_expansion.py
-----------------
Encuentra el valor óptimo de expansion para la Fase 1.

Prueba varios valores de expansion y mide el recall promedio
sobre todo el dataset para ambos modelos.

Uso:
    python tune_expansion.py
"""

import json
from pathlib import Path
from embeddings import get_cut_candidates, MODEL_A, MODEL_B


def evaluate_expansion(dataset: list[dict], model_path: str, model_label: str, expansions: list[int]) -> None:
    print(f"\n{'─'*60}")
    print(f"Modelo: {model_label}")
    print(f"{'─'*60}")
    print(f"{'Expansion':>10} | {'Recall %':>8} | {'Candidatos promedio':>20} | {'% espacio':>10}")
    print(f"{'─'*10}-+-{'─'*8}-+-{'─'*20}-+-{'─'*10}")

    for expansion in expansions:
        recalls = []
        n_candidates_list = []
        space_reductions = []

        for instance in dataset:
            sentences = instance["sentences"]
            K         = instance["K"]
            gt_cuts   = set(instance["ground_truth_cuts"])
            n         = instance["n_sentences"]

            candidates = get_cut_candidates(sentences, K, model_path, expansion=expansion)
            cand_set   = set(candidates)

            recall = len(gt_cuts & cand_set) / len(gt_cuts) if gt_cuts else 0.0
            recalls.append(recall)
            n_candidates_list.append(len(candidates))
            space_reductions.append(len(candidates) / (n - 1) if n > 1 else 1.0)

        avg_recall     = sum(recalls) / len(recalls)
        avg_candidates = sum(n_candidates_list) / len(n_candidates_list)
        avg_space      = sum(space_reductions) / len(space_reductions)

        print(f"{expansion:>10} | {avg_recall*100:>7.1f}% | {avg_candidates:>20.1f} | {avg_space*100:>9.1f}%")


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

    for model_label, model_path in [("Modelo A (MiniLM)", MODEL_A), ("Modelo B (mpnet)", MODEL_B)]:
        evaluate_expansion(dataset, model_path, model_label, expansions)

    print(f"\n{'─'*60}")
    print("Interpretación:")
    print("  Recall    → % de cortes reales capturados (queremos ≥ 80%)")
    print("  Candidatos → cuántas posiciones pasan al SA (menos = más rápido)")
    print("  % espacio  → qué fracción del espacio total representan")
