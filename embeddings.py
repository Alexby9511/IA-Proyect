"""
embeddings.py
-------------
Cálculo de embeddings semánticos, perfil de similitud, detección de
valles candidatos y generación de cortes iniciales para la segmentación.

Modelos (cargados desde carpetas locales):
    A) models/paraphrase-multilingual-MiniLM-L12-v2  (rápido)
    B) models/paraphrase-multilingual-mpnet-base-v2  (más preciso)

Requisitos:
    pip install sentence-transformers scipy
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import argrelextrema
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Rutas locales a los modelos
# ---------------------------------------------------------------------------
MODELS_DIR = Path("./models")
MODEL_A = str(MODELS_DIR / "paraphrase-multilingual-MiniLM-L12-v2")
MODEL_B = str(MODELS_DIR / "paraphrase-multilingual-mpnet-base-v2")

_model_cache: dict[str, SentenceTransformer] = {}

def load_model(model_path: str) -> SentenceTransformer:
    if model_path not in _model_cache:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        print(f"  Cargando modelo: {Path(model_path).name} ...")
        _model_cache[model_path] = SentenceTransformer(model_path)
        print(f"  Modelo cargado.")
    return _model_cache[model_path]

def compute_embeddings(sentences: list[str], model_path: str) -> np.ndarray:
    model = load_model(model_path)
    return model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def compute_similarity_profile(embeddings: np.ndarray) -> list[float]:
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
    return similarities

def get_ranked_candidates(
    sentences: list[str],
    model_path: str,
) -> list[tuple[int, float]]:
    n = len(sentences)
    if n < 2:
        return []
    embeddings = compute_embeddings(sentences, model_path)
    similarities = compute_similarity_profile(embeddings)
    ranked = [(i + 1, sim) for i, sim in enumerate(similarities)]
    ranked.sort(key=lambda x: x[1])
    return ranked

def get_initial_cuts(
    ranked_candidates: list[tuple[int, float]],
    K: int,
) -> list[int]:
    best = ranked_candidates[:K - 1]
    return sorted([pos for pos, _ in best])

def find_local_valleys(similarities: list[float]) -> list[tuple[int, float]]:
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

def evaluate_recall(dataset: list[dict], model_path: str) -> dict:
    recalls_k, recalls_2k = [], []
    for instance in dataset:
        sentences = instance["sentences"]
        K = instance["K"]
        gt_set = set(instance["ground_truth_cuts"])
        ranked = get_ranked_candidates(sentences, model_path)
        top_k = set(pos for pos, _ in ranked[:K - 1])
        top_2k = set(pos for pos, _ in ranked[:(K - 1) * 2])
        recalls_k.append(len(gt_set & top_k) / len(gt_set) if gt_set else 0.0)
        recalls_2k.append(len(gt_set & top_2k) / len(gt_set) if gt_set else 0.0)
    return {
        "recall_k": sum(recalls_k) / len(recalls_k),
        "recall_2k": sum(recalls_2k) / len(recalls_2k),
    }

def test_instance(instance: dict, model_path: str, model_label: str) -> None:
    sentences = instance["sentences"]
    K = instance["K"]
    gt_cuts = instance["ground_truth_cuts"]
    n = instance["n_sentences"]
    ranked = get_ranked_candidates(sentences, model_path)
    initial_cuts = get_initial_cuts(ranked, K)
    gt_set = set(gt_cuts)
    top_k = set(pos for pos, _ in ranked[:K - 1])
    top_2k = set(pos for pos, _ in ranked[:(K - 1) * 2])
    recall_k = len(gt_set & top_k) / len(gt_set) if gt_set else 0.0
    recall_2k = len(gt_set & top_2k) / len(gt_set) if gt_set else 0.0
    print(f"\n{'─'*60}")
    print(f"Instancia       : {instance['title']}")
    print(f"Oraciones       : {n}  |  K={K}  |  GT: {gt_cuts}")
    print(f"Modelo          : {model_label}")
    print(f"Cortes iniciales: {initial_cuts}")
    print(f"Top-{K-1} posiciones: {[pos for pos, _ in ranked[:K-1]]}")
    print(f"Recall top-K    : {recall_k:.0%}  ({len(gt_set & top_k)}/{len(gt_set)})")
    print(f"Recall top-2K   : {recall_2k:.0%}  ({len(gt_set & top_2k)}/{len(gt_set)})")

if __name__ == "__main__":
    print("Verificando modelos...")
    for label, path in [("Modelo A (MiniLM)", MODEL_A), ("Modelo B (mpnet)", MODEL_B)]:
        status = "✓" if Path(path).exists() else "✗ NO ENCONTRADO"
        print(f"  {status}  {label}: {path}")

    datasets = {}
    for name, fname in [("Wikipedia", "dataset.json"), ("Sintético", "dataset_synthetic.json")]:
        p = Path(fname)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                datasets[name] = json.load(f)
            print(f"  ✓  Dataset {name}: {len(datasets[name])} instancias")
        else:
            print(f"  ✗  Dataset {name}: no encontrado")

    if not datasets:
        print("ERROR: No se encontró ningún dataset.")
        exit(1)

    print(f"\n{'='*60}")
    print("RECALL AGREGADO POR DATASET Y MODELO")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Modelo':<22} {'top-K':>7} {'top-2K':>8}")
    print(f"{'─'*12} {'─'*22} {'─'*7} {'─'*8}")
    for ds_name, dataset in datasets.items():
        for label, path in [("MiniLM (A)", MODEL_A), ("mpnet (B)", MODEL_B)]:
            if not Path(path).exists():
                continue
            stats = evaluate_recall(dataset, path)
            print(f"{ds_name:<12} {label:<22} {stats['recall_k']:>6.0%} {stats['recall_2k']:>7.0%}")

    for ds_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"DETALLE — 3 PRIMERAS INSTANCIAS ({ds_name})")
        print(f"{'='*60}")
        for instance in dataset[:3]:
            for label, path in [("MiniLM (A)", MODEL_A), ("mpnet (B)", MODEL_B)]:
                if Path(path).exists():
                    test_instance(instance, path, label)