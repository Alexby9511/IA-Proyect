"""
run_experiments.py
------------------
Ejecuta el pipeline (DP + grafo esparcido con poda por embeddings) sobre
todas las instancias de los datasets disponibles (dataset.json,
dataset_synthetic.json), variando:
  - Modelo de embeddings (A / B)
  - lambda_penalty (penalización estructural por segmento)
  - embedding_weight (peso de la cohesión de embeddings frente al LLM)

Genera results.csv y muestra tabla resumen.

Nota: cada combinación de hiperparámetros implica una corrida completa
del DP, con sus propias llamadas al LLM (del orden de 15-30 por
instancia, según las podas). Si el dataset es grande, usa --quick para
limitar a 3 instancias por dataset durante pruebas.

Uso:
    python run_experiments.py
    python run_experiments.py --quick   # solo 3 instancias por dataset
"""

import csv
import json
import random
import argparse
import time
from pathlib import Path

from embeddings import MODEL_A, MODEL_B
from pipeline import run_pipeline
from adaptive_segmentation import compute_f1

DATASETS          = ["dataset.json", "dataset_synthetic.json"]
LAMBDA_PENALTIES  = [1.0, 3.0, 5.0]
EMBEDDING_WEIGHT  = 0.3   # fijo; documentar en el informe si se quiere
                          # barrer también este parámetro
MIN_COHESION      = 0.5
MODEL_OPTIONS     = {"MODEL_A": MODEL_A, "MODEL_B": MODEL_B}

FIELDNAMES = [
    "dataset", "instance_id", "title", "n_sentences", "K_real",
    "model", "lambda_penalty", "embedding_weight",
    "K_detected", "F1_tol0", "Precision_tol0", "Recall_tol0",
    "F1_tol1", "Precision_tol1", "Recall_tol1",
    "time_sec", "llm_calls", "cache_hit_rate",
    "edges_total", "edges_pruned_length", "edges_pruned_cohesion", "edges_evaluated_llm",
    "evaluation_mode", "global_score",
]


def run_all(quick: bool = False) -> None:
    random.seed(42)
    results = []

    print("=" * 70)
    print("EJECUCIÓN DE EXPERIMENTOS")
    print("=" * 70)

    for ds_name in DATASETS:
        ds_path = Path(ds_name)
        if not ds_path.exists():
            print(f"  [!] Dataset {ds_name} no encontrado, se omite.")
            continue

        with open(ds_path, encoding="utf-8") as f:
            dataset = json.load(f)

        if quick:
            dataset = dataset[:3]

        print(f"\nDataset: {ds_name} ({len(dataset)} instancias)")

        for idx, instance in enumerate(dataset):
            sentences = instance["sentences"]
            gt_cuts   = instance.get("ground_truth_cuts", [])
            title     = instance.get("title", f"instancia_{idx}")
            n         = instance["n_sentences"]
            K_real    = instance["K"]

            for model_name, model_path in MODEL_OPTIONS.items():
                if not Path(model_path).exists():
                    print(f"  [!] Modelo {model_name} no encontrado, se omite.")
                    continue

                for lp in LAMBDA_PENALTIES:
                    tag = f"{model_name} | lambda={lp}"
                    print(f"  [{idx+1}/{len(dataset)}] {title[:40]} | {tag} ...", end=" ", flush=True)

                    t0 = time.time()
                    try:
                        res = run_pipeline(
                            sentences,
                            model_path       = model_path,
                            min_seg          = 3,
                            lambda_penalty   = lp,
                            min_cohesion     = MIN_COHESION,
                            embedding_weight = EMBEDDING_WEIGHT,
                        )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue

                    elapsed  = time.time() - t0
                    f1_0, p0, r0 = compute_f1(res["best_cuts"], gt_cuts, tolerance=0)
                    f1_1, p1, r1 = compute_f1(res["best_cuts"], gt_cuts, tolerance=1)
                    calls    = res.get("total_llm_calls", 0)
                    hit_rate = res.get("cache_stats", {}).get("hit_rate", 0.0)
                    dp_stats = res.get("dp_stats", {})

                    results.append({
                        "dataset":               ds_name,
                        "instance_id":           idx,
                        "title":                 title,
                        "n_sentences":           n,
                        "K_real":                K_real,
                        "model":                 model_name,
                        "lambda_penalty":        lp,
                        "embedding_weight":      EMBEDDING_WEIGHT,
                        "K_detected":            res["best_K"],
                        "F1_tol0":               f1_0,
                        "Precision_tol0":        p0,
                        "Recall_tol0":           r0,
                        "F1_tol1":               f1_1,
                        "Precision_tol1":        p1,
                        "Recall_tol1":           r1,
                        "time_sec":              elapsed,
                        "llm_calls":             calls,
                        "cache_hit_rate":        hit_rate,
                        "edges_total":           dp_stats.get("edges_total", 0),
                        "edges_pruned_length":   dp_stats.get("edges_pruned_length", 0),
                        "edges_pruned_cohesion": dp_stats.get("edges_pruned_cohesion", 0),
                        "edges_evaluated_llm":   dp_stats.get("edges_evaluated_llm", 0),
                        "evaluation_mode":       dp_stats.get("evaluation_mode"),
                        "global_score":          dp_stats.get("global_score"),
                    })
                    print(f"K={res['best_K']} F1={f1_0:.3f} ({elapsed:.1f}s)")

    # Exportar CSV
    csv_path = "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResultados guardados en: {csv_path}")

    # Tabla resumen
    if not results:
        print("Sin resultados para mostrar.")
        return

    print("\n" + "=" * 90)
    print("RESUMEN POR CONFIGURACIÓN (promedios)")
    print("=" * 90)
    print(f"{'Modelo':<10} {'lambda':<8} {'N':<5} "
          f"{'F1(tol0)':<9} {'F1(tol1)':<9} {'t(s)':<7} {'LLM':<6} {'PodCoh':<7}")
    print("-" * 90)

    groups: dict = {}
    for r in results:
        key = (r["model"], r["lambda_penalty"])
        groups.setdefault(key, []).append(r)

    for (model, lp), items in sorted(groups.items()):
        avg_f1_0  = sum(it["F1_tol0"]   for it in items) / len(items)
        avg_f1_1  = sum(it["F1_tol1"]   for it in items) / len(items)
        avg_t     = sum(it["time_sec"]  for it in items) / len(items)
        avg_calls = sum(it["llm_calls"] for it in items) / len(items)
        avg_pruned_coh = sum(it["edges_pruned_cohesion"] for it in items) / len(items)
        print(f"{model:<10} {lp:<8.1f} {len(items):<5} "
              f"{avg_f1_0:<9.3f} {avg_f1_1:<9.3f} {avg_t:<7.1f} {avg_calls:<6.0f} {avg_pruned_coh:<7.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Ejecutar solo 3 instancias por dataset (para pruebas)")
    args = parser.parse_args()
    run_all(quick=args.quick)