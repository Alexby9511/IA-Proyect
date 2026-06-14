"""
run_experiments.py
------------------
Ejecuta el pipeline sobre todas las instancias de los datasets
disponibles (dataset.json, dataset_synthetic.json), variando
configuraciones:
  - Modelo de embeddings (A / B)
  - llm_weight (0.0, 0.4, 0.8)
  - Con SA vs sin SA

Genera results.csv y muestra tabla resumen.

Nota sobre cache_hit_rate: con la arquitectura LLM-al-final, el cache se
reinicia (reset_cache) antes de cada evaluación final por K, y cada
segmento de esa partición se evalúa una sola vez. Por lo tanto
cache_hit_rate será 0.0 en el caso normal (no hay reevaluaciones dentro
de la misma partición); se mantiene en el CSV por completitud y para
detectar particiones degeneradas con segmentos repetidos.

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

DATASETS        = ["dataset.json", "dataset_synthetic.json"]
LLM_WEIGHTS     = [0.0, 0.4, 0.8]
USE_SA_OPTIONS  = [True, False]
MODEL_OPTIONS   = {"MODEL_A": MODEL_A, "MODEL_B": MODEL_B}
MAX_ITER_CAP    = 2000  # el SA ya no llama al LLM, por lo que un cap alto
                        # tiene costo despreciable (solo CPU)

FIELDNAMES = [
    "dataset", "instance_id", "title", "n_sentences", "K_real",
    "model", "llm_weight", "use_sa",
    "K_detected", "F1_tol0", "Precision_tol0", "Recall_tol0",
    "F1_tol1", "Precision_tol1", "Recall_tol1",
    "time_sec", "llm_calls", "cache_hit_rate",
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

                for lw in LLM_WEIGHTS:
                    for use_sa in USE_SA_OPTIONS:
                        tag = f"{model_name} | llm_w={lw} | SA={use_sa}"
                        print(f"  [{idx+1}/{len(dataset)}] {title[:40]} | {tag} ...", end=" ", flush=True)

                        t0 = time.time()
                        try:
                            res = run_pipeline(
                                sentences,
                                model_path   = model_path,
                                min_seg      = 3,
                                max_k        = 8,
                                llm_weight   = lw,
                                max_iter_cap = MAX_ITER_CAP,
                                use_sa       = use_sa,
                            )
                        except Exception as e:
                            print(f"ERROR: {e}")
                            continue

                        elapsed  = time.time() - t0
                        f1_0, p0, r0 = compute_f1(res["best_cuts"], gt_cuts, tolerance=0)
                        f1_1, p1, r1 = compute_f1(res["best_cuts"], gt_cuts, tolerance=1)
                        calls    = res.get("total_llm_calls", 0)
                        hit_rate = res.get("cache_stats", {}).get("hit_rate", 0.0)

                        results.append({
                            "dataset":         ds_name,
                            "instance_id":     idx,
                            "title":           title,
                            "n_sentences":     n,
                            "K_real":          K_real,
                            "model":           model_name,
                            "llm_weight":      lw,
                            "use_sa":          use_sa,
                            "K_detected":      res["best_K"],
                            "F1_tol0":         f1_0,
                            "Precision_tol0":  p0,
                            "Recall_tol0":     r0,
                            "F1_tol1":         f1_1,
                            "Precision_tol1":  p1,
                            "Recall_tol1":     r1,
                            "time_sec":        elapsed,
                            "llm_calls":       calls,
                            "cache_hit_rate":  hit_rate,
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
    print(f"{'Modelo':<10} {'llm_w':<6} {'SA':<6} {'N':<5} "
          f"{'F1(tol0)':<9} {'F1(tol1)':<9} {'t(s)':<7} {'LLM':<8}")
    print("-" * 90)

    groups: dict = {}
    for r in results:
        key = (r["model"], r["llm_weight"], r["use_sa"])
        groups.setdefault(key, []).append(r)

    for (model, lw, sa), items in sorted(groups.items()):
        avg_f1_0  = sum(it["F1_tol0"]   for it in items) / len(items)
        avg_f1_1  = sum(it["F1_tol1"]   for it in items) / len(items)
        avg_t     = sum(it["time_sec"]   for it in items) / len(items)
        avg_calls = sum(it["llm_calls"]  for it in items) / len(items)
        print(f"{model:<10} {lw:<6.1f} {str(sa):<6} {len(items):<5} "
              f"{avg_f1_0:<9.3f} {avg_f1_1:<9.3f} {avg_t:<7.1f} {avg_calls:<8.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Ejecutar solo 3 instancias por dataset (para pruebas)")
    args = parser.parse_args()
    run_all(quick=args.quick)