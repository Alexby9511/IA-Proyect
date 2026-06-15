"""
test_strict_prompts.py
-----------------------
Compara variantes de prompts estrictos (N1‑N8) sobre casos
derivados de una instancia real del dataset, para ver cuál distingue
mejor entre segmentos dentro de una misma sección (HIGH) y ventanas
que cruzan una frontera real (LOW).

Las variantes N5‑N8 mejoran a N2 añadiendo conciencia de narrativa
cronológica para reducir falsos positivos en secciones de historia.

Uso:
    python test_strict_prompts.py --dataset dataset.json --index 0
    python test_strict_prompts.py --dataset dataset.json --index 0 --variants N5_narrative_aware_cut_point,N6_narrative_examples
    python test_strict_prompts.py --dataset dataset.json --index 0 --sleep 5
"""

import argparse
import json
import re
import time
from pathlib import Path

from llm_evaluator import _call_llm, _parse_score


# ---------------------------------------------------------------------------
# Casos de prueba fijos (independientes del dataset) — solo con --include-fixed
# ---------------------------------------------------------------------------

COHESIVE_CLASICO = (
    "La fotosíntesis es el proceso por el cual las plantas convierten luz solar en energía química. "
    "Las plantas capturan la luz mediante la clorofila, un pigmento verde presente en los cloroplastos. "
    "Durante la fotosíntesis se producen glucosa y oxígeno a partir de dióxido de carbono y agua. "
    "La glucosa obtenida sirve como fuente de energía para el crecimiento y desarrollo de la planta. "
    "Sin fotosíntesis, la cadena alimentaria y la producción de oxígeno en la Tierra serían imposibles."
)

MIXED_CLASICO = (
    "La fotosíntesis convierte la luz solar en energía química almacenada en glucosa. "
    "El Imperio romano dominó gran parte de Europa, el norte de África y el Medio Oriente. "
    "La Vía Láctea es una galaxia espiral barrada que contiene entre 200 y 400 mil millones de estrellas. "
    "Las plantas usan la clorofila para absorber la energía lumínica necesaria para la fotosíntesis. "
    "Roma fundó instituciones que influyeron profundamente en el derecho y la cultura occidentales."
)

NARRATIVE_CLASICO = (
    "En el siglo XVII, William Harvey describió por primera vez la circulación de la sangre, "
    "proponiendo que el corazón actuaba como una bomba que impulsaba el líquido a través de los vasos. "
    "Durante el siglo XVIII, Stephen Hales realizó las primeras mediciones directas de la presión sanguínea "
    "en animales, insertando tubos de vidrio en arterias de caballos. "
    "A comienzos del siglo XIX, Jean Poiseuille desarrolló ecuaciones matemáticas para describir el flujo "
    "de líquidos en tubos estrechos, sentando las bases de la hemodinámica moderna. "
    "Más adelante, en la segunda mitad del siglo XIX, Carl Ludwig inventó el quimógrafo, un instrumento "
    "que permitía registrar gráficamente las variaciones de presión arterial en tiempo real. "
    "A principios del siglo XX, Scipione Riva-Rocci diseñó el esfigmomanómetro de manguito, "
    "precursor directo de los aparatos de medición de presión arterial empleados en la actualidad. "
    "Poco después, Nikolai Korotkov describió los sonidos que se escuchan al desinflar el manguito, "
    "lo que permitió determinar tanto la presión sistólica como la diastólica. "
    "Durante el siglo XX, la introducción de dispositivos electrónicos automatizó por completo "
    "el proceso, eliminando la necesidad de un observador entrenado con estetoscopio. "
    "En las últimas décadas, los monitores ambulatorios permiten registrar la presión arterial "
    "de un paciente de forma continua durante 24 horas en su vida diaria."
)

FIXED_CASES = [
    {"name": "COHESIVE_clasico", "text": COHESIVE_CLASICO, "expected": "HIGH"},
    {"name": "MIXED_clasico",    "text": MIXED_CLASICO,    "expected": "LOW"},
    {"name": "NARRATIVE_clasico", "text": NARRATIVE_CLASICO, "expected": "HIGH"},
]


# ---------------------------------------------------------------------------
# Variantes de prompt (N1‑N8)
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    

    "N5_narrative_aware_cut_point": """You are a text segmentation algorithm. Below is a text that may contain a transition between two different sub-topics. Your job is to find the EXACT sentence index where the topic changes.

IMPORTANT: A chronological narrative (e.g., "how X was discovered over time") that mentions different people, dates, or specific findings is considered ONE continuous sub-topic. Do NOT mark a cut just because a new author or century appears. Only mark a cut if the text changes the KIND of content — e.g., from history to physical structure, or from one mechanism to a different, independent mechanism.

Read the sentences one by one. Identify the FIRST sentence that clearly belongs to a new sub-topic, different from the preceding sentences, taking into account the rule above. Then output:

- The 0-based index of that sentence (0 = first sentence), OR
- -1 if the whole text discusses the same sub-topic (including a continuous narrative).

Text:
{texto}

Put your reasoning inside <think></think> tags. Output your final decision exactly as: [INDEX: N] where N is the integer index, or [INDEX: -1] if no change.""",

    "N6_narrative_examples": """You are a text segmentation algorithm. Find if there is a sub-topic boundary in the text.

A sub-topic boundary means the text changes its NATURE: e.g., from describing a historical discovery process to describing a physical structure, or from one scientific mechanism to a completely different mechanism.

A continuous chronological narrative that mentions many different scientists, years, or experiments is NOT a boundary — it is ONE narrative thread (e.g., "the history of how X was discovered"). Do NOT mark a cut just because a new person or date appears.

Examples of NO boundary (single narrative thread):
- "Aristotle proposed X. In the 18th century, Priestley discovered Y. Later, Calvin traced Z." (all history of photosynthesis discovery)

Examples of YES boundary (true sub-topic change):
- "The history of photosynthesis discovery. The structure of chloroplasts." (history → anatomy)
- "The light-dependent reactions of photosynthesis. The Calvin cycle." (two different biochemical mechanisms)

Now analyze this text. Output the 0-based index of the first sentence that starts a NEW sub-topic, or -1 if the whole text is one continuous thread.

Text:
{texto}

Output exactly as: [INDEX: N] or [INDEX: -1]""",

    "N7_question_based": """You are a text segmentation algorithm. Your job is to decide if the text answers ONE single overarching question or MULTIPLE different questions.

If the entire text answers the SAME question (e.g., "How was X discovered?", "What is the structure of X?", "How does process X work?"), then it is one continuous sub-topic and you should output -1.

If the text changes the question it answers (e.g., from "How was X discovered?" to "What is the structure of X?"), then it contains a sub-topic boundary. Find the first sentence that starts answering a DIFFERENT question and output its 0-based index.

Read the sentences one by one. Identify if and where the overarching question changes.

Text:
{texto}

Output exactly as: [INDEX: N] or [INDEX: -1]""",

    "N8_strict_narrative_index": """You are a text segmenter. Find the first sentence that changes the sub-topic.

RULES:
1. A chronological narrative (e.g., history of a discovery) with many different people, dates, or experiments is ONE sub-topic. Do NOT cut inside a narrative.
2. A cut ONLY happens when the text changes TYPE of content: history → structure, mechanism → different mechanism, theory → experiment, etc.
3. If the whole text is one continuous narrative or one single sub-topic, output -1.

Read the sentences one by one. Find the first sentence that violates the current sub-topic. Output the 0-based index, or -1.

Text:
{texto}

Output exactly as: [INDEX: N] or [INDEX: -1]""",
}


# ---------------------------------------------------------------------------
# Construcción de casos de prueba a partir del dataset
# ---------------------------------------------------------------------------

def build_dataset_cases(
    instance: dict,
    boundary_window: int = 6,
) -> list[dict]:
    sentences = instance["sentences"]
    cuts = sorted(instance.get("ground_truth_cuts", []))
    n = len(sentences)
    boundaries = [0] + cuts + [n]

    within_cases = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 4:
            continue
        within_cases.append({
            "name": f"within_section_{i + 1}_[{start}:{end})",
            "text": " ".join(sentences[start:end]),
            "expected": "HIGH",
        })

    cross_cases = []
    for cut in cuts:
        start = max(0, cut - boundary_window)
        end = min(n, cut + boundary_window)
        cross_cases.append({
            "name": f"cross_boundary_{cut}_[{start}:{end})",
            "text": " ".join(sentences[start:end]),
            "expected": "LOW",
        })

    return within_cases + cross_cases


# ---------------------------------------------------------------------------
# Clasificación de resultados
# ---------------------------------------------------------------------------

def classify(score: float) -> str:
    if score >= 7.0:
        return "HIGH"
    if score <= 5.0:
        return "LOW"
    return "BORDERLINE"


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def run_tests(cases: list[dict], variant_names: list[str], sleep_s: float) -> None:
    print("=" * 90)
    print(f"COMPARACIÓN DE PROMPTS ESTRICTOS — {len(variant_names)} variantes x {len(cases)} casos")
    print("=" * 90)

    results: dict[str, dict[str, tuple]] = {}

    for variant_name in variant_names:
        template = PROMPT_VARIANTS[variant_name]
        print(f"\n{'─'*90}")
        print(f"Variante: {variant_name}")
        print(f"{'─'*90}")

        results[variant_name] = {}

        for case in cases:
            prompt = template.format(texto=case["text"])
            try:
                response = _call_llm(prompt)

                # Parseo especial para variantes que devuelven índice
                if variant_name in (
                    "N2_find_cut_point", "N3_sliding_window",
                    "N5_narrative_aware_cut_point", "N6_narrative_examples",
                    "N7_question_based", "N8_strict_narrative_index"
                ):
                    idx_match = re.search(r'\[INDEX:\s*(-?\d+)\]', response, re.IGNORECASE)
                    if idx_match:
                        idx = int(idx_match.group(1))
                        score = 2.0 if idx >= 0 else 9.5
                    else:
                        score = _parse_score(response)
                else:
                    score = _parse_score(response)

            except Exception as e:
                print(f"  {case['name']:<35} ERROR: {e}")
                results[variant_name][case["name"]] = (None, "ERROR", False)
                time.sleep(sleep_s)
                continue

            cls = classify(score)
            correct = (cls == case["expected"])
            status = "✓" if correct else "✗"

            preview = re.sub(r'<think>.*?</think>', '[…]', response, flags=re.DOTALL).strip()
            preview = preview.replace("\n", " ")[:70]

            print(f"  {case['name']:<35} esperado={case['expected']:<10} "
                  f"score={score:>4.1f} → {cls:<10} {status}  | {preview}")

            results[variant_name][case["name"]] = (score, cls, correct)
            time.sleep(sleep_s)

    # -----------------------------------------------------------------------
    # Tabla resumen
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("RESUMEN POR VARIANTE")
    print(f"{'='*90}")
    print(f"{'Variante':<28} {'Accuracy':>10} {'Score HIGH (avg)':>18} {'Score LOW (avg)':>17} {'Diff':>8}")
    print(f"{'─'*28} {'─'*10} {'─'*18} {'─'*17} {'─'*8}")

    summary = []
    for variant_name in variant_names:
        per_case = results[variant_name]
        valid = {k: v for k, v in per_case.items() if v[0] is not None}
        if not valid:
            print(f"{variant_name:<28} {'ERROR':>10}")
            continue

        n_correct = sum(1 for _, _, c in valid.values() if c)
        accuracy = n_correct / len(valid)

        high_scores = [v[0] for k, v in valid.items()
                       if next(c for c in cases if c["name"] == k)["expected"] == "HIGH"]
        low_scores = [v[0] for k, v in valid.items()
                      if next(c for c in cases if c["name"] == k)["expected"] == "LOW"]

        avg_high = sum(high_scores) / len(high_scores) if high_scores else float("nan")
        avg_low = sum(low_scores) / len(low_scores) if low_scores else float("nan")
        diff = avg_high - avg_low if high_scores and low_scores else float("nan")

        summary.append((variant_name, accuracy, avg_high, avg_low, diff))
        print(f"{variant_name:<28} {accuracy:>9.0%} {avg_high:>18.2f} {avg_low:>17.2f} {diff:>+8.2f}")

    if summary:
        best = max(summary, key=lambda x: (x[1], x[4] if x[4] == x[4] else -999))
        print(f"\n✓ Mejor variante: '{best[0]}' (accuracy={best[1]:.0%}, diff={best[4]:+.2f})")
        print("  Recomendación: usar esta variante como PROMPT_STRICT en llm_evaluator.py.")


def main():
    parser = argparse.ArgumentParser(description="Compara variantes de prompt estricto (N1‑N8)")
    parser.add_argument("--dataset", type=Path, default=Path("dataset.json"),
                         help="Ruta a dataset.json (default: dataset.json)")
    parser.add_argument("--index", type=int, default=0,
                         help="Índice de la instancia a usar (default: 0)")
    parser.add_argument("--boundary-window", type=int, default=6,
                         help="Oraciones a cada lado del corte para casos CROSS_BOUNDARY (default: 6)")
    parser.add_argument("--variants", type=str, default=None,
                         help="Lista separada por comas de variantes a probar (default: todas N1‑N8).")
    parser.add_argument("--sleep", type=float, default=3.0,
                         help="Segundos de espera entre llamadas al LLM (default: 3.0)")
    parser.add_argument("--include-fixed", action="store_true",
                         help="Incluir los casos fijos clásicos además de los derivados del dataset")
    args = parser.parse_args()

    variant_names = list(PROMPT_VARIANTS.keys())
    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
        for v in variant_names:
            if v not in PROMPT_VARIANTS:
                raise ValueError(f"Variante desconocida: '{v}'. "
                                 f"Disponibles: {list(PROMPT_VARIANTS.keys())}")

    cases = []
    if args.include_fixed:
        cases.extend(FIXED_CASES)

    if args.dataset.exists():
        with open(args.dataset, encoding="utf-8") as f:
            dataset = json.load(f)
        instance = dataset[args.index]
        print(f"Instancia: {instance.get('title', f'instancia_{args.index}')} "
              f"({instance['n_sentences']} oraciones, "
              f"ground_truth_cuts={instance.get('ground_truth_cuts', [])})")
        cases.extend(build_dataset_cases(instance, args.boundary_window))
    else:
        print(f"[!] {args.dataset} no encontrado, solo se usarán los casos fijos (si se activaron).")

    if not cases:
        print("ERROR: no hay casos de prueba para ejecutar.")
        return

    n_calls = len(variant_names) * len(cases)
    print(f"\nVariantes a probar : {variant_names}")
    print(f"Casos de prueba    : {len(cases)} "
          f"({[c['name'] for c in cases]})")
    print(f"Llamadas totales   : {n_calls} "
          f"(~{n_calls * args.sleep:.0f}s solo en sleeps, sin contar el tiempo del LLM)\n")

    run_tests(cases, variant_names, args.sleep)


if __name__ == "__main__":
    main()