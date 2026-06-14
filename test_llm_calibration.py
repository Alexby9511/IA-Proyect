"""
test_llm_calibration.py
-----------------------
Verifica rápidamente si el LLM distingue entre segmentos cohesivos
y segmentos mezclados ANTES de ejecutar run_experiments.py completo.

Uso:
    python test_llm_calibration.py

Salida esperada:
  - Segmento cohesivo: score ~7-10
  - Segmento mezclado: score ~1-4
  - Diferencia >= 3 → LLM calibrado, llm_weight puede usarse con confianza
  - Diferencia < 3  → LLM poco discriminativo, reducir llm_weight o documentar
"""

from llm_evaluator import evaluate_segment, get_cache_stats, reset_cache

# ---------------------------------------------------------------------------
# Ejemplos manuales fijos
# ---------------------------------------------------------------------------

# (a) Segmento claramente cohesivo: todo sobre fotosíntesis
COHESIVE = [
    "La fotosíntesis es el proceso por el cual las plantas convierten luz solar en energía química.",
    "Las plantas capturan la luz mediante la clorofila, un pigmento verde presente en los cloroplastos.",
    "Durante la fotosíntesis se producen glucosa y oxígeno a partir de dióxido de carbono y agua.",
    "La glucosa obtenida sirve como fuente de energía para el crecimiento y desarrollo de la planta.",
    "Sin fotosíntesis, la cadena alimentaria y la producción de oxígeno en la Tierra serían imposibles.",
]

# (b) Segmento mezclado: fotosíntesis + Imperio romano + astronomía
MIXED = [
    "La fotosíntesis convierte la luz solar en energía química almacenada en glucosa.",
    "El Imperio romano dominó gran parte de Europa, el norte de África y el Medio Oriente.",
    "La Vía Láctea es una galaxia espiral barrada que contiene entre 200 y 400 mil millones de estrellas.",
    "Las plantas usan la clorofila para absorber la energía lumínica necesaria para la fotosíntesis.",
    "Roma fundó instituciones que influyeron profundamente en el derecho y la cultura occidentales.",
]

# (c) Segmento moderadamente cohesivo: historia de la ciencia (relacionado pero no idéntico)
MODERATE = [
    "Isaac Newton formuló las leyes del movimiento y la gravitación universal en el siglo XVII.",
    "Albert Einstein desarrolló la teoría de la relatividad especial y general a principios del siglo XX.",
    "Marie Curie fue pionera en el estudio de la radiactividad y primera mujer en ganar el Nobel.",
    "Charles Darwin propuso la teoría de la evolución por selección natural en 1859.",
    "Galileo Galilei fue uno de los primeros científicos en usar el telescopio para observar el cielo.",
]


def run_calibration():
    reset_cache()
    print("=" * 60)
    print("TEST DE CALIBRACIÓN DEL LLM")
    print("=" * 60)

    print("\n[1/3] Segmento COHESIVO (fotosíntesis):")
    score_coh = evaluate_segment(COHESIVE, 0, len(COHESIVE))
    print(f"      Score: {score_coh:.1f}/10  (esperado: 7-10)")

    print("\n[2/3] Segmento MEZCLADO (fotosíntesis + Roma + astronomía):")
    score_mix = evaluate_segment(MIXED, 0, len(MIXED))
    print(f"      Score: {score_mix:.1f}/10  (esperado: 1-4)")

    print("\n[3/3] Segmento MODERADO (historia de la ciencia):")
    score_mod = evaluate_segment(MODERATE, 0, len(MODERATE))
    print(f"      Score: {score_mod:.1f}/10  (esperado: 5-7)")

    print(f"\n{'─'*60}")
    print("RESULTADO")
    print(f"{'─'*60}")
    print(f"  Cohesivo  : {score_coh:.1f}")
    print(f"  Moderado  : {score_mod:.1f}")
    print(f"  Mezclado  : {score_mix:.1f}")
    diff = score_coh - score_mix
    print(f"  Diferencia cohesivo - mezclado: {diff:.1f}")

    print()
    if diff >= 3:
        print("✓ El LLM distingue bien entre cohesivo y mezclado.")
        print("  llm_weight puede usarse con confianza en pipeline.py.")
    elif diff >= 1:
        print("⚠ El LLM distingue poco. Diferencia < 3.")
        print("  Recomendación: reducir llm_weight a 0.2 o menos.")
        print("  Documenta esta limitación en el informe.")
    else:
        print("✗ El LLM NO distingue entre cohesivo y mezclado.")
        print("  Recomendación: usar llm_weight=0.0 (solo embeddings)")
        print("  o probar con un modelo mayor.")
        print("  Documenta esta limitación en el informe.")

    stats = get_cache_stats()
    print(f"\nLlamadas LLM: {stats['misses']} | Tiempo estimado: {stats['misses'] * 10}s aprox.")


if __name__ == "__main__":
    run_calibration()
