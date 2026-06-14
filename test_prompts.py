"""
test_prompts.py
---------------
Prueba 3 estilos de prompt distintos para encontrar cuál hace que
mi-llama:latest distinga entre texto cohesivo y texto mezclado.

Uso:
    python test_prompts.py
"""

import re
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "mi-llama:latest"

# ---------------------------------------------------------------------------
# Textos de prueba
# ---------------------------------------------------------------------------

COHESIVE_TEXT = (
    "La fotosíntesis es el proceso por el cual las plantas convierten luz solar en energía química. "
    "Las plantas capturan la luz mediante la clorofila, un pigmento verde presente en los cloroplastos. "
    "Durante la fotosíntesis se producen glucosa y oxígeno a partir de dióxido de carbono y agua. "
    "La glucosa sirve como fuente de energía para el crecimiento de la planta. "
    "Sin fotosíntesis, la producción de oxígeno en la Tierra sería imposible."
)

MIXED_TEXT = (
    "La fotosíntesis convierte la luz solar en energía química almacenada en glucosa. "
    "El Imperio romano dominó gran parte de Europa, el norte de África y el Medio Oriente. "
    "La Vía Láctea es una galaxia espiral barrada con entre 200 y 400 mil millones de estrellas. "
    "Las plantas usan la clorofila para absorber la energía lumínica necesaria. "
    "Roma fundó instituciones que influyeron profundamente en el derecho occidental."
)

# ---------------------------------------------------------------------------
# Prompts a probar
# ---------------------------------------------------------------------------

PROMPTS = {
    "A — Pregunta directa con escala": (
        "En una escala del 1 al 10, ¿qué tan cohesionado temáticamente es este texto?\n"
        "1 = trata varios temas sin relación. 10 = trata un único tema de principio a fin.\n"
        "Responde SOLO con el número.\n\n"
        "Texto: {texto}\n\n"
        "Puntuación:"
    ),
    "B — Pregunta binaria primero, luego número": (
        "Lee este texto y decide:\n"
        "¿Todas las oraciones tratan el MISMO tema? (sí/no)\n"
        "Luego da una puntuación de cohesión del 1 al 10.\n"
        "Formato de respuesta: [sí/no] [número]\n\n"
        "Texto: {texto}\n\n"
        "Respuesta:"
    ),
    "C — Instrucción en inglés (modelos pequeños responden mejor)": (
        "Rate the thematic coherence of this text from 1 to 10.\n"
        "1 = completely unrelated topics mixed together.\n"
        "10 = all sentences discuss exactly the same topic.\n"
        "Reply with ONLY a single number.\n\n"
        "Text: {texto}\n\n"
        "Score:"
    ),
}


def call_llm(prompt: str, num_predict: int = 10) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        }
    }
    t0 = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    elapsed = time.time() - t0
    content = response.json().get("response", "").strip()
    return content, elapsed


def parse_score(response: str) -> float:
    numbers = re.findall(r'\b(10|[1-9])\b', response)
    if numbers:
        return float(numbers[0])
    digits = re.findall(r'\d', response)
    if digits:
        return float(min(10, max(1, int(digits[0]))))
    return -1.0  # -1 indica que no se encontró número


def run_tests():
    print("=" * 65)
    print(f"PRUEBA DE PROMPTS — modelo: {MODEL}")
    print("=" * 65)

    results = {}

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n{'─'*65}")
        print(f"Prompt {prompt_name}")
        print(f"{'─'*65}")

        for label, text in [("COHESIVO", COHESIVE_TEXT), ("MEZCLADO", MIXED_TEXT)]:
            prompt = prompt_template.format(texto=text)
            try:
                response, elapsed = call_llm(prompt, num_predict=15)
                score = parse_score(response)
                status = f"{score:.0f}/10" if score >= 0 else "NO NÚMERO"
                print(f"  {label:<10}: '{response[:40]}' → {status}  ({elapsed:.1f}s)")
                results.setdefault(prompt_name, {})[label] = score
            except Exception as e:
                print(f"  {label:<10}: ERROR — {e}")
                results.setdefault(prompt_name, {})[label] = -1

    # Resumen
    print(f"\n{'='*65}")
    print("RESUMEN — ¿Qué prompt distingue mejor cohesivo vs mezclado?")
    print(f"{'='*65}")
    print(f"{'Prompt':<45} {'Cohesivo':>8} {'Mezclado':>8} {'Diff':>6}")
    print(f"{'─'*45} {'─'*8} {'─'*8} {'─'*6}")

    best_diff = -999
    best_prompt = None

    for pname, scores in results.items():
        coh = scores.get("COHESIVO", -1)
        mix = scores.get("MEZCLADO", -1)
        diff = coh - mix if coh >= 0 and mix >= 0 else -999
        label = "✓ MEJOR" if diff == max(
            (s.get("COHESIVO", -1) - s.get("MEZCLADO", -1))
            for s in results.values()
        ) else ""
        print(f"  {pname[:43]:<43} {coh:>8.1f} {mix:>8.1f} {diff:>+6.1f}  {label}")
        if diff > best_diff:
            best_diff = diff
            best_prompt = pname

    print()
    if best_diff >= 3:
        print(f"✓ Prompt '{best_prompt}' funciona bien (diff={best_diff:.1f}).")
        print("  Actualiza EVAL_PROMPT en llm_evaluator.py con ese prompt.")
    elif best_diff >= 1:
        print(f"⚠ Prompt '{best_prompt}' distingue poco (diff={best_diff:.1f}).")
        print("  Considera bajar llm_weight a 0.1-0.2.")
    else:
        print("✗ Ningún prompt funciona bien con este modelo.")
        print("  Recomendación: usar llm_weight=0.0 o cambiar de modelo.")


if __name__ == "__main__":
    run_tests()
