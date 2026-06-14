"""
test_prompts.py
---------------
Prueba 3 estilos de prompt distintos para encontrar cuál hace que
deepseek-local:1.5b distinga entre texto cohesivo y texto mezclado.
"""

import re
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "deepseek-local:1.5b"

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
        "Responde SOLO con el número al final.\n\n"
        "Texto: {texto}\n\n"
        "Puntuación:"
    ),
    "B — Explicación breve y escala": (
        "Lee este texto y evalúa si trata un solo tema o mezcla varios.\n"
        "Piensa brevemente en tu decisión y luego da una puntuación del 1 al 10.\n"
        "1 = Temas mezclados, 10 = Un solo tema coherente.\n\n"
        "Texto: {texto}\n\n"
        "Respuesta:"
    ),
    "C — Instrucción en inglés (recomendado)": (
        "Rate the thematic coherence of this text from 1 to 10.\n"
        "1 = completely unrelated topics mixed together.\n"
        "10 = all sentences discuss exactly the same topic.\n"
        "Briefly explain your reasoning, then reply with the final number.\n\n"
        "Text: {texto}\n\n"
        "Score:"
    ),
    "D — Strict role and constraint": (
        "You are an expert text analyzer. Your task is to rate the thematic cohesion of the text.\n"
        "Rules:\n"
        "- If the text discusses a SINGLE topic consistently, score it 10.\n"
        "- If the text abruptly switches topics or mixes completely unrelated subjects, score it 1.\n"
        "Briefly explain your thought process, then output the final score.\n\n"
        "Text: {texto}\n\n"
        "Score:"
    ),
    "E — Few-shot examples (Highly Recommended)": (
        "Rate the thematic cohesion from 1 (mixed topics) to 10 (single topic).\n\n"
        "Example 1:\n"
        "Text: The sun is a star. Cars need fuel to run. Water freezes at zero degrees.\n"
        "Score: 1\n\n"
        "Example 2:\n"
        "Text: Mitochondria is the powerhouse of the cell. It generates most of the cell's supply of ATP, used as a source of chemical energy.\n"
        "Score: 10\n\n"
        "Now rate this text.\n"
        "Text: {texto}\n\n"
        "Score:"
    ),
    "F — Chain of Thought trigger": (
        "Determine if the following text is thematically cohesive (discusses one topic) or mixed (discusses multiple unrelated topics).\n"
        "Step 1: Identify the main topics.\n"
        "Step 2: Decide if they are related or completely different.\n"
        "Step 3: Assign a score (10 for purely cohesive, 1 for heavily mixed).\n\n"
        "Text: {texto}\n\n"
        "Score:"
    ),
}

def call_llm(prompt: str, num_predict: int = 300) -> str:
    """Aumentamos num_predict a 300 para darle espacio a la fase de <think>"""
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
    """Extrae el número final ignorando la fase de pensamiento."""
    
    # Si empezó a pensar pero nunca terminó, la respuesta está truncada
    if "<think>" in response and "</think>" not in response:
        print(" [!] Respuesta truncada por num_predict")
        return -1.0
        
    # Borrar todo el bloque de razonamiento
    clean_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Buscar números (del 1 al 10) en la respuesta limpia
    numbers = re.findall(r'\b(10|[1-9])\b', clean_text)
    if numbers:
        return float(numbers[-1]) # Tomamos el último número emitido
    
    digits = re.findall(r'\d', clean_text)
    if digits:
        return float(min(10, max(1, int(digits[-1]))))
        
    return -1.0 


def run_tests():
    print("=" * 65)
    print(f"PRUEBA DE PROMPTS — modelo: {MODEL}")
    print("=" * 65)

    results = {}

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n{'─'*65}")
        print(f"Prompt: {prompt_name}")
        print(f"{'─'*65}")

        for label, text in [("COHESIVO", COHESIVE_TEXT), ("MEZCLADO", MIXED_TEXT)]:
            prompt = prompt_template.format(texto=text)
            try:
                # Le damos 300 tokens para que pueda pensar y responder
                response, elapsed = call_llm(prompt, num_predict=300)
                score = parse_score(response)
                
                status = f"{score:.0f}/10" if score >= 0 else "FALLO/TRUNCADO"
                
                # Mostrar un fragmento de la respuesta limpia para debug
                clean_preview = re.sub(r'<think>.*?</think>', '[PENSAMIENTO OCULTO]', response, flags=re.DOTALL).replace('\n', ' ')
                
                print(f"  {label:<10}: {status} ({elapsed:.1f}s)")
                print(f"  ↳ Raw: {clean_preview[:80]}...")
                
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
    elif best_diff >= 1:
        print(f"⚠ Prompt '{best_prompt}' distingue poco (diff={best_diff:.1f}).")
    else:
        print("✗ Ningún prompt funciona bien con este modelo.")

if __name__ == "__main__":
    run_tests()