import time
import re
from llm_evaluator import _call_llm

TEXT_HIGHLY_COHESIVE = (
    "Los cloroplastos se distinguen por ser unas estructuras polimorfas de color verde, siendo la "
    "coloración que presentan consecuencia directa de la presencia del pigmento clorofila en su interior. "
    "Los cloroplastos están delimitados por una envoltura formada, en la mayoría de las algas y en todas las plantas, "
    "por dos membranas (externa e interna) llamadas envueltas, que son ricas en galactolípidos y sulfolípidos. "
    "En algunas algas, las envueltas están formadas por tres o cuatro membranas, lo que se considera prueba de "
    "que se han originado por procesos de endosimbiosis secundaria o terciaria. "
    "Las envueltas de los cloroplastos regulan el tráfico de sustancias entre el citosol y el interior de estos orgánulos."
)

TEXT_SUBTLE_BOUNDARY = (
    "A Sachs se debe la formulación de la ecuación básica de la fotosíntesis: 6 CO2 + 6 H2O → C6H12O6 + 6 O2. "
    "Andreas Franz Wilhelm Schimper daría el nombre de cloroplastos a los cuerpos coloreados de Sachs. "
    "En el último tercio del siglo XIX se sucederían los esfuerzos por establecer las propiedades físico-químicas "
    "de las clorofilas y se comienzan a estudiar los aspectos ecofisiológicos de la fotosíntesis. "
    "En 1905, Frederick Frost Blackman midió la velocidad a la que se produce la fotosíntesis en diferentes condiciones."
)

PROMPTS = {
    "1_ANALISIS_TITULOS": """You are an editor for Wikipedia. Your job is to decide if a text block belongs under ONE single specific subheading, or if it needs to be split.

Step 1: Write a highly specific subheading (max 4 words) for the FIRST half of the text.
Step 2: Write a highly specific subheading (max 4 words) for the SECOND half of the text.
Step 3: Compare them. Are they addressing the exact same specific concept, or has the focus shifted (e.g., from one historical era to another, or from a general theory to a specific experiment)?
Step 4: Score the text from 1.0 to 10.0.
- Score 9.0-10.0 ONLY if the exact same specific subheading applies perfectly to all sentences.
- Score 1.0-4.0 if the subheadings would naturally be different.

Text:
{text}

Write your analysis inside <think></think> tags. Output the final score EXACTLY like this: [SCORE: X.X]""",

    "2_DETECTOR_PIVOTE": """Analyze the text strictly to find a "Pivot". 
A Pivot is a point where the text changes its specific focus. Examples of Pivots:
- Moving from a general timeline to a specific scientist's experiment.
- Moving from macroscopic biology to molecular chemistry.
- Moving from the 19th century to the 20th century.

Rules:
- If there is ZERO Pivot and the text is a static description of a single concept, score it 10.0.
- If you detect EVEN A MINOR Pivot that shifts the focus, you must score it 3.0 or lower. Do not be generous.

Text:
{text}

Explain your reasoning in 2 sentences inside <think></think> tags. Conclude with: [SCORE: X.X]""",

    "3_EDITOR_IMPLACABLE": """You are a ruthless text segmentation algorithm. Your only job is to find reasons to cut paragraphs.

Evaluate the text's homogeneity.
- If the text is 100% homogeneous (e.g., all sentences describe the exact same physical structure, OR all sentences describe the exact same step in a process), score it 10.0.
- If the text transitions between sub-topics (e.g., from author A to author B, from historical context to physical measurement, or from one century to another), you MUST cut it. Score it 2.0.

Text:
{text}

Put your reasoning inside <think></think> tags. Output your final decision exactly as: [SCORE: X.X]"""
}

def test_parser(response: str) -> float:
    match = re.search(r'\[SCORE:\s*([0-9]*\.?[0-9]+)\]', response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    numbers = re.findall(r'\b(10(?:\.0)?|[1-9](?:\.\d)?)\b', clean)
    if numbers:
        return float(numbers[-1])
    return 5.0

def run_experiment():
    print("="*80)
    print(" ROUND 2: BUSCANDO UN CONTRASTE AGRESIVO EN CEREBRAS")
    print("="*80)
    
    scenarios = [
        ("COHESIVO PURO (Esperado: ~ 10.0)", TEXT_HIGHLY_COHESIVE),
        ("TRANSICIÓN CRÍTICA (Esperado: < 4.0)", TEXT_SUBTLE_BOUNDARY)
    ]
    
    for prompt_name, prompt_tmpl in PROMPTS.items():
        print(f"\n🔥 PROBANDO: {prompt_name}")
        print("-" * 60)
        
        for case_name, text_content in scenarios:
            formatted_prompt = prompt_tmpl.format(text=text_content)
            
            print(f"-> Escenario: {case_name}")
            try:
                response = _call_llm(formatted_prompt)
                score = test_parser(response)
                
                print(f"   [Resultado] Score: {score}/10")
                
                if "<think>" in response and "</think>" in response:
                    reasoning = response.split("<think>")[1].split("</think>")[0].strip()
                    print(f"   [Pensamiento]: {reasoning[:200]}...")
                else:
                    print(f"   [Respuesta]: {response.strip()[:150]}...")
                    
            except Exception as e:
                print(f"   [!] Error: {e}")
            
            time.sleep(3)
        print("-" * 60)

if __name__ == "__main__":
    run_experiment()