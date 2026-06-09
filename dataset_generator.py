"""
dataset_generator.py
--------------------
Descarga artículos de Wikipedia en español, extrae sus secciones
y construye instancias de prueba con ground truth para el proyecto
de segmentación óptima de contenido.

Requisitos:
    pip install wikipedia-api

Uso:
    python dataset_generator.py
    → genera dataset.json en el directorio actual
"""

import json
import re
import time
import wikipediaapi

# ---------------------------------------------------------------------------
# Artículos seleccionados por nivel de dificultad
# Dificultad = qué tan distintas son las secciones entre sí temáticamente
# ---------------------------------------------------------------------------

ARTICLES = {
    "facil": [
        "Fotosíntesis",
        "Segunda Guerra Mundial",
        "Sistema solar",
        "Revolución industrial",
        "Dinosaurio",
        "Internet",
        "Vacuna",
        "Volcán",
    ],
    "medio": [
        "Inteligencia artificial",
        "Cambio climático",
        "Evolución",
        "Democracia",
        "Energía solar",
        "Neurona",
        "Antibiótico",
        "Globalización",
        "Genoma",
        "Economía",
        "Filosofía",
        "Psicología",
    ],
    "dificil": [
        "Relatividad general",
        "Mecánica cuántica",
        "Termodinámica",
        "Lingüística",
        "Epistemología",
        "Fenomenología",
        "Dialéctica",
        "Hermenéutica",
    ],
}

# Secciones de Wikipedia que no aportan contenido útil
SKIP_SECTIONS = {
    "véase también", "referencias", "bibliografía", "enlaces externos",
    "notas", "fuentes", "lecturas adicionales", "galería",
    "galería de imágenes", "notas al pie", "notas y referencias",
}

# Parámetros de filtrado
MIN_SENTENCES_PER_SECTION = 3
MAX_SENTENCES_PER_SECTION = 30
MIN_SECTIONS = 3
MAX_SECTIONS = 7


def split_sentences(text: str) -> list[str]:
    """
    Divide texto en oraciones usando regex.
    Sin dependencias externas, funciona bien para español.
    """
    # Cortar en . ! ? seguidos de espacio y letra mayúscula
    # Excluir abreviaturas comunes precedidas por punto
    pattern = r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÜ])'
    parts = re.split(pattern, text.strip())
    # Filtrar fragmentos muy cortos (artefactos, títulos residuales)
    return [s.strip() for s in parts if len(s.split()) >= 5]


def clean_text(text: str) -> str:
    """Limpia artefactos típicos del texto de Wikipedia."""
    text = re.sub(r'\[[\d]+\]', '', text)   # referencias [1], [2]
    text = re.sub(r'\s+', ' ', text)         # espacios múltiples
    return text.strip()


def fetch_article(wiki, title: str) -> dict | None:
    """
    Descarga un artículo y extrae secciones válidas.
    Devuelve un dict con la estructura del artículo o None si falla.
    """
    page = wiki.page(title)
    if not page.exists():
        print(f"  [!] No encontrado: {title}")
        return None

    sections_data = []

    def process_section(section):
        if len(sections_data) >= MAX_SECTIONS:
            return
        if section.title.lower() in SKIP_SECTIONS:
            return

        text = clean_text(section.text)
        if text:
            sentences = split_sentences(text)
            if len(sentences) >= MIN_SENTENCES_PER_SECTION:
                sentences = sentences[:MAX_SENTENCES_PER_SECTION]
                sections_data.append({
                    "title": section.title,
                    "sentences": sentences,
                })

        for child in section.sections:
            if len(sections_data) >= MAX_SECTIONS:
                break
            process_section(child)

    for section in page.sections:
        if len(sections_data) >= MAX_SECTIONS:
            break
        process_section(section)

    if len(sections_data) < MIN_SECTIONS:
        print(f"  [!] Pocas secciones válidas en '{title}': {len(sections_data)}, descartado")
        return None

    return {
        "title": title,
        "sections": sections_data[:MAX_SECTIONS],
    }


def build_instance(article_data: dict, difficulty: str) -> dict:
    """
    Construye la instancia de prueba a partir de los datos del artículo.

    Estructura de salida:
    {
        "id": str,
        "title": str,
        "difficulty": str,
        "K": int,                    # número de segmentos (= número de secciones)
        "sentences": [...],          # todas las oraciones concatenadas
        "ground_truth_cuts": [...],  # índice donde empieza cada nueva sección (sin el 0)
        "section_titles": [...],     # títulos de sección (referencia / ground truth)
        "n_sentences": int
    }
    """
    all_sentences = []
    cut_positions = []
    section_titles = []

    for section in article_data["sections"]:
        cut_positions.append(len(all_sentences))
        all_sentences.extend(section["sentences"])
        section_titles.append(section["title"])

    return {
        "id": re.sub(r'\s+', '_', article_data["title"].lower()),
        "title": article_data["title"],
        "difficulty": difficulty,
        "K": len(article_data["sections"]),
        "sentences": all_sentences,
        "ground_truth_cuts": cut_positions[1:],  # excluir posición 0 (trivial)
        "section_titles": section_titles,
        "n_sentences": len(all_sentences),
    }


def generate_dataset() -> list[dict]:
    """Función principal: descarga todos los artículos y construye el dataset."""
    wiki = wikipediaapi.Wikipedia(
        language="es",
        user_agent="ProyectoIA-Segmentacion/1.0 (proyecto universitario)",
    )

    dataset = []
    total = sum(len(v) for v in ARTICLES.values())
    processed = 0

    for difficulty, titles in ARTICLES.items():
        print(f"\n{'='*50}")
        print(f"Dificultad: {difficulty.upper()}")
        print(f"{'='*50}")

        for title in titles:
            processed += 1
            print(f"\n[{processed}/{total}] {title}")

            article_data = fetch_article(wiki, title)
            if article_data is None:
                continue

            instance = build_instance(article_data, difficulty)

            if instance["n_sentences"] < 10:
                print(f"  [!] Muy corto ({instance['n_sentences']} oraciones), descartado")
                continue

            print(f"  ✓ K={instance['K']} secciones | {instance['n_sentences']} oraciones")
            print(f"    Cortes ground truth: {instance['ground_truth_cuts']}")
            dataset.append(instance)

            time.sleep(0.5)

    return dataset


def print_summary(dataset: list[dict]) -> None:
    print(f"\n{'='*50}")
    print("RESUMEN DEL DATASET")
    print(f"{'='*50}")
    print(f"Total instancias: {len(dataset)}")

    for diff in ["facil", "medio", "dificil"]:
        subset = [d for d in dataset if d["difficulty"] == diff]
        if not subset:
            continue
        avg_s = sum(d["n_sentences"] for d in subset) / len(subset)
        avg_k = sum(d["K"] for d in subset) / len(subset)
        print(f"\n  {diff.upper()} — {len(subset)} instancias")
        print(f"    Oraciones promedio : {avg_s:.1f}")
        print(f"    K promedio         : {avg_k:.1f}")


if __name__ == "__main__":
    print("Generando dataset de Wikipedia en español...")
    print("Requisito: pip install wikipedia-api\n")

    dataset = generate_dataset()
    print_summary(dataset)

    output_path = "dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Dataset guardado en: {output_path}")
    print(f"  {len(dataset)} instancias listas")