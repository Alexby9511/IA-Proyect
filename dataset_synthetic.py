"""
dataset_synthetic.py
--------------------
Genera un dataset sintético con cambios de tema muy marcados
para verificar que los modelos de embeddings funcionan correctamente.

Estrategia: combinar fragmentos de artículos sobre temas completamente
distintos en un solo texto. Los cortes reales son exactamente donde
cambia el tema — debería ser fácil de detectar para los embeddings.

Uso:
    python dataset_synthetic.py
    → genera dataset_synthetic.json
"""

import json
import re
import time
import random
from pathlib import Path
import wikipediaapi

# ---------------------------------------------------------------------------
# Grupos de temas muy distintos entre sí
# Cada instancia sintética combina K artículos de grupos diferentes
# ---------------------------------------------------------------------------

TOPIC_GROUPS = {
    "ciencias_naturales": [
        "Fotosíntesis", "Mitocondria", "Célula", "ADN", "Evolución"
    ],
    "historia": [
        "Imperio romano", "Revolución francesa", "Segunda Guerra Mundial",
        "Edad Media", "Conquista de América"
    ],
    "astronomia": [
        "Agujero negro", "Sistema solar", "Via Láctea", "Estrella de neutrones", "Cometa"
    ],
    "tecnologia": [
        "Internet", "Inteligencia artificial", "Computadora", "Robot", "Satélite artificial"
    ],
    "arte_cultura": [
        "Pintura", "Música clásica", "Literatura", "Escultura", "Arquitectura"
    ],
    "geografia": [
        "Amazonas", "Cordillera de los Andes", "Desierto del Sahara", "Océano Pacífico", "Himalaya"
    ],
}

# Configuración de instancias sintéticas
INSTANCES_CONFIG = [
    # (K, grupos a combinar) — K = número de segmentos
    (3, ["ciencias_naturales", "historia", "astronomia"]),
    (3, ["tecnologia", "arte_cultura", "geografia"]),
    (3, ["historia", "tecnologia", "ciencias_naturales"]),
    (4, ["ciencias_naturales", "historia", "astronomia", "tecnologia"]),
    (4, ["arte_cultura", "geografia", "historia", "ciencias_naturales"]),
    (4, ["tecnologia", "astronomia", "arte_cultura", "geografia"]),
    (5, ["ciencias_naturales", "historia", "astronomia", "tecnologia", "arte_cultura"]),
    (5, ["historia", "geografia", "tecnologia", "ciencias_naturales", "astronomia"]),
    (5, ["arte_cultura", "historia", "geografia", "astronomia", "tecnologia"]),
    (6, ["ciencias_naturales", "historia", "astronomia", "tecnologia", "arte_cultura", "geografia"]),
    (6, ["historia", "tecnologia", "ciencias_naturales", "astronomia", "geografia", "arte_cultura"]),
    (3, ["astronomia", "arte_cultura", "geografia"]),
    (4, ["historia", "ciencias_naturales", "geografia", "tecnologia"]),
    (5, ["astronomia", "arte_cultura", "historia", "tecnologia", "geografia"]),
]

SENTENCES_PER_SEGMENT = 8   # oraciones por segmento
SKIP_SECTIONS = {
    "véase también", "referencias", "bibliografía", "enlaces externos",
    "notas", "fuentes", "lecturas adicionales", "galería",
    "galería de imágenes", "notas al pie", "notas y referencias",
}


def split_sentences(text: str) -> list[str]:
    """Divide texto en oraciones usando regex."""
    pattern = r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÜ])'
    parts = re.split(pattern, text.strip())
    return [s.strip() for s in parts if len(s.split()) >= 5]


def clean_text(text: str) -> str:
    text = re.sub(r'\[[\d]+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_sentences_from_article(wiki, title: str, n_sentences: int) -> list[str] | None:
    """
    Descarga un artículo y extrae exactamente n_sentences oraciones
    del cuerpo principal (evitando secciones de referencias etc.)
    """
    page = wiki.page(title)
    if not page.exists():
        return None

    all_sentences = []

    # Primero intentar el resumen/introducción
    if page.summary:
        text = clean_text(page.summary)
        all_sentences.extend(split_sentences(text))

    # Luego las secciones principales
    for section in page.sections:
        if len(all_sentences) >= n_sentences * 2:
            break
        if section.title.lower() in SKIP_SECTIONS:
            continue
        text = clean_text(section.text)
        if text:
            all_sentences.extend(split_sentences(text))

    if len(all_sentences) < n_sentences:
        return None

    # Tomar exactamente n_sentences del inicio
    return all_sentences[:n_sentences]


def build_synthetic_instance(
    wiki,
    instance_id: int,
    K: int,
    group_names: list[str],
    topic_groups: dict,
    used_articles: set,
) -> dict | None:
    """
    Construye una instancia sintética combinando K fragmentos
    de artículos sobre temas completamente distintos.
    """
    segments = []
    section_titles = []

    for group_name in group_names:
        candidates = topic_groups[group_name]
        # Elegir un artículo no usado de este grupo
        available = [a for a in candidates if a not in used_articles]
        if not available:
            available = candidates  # reusar si no hay disponibles

        article_title = random.choice(available)
        used_articles.add(article_title)

        print(f"    Descargando: {article_title} ({group_name})")
        sentences = fetch_sentences_from_article(wiki, article_title, SENTENCES_PER_SEGMENT)

        if sentences is None:
            print(f"    [!] No se pudo obtener oraciones de: {article_title}")
            return None

        segments.append({
            "article": article_title,
            "group": group_name,
            "sentences": sentences,
        })
        section_titles.append(f"{article_title} ({group_name})")
        time.sleep(0.3)

    # Construir instancia
    all_sentences = []
    cut_positions = []

    for seg in segments:
        cut_positions.append(len(all_sentences))
        all_sentences.extend(seg["sentences"])

    return {
        "id": f"synthetic_{instance_id:03d}",
        "title": " | ".join(seg["article"] for seg in segments),
        "difficulty": "synthetic_marcado",
        "K": K,
        "sentences": all_sentences,
        "ground_truth_cuts": cut_positions[1:],
        "section_titles": section_titles,
        "n_sentences": len(all_sentences),
        "segments_info": [
            {"article": seg["article"], "group": seg["group"]}
            for seg in segments
        ],
    }


def generate_synthetic_dataset() -> list[dict]:
    wiki = wikipediaapi.Wikipedia(
        language="es",
        user_agent="ProyectoIA-Segmentacion/1.0 (proyecto universitario)",
    )

    dataset = []
    used_articles: set = set()

    for i, (K, group_names) in enumerate(INSTANCES_CONFIG):
        print(f"\n[{i+1}/{len(INSTANCES_CONFIG)}] K={K} — {' + '.join(group_names)}")

        instance = build_synthetic_instance(
            wiki, i + 1, K, group_names, TOPIC_GROUPS, used_articles
        )

        if instance is None:
            print(f"  [!] Instancia {i+1} descartada")
            continue

        print(f"  ✓ {instance['n_sentences']} oraciones | cortes en: {instance['ground_truth_cuts']}")
        dataset.append(instance)

    return dataset


if __name__ == "__main__":
    random.seed(42)

    print("Generando dataset sintético con cambios marcados de tema...")
    print(f"Instancias a generar: {len(INSTANCES_CONFIG)}\n")

    dataset = generate_synthetic_dataset()

    output_path = "dataset_synthetic.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"✓ Dataset sintético guardado en: {output_path}")
    print(f"  {len(dataset)} instancias generadas")
    print(f"\nEstructura:")
    print(f"  Oraciones por segmento : {SENTENCES_PER_SEGMENT}")
    print(f"  K range                : {min(K for K,_ in INSTANCES_CONFIG)}-{max(K for K,_ in INSTANCES_CONFIG)}")
    print(f"\nEjemplo de instancia:")
    if dataset:
        ex = dataset[0]
        print(f"  Título  : {ex['title']}")
        print(f"  K       : {ex['K']}")
        print(f"  Cortes  : {ex['ground_truth_cuts']}")
        print(f"  Temas   : {[s['group'] for s in ex['segments_info']]}")
