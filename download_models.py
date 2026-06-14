"""
download_models.py
------------------
Descarga los modelos de embeddings y los guarda en ./models/.
Modelo A: paraphrase-multilingual-MiniLM-L12-v2
Modelo B: paraphrase-multilingual-mpnet-base-v2

Requisitos:
    pip install sentence-transformers
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path("./models")
MODEL_NAMES = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
]

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name in MODEL_NAMES:
        print(f"Descargando {name} ...")
        model = SentenceTransformer(name)
        save_path = MODELS_DIR / name
        model.save(str(save_path))
        print(f"  Guardado en {save_path}")

if __name__ == "__main__":
    main()