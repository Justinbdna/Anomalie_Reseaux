"""API FastAPI du projet Anomalie Réseaux.

Pour lancer l'API en local :
    uvicorn api:app --reload
puis ouvrir http://localhost:8000/docs
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI

# Chemin du modèle exporté par train.py.
# On peut le changer avec la variable d'environnement MODEL_PATH
# (utile pour les tests ou pour Docker).
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

app = FastAPI(title="Anomalie Réseaux API")


@lru_cache(maxsize=1)
def load_model():
    """Charge le modèle une seule fois puis le garde en mémoire (cache).

    Le fichier contient un dictionnaire avec les clés
    "pipeline", "threshold" et "features".
    Renvoie None si le fichier n'existe pas encore.
    """
    chemin = Path(MODEL_PATH)
    if not chemin.exists():
        return None
    return joblib.load(chemin)


@app.get("/health")
def health():
    """Vérifie que l'API répond et indique si le modèle est chargé."""
    modele = load_model()
    return {"status": "ok", "model_loaded": modele is not None}
