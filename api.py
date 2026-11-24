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
from pydantic import BaseModel, Field

# Chemin du modèle exporté par train.py.
# On peut le changer avec la variable d'environnement MODEL_PATH
# (utile pour les tests ou pour Docker).
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

app = FastAPI(title="Anomalie Réseaux API")


class LogLine(BaseModel):
    """Une ligne du journal de pare-feu : les 4 mesures utilisées par le modèle.

    ge=0 veut dire "plus grand ou égal à 0" : une valeur négative est
    refusée automatiquement par FastAPI avec une erreur HTTP 422.
    """

    bytes_sent: float = Field(..., ge=0, description="Octets envoyés")
    bytes_received: float = Field(..., ge=0, description="Octets reçus")
    packets: float = Field(..., ge=0, description="Nombre de paquets")
    elapsed_time_sec: float = Field(..., ge=0, description="Durée en secondes")

    # Exemple affiché (et pré-rempli) dans /docs : une ligne "allow" habituelle.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bytes_sent": 341,
                    "bytes_received": 321,
                    "packets": 38,
                    "elapsed_time_sec": 50,
                }
            ]
        }
    }


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


@app.post("/predict")
def predict(ligne: LogLine):
    """Reçoit une ligne de journal validée.

    Pour l'instant (issue #18), on renvoie simplement les valeurs reçues.
    Le calcul de "score" et "is_anomaly" avec le modèle arrive dans l'issue #19.
    """
    return {"received": ligne.model_dump()}
