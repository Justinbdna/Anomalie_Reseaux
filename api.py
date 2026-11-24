"""API FastAPI du projet Anomalie Réseaux.

Pour lancer l'API en local :
    uvicorn api:app --reload
puis ouvrir http://localhost:8000/docs
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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


class Prediction(BaseModel):
    """Réponse de POST /predict (contrat commun du README)."""

    score: float = Field(..., description="Score d'anomalie : plus il est grand, plus la ligne est atypique")
    is_anomaly: bool = Field(..., description="True si le score atteint le seuil du modèle")


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


@app.post("/predict", response_model=Prediction)
def predict(ligne: LogLine):
    """Calcule le score d'anomalie d'une ligne de journal et dit si c'est une alerte."""
    modele = load_model()

    # Pas de modèle : on répond 503 ("service indisponible") avec un message clair.
    if modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle introuvable : lancez d'abord python train.py",
        )

    # Le pipeline a été entraîné sur un DataFrame : on lui donne une ligne
    # avec les mêmes colonnes, dans le même ordre.
    X = pd.DataFrame([ligne.model_dump()])[modele["features"]]

    # Même calcul que train.py : score_samples() est plus bas quand la ligne
    # est atypique, donc on inverse le signe (plus grand = plus atypique).
    score = float(-modele["pipeline"].score_samples(X)[0])
    is_anomaly = score >= modele["threshold"]

    return Prediction(score=score, is_anomaly=is_anomaly)
