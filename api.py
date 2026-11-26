"""API FastAPI du projet Anomalie Réseaux.

Pour lancer l'API en local :
    uvicorn api:app --reload
puis ouvrir http://localhost:8000/docs
"""

# os : lire les variables d'environnement. lru_cache : garder un résultat en mémoire.
# Path : manipuler des chemins de fichiers.
import os
from functools import lru_cache
from pathlib import Path

# joblib : relire le modèle sauvegardé. pandas : construire le tableau donné au modèle.
# FastAPI : le framework web. HTTPException : renvoyer une erreur HTTP (ex. 503).
# RequestValidationError : l'erreur levée quand les données reçues sont invalides (422).
# JSONResponse : construire nous-mêmes une réponse JSON. BaseModel / Field : décrire et valider les données.
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Chemin du modèle exporté par train.py.
# On peut le changer avec la variable d'environnement MODEL_PATH
# (utile pour les tests ou pour Docker).
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

# Création de l'application. Le titre s'affiche en haut de la page /docs.
app = FastAPI(title="Anomalie Réseaux API")

# Nom en français de chaque champ, utilisé dans les messages d'erreur.
LIBELLES = {
    "bytes_sent": "Octets envoyés",
    "bytes_received": "Octets reçus",
    "packets": "Nombre de paquets",
    "elapsed_time_sec": "Durée",
}

# Champs au féminin ("Durée ne peut pas être négative").
CHAMPS_FEMININS = {"elapsed_time_sec"}


def message_en_francais(erreur):
    """Transforme une erreur de Pydantic en {"field": ..., "message": ...}."""
    # erreur["loc"] ressemble à ("body", "bytes_sent") : le dernier élément
    # est le nom du champ. S'il n'y a que ("body",), c'est tout le corps
    # de la requête qui pose problème (vide ou JSON invalide).
    loc = erreur["loc"]
    champ = loc[-1] if len(loc) > 1 and isinstance(loc[-1], str) else "body"

    # Si ce n'est pas un de nos 4 champs, le problème vient du corps entier de la requête.
    if champ not in LIBELLES:
        return {
            "field": "body",
            "message": "Le corps de la requête doit être un JSON avec les 4 mesures.",
        }

    # Nom français du champ et type d'erreur donné par Pydantic.
    libelle = LIBELLES[champ]
    type_erreur = erreur["type"]

    # On choisit le message selon le type d'erreur :
    # "greater_than_equal" = valeur négative, "missing" = champ absent,
    # "float_parsing" / "float_type" = ce n'est pas un nombre.
    if type_erreur == "greater_than_equal":
        negatif = "négative" if champ in CHAMPS_FEMININS else "négatif"
        message = f"{libelle} ne peut pas être {negatif}."
    elif type_erreur == "missing":
        message = f"{libelle} est obligatoire."
    elif type_erreur in ("float_parsing", "float_type"):
        message = f"{libelle} doit être un nombre."
    else:
        message = f"{libelle} : valeur invalide."

    return {"field": champ, "message": message}


# Ce décorateur dit à FastAPI : "quand une erreur de validation arrive, appelle cette fonction".
@app.exception_handler(RequestValidationError)
def erreurs_de_validation(request: Request, exc: RequestValidationError):
    """Remplace les erreurs 422 de FastAPI (en anglais) par des messages en français."""
    # Un message français par erreur, puis réponse 422 au format {"detail": [...]}.
    details = [message_en_francais(erreur) for erreur in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


# Modèle Pydantic de l'ENTRÉE : FastAPI vérifie automatiquement chaque requête avec cette classe.
class LogLine(BaseModel):
    """Une ligne du journal de pare-feu : les 4 mesures utilisées par le modèle.

    ge=0 veut dire "plus grand ou égal à 0" : une valeur négative est
    refusée automatiquement par FastAPI avec une erreur HTTP 422.
    """

    # Les 4 champs : nombre décimal (float), obligatoire (...), et >= 0.
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


# Modèle Pydantic de la SORTIE : décrit la réponse de POST /predict dans /docs.
class Prediction(BaseModel):
    """Réponse de POST /predict (contrat commun du README)."""

    score: float = Field(..., description="Score d'anomalie : plus il est grand, plus la ligne est atypique")
    is_anomaly: bool = Field(..., description="True si le score atteint le seuil du modèle")


# lru_cache(maxsize=1) : le premier appel lit le fichier, les suivants réutilisent le résultat en mémoire.
@lru_cache(maxsize=1)
def load_model():
    """Charge le modèle une seule fois puis le garde en mémoire (cache).

    Le fichier contient un dictionnaire avec les clés
    "pipeline", "threshold" et "features".
    Renvoie None si le fichier n'existe pas encore.
    """
    # Si le fichier n'existe pas, on renvoie None (l'API ne plante pas).
    chemin = Path(MODEL_PATH)
    if not chemin.exists():
        return None
    return joblib.load(chemin)


# Route GET /health : sert à vérifier que l'API est vivante (utilisée aussi par Docker).
@app.get("/health")
def health():
    """Vérifie que l'API répond et indique si le modèle est chargé."""
    modele = load_model()
    return {"status": "ok", "model_loaded": modele is not None}


# Route POST /predict : reçoit une LogLine (déjà validée par FastAPI) et renvoie une Prediction.
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
    # Alerte si le score atteint le seuil fixé par train.py.
    is_anomaly = score >= modele["threshold"]

    return Prediction(score=score, is_anomaly=is_anomaly)
