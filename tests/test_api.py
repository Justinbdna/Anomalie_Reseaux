"""Tests de l'API FastAPI (api.py)."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# api.py est à la racine du dépôt : on ajoute ce dossier au chemin
# de recherche de Python pour pouvoir faire "from api import app".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import app  # noqa: E402

# Client de test : il appelle l'API sans lancer de serveur.
client = TestClient(app)


def test_health_renvoie_200_et_status_ok():
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"
