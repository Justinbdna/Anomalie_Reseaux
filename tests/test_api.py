"""Tests de l'API FastAPI (api.py).

La fixture `client` vient de tests/conftest.py : c'est un TestClient
branché sur un petit modèle temporaire (pas besoin de lancer le serveur).
"""

# Une ligne "allow" habituelle, valide (toutes les valeurs sont >= 0).
LIGNE_VALIDE = {
    "bytes_sent": 341,
    "bytes_received": 321,
    "packets": 38,
    "elapsed_time_sec": 50,
}


def test_health_renvoie_200_et_status_ok(client):
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_predict_accepte_une_ligne_valide(client):
    reponse = client.post("/predict", json=LIGNE_VALIDE)

    assert reponse.status_code == 200


def test_predict_refuse_bytes_sent_negatif(client):
    ligne = {**LIGNE_VALIDE, "bytes_sent": -10}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422


def test_predict_refuse_bytes_received_negatif(client):
    ligne = {**LIGNE_VALIDE, "bytes_received": -1}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422
