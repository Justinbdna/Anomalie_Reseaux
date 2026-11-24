"""Tests de l'API FastAPI (api.py).

La fixture `client` vient de tests/conftest.py : c'est un TestClient
branché sur un petit modèle temporaire (pas besoin de lancer le serveur).
"""

import api

# Une ligne "allow" habituelle, valide (toutes les valeurs sont >= 0).
LIGNE_VALIDE = {
    "bytes_sent": 341,
    "bytes_received": 321,
    "packets": 38,
    "elapsed_time_sec": 50,
}

# Cas synthétique d'exfiltration (1re ligne de artifacts/cas_synthetiques.csv) :
# énormément d'octets envoyés et de paquets.
LIGNE_EXFILTRATION = {
    "bytes_sent": 9016292,
    "bytes_received": 1534,
    "packets": 29647,
    "elapsed_time_sec": 257,
}


def test_health_renvoie_200_et_status_ok(client):
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_predict_accepte_une_ligne_valide(client):
    reponse = client.post("/predict", json=LIGNE_VALIDE)

    assert reponse.status_code == 200
    resultat = reponse.json()
    assert isinstance(resultat["score"], float)
    assert isinstance(resultat["is_anomaly"], bool)


def test_predict_ligne_habituelle_pas_d_alerte(client):
    reponse = client.post("/predict", json=LIGNE_VALIDE)

    assert reponse.json()["is_anomaly"] is False


def test_predict_cas_synthetique_declenche_une_alerte(client):
    reponse = client.post("/predict", json=LIGNE_EXFILTRATION)

    assert reponse.status_code == 200
    assert reponse.json()["is_anomaly"] is True


def test_predict_sans_modele_renvoie_503(client, monkeypatch, tmp_path):
    # On fait pointer l'API vers un fichier qui n'existe pas.
    monkeypatch.setattr(api, "MODEL_PATH", str(tmp_path / "absent.joblib"))
    api.load_model.cache_clear()
    try:
        reponse = client.post("/predict", json=LIGNE_VALIDE)

        assert reponse.status_code == 503
        assert "train.py" in reponse.json()["detail"]
    finally:
        # On vide le cache pour que les autres tests rechargent le vrai modèle.
        monkeypatch.undo()
        api.load_model.cache_clear()


def test_predict_refuse_bytes_sent_negatif(client):
    ligne = {**LIGNE_VALIDE, "bytes_sent": -10}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422


def test_predict_refuse_bytes_received_negatif(client):
    ligne = {**LIGNE_VALIDE, "bytes_received": -1}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422
