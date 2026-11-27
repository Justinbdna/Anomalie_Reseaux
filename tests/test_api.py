"""Tests de l'API FastAPI (api.py).

La fixture `client` vient de tests/conftest.py : c'est un TestClient
branché sur un petit modèle temporaire (pas besoin de lancer le serveur).
"""

# On importe le module api pour pouvoir modifier MODEL_PATH dans le test "sans modèle".
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

# Chaque test suit le même schéma : 1) préparer la requête, 2) appeler l'API, 3) vérifier (assert).
# Si un "assert" est faux, le test échoue et pytest affiche lequel.


# ---------------------------------------------------------------- GET /health

def test_health_renvoie_200_et_status_ok(client):
    # L'API doit répondre 200 et {"status": "ok", ...}.
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


# ---------------------------------------------------------------- POST /predict : cas valides

def test_predict_accepte_une_ligne_valide(client):
    # Une ligne valide renvoie 200, avec un score (nombre décimal) et is_anomaly (vrai/faux).
    reponse = client.post("/predict", json=LIGNE_VALIDE)

    assert reponse.status_code == 200
    resultat = reponse.json()
    assert isinstance(resultat["score"], float)
    assert isinstance(resultat["is_anomaly"], bool)


def test_predict_ligne_habituelle_pas_d_alerte(client):
    # Une ligne de trafic normal ne doit PAS déclencher d'alerte.
    reponse = client.post("/predict", json=LIGNE_VALIDE)

    assert reponse.json()["is_anomaly"] is False


def test_predict_cas_synthetique_declenche_une_alerte(client):
    # Le cas d'exfiltration DOIT déclencher une alerte.
    reponse = client.post("/predict", json=LIGNE_EXFILTRATION)

    assert reponse.status_code == 200
    assert reponse.json()["is_anomaly"] is True


def test_predict_sans_modele_renvoie_503(client, monkeypatch, tmp_path):
    # On fait pointer l'API vers un fichier qui n'existe pas.
    # monkeypatch : remplace temporairement une valeur le temps du test.
    monkeypatch.setattr(api, "MODEL_PATH", str(tmp_path / "absent.joblib"))
    api.load_model.cache_clear()
    try:
        # Sans modèle, l'API doit répondre 503 avec un message qui parle de train.py.
        reponse = client.post("/predict", json=LIGNE_VALIDE)

        assert reponse.status_code == 503
        assert "train.py" in reponse.json()["detail"]
    finally:
        # On vide le cache pour que les autres tests rechargent le vrai modèle.
        monkeypatch.undo()
        api.load_model.cache_clear()


# ---------------------------------------------------------------- POST /predict : valeurs négatives (422)
# {**LIGNE_VALIDE, "champ": valeur} : copie de la ligne valide en changeant une seule valeur.

def test_predict_refuse_bytes_sent_negatif(client):
    ligne = {**LIGNE_VALIDE, "bytes_sent": -10}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422


def test_predict_refuse_bytes_received_negatif(client):
    ligne = {**LIGNE_VALIDE, "bytes_received": -1}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422


def test_predict_refuse_packets_negatif(client):
    ligne = {**LIGNE_VALIDE, "packets": -3}

    reponse = client.post("/predict", json=ligne)

    # On vérifie aussi le message exact en français.
    assert reponse.status_code == 422
    assert reponse.json()["detail"] == [
        {"field": "packets", "message": "Nombre de paquets ne peut pas être négatif."}
    ]


def test_predict_refuse_duree_negative(client):
    ligne = {**LIGNE_VALIDE, "elapsed_time_sec": -5}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422
    # "Durée" est au féminin : le message dit "négative".
    assert reponse.json()["detail"] == [
        {"field": "elapsed_time_sec", "message": "Durée ne peut pas être négative."}
    ]


# ---------------------------------------------------------------- Messages d'erreur en français (issue #20)

def test_erreur_octets_negatifs_message_en_francais(client):
    # Critère d'acceptation de l'issue #20 : message exact pour bytes_sent = -1.
    ligne = {**LIGNE_VALIDE, "bytes_sent": -1}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422
    assert reponse.json()["detail"] == [
        {"field": "bytes_sent", "message": "Octets envoyés ne peut pas être négatif."}
    ]


def test_erreur_champ_manquant(client):
    # On copie la ligne valide puis on supprime le champ "packets".
    ligne = {**LIGNE_VALIDE}
    del ligne["packets"]

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422
    assert reponse.json()["detail"] == [
        {"field": "packets", "message": "Nombre de paquets est obligatoire."}
    ]


def test_erreur_texte_au_lieu_d_un_nombre(client):
    # Du texte à la place d'un nombre.
    ligne = {**LIGNE_VALIDE, "bytes_received": "beaucoup"}

    reponse = client.post("/predict", json=ligne)

    assert reponse.status_code == 422
    assert reponse.json()["detail"] == [
        {"field": "bytes_received", "message": "Octets reçus doit être un nombre."}
    ]


def test_erreur_corps_vide(client):
    # Requête sans aucun contenu : l'erreur porte sur "body" (tout le corps de la requête).
    reponse = client.post("/predict")

    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["field"] == "body"
