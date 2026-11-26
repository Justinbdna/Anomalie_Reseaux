"""Interface Streamlit du projet Anomalie Réseaux.

Le formulaire envoie les 4 mesures d'une ligne de journal à l'API FastAPI
(POST /predict) et affiche le score et l'alerte.

Pour lancer en local (l'API doit tourner dans un autre terminal) :
    uvicorn api:app --reload
    streamlit run app.py
"""

import os

import requests
import streamlit as st

# Adresse de l'API : localhost en local, http://api:8000 dans Docker Compose.
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Nom des 4 champs (comme dans l'API) et leur libellé dans le formulaire.
CHAMPS = {
    "bytes_sent": "Octets envoyés",
    "bytes_received": "Octets reçus",
    "packets": "Nombre de paquets",
    "elapsed_time_sec": "Durée (secondes)",
}

# Cas habituel : vraie ligne "allow" choisie par Aymen (issue #6, README).
CAS_HABITUEL = {
    "bytes_sent": 168,
    "bytes_received": 175,
    "packets": 3,
    "elapsed_time_sec": 30,
}

# Cas synthétique : exfiltration (1re ligne de artifacts/cas_synthetiques.csv).
CAS_SYNTHETIQUE = {
    "bytes_sent": 9016292,
    "bytes_received": 1534,
    "packets": 29647,
    "elapsed_time_sec": 257,
}


def pre_remplir(cas):
    """Met les valeurs d'un cas dans le formulaire (via st.session_state)."""
    for champ, valeur in cas.items():
        st.session_state[champ] = float(valeur)


def afficher_resultat(reponse):
    """Affiche la réponse de l'API selon son code HTTP."""
    if reponse.status_code == 200:
        resultat = reponse.json()
        st.metric("Score d'anomalie", f"{resultat['score']:.3f}")
        if resultat["is_anomaly"]:
            st.error("🚨 Alerte : trafic atypique")
        else:
            st.success("✅ Trafic habituel : pas d'alerte")

    elif reponse.status_code == 422:
        # Messages en français renvoyés par l'API (issue #20).
        st.warning("Saisie à corriger :")
        for erreur in reponse.json()["detail"]:
            st.markdown(f"- {erreur['message']}")

    elif reponse.status_code == 503:
        # Le modèle n'est pas chargé côté API.
        st.error(reponse.json()["detail"])

    else:
        st.error(f"Erreur inattendue de l'API (code {reponse.status_code}).")


# ---------------------------------------------------------------- Page

st.set_page_config(page_title="Anomalie Réseaux", page_icon="🛡️")
st.title("🛡️ Anomalie Réseaux")
st.write(
    "Saisissez les 4 mesures d'une ligne du journal de pare-feu. "
    "Le modèle (Isolation Forest) calcule un score : plus il est élevé, "
    "plus le trafic est atypique."
)

# Valeurs de départ du formulaire : le cas habituel.
for champ, valeur in CAS_HABITUEL.items():
    st.session_state.setdefault(champ, float(valeur))

# Boutons de pré-remplissage (issue #22).
# on_click s'exécute AVANT l'affichage du formulaire : c'est ce qui permet
# de changer les valeurs des champs.
col1, col2 = st.columns(2)
col1.button(
    "Cas habituel", on_click=pre_remplir, args=(CAS_HABITUEL,), use_container_width=True
)
col2.button(
    "Cas synthétique (exfiltration)",
    on_click=pre_remplir,
    args=(CAS_SYNTHETIQUE,),
    use_container_width=True,
)

# Formulaire (issue #21). Pas de min_value=0 : c'est l'API qui contrôle
# les valeurs et renvoie un message clair si une valeur est négative.
with st.form("ligne_de_journal"):
    for champ, libelle in CHAMPS.items():
        st.number_input(libelle, key=champ, step=1.0)
    envoye = st.form_submit_button("Analyser", type="primary")

# Appel de l'API et affichage du résultat (issue #23).
if envoye:
    ligne = {champ: st.session_state[champ] for champ in CHAMPS}
    try:
        reponse = requests.post(f"{API_URL}/predict", json=ligne, timeout=5)
    except requests.exceptions.RequestException:
        st.error(
            f"L'API ne répond pas ({API_URL}). "
            "Vérifiez qu'elle est lancée : uvicorn api:app --reload"
        )
    else:
        afficher_resultat(reponse)
