"""Interface Streamlit du projet Anomalie Réseaux.

Le formulaire envoie les 4 mesures d'une ligne de journal à l'API FastAPI
(POST /predict) et affiche le score et l'alerte.

Pour lancer en local (l'API doit tourner dans un autre terminal) :
    uvicorn api:app --reload
    streamlit run app.py
"""

# os : lire la variable d'environnement API_URL.
import os

# requests : envoyer des requêtes HTTP à l'API. streamlit : construire la page web.
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
    # st.session_state : la "mémoire" de la page. Chaque champ du formulaire y a une case à son nom.
    for champ, valeur in cas.items():
        st.session_state[champ] = float(valeur)


def afficher_resultat(reponse):
    """Affiche la réponse de l'API selon son code HTTP."""
    # 200 = tout va bien : on affiche le score, puis un bandeau rouge (alerte) ou vert (normal).
    if reponse.status_code == 200:
        resultat = reponse.json()
        st.metric("Score d'anomalie", f"{resultat['score']:.3f}")
        if resultat["is_anomaly"]:
            st.error("🚨 Alerte : trafic atypique")
        else:
            st.success("✅ Trafic habituel : pas d'alerte")

    # 422 = saisie invalide : on affiche chaque message, un par ligne.
    elif reponse.status_code == 422:
        # Messages en français renvoyés par l'API (issue #20).
        st.warning("Saisie à corriger :")
        for erreur in reponse.json()["detail"]:
            st.markdown(f"- {erreur['message']}")

    elif reponse.status_code == 503:
        # Le modèle n'est pas chargé côté API.
        st.error(reponse.json()["detail"])

    # Tout autre code : erreur générique.
    else:
        st.error(f"Erreur inattendue de l'API (code {reponse.status_code}).")


# ---------------------------------------------------------------- Page
# Streamlit relit tout le fichier de haut en bas à chaque clic : la page est redessinée à chaque fois.

# Titre de l'onglet du navigateur, titre de la page et phrase d'explication.
st.set_page_config(page_title="Anomalie Réseaux", page_icon="🛡️")
st.title("🛡️ Anomalie Réseaux")
st.write(
    "Saisissez les 4 mesures d'une ligne du journal de pare-feu. "
    "Le modèle (Isolation Forest) calcule un score : plus il est élevé, "
    "plus le trafic est atypique."
)

# Valeurs de départ du formulaire : le cas habituel.
# setdefault : n'écrit la valeur que si elle n'existe pas encore (on n'écrase pas une saisie).
for champ, valeur in CAS_HABITUEL.items():
    st.session_state.setdefault(champ, float(valeur))

# Boutons de pré-remplissage (issue #22).
# on_click s'exécute AVANT l'affichage du formulaire : c'est ce qui permet
# de changer les valeurs des champs.
# Deux colonnes côte à côte, un bouton dans chacune ; args = le cas à envoyer à pre_remplir().
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
# Une case numérique par champ ; key=champ relie la case à st.session_state[champ].
# envoye vaut True uniquement au moment où on clique sur "Analyser".
with st.form("ligne_de_journal"):
    for champ, libelle in CHAMPS.items():
        st.number_input(libelle, key=champ, step=1.0)
    envoye = st.form_submit_button("Analyser", type="primary")

# Appel de l'API et affichage du résultat (issue #23).
if envoye:
    # On rassemble les 4 valeurs saisies dans un dictionnaire (envoyé en JSON).
    ligne = {champ: st.session_state[champ] for champ in CHAMPS}
    try:
        # Appel de POST /predict ; timeout=5 : on n'attend pas plus de 5 secondes.
        reponse = requests.post(f"{API_URL}/predict", json=ligne, timeout=5)
    except requests.exceptions.RequestException:
        # L'API est éteinte ou injoignable : message clair au lieu d'un plantage.
        st.error(
            f"L'API ne répond pas ({API_URL}). "
            "Vérifiez qu'elle est lancée : uvicorn api:app --reload"
        )
    else:
        # L'API a répondu (quel que soit le code) : on affiche le résultat.
        afficher_resultat(reponse)
