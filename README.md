# Anomalie Réseaux

Mini-hackathon Data & IA (HETIC) — Détection d'anomalies dans des journaux de pare-feu, en local.

## Équipe, rôles & branches Git

| Membre | Branche | Rôle | Fichiers principaux |
| --- | --- | --- | --- |
| **Aymen** | `aymen` | Données & Exploration | `data/log2.csv`, `features.py`, `explore.py`, `artifacts/*.png` |
| **Justin** | `justin` | Modèle IA | `train.py`, `artifacts/model.joblib` |
| **Nassima** | `nassima` | Application (API + UI) | `api.py` (FastAPI), `app.py` (Streamlit) |
| **Quentin** | `quentin` | DevOps, Tests & Intégration | `tests/`, `Dockerfile`, `docker-compose.yml` |

- **`staging`** : branche d'intégration et de tests avant fusion dans `main`.
- **`main`** : version stable prête pour la soutenance.

## Sujet & jeu de données
- **Source** : [Internet Firewall Data · UCI](https://archive.ics.uci.edu/dataset/542/internet+firewall+data) (`log2.csv`)
- **Mesures** : `Bytes Sent`, `Bytes Received`, `Packets`, `Elapsed Time (sec)` (`Action` gardé à part pour isoler `allow`)

## Contrat commun (pour travailler en parallèle)
- **Modèle exporté** : `artifacts/model.joblib`
- **API FastAPI** :
  - `GET /health`
  - `POST /predict` — entrée JSON : `bytes_sent`, `bytes_received`, `packets`, `elapsed_time_sec` (tous `>= 0`, rejet `HTTP 422` si négatif) → sortie JSON : `score` (float) et `is_anomaly` (bool)
- **Interface** : formulaire Streamlit relié à `POST /predict` + tests `pytest` + Docker Compose

## Modèle IA — Cas synthétiques atypiques (`artifacts/cas_synthetiques.csv`)
Le jeu `log2.csv` ne contenant pas d'étiquette « attaque », 200 lignes synthétiques (50 par type, graine fixe `42`) simulent 4 comportements réseau atypiques :
- **`exfiltration`** : volume d'octets envoyés massif pour peu de données reçues, typique d'une fuite de données vers l'extérieur.
- **`flood`** : rafale de centaines à milliers de paquets quasi vides concentrés en `0 s`, caractéristique d'une saturation ou d'un scan agressif.
- **`connexion_fantome`** : session ouverte très longtemps (1 h à 4 h) avec un seul paquet échangé, typique d'une connexion maintenue anormalement en vie.
- **`telechargement_massif`** : volume d'octets reçus anormalement élevé face à un trafic sortant minime, simulant un rapatriement massif non habituel.
