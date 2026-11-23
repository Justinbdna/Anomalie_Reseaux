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
