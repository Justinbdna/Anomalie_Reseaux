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

## Flux de travail Git

```
branche perso  ──PR──►  staging  ──PR──►  main
(aymen, justin,        (intégration       (version stable,
 nassima, quentin)      et tests)          soutenance)
```

1. **Avant de commencer** : `git checkout <ma-branche>` puis `git pull origin staging` pour partir de la dernière version intégrée.
2. **Je travaille** sur ma branche perso, un commit par issue (ex. `feat: ... (#29)`).
3. **PR vers `staging`** : la description contient `Closes #n` pour fermer l'issue automatiquement au merge.
4. **Quand `staging` est vert** (tests `pytest` OK), PR de `staging` vers `main`.
5. Personne ne pousse directement sur `staging` ni `main`.

**Suivi** : les issues sont suivies dans le board GitHub Projects (colonnes *To do*, *In progress*, *Review*, *Done*).

## Sujet & jeu de données
- **Source** : [Internet Firewall Data · UCI](https://archive.ics.uci.edu/dataset/542/internet+firewall+data) (`log2.csv`)
- **Mesures** : `Bytes Sent`, `Bytes Received`, `Packets`, `Elapsed Time (sec)` (`Action` gardé à part pour isoler `allow`)

## Contrat commun (pour travailler en parallèle)
- **Modèle exporté** : `artifacts/model.joblib`
- **API FastAPI** :
  - `GET /health`
  - `POST /predict` — entrée JSON : `bytes_sent`, `bytes_received`, `packets`, `elapsed_time_sec` (tous `>= 0`, rejet `HTTP 422` si négatif) → sortie JSON : `score` (float) et `is_anomaly` (bool)
- **Interface** : formulaire Streamlit relié à `POST /predict` + tests `pytest` + Docker Compose
