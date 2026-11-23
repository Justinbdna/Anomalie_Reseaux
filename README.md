# Anomalie Réseaux

Mini-hackathon Data & IA (HETIC) — Détection d'anomalies dans des journaux de pare-feu, en local.

## Équipe & branches Git

| Branche | Rôle |
| --- | --- |
| `main` | Version stable (soutenance) |
| `staging` | Intégration et tests avant fusion dans `main` |
| `nassima` | Développement — Nassima |
| `quentin` | Développement — Quentin |
| `aymen` | Développement — Aymen |
| `justin` | Développement — Justin |

## Sujet & jeu de données
- **Source** : [Internet Firewall Data · UCI](https://archive.ics.uci.edu/dataset/542/internet+firewall+data) (`log2.csv`)
- **Mesures** : `Bytes Sent`, `Bytes Received`, `Packets`, `Elapsed Time (sec)` (`Action` gardé à part pour isoler `allow`)
- **API prévue** : `GET /health` et `POST /predict` (score + alerte, rejet `HTTP 422` si octets négatifs)
- **Interface** : formulaire Streamlit relié à l'API + tests `pytest` + Docker Compose
