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

## Sujet & jeu de données (P1 — Données)
- **Nom du jeu** : *Internet Firewall Data* — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/542/internet+firewall+data)
- **Fichier versionné** : `data/log2.csv` (**65 532 lignes** × **12 colonnes**)
- **DOI** : `10.24432/C5131M`
- **Licence & droits** : **CC BY 4.0** (Creative Commons Attribution 4.0 — réutilisation, partage et adaptation autorisés en citant la source)

### Description des 12 colonnes (`data/log2.csv`)
| # | Colonne brute | Rôle dans le projet | Nom contrat API |
| ---: | --- | --- | --- |
| 1 | `Source Port` | Non utilisée (port source) | — |
| 2 | `Destination Port` | Non utilisée (port destination) | — |
| 3 | `NAT Source Port` | Non utilisée (port NAT source) | — |
| 4 | `NAT Destination Port` | Non utilisée (port NAT destination) | — |
| 5 | `Action` | **Gardée à part** (filtre pour isoler `allow`, jamais en entrée du modèle) | — |
| 6 | `Bytes` | Non utilisée (redondante : `Bytes Sent + Bytes Received`) | — |
| 7 | `Bytes Sent` | **Mesure utilisée (1/4)** — octets envoyés | `bytes_sent` |
| 8 | `Bytes Received` | **Mesure utilisée (2/4)** — octets reçus | `bytes_received` |
| 9 | `Packets` | **Mesure utilisée (3/4)** — nombre total de paquets | `packets` |
| 10 | `Elapsed Time (sec)` | **Mesure utilisée (4/4)** — durée de session en secondes | `elapsed_time_sec` |
| 11 | `pkts_sent` | Non utilisée (sous-détail de `Packets`) | — |
| 12 | `pkts_received` | Non utilisée (sous-détail de `Packets`) | — |

### Premier contrôle : répartition par action (`Action`) & valeurs manquantes (D2)
Contrôle exécuté via `python explore.py` (`first_control()`) : **0 valeur manquante (`NaN`)** détectée sur l'ensemble des 12 colonnes (`65 532` lignes complètes). La classe **`allow`** (`37 640` lignes, soit `57,44 %`) constitue la référence de trafic habituel pour l'entraînement non supervisé.

| Action | Nombre de lignes | Part |
| --- | ---: | ---: |
| `allow` | **37 640** | 57,44 % |
| `deny` | **14 987** | 22,87 % |
| `drop` | **12 851** | 19,61 % |
| `reset-both` | **54** | 0,08 % |
| **Total** | **65 532** | **100,00 %** |

### Contrôle qualité & nettoyage
- **Valeurs manquantes (`NaN`)** : `0`
- **Valeurs numériques illisibles** : `0`
- **Valeurs négatives (`< 0`)** : `0` (cohérence vérifiée : `Bytes == Bytes Sent + Bytes Received` et `Packets == pkts_sent + pkts_received` sur 100 % des lignes)
- **Doublons** :
  - `8 362` doublons exacts sur les 12 colonnes (12,76 % du total : `6 945` en `deny`, `1 216` en `drop`, `201` en `allow`, `0` en `reset-both`).
  - Sur les 4 variables dans `allow` (`37 640` lignes) : `19 777` doublons (`52,54 %`, soit `17 863` profils uniques).
  - **Décision : doublons conservés.** Le fichier ne contient ni horodatage ni adresse IP : chaque ligne correspond à une session pare-feu distincte. Des trafics normaux répétitifs (notamment DNS sur le port 53) produisent naturellement les mêmes volumes ; les dédupliquer fausserait la distribution empirique apprise par le détecteur d'anomalies.

### Statistiques du trafic normal (`allow`, `n = 37 640`)
| Variable | Min | Q25 | Médiane | Q75 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Bytes Sent` | 60 | 94 | **126** | 1 586 | 9 293 | 64 722 | 948 477 220 |
| `Bytes Received` | 0 | 91 | **220** | 3 063 | 58 599 | 1 225 556 | 320 881 795 |
| `Packets` | 1 | 2 | **3** | 19 | 107 | 1 469 | 1 036 116 |
| `Elapsed Time (sec)` | 0 | 27 | **30** | 49 | 360 | 1 201 | 10 824 |

- **Visualisation (D4)** : `artifacts/trafic_allow.png` (générée par `python explore.py`).
  - **Ordres de grandeur** : la majorité du trafic `allow` correspond à de petits échanges rapides (médiane à `126` octets envoyés, `220` octets reçus, `3` paquets, `30 s` de durée, dominés par le DNS port 53 et le web HTTPS/HTTP).
  - **Distributions très étalées (queue lourde)** : les volumes s'étendent sur 6 à 9 ordres de grandeur (jusqu'à `948 Mo` envoyés, `321 Mo` reçus, `1,04 M` paquets et `3 h` de session), d'où la nécessité d'une échelle logarithmique (`log10(1 + x)`) pour visualiser la distribution et de la transformation `log1p` avant Isolation Forest.
  - **Valeurs extrêmes** : dès le `99e` percentile (`P99` : `64 722` octets envoyés, `1,23 Mo` reçus, `1 469` paquets, `1 201 s`), on entre dans des transferts massifs ou sessions longues qui représentent des candidats naturels aux anomalies de volume.

### Cas habituel de référence pour la démo (P3)
Vraie ligne `allow` la plus proche de la médiane des 4 mesures (index pandas `12871`, ligne CSV `12873`, trafic DNS sur le port destination `53`) :
```json
{
  "bytes_sent": 168,
  "bytes_received": 175,
  "packets": 3,
  "elapsed_time_sec": 30
}
```

## Contrat commun (pour travailler en parallèle)
- **Modèle exporté** : `artifacts/model.joblib`
- **API FastAPI** :
  - `GET /health`
  - `POST /predict` — entrée JSON : `bytes_sent`, `bytes_received`, `packets`, `elapsed_time_sec` (tous `>= 0`, rejet `HTTP 422` si négatif) → sortie JSON : `score` (float) et `is_anomaly` (bool)
- **Interface** : formulaire Streamlit relié à `POST /predict` + tests `pytest` + Docker Compose
