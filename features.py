"""Chargement, validation et préparation des variables du jeu de données pare-feu.

Contrat commun du projet Anomalie Réseaux :
- Source : data/log2.csv (UCI Internet Firewall Data)
- Mesures retenues : Bytes Sent, Bytes Received, Packets, Elapsed Time (sec)
- Cible d'entraînement non supervisé : flux normaux (Action == "allow")
- Doublons : conservés volontairement (chaque ligne représente une session
  réseau distincte sans horodatage ni IP ; les supprimer fausserait la
  fréquence réelle du trafic courant comme les requêtes DNS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_PATH = Path("data/log2.csv")
ACTION_COLUMN = "Action"
ALLOW_ACTION = "allow"

FEATURE_COLUMNS: list[str] = [
    "Bytes Sent",
    "Bytes Received",
    "Packets",
    "Elapsed Time (sec)",
]

API_TO_FEATURE: dict[str, str] = {
    "bytes_sent": "Bytes Sent",
    "bytes_received": "Bytes Received",
    "packets": "Packets",
    "elapsed_time_sec": "Elapsed Time (sec)",
}

FEATURE_TO_API: dict[str, str] = {v: k for k, v in API_TO_FEATURE.items()}


def load_raw_logs(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le fichier CSV brut des journaux de pare-feu."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")
    return pd.read_csv(csv_path)


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Valide et nettoie les 4 colonnes numériques de mesures.

    - Convertit les 4 mesures en numérique (les valeurs illisibles deviennent NaN).
    - Supprime les lignes ayant des valeurs manquantes/illisibles ou négatives.
    - Conserve les doublons pour préserver la distribution empirique du trafic.
    """
    missing_cols = [c for c in [*FEATURE_COLUMNS, ACTION_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le jeu de données : {missing_cols}")

    cleaned = df.copy()
    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    valid_mask = cleaned[FEATURE_COLUMNS].notna().all(axis=1)
    non_negative_mask = (cleaned[FEATURE_COLUMNS] >= 0).all(axis=1)
    return cleaned.loc[valid_mask & non_negative_mask].copy()


def load_allow_logs(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge, nettoie et isole les lignes de trafic autorisé (Action == 'allow')."""
    df = clean_logs(load_raw_logs(path))
    return df.loc[df[ACTION_COLUMN] == ALLOW_ACTION].copy()


def load_allow_features(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Retourne uniquement la matrice des 4 variables numériques pour Action == 'allow'."""
    allow_df = load_allow_logs(path)
    return allow_df[FEATURE_COLUMNS].copy()


def from_api_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convertit un payload JSON de l'API en DataFrame (1 ligne, 4 colonnes)."""
    row: dict[str, float] = {}
    for api_key, feature_col in API_TO_FEATURE.items():
        if api_key not in payload:
            raise ValueError(f"Champ requis manquant : {api_key}")
        value = float(payload[api_key])
        if np.isnan(value) or value < 0:
            raise ValueError(f"La valeur de '{api_key}' doit être >= 0 (reçu : {value})")
        row[feature_col] = value
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def find_typical_allow_row(allow_df: pd.DataFrame) -> pd.Series:
    """Sélectionne la vraie ligne 'allow' la plus proche de la médiane (en log1p/IQR)."""
    x_log = np.log1p(allow_df[FEATURE_COLUMNS].astype(float))
    med_log = x_log.median()
    iqr_log = (x_log.quantile(0.75) - x_log.quantile(0.25)).replace(0, 1.0)
    dist = (((x_log - med_log) / iqr_log) ** 2).sum(axis=1)
    best_idx = dist.idxmin()
    return allow_df.loc[best_idx]
