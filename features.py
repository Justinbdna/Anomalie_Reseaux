"""Chargement, validation et nettoyage des variables du jeu de données pare-feu (D3).

Contrat commun du projet Anomalie Réseaux :
- Source : data/log2.csv (UCI Internet Firewall Data, DOI 10.24432/C5131M, CC BY 4.0)
- 4 mesures renommées selon le contrat API :
    Bytes Sent         -> bytes_sent
    Bytes Received     -> bytes_received
    Packets            -> packets
    Elapsed Time (sec) -> elapsed_time_sec
- Colonne Action conservée à part (jamais fournie en entrée du modèle) pour isoler
  le trafic normal ('allow').
- Décision sur les doublons : CONSERVÉS volontairement.
  Le fichier ne contient ni horodatage ni adresse IP : chaque ligne correspond à
  une session pare-feu distincte. Les trafics courants répétitifs (ex. DNS port 53)
  produisent naturellement les mêmes volumes ; les dédupliquer fausserait la
  distribution empirique apprise par le détecteur d'anomalies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_PATH = Path("data/log2.csv")
ACTION_COLUMN = "Action"
ALLOW_ACTION = "allow"
RANDOM_STATE = 42

COLUMN_MAPPING: dict[str, str] = {
    "Bytes Sent": "bytes_sent",
    "Bytes Received": "bytes_received",
    "Packets": "packets",
    "Elapsed Time (sec)": "elapsed_time_sec",
}

FEATURES: list[str] = [
    "bytes_sent",
    "bytes_received",
    "packets",
    "elapsed_time_sec",
]

# Alias conservés pour compatibilité avec les scripts d'exploration
FEATURE_COLUMNS: list[str] = list(COLUMN_MAPPING.keys())
API_TO_FEATURE: dict[str, str] = {v: k for k, v in COLUMN_MAPPING.items()}
FEATURE_TO_API: dict[str, str] = dict(COLUMN_MAPPING)


def load_raw_logs(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le fichier CSV brut des journaux de pare-feu sans transformation."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")
    return pd.read_csv(csv_path)


def load_and_clean(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV, renomme les 4 mesures au format API et retire les lignes impossibles.

    Étapes :
    1. Renomme Bytes Sent, Bytes Received, Packets, Elapsed Time (sec) en
       bytes_sent, bytes_received, packets, elapsed_time_sec.
    2. Convertit les 4 mesures en numérique (les valeurs illisibles deviennent NaN).
    3. Retire toute ligne ayant une valeur manquante/illisible ou négative (< 0).
    4. Normalise Action en minuscules sans espaces superflus.
    5. Conserve les doublons (sessions distinctes sans horodatage ni IP).

    Retourne un DataFrame contenant exactement les 4 mesures + 'Action'.
    """
    raw_df = load_raw_logs(path)
    expected_cols = [*COLUMN_MAPPING.keys(), ACTION_COLUMN]
    missing_cols = [c for c in expected_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le jeu de données : {missing_cols}")

    df = raw_df.rename(columns=COLUMN_MAPPING)[[*FEATURES, ACTION_COLUMN]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[*FEATURES, ACTION_COLUMN])
    df[ACTION_COLUMN] = df[ACTION_COLUMN].astype(str).str.strip().str.lower()
    valid_mask = (df[FEATURES] >= 0).all(axis=1)
    return df.loc[valid_mask].reset_index(drop=True)


def audit_cleaning(path: str | Path = DATA_PATH) -> dict[str, Any]:
    """Mesure précisément les valeurs manquantes, illisibles, négatives et doublons."""
    raw_str = pd.read_csv(path, dtype=str)
    unreadable_by_col = {
        col: int(pd.to_numeric(raw_str[col], errors="coerce").isna().sum())
        for col in COLUMN_MAPPING
    }

    raw_df = load_raw_logs(path)
    cleaned_df = load_and_clean(path)
    allow_raw = raw_df.loc[raw_df[ACTION_COLUMN].str.strip().str.lower() == ALLOW_ACTION]
    allow_clean = cleaned_df.loc[cleaned_df[ACTION_COLUMN] == ALLOW_ACTION]

    negative_by_col = {
        col: int((pd.to_numeric(raw_df[col], errors="coerce") < 0).sum())
        for col in COLUMN_MAPPING
    }

    return {
        "rows_raw": len(raw_df),
        "rows_clean": len(cleaned_df),
        "rows_dropped": len(raw_df) - len(cleaned_df),
        "missing_total": int(raw_df.isna().sum().sum()),
        "unreadable_by_col": unreadable_by_col,
        "unreadable_total": sum(unreadable_by_col.values()),
        "negative_by_col": negative_by_col,
        "negative_total": sum(negative_by_col.values()),
        "exact_duplicates_all_cols": int(raw_df.duplicated().sum()),
        "exact_duplicates_allow_all_cols": int(allow_raw.duplicated().sum()),
        "feature_duplicates_allow": int(allow_clean.duplicated(subset=FEATURES).sum()),
        "unique_feature_profiles_allow": int(allow_clean[FEATURES].drop_duplicates().shape[0]),
        "duplicates_policy": "conserves",
    }


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un DataFrame déjà chargé en conservant toutes ses colonnes."""
    missing_cols = [c for c in [*FEATURE_COLUMNS, ACTION_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le jeu de données : {missing_cols}")

    cleaned = df.copy()
    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    valid_mask = cleaned[FEATURE_COLUMNS].notna().all(axis=1)
    non_negative_mask = (cleaned[FEATURE_COLUMNS] >= 0).all(axis=1)
    return cleaned.loc[valid_mask & non_negative_mask].copy()


def from_api_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convertit un payload JSON de l'API en DataFrame (1 ligne, 4 colonnes contrat API)."""
    row: dict[str, float] = {}
    for col in FEATURES:
        if col not in payload:
            raise ValueError(f"Champ requis manquant : {col}")
        value = float(payload[col])
        if np.isnan(value) or value < 0:
            raise ValueError(f"La valeur de '{col}' doit être >= 0 (reçu : {value})")
        row[col] = value
    return pd.DataFrame([row], columns=FEATURES)


def find_typical_allow_row(allow_df: pd.DataFrame) -> pd.Series:
    """Sélectionne la vraie ligne 'allow' la plus proche de la médiane (en log1p/IQR)."""
    cols = FEATURES if all(c in allow_df.columns for c in FEATURES) else FEATURE_COLUMNS
    x_log = np.log1p(allow_df[cols].astype(float))
    med_log = x_log.median()
    iqr_log = (x_log.quantile(0.75) - x_log.quantile(0.25)).replace(0, 1.0)
    dist = (((x_log - med_log) / iqr_log) ** 2).sum(axis=1)
    best_idx = dist.idxmin()
    return allow_df.loc[best_idx]
