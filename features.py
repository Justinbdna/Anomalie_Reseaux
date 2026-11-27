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

# Permet d'écrire les types modernes (ex. "str | Path") même sur d'anciennes versions de Python.
from __future__ import annotations

# Path : manipuler des chemins de fichiers. Any : "n'importe quel type" (pour les annotations).
from pathlib import Path
from typing import Any

# numpy : calcul numérique. pandas : tableaux de données (DataFrame).
# train_test_split : découper un tableau en deux parties au hasard.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- Constantes
# Chemin du fichier CSV, nom de la colonne de décision du pare-feu et valeur "trafic autorisé".
DATA_PATH = Path("data/log2.csv")
ACTION_COLUMN = "Action"
ALLOW_ACTION = "allow"
# Graine du hasard : avec la même graine, le découpage est toujours identique (reproductible).
RANDOM_STATE = 42
# Nombre maximum de lignes "allow" gardées (échantillonnage si le fichier est plus gros).
MAX_ALLOW_ROWS = 50_000

# Correspondance "nom dans le CSV" -> "nom utilisé par l'API et le modèle".
COLUMN_MAPPING: dict[str, str] = {
    "Bytes Sent": "bytes_sent",
    "Bytes Received": "bytes_received",
    "Packets": "packets",
    "Elapsed Time (sec)": "elapsed_time_sec",
}

# Les 4 mesures données au modèle, toujours dans cet ordre.
FEATURES: list[str] = [
    "bytes_sent",
    "bytes_received",
    "packets",
    "elapsed_time_sec",
]

# Alias conservés pour compatibilité avec les scripts d'exploration
# FEATURE_COLUMNS : les noms d'origine du CSV.
# API_TO_FEATURE : dictionnaire inversé (nom API -> nom CSV). FEATURE_TO_API : copie du mapping.
FEATURE_COLUMNS: list[str] = list(COLUMN_MAPPING.keys())
API_TO_FEATURE: dict[str, str] = {v: k for k, v in COLUMN_MAPPING.items()}
FEATURE_TO_API: dict[str, str] = dict(COLUMN_MAPPING)


def load_raw_logs(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le fichier CSV brut des journaux de pare-feu sans transformation."""
    # Si le fichier n'existe pas, on arrête avec un message clair au lieu d'une erreur obscure.
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")
    # Lecture du CSV dans un DataFrame pandas, tel quel.
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
    # On charge le CSV brut, puis on vérifie que les 5 colonnes utiles sont bien présentes.
    raw_df = load_raw_logs(path)
    expected_cols = [*COLUMN_MAPPING.keys(), ACTION_COLUMN]
    missing_cols = [c for c in expected_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le jeu de données : {missing_cols}")

    # Renommage au format API, on ne garde que les 4 mesures + Action.
    # errors="coerce" : une valeur qui n'est pas un nombre devient NaN (valeur manquante).
    df = raw_df.rename(columns=COLUMN_MAPPING)[[*FEATURES, ACTION_COLUMN]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # On retire les lignes avec une valeur manquante, puis on met Action en minuscules sans espaces.
    df = df.dropna(subset=[*FEATURES, ACTION_COLUMN])
    df[ACTION_COLUMN] = df[ACTION_COLUMN].astype(str).str.strip().str.lower()
    # On garde seulement les lignes où les 4 mesures sont >= 0 (une valeur négative est impossible).
    valid_mask = (df[FEATURES] >= 0).all(axis=1)
    return df.loc[valid_mask].reset_index(drop=True)


def split_allow(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    max_rows: int | None = MAX_ALLOW_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe les lignes 'allow' en train (60 %), validation (20 %) et test (20 %) (D5).

    - Filtre uniquement le trafic normal (Action == 'allow').
    - Échantillonne à `max_rows` (50 000 par défaut) si le volume dépasse ce seuil.
    - Garantit l'absence de recouvrement entre les 3 jeux et une reproductibilité
      stricte grâce à `random_state=42`.
    - Signature alignée sur `train.py` (Justin).
    """
    # On ne garde que le trafic autorisé ("allow") : c'est notre référence du trafic normal.
    action_norm = df[ACTION_COLUMN].astype(str).str.strip().str.lower()
    allow_df = df.loc[action_norm == ALLOW_ACTION].copy()

    # S'il y a trop de lignes, on en tire au hasard max_rows (toujours les mêmes grâce à la graine).
    if max_rows is not None and len(allow_df) > max_rows:
        allow_df = allow_df.sample(n=max_rows, random_state=random_state)

    # 1er découpage : 60 % pour l'entraînement, 40 % mis de côté.
    train_df, temp_df = train_test_split(
        allow_df,
        test_size=0.4,
        random_state=random_state,
    )
    # 2e découpage : les 40 % sont coupés en deux -> 20 % validation et 20 % test.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
    )
    # On renvoie les 3 jeux avec des index remis à 0, 1, 2, ...
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def audit_cleaning(path: str | Path = DATA_PATH) -> dict[str, Any]:
    """Mesure précisément les valeurs manquantes, illisibles, négatives et doublons."""
    # On relit le CSV en texte pour compter, par colonne, les valeurs qui ne sont pas des nombres.
    raw_str = pd.read_csv(path, dtype=str)
    unreadable_by_col = {
        col: int(pd.to_numeric(raw_str[col], errors="coerce").isna().sum())
        for col in COLUMN_MAPPING
    }

    # Version brute et version nettoyée, et leurs lignes "allow", pour comparer avant / après.
    raw_df = load_raw_logs(path)
    cleaned_df = load_and_clean(path)
    allow_raw = raw_df.loc[raw_df[ACTION_COLUMN].str.strip().str.lower() == ALLOW_ACTION]
    allow_clean = cleaned_df.loc[cleaned_df[ACTION_COLUMN] == ALLOW_ACTION]

    # Nombre de valeurs négatives par colonne.
    negative_by_col = {
        col: int((pd.to_numeric(raw_df[col], errors="coerce") < 0).sum())
        for col in COLUMN_MAPPING
    }

    # Rapport final : un dictionnaire avec tous les comptages (lignes, manquants, négatifs, doublons).
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
    # Vérifie que les colonnes nécessaires (noms d'origine du CSV) sont présentes.
    missing_cols = [c for c in [*FEATURE_COLUMNS, ACTION_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le jeu de données : {missing_cols}")

    # Conversion des 4 mesures en nombres (texte illisible -> NaN), sur une copie du tableau.
    cleaned = df.copy()
    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # On garde les lignes sans valeur manquante ET sans valeur négative.
    valid_mask = cleaned[FEATURE_COLUMNS].notna().all(axis=1)
    non_negative_mask = (cleaned[FEATURE_COLUMNS] >= 0).all(axis=1)
    return cleaned.loc[valid_mask & non_negative_mask].copy()


def from_api_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convertit un payload JSON de l'API en DataFrame (1 ligne, 4 colonnes contrat API)."""
    # Pour chaque mesure : on vérifie qu'elle est présente, que c'est un nombre et qu'il est >= 0.
    row: dict[str, float] = {}
    for col in FEATURES:
        if col not in payload:
            raise ValueError(f"Champ requis manquant : {col}")
        value = float(payload[col])
        if np.isnan(value) or value < 0:
            raise ValueError(f"La valeur de '{col}' doit être >= 0 (reçu : {value})")
        row[col] = value
    # Tableau d'une seule ligne, avec les colonnes dans l'ordre attendu par le modèle.
    return pd.DataFrame([row], columns=FEATURES)


def find_typical_allow_row(allow_df: pd.DataFrame) -> pd.Series:
    """Sélectionne la vraie ligne 'allow' la plus proche de la médiane (en log1p/IQR)."""
    # On utilise les noms API s'ils existent, sinon les noms d'origine du CSV.
    cols = FEATURES if all(c in allow_df.columns for c in FEATURES) else FEATURE_COLUMNS
    # log1p écrase les très grandes valeurs ; on calcule la médiane de chaque mesure.
    x_log = np.log1p(allow_df[cols].astype(float))
    med_log = x_log.median()
    # IQR = écart entre le 3e et le 1er quartile : sert d'échelle (0 remplacé par 1 pour ne pas diviser par 0).
    iqr_log = (x_log.quantile(0.75) - x_log.quantile(0.25)).replace(0, 1.0)
    # Distance de chaque ligne à la médiane ; on renvoie la ligne la plus proche (la plus "typique").
    dist = (((x_log - med_log) / iqr_log) ** 2).sum(axis=1)
    best_idx = dist.idxmin()
    return allow_df.loc[best_idx]
