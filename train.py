"""Entraînement du modèle Isolation Forest sur les 4 mesures réseau (issue #10)."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

FEATURES = ["bytes_sent", "bytes_received", "packets", "elapsed_time_sec"]
DATA_PATH = Path("data/log2.csv")
RANDOM_STATE = 42

COLUMN_MAPPING = {
    "Bytes Sent": "bytes_sent",
    "Bytes Received": "bytes_received",
    "Packets": "packets",
    "Elapsed Time (sec)": "elapsed_time_sec",
}

# Fallback temporaire tant que features.py d'Aymen n'est pas mergé sur staging.
try:
    from features import load_and_clean, split_allow
except ImportError:

    def load_and_clean(path: str | Path = DATA_PATH) -> pd.DataFrame:
        """Charge le CSV, renomme les 4 mesures et retire les lignes invalides."""
        df = pd.read_csv(path).rename(columns=COLUMN_MAPPING)
        df = df[FEATURES + ["Action"]].copy()
        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=FEATURES + ["Action"])
        df["Action"] = df["Action"].astype(str).str.strip().str.lower()
        valid_mask = (df[FEATURES] >= 0).all(axis=1)
        return df.loc[valid_mask].reset_index(drop=True)

    def split_allow(
        df: pd.DataFrame, random_state: int = RANDOM_STATE
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Découpe les lignes allow en train (60 %), validation (20 %) et test (20 %)."""
        # Référence de trafic normal uniquement ; 3 splits pour éviter toute fuite de données.
        allow_df = df.loc[df["Action"] == "allow"].copy()
        train_df, temp_df = train_test_split(
            allow_df, test_size=0.4, random_state=random_state
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=random_state
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )


def build_isolation_forest_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Construit le pipeline log1p + IsolationForest(n_estimators=200)."""
    return Pipeline(
        steps=[
            # log1p compresse les grands écarts d'octets tout en gérant proprement les 0.
            ("log1p", FunctionTransformer(np.log1p)),
            (
                # 200 arbres avec graine fixe pour isoler les cas atypiques de façon reproductible.
                "isolation_forest",
                IsolationForest(n_estimators=200, random_state=random_state),
            ),
        ]
    )


def train_isolation_forest(
    train_df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> Pipeline:
    """Entraîne le pipeline uniquement sur les 4 mesures du jeu d'entraînement allow."""
    pipeline = build_isolation_forest_pipeline(random_state=random_state)
    pipeline.fit(train_df[FEATURES])
    return pipeline


def compute_anomaly_scores(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Calcule le score d'anomalie (-score_samples) : plus grand = plus atypique."""
    # Inversion du signe car score_samples() est d'autant plus bas que la ligne est atypique.
    return -pipeline.score_samples(df[FEATURES])


def main() -> None:
    """Charge les données, entraîne Isolation Forest et affiche un aperçu des scores."""
    df = load_and_clean(DATA_PATH)
    train_df, val_df, test_df = split_allow(df, random_state=RANDOM_STATE)

    start = time.perf_counter()
    pipeline = train_isolation_forest(train_df, random_state=RANDOM_STATE)
    elapsed = time.perf_counter() - start

    val_scores = compute_anomaly_scores(pipeline, val_df)

    print(
        f"Lignes allow : train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(f"Entraînement Isolation Forest terminé en {elapsed:.2f} s")
    print(
        "Scores validation (-score_samples) : "
        f"min={val_scores.min():.4f}, "
        f"médiane={np.median(val_scores):.4f}, "
        f"p99={np.percentile(val_scores, 99):.4f}, "
        f"max={val_scores.max():.4f}"
    )


if __name__ == "__main__":
    main()
