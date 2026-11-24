"""Fixtures pytest communes (issue #30).

Les tests ne dépendent ni de data/log2.csv (65 532 lignes) ni de
artifacts/model.joblib : on entraîne un petit modèle temporaire sur le faux
journal tests/fixtures/sample_log2.csv, écrit dans un dossier temporaire.
Rien n'est modifié dans data/ ni artifacts/.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

RACINE = Path(__file__).resolve().parent.parent
FAUX_JOURNAL = Path(__file__).resolve().parent / "fixtures" / "sample_log2.csv"

# api.py est à la racine du dépôt.
sys.path.insert(0, str(RACINE))

FEATURES = ["bytes_sent", "bytes_received", "packets", "elapsed_time_sec"]
COLONNES = {
    "Bytes Sent": "bytes_sent",
    "Bytes Received": "bytes_received",
    "Packets": "packets",
    "Elapsed Time (sec)": "elapsed_time_sec",
}


@pytest.fixture(scope="session")
def faux_journal() -> pd.DataFrame:
    """Le faux journal, avec les 12 colonnes d'origine."""
    return pd.read_csv(FAUX_JOURNAL)


@pytest.fixture(scope="session")
def modele_temporaire(tmp_path_factory, faux_journal) -> Path:
    """Entraîne un Isolation Forest sur les lignes allow du faux journal.

    Même format que artifacts/model.joblib : un dict avec "pipeline",
    "threshold" et "features". Le fichier vit dans un dossier temporaire.
    """
    allow = faux_journal.loc[faux_journal["Action"] == "allow"]
    X = allow.rename(columns=COLONNES)[FEATURES]

    pipeline = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("forest", IsolationForest(n_estimators=50, random_state=42)),
        ]
    ).fit(X)

    # Contrat de train.py : score = -score_samples (plus grand = plus atypique)
    # et is_anomaly = score >= seuil.
    # Seuil = percentile 90 (et non 99 comme train.py) : le faux journal n'a que
    # 160 lignes allow, un P99 (0.5868) laisserait passer un cas d'exfiltration
    # (score 0.5498). Avec P90 (0.5466) : normal 0.507 -> False, exfiltration -> True.
    seuil = float(np.quantile(-pipeline.score_samples(X), 0.90))

    chemin = tmp_path_factory.mktemp("modele") / "model.joblib"
    joblib.dump(
        {"pipeline": pipeline, "threshold": seuil, "features": FEATURES}, chemin
    )
    return chemin


@pytest.fixture(scope="session")
def client(modele_temporaire):
    """TestClient FastAPI branché sur le modèle temporaire."""
    # Import tardif : MODEL_PATH est lu à l'import de api.py.
    import importlib

    import os

    os.environ["MODEL_PATH"] = str(modele_temporaire)
    import api

    api = importlib.reload(api)
    api.load_model.cache_clear()

    from fastapi.testclient import TestClient

    return TestClient(api.app)
