"""Tests des contrats d'entraînement et d'export du modèle."""

import json

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, split_allow
from train import (
    compute_anomaly_scores,
    compute_threshold,
    evaluate_detector,
    save_metrics,
    save_model_artifact,
    train_isolation_forest,
)


def _training_data() -> pd.DataFrame:
    values = np.arange(60, dtype=float)
    return pd.DataFrame(
        {
            "bytes_sent": values + 10,
            "bytes_received": values + 20,
            "packets": values % 8 + 1,
            "elapsed_time_sec": values % 30,
            "Action": ["allow"] * 60,
        }
    )


def test_seuil_est_calcule_sur_les_scores_de_validation():
    train_df, val_df, _ = split_allow(_training_data(), random_state=42)
    pipeline = train_isolation_forest(train_df, random_state=42)
    val_scores = compute_anomaly_scores(pipeline, val_df)

    threshold = compute_threshold(val_scores, alert_rate=0.10)

    assert threshold == np.quantile(val_scores, 0.90)
    assert np.isfinite(threshold)


def test_metriques_et_export_sont_lisibles(tmp_path):
    train_df, val_df, _ = split_allow(_training_data(), random_state=42)
    pipeline = train_isolation_forest(train_df, random_state=42)
    val_scores = compute_anomaly_scores(pipeline, val_df)
    synthetic = pd.DataFrame({column: [10_000.0] for column in FEATURES})
    synthetic_scores = compute_anomaly_scores(pipeline, synthetic)
    threshold = compute_threshold(val_scores, alert_rate=0.10)
    metrics = evaluate_detector(
        val_scores, synthetic_scores, pd.Series(["exfiltration"]), threshold
    )

    model_path = save_model_artifact(pipeline, threshold, path=tmp_path / "model.joblib")
    metrics_path = save_metrics(metrics, tmp_path / "metrics.json")

    artifact = joblib.load(model_path)
    loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(artifact) == {"pipeline", "threshold", "features"}
    assert artifact["features"] == FEATURES
    assert loaded_metrics["test_size"] == len(val_scores)
    assert loaded_metrics["synthetic_size"] == 1
