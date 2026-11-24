"""Entraînement du modèle Isolation Forest sur les 4 mesures réseau (issue #10)."""

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

FEATURES = ["bytes_sent", "bytes_received", "packets", "elapsed_time_sec"]
DATA_PATH = Path("data/log2.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SYNTHETIC_PATH = ARTIFACTS_DIR / "cas_synthetiques.csv"
SYNTHETIC_PER_TYPE = 50
ALERT_RATE = 0.01
ALERT_RATES_STUDY = (0.005, 0.01, 0.02)
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


def compute_baseline_scores(df: pd.DataFrame) -> np.ndarray:
    """Calcule le score baseline (bytes_sent + bytes_received) : plus grand = plus atypique."""
    # Volume total d'octets échangés comme référence simple face à Isolation Forest.
    return (df["bytes_sent"] + df["bytes_received"]).to_numpy(dtype=float)


def compute_threshold(
    val_scores: np.ndarray, alert_rate: float = ALERT_RATE
) -> float:
    """Fixe le seuil au quantile (1 - alert_rate) des scores de validation uniquement."""
    # Seuil calibré uniquement sur la validation pour ne jamais contaminer le jeu de test.
    return float(np.quantile(val_scores, 1.0 - alert_rate))


def predict_anomalies(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Déclenche une alerte (True) pour chaque score supérieur ou égal au seuil."""
    return scores >= threshold


def generate_synthetic_anomalies(
    n_per_type: int = SYNTHETIC_PER_TYPE, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Génère 4 types de lignes réseau atypiques reproductibles et sans valeurs négatives."""
    # Aucun label d'attaque dans log2.csv : ces 4 profils servent à mesurer le rappel.
    rng = np.random.default_rng(random_state)

    exfiltration = pd.DataFrame(
        {
            "bytes_sent": rng.integers(5_000_000, 50_000_000, size=n_per_type),
            "bytes_received": rng.integers(60, 2_000, size=n_per_type),
            "packets": rng.integers(3_000, 35_000, size=n_per_type),
            "elapsed_time_sec": rng.integers(5, 300, size=n_per_type),
            "type": "exfiltration",
        }
    )
    flood = pd.DataFrame(
        {
            "bytes_sent": rng.integers(0, 600, size=n_per_type),
            "bytes_received": np.zeros(n_per_type, dtype=int),
            "packets": rng.integers(500, 5_000, size=n_per_type),
            "elapsed_time_sec": np.zeros(n_per_type, dtype=int),
            "type": "flood",
        }
    )
    connexion_fantome = pd.DataFrame(
        {
            "bytes_sent": rng.integers(60, 200, size=n_per_type),
            "bytes_received": rng.integers(0, 100, size=n_per_type),
            "packets": np.ones(n_per_type, dtype=int),
            "elapsed_time_sec": rng.integers(3_600, 14_400, size=n_per_type),
            "type": "connexion_fantome",
        }
    )
    telechargement_massif = pd.DataFrame(
        {
            "bytes_sent": rng.integers(100, 5_000, size=n_per_type),
            "bytes_received": rng.integers(5_000_000, 50_000_000, size=n_per_type),
            "packets": rng.integers(3_000, 35_000, size=n_per_type),
            "elapsed_time_sec": rng.integers(10, 600, size=n_per_type),
            "type": "telechargement_massif",
        }
    )

    synthetic_df = pd.concat(
        [exfiltration, flood, connexion_fantome, telechargement_massif],
        ignore_index=True,
    )
    return synthetic_df[FEATURES + ["type"]]


def save_synthetic_anomalies(
    synthetic_df: pd.DataFrame, path: Path = SYNTHETIC_PATH
) -> Path:
    """Sauvegarde les cas synthétiques dans artifacts/cas_synthetiques.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(path, index=False)
    return path


def save_model_artifact(
    pipeline: Pipeline,
    threshold: float,
    features: list[str] = FEATURES,
    path: Path = MODEL_PATH,
) -> Path:
    """Sauvegarde le dict {pipeline, threshold, features} selon le contrat de l'API."""
    # Contrat commun avec api.py : exactement ces 3 clés et l'ordre des 4 colonnes.
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "features": list(features),
    }
    joblib.dump(artifact, path)
    return path


def evaluate_detector(
    test_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    synthetic_types: pd.Series,
    threshold: float,
) -> dict[str, object]:
    """Mesure les fausses alertes sur test allow et le rappel global/par type sur synthétiques."""
    # Séparation stricte : test allow mesure les fausses alertes, les synthétiques le rappel.
    test_alerts = predict_anomalies(test_scores, threshold)
    synth_alerts = predict_anomalies(synthetic_scores, threshold)

    recall_by_type: dict[str, float] = {}
    for anomaly_type in pd.unique(synthetic_types):
        mask = (synthetic_types == anomaly_type).to_numpy()
        recall_by_type[str(anomaly_type)] = round(float(synth_alerts[mask].mean()), 6)

    return {
        "threshold": round(float(threshold), 6),
        "false_alarm_rate": round(float(test_alerts.mean()), 6),
        "false_alarms_count": int(test_alerts.sum()),
        "test_size": int(len(test_alerts)),
        "synthetic_recall": round(float(synth_alerts.mean()), 6),
        "synthetic_detected": int(synth_alerts.sum()),
        "synthetic_size": int(len(synth_alerts)),
        "recall_by_type": recall_by_type,
    }


def evaluate_alert_rates(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    synthetic_types: pd.Series,
    alert_rates: tuple[float, ...] = ALERT_RATES_STUDY,
) -> dict[str, dict[str, object]]:
    """Compare plusieurs taux d'alertes cibles (seuils fixés sur la validation)."""
    # Permet d'objectiver le compromis fausses alertes / rappel sans toucher au test.
    results: dict[str, dict[str, object]] = {}
    for rate in alert_rates:
        threshold = compute_threshold(val_scores, alert_rate=rate)
        results[f"{rate * 100:g}%"] = evaluate_detector(
            test_scores, synthetic_scores, synthetic_types, threshold
        )
    return results


def save_metrics(
    metrics: dict[str, object], path: Path = METRICS_PATH
) -> Path:
    """Écrit le résumé d'évaluation dans artifacts/metrics.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Charge les données, entraîne Isolation Forest et génère les cas synthétiques."""
    df = load_and_clean(DATA_PATH)
    train_df, val_df, test_df = split_allow(df, random_state=RANDOM_STATE)

    start = time.perf_counter()
    pipeline = train_isolation_forest(train_df, random_state=RANDOM_STATE)
    elapsed = time.perf_counter() - start

    val_scores = compute_anomaly_scores(pipeline, val_df)
    baseline_val_scores = compute_baseline_scores(val_df)

    iforest_threshold = compute_threshold(val_scores, alert_rate=ALERT_RATE)
    baseline_threshold = compute_threshold(baseline_val_scores, alert_rate=ALERT_RATE)
    iforest_val_alerts = predict_anomalies(val_scores, iforest_threshold)
    baseline_val_alerts = predict_anomalies(baseline_val_scores, baseline_threshold)

    synthetic_df = generate_synthetic_anomalies(
        n_per_type=SYNTHETIC_PER_TYPE, random_state=RANDOM_STATE
    )
    synthetic_path = save_synthetic_anomalies(synthetic_df, SYNTHETIC_PATH)
    model_path = save_model_artifact(pipeline, iforest_threshold, FEATURES, MODEL_PATH)

    test_scores = compute_anomaly_scores(pipeline, test_df)
    baseline_test_scores = compute_baseline_scores(test_df)
    synth_scores = compute_anomaly_scores(pipeline, synthetic_df)
    baseline_synth_scores = compute_baseline_scores(synthetic_df)

    metrics = {
        "alert_rate_target": ALERT_RATE,
        "random_state": RANDOM_STATE,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "synthetic": len(synthetic_df),
        },
        "baseline": evaluate_detector(
            baseline_test_scores,
            baseline_synth_scores,
            synthetic_df["type"],
            baseline_threshold,
        ),
        "isolation_forest": evaluate_detector(
            test_scores,
            synth_scores,
            synthetic_df["type"],
            iforest_threshold,
        ),
        "alert_rate_study": {
            "baseline": evaluate_alert_rates(
                baseline_val_scores,
                baseline_test_scores,
                baseline_synth_scores,
                synthetic_df["type"],
                ALERT_RATES_STUDY,
            ),
            "isolation_forest": evaluate_alert_rates(
                val_scores,
                test_scores,
                synth_scores,
                synthetic_df["type"],
                ALERT_RATES_STUDY,
            ),
        },
    }
    metrics_path = save_metrics(metrics, METRICS_PATH)

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
    print(
        "Scores baseline validation (octets totaux) : "
        f"min={baseline_val_scores.min():.1f}, "
        f"médiane={np.median(baseline_val_scores):.1f}, "
        f"p99={np.percentile(baseline_val_scores, 99):.1f}, "
        f"max={baseline_val_scores.max():.1f}"
    )
    print(
        f"Seuil Isolation Forest (val, cible {ALERT_RATE * 100:.1f} %) : "
        f"{iforest_threshold:.4f} -> taux observé val = "
        f"{iforest_val_alerts.mean() * 100:.2f} % "
        f"({int(iforest_val_alerts.sum())}/{len(val_df)})"
    )
    print(
        f"Seuil baseline (val, cible {ALERT_RATE * 100:.1f} %) : "
        f"{baseline_threshold:.1f} -> taux observé val = "
        f"{baseline_val_alerts.mean() * 100:.2f} % "
        f"({int(baseline_val_alerts.sum())}/{len(val_df)})"
    )
    print(
        f"Cas synthétiques sauvegardés : {len(synthetic_df)} lignes "
        f"({synthetic_df['type'].nunique()} types) -> {synthetic_path}"
    )
    print(f"Modèle exporté : {model_path} (seuil={iforest_threshold:.4f})")
    for name in ("baseline", "isolation_forest"):
        res = metrics[name]
        print(
            f"Évaluation {name} : fausses alertes test = "
            f"{res['false_alarm_rate'] * 100:.2f} % "
            f"({res['false_alarms_count']}/{res['test_size']}) | "
            f"rappel synthétiques = {res['synthetic_recall'] * 100:.2f} % "
            f"({res['synthetic_detected']}/{res['synthetic_size']})"
        )
    print("Étude des taux d'alertes (0.5 %, 1 %, 2 %) :")
    for rate_key in ("0.5%", "1%", "2%"):
        bl = metrics["alert_rate_study"]["baseline"][rate_key]
        iso = metrics["alert_rate_study"]["isolation_forest"][rate_key]
        print(
            f"  cible {rate_key:>4} | baseline: FA={bl['false_alarm_rate'] * 100:.2f}% "
            f"rappel={bl['synthetic_recall'] * 100:.2f}% | "
            f"isolation_forest: seuil={iso['threshold']:.4f} "
            f"FA={iso['false_alarm_rate'] * 100:.2f}% "
            f"rappel={iso['synthetic_recall'] * 100:.2f}%"
        )
    print(f"Métriques exportées : {metrics_path}")


if __name__ == "__main__":
    main()
