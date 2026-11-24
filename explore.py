"""Exploration, contrôle qualité et analyse comparative des journaux pare-feu (D2, D4, D7).

Exécuter avec :
    python explore.py

Produit :
- Un rapport complet sur la console (premier contrôle D2, audit nettoyage D3,
  quantiles du trafic allow, découpage D5, vérification du cas typique D6,
  comparaison allow vs deny/drop D7).
- Les graphiques dans artifacts/ :
  - artifacts/trafic_allow.png (D4)
  - artifacts/comparaison_allow_deny_drop.png (D7)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from features import (
    ACTION_COLUMN,
    ALLOW_ACTION,
    DATA_PATH,
    FEATURES,
    RANDOM_STATE,
    audit_cleaning,
    find_typical_allow_row,
    from_api_payload,
    load_and_clean,
    load_raw_logs,
    split_allow,
)

ARTIFACTS_DIR = Path("artifacts")
TRAFFIC_ALLOW_PLOT = ARTIFACTS_DIR / "trafic_allow.png"
COMPARISON_PLOT = ARTIFACTS_DIR / "comparaison_allow_deny_drop.png"


def _format_log1p_tick(log_val: float, _pos: int) -> str:
    """Formate un tick en échelle log10(1 + x) de façon lisible (0, 10, 1k, 1M, 1G)."""
    raw = round((10**log_val) - 1)
    if raw <= 0:
        return "0"
    if raw < 1_000:
        return f"{raw}"
    if raw < 1_000_000:
        val = raw / 1_000
        return f"{val:.0f}k" if val >= 10 or abs(val - round(val)) < 0.05 else f"{val:.1f}k"
    if raw < 1_000_000_000:
        val = raw / 1_000_000
        return f"{val:.0f}M" if val >= 10 or abs(val - round(val)) < 0.05 else f"{val:.1f}M"
    val = raw / 1_000_000_000
    return f"{val:.0f}G" if val >= 10 or abs(val - round(val)) < 0.05 else f"{val:.1f}G"


def _format_int(val: float) -> str:
    """Formate un nombre arrondi à l'entier avec espace comme séparateur de milliers."""
    return f"{round(val):,}".replace(",", " ")


def first_control(path: Path = DATA_PATH) -> dict[str, object]:
    """Premier contrôle (D2) : comptage par Action et vérification des valeurs manquantes."""
    df_raw = load_raw_logs(path)
    action_counts = df_raw[ACTION_COLUMN].value_counts(dropna=False).to_dict()
    missing_by_col = {col: int(cnt) for col, cnt in df_raw.isna().sum().items()}
    return {
        "total_rows": len(df_raw),
        "n_columns": int(df_raw.shape[1]),
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "missing_by_column": missing_by_col,
        "total_missing": sum(missing_by_col.values()),
    }


def _fit_reference_model(train_df: pd.DataFrame) -> Pipeline:
    """Entraîne le pipeline Isolation Forest de référence (identique à train.py)."""
    pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p)),
            (
                "isolation_forest",
                IsolationForest(n_estimators=200, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(train_df[FEATURES])
    return pipeline


def plot_allow_traffic(
    allow_df: pd.DataFrame,
    typical_row: pd.Series,
    output_path: Path = TRAFFIC_ALLOW_PLOT,
) -> Path:
    """Génère le graphique 2x2 des distributions du trafic 'allow' en échelle log10(1+x) (D4)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), dpi=180)
    fig.patch.set_facecolor("#ffffff")

    feature_meta = [
        ("bytes_sent", "Octets envoyés (bytes_sent)", "octets", "#2563eb"),
        ("bytes_received", "Octets reçus (bytes_received)", "octets", "#0891b2"),
        ("packets", "Nombre de paquets (packets)", "paquets", "#7c3aed"),
        ("elapsed_time_sec", "Durée de session (elapsed_time_sec)", "s", "#059669"),
    ]

    for ax, (col, title, unit, color) in zip(axes.flat, feature_meta):
        series = allow_df[col].astype(float)
        log_vals = np.log10(1.0 + series)

        q25 = float(series.quantile(0.25))
        median = float(series.median())
        q75 = float(series.quantile(0.75))
        p99 = float(series.quantile(0.99))
        max_val = float(series.max())
        typical_val = float(typical_row[col])

        log_q25 = np.log10(1.0 + q25)
        log_med = np.log10(1.0 + median)
        log_q75 = np.log10(1.0 + q75)
        log_p99 = np.log10(1.0 + p99)
        log_typ = np.log10(1.0 + typical_val)

        ax.set_facecolor("#f8fafc")
        ax.hist(
            log_vals,
            bins=48,
            color=color,
            alpha=0.78,
            edgecolor="#ffffff",
            linewidth=0.35,
        )

        ax.axvspan(
            log_q25,
            log_q75,
            color=color,
            alpha=0.14,
            zorder=2,
            label=f"IQR (Q25–Q75) : {_format_int(q25)}–{_format_int(q75)}",
        )
        ax.axvline(
            log_typ,
            color="#f59e0b",
            linestyle="-",
            linewidth=2.8,
            alpha=0.9,
            zorder=3,
            label=f"Cas typique démo : {_format_int(typical_val)} {unit}",
        )
        ax.axvline(
            log_med,
            color="#0f172a",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Médiane : {_format_int(median)} {unit}",
        )
        ax.axvline(
            log_p99,
            color="#dc2626",
            linestyle=":",
            linewidth=1.8,
            zorder=4,
            label=f"P99 : {_format_int(p99)} {unit} (max : {_format_int(max_val)})",
        )

        max_exp = int(np.ceil(log_vals.max()))
        tick_raw = [0] + [10**e for e in range(1, max_exp + 1)]
        tick_pos = [np.log10(1.0 + v) for v in tick_raw]
        ax.set_xticks(tick_pos)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_log1p_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _format_int(y)))

        ax.set_title(title, fontsize=11.5, fontweight="bold", color="#0f172a", pad=8)
        ax.set_xlabel(f"Valeur ({unit}, échelle log10(1 + x))", fontsize=9.5, color="#334155")
        ax.set_ylabel("Nombre de flux", fontsize=9.5, color="#334155")
        ax.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="upper right",
            fontsize=8.2,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#e2e8f0",
            framealpha=0.94,
        )

    fig.suptitle(
        f"Distribution du trafic autorisé (Action = allow) — {_format_int(len(allow_df))} flux",
        fontsize=14.5,
        fontweight="bold",
        color="#0f172a",
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        (
            "Source : UCI Internet Firewall Data (data/log2.csv, 65 532 lignes)  •  "
            f"Cas typique démo : ligne #{int(typical_row.name)} "
            f"({int(typical_row['bytes_sent'])} B envoyés, "
            f"{int(typical_row['bytes_received'])} B reçus, "
            f"{int(typical_row['packets'])} paquets, "
            f"{int(typical_row['elapsed_time_sec'])} s)"
        ),
        ha="center",
        fontsize=9.3,
        color="#475569",
    )

    fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.91))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def compare_allow_vs_refused(
    df_clean: pd.DataFrame,
    pipeline: Pipeline,
    val_scores: np.ndarray,
    test_df: pd.DataFrame,
    output_path: Path = COMPARISON_PLOT,
) -> dict[str, dict[str, float]]:
    """Compare allow vs deny/drop sur les 4 mesures et sur les scores du modèle (D7)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allow_df = df_clean.loc[df_clean[ACTION_COLUMN] == ALLOW_ACTION]
    deny_df = df_clean.loc[df_clean[ACTION_COLUMN] == "deny"]
    drop_df = df_clean.loc[df_clean[ACTION_COLUMN] == "drop"]
    refused_df = df_clean.loc[df_clean[ACTION_COLUMN].isin(["deny", "drop"])]

    p95 = float(np.percentile(val_scores, 95))
    p99 = float(np.percentile(val_scores, 99))

    scores = {
        "allow_test": -pipeline.score_samples(test_df[FEATURES]),
        "deny": -pipeline.score_samples(deny_df[FEATURES]),
        "drop": -pipeline.score_samples(drop_df[FEATURES]),
        "deny_drop": -pipeline.score_samples(refused_df[FEATURES]),
    }

    summary: dict[str, dict[str, float]] = {}
    for key, arr in scores.items():
        summary[key] = {
            "count": float(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "pct_above_p95": float((arr > p95).mean() * 100.0),
            "pct_above_p99": float((arr > p99).mean() * 100.0),
        }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
        }
    )

    fig = plt.figure(figsize=(14.0, 10.6), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.15], hspace=0.42, wspace=0.22)

    feature_meta = [
        ("bytes_sent", "Octets envoyés (bytes_sent)", "octets"),
        ("bytes_received", "Octets reçus (bytes_received)", "octets"),
        ("packets", "Nombre de paquets (packets)", "paquets"),
        ("elapsed_time_sec", "Durée de session (elapsed_time_sec)", "s"),
    ]

    for idx, (col, title, unit) in enumerate(feature_meta):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#f8fafc")

        allow_log = np.log10(1.0 + allow_df[col].astype(float))
        refused_log = np.log10(1.0 + refused_df[col].astype(float))
        upper = max(float(allow_log.max()), float(refused_log.max()))
        bins = np.linspace(0.0, upper, 42)

        ax.hist(
            allow_log,
            bins=bins,
            density=True,
            color="#2563eb",
            alpha=0.55,
            edgecolor="#ffffff",
            linewidth=0.3,
            label=f"allow (n={_format_int(len(allow_df))}, méd. {_format_int(allow_df[col].median())})",
        )
        ax.hist(
            refused_log,
            bins=bins,
            density=True,
            color="#dc2626",
            alpha=0.55,
            edgecolor="#ffffff",
            linewidth=0.3,
            label=(
                f"deny + drop (n={_format_int(len(refused_df))}, "
                f"méd. {_format_int(refused_df[col].median())})"
            ),
        )

        max_exp = int(np.ceil(upper))
        tick_raw = [0] + [10**e for e in range(1, max_exp + 1)]
        ax.set_xticks([np.log10(1.0 + v) for v in tick_raw])
        ax.xaxis.set_major_formatter(FuncFormatter(_format_log1p_tick))
        ax.set_title(title, fontsize=11.0, fontweight="bold", color="#0f172a", pad=7)
        ax.set_xlabel(f"Valeur ({unit}, échelle log10(1 + x))", fontsize=9.2, color="#334155")
        ax.set_ylabel("Densité relative", fontsize=9.2, color="#334155")
        ax.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="upper right",
            fontsize=8.3,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#e2e8f0",
            framealpha=0.94,
        )

    # Panneau inférieur pleine largeur : scores du modèle Isolation Forest
    ax_score = fig.add_subplot(gs[2, :])
    ax_score.set_facecolor("#f8fafc")

    score_bins = np.linspace(
        min(scores["allow_test"].min(), scores["deny_drop"].min()),
        max(scores["allow_test"].max(), scores["deny_drop"].max()),
        55,
    )
    ax_score.hist(
        scores["allow_test"],
        bins=score_bins,
        density=True,
        color="#2563eb",
        alpha=0.55,
        edgecolor="#ffffff",
        linewidth=0.3,
        label=(
            f"allow — test (n={_format_int(len(test_df))})  |  "
            f"moy. = {summary['allow_test']['mean']:.4f}"
        ),
    )
    ax_score.hist(
        scores["deny"],
        bins=score_bins,
        density=True,
        color="#dc2626",
        alpha=0.52,
        edgecolor="#ffffff",
        linewidth=0.3,
        label=(
            f"deny (n={_format_int(len(deny_df))})  |  "
            f"moy. = {summary['deny']['mean']:.4f}  "
            f"({summary['deny']['pct_above_p95']:.1f} % > P95)"
        ),
    )
    ax_score.hist(
        scores["drop"],
        bins=score_bins,
        density=True,
        color="#f59e0b",
        alpha=0.50,
        edgecolor="#ffffff",
        linewidth=0.3,
        label=(
            f"drop (n={_format_int(len(drop_df))})  |  "
            f"moy. = {summary['drop']['mean']:.4f}  "
            f"({summary['drop']['pct_above_p95']:.1f} % > P95)"
        ),
    )

    ax_score.axvline(
        p95,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.9,
        label=f"Seuil validation P95 = {p95:.4f}",
    )
    ax_score.axvline(
        p99,
        color="#0f172a",
        linestyle=":",
        linewidth=2.0,
        label=f"Seuil validation P99 = {p99:.4f}",
    )

    ax_score.set_title(
        "Scores d'anomalie Isolation Forest (-score_samples) — entraîné uniquement sur allow",
        fontsize=11.5,
        fontweight="bold",
        color="#0f172a",
        pad=8,
    )
    ax_score.set_xlabel("Score d'anomalie (plus élevé = plus atypique par rapport à allow)", fontsize=9.5, color="#334155")
    ax_score.set_ylabel("Densité relative", fontsize=9.5, color="#334155")
    ax_score.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8")
    ax_score.set_axisbelow(True)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    ax_score.legend(
        loc="upper right",
        fontsize=8.4,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#e2e8f0",
        framealpha=0.95,
    )

    fig.suptitle(
        "Comparaison trafic autorisé (allow) vs trafic refusé (deny / drop) — Bonus D7",
        fontsize=14.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    fig.text(
        0.5,
        0.946,
        (
            "Les flux deny/drop sont bloqués dès le 1er paquet (0 octet reçu, 1 paquet, 0 s)  •  "
            f"Score moyen : allow = {summary['allow_test']['mean']:.4f} vs "
            f"deny+drop = {summary['deny_drop']['mean']:.4f} (sans utiliser deny/drop à l'entraînement)"
        ),
        ha="center",
        fontsize=9.3,
        color="#475569",
    )

    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.06, right=0.98)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def run_exploration(path: Path = DATA_PATH) -> None:
    """Exécute le contrôle qualité complet et génère les graphiques dans artifacts/."""
    ctrl = first_control(path)
    print("=== 1. PREMIER CONTRÔLE (D2 — data/log2.csv) ===")
    print(f"Lignes totales : {ctrl['total_rows']:,} | Colonnes : {ctrl['n_columns']}")
    for action, count in ctrl["action_counts"].items():
        pct = 100.0 * count / ctrl["total_rows"]
        print(f"  - {action:12s}: {count:6d} ({pct:5.2f} %)")
    print(f"Valeurs manquantes (NaN, sur les 12 colonnes) : {ctrl['total_missing']}")

    audit = audit_cleaning(path)
    print("\n=== 2. NETTOYAGE & QUALITÉ (D3) ===")
    print(f"Valeurs manquantes (NaN)           : {audit['missing_total']}")
    print(f"Valeurs numériques illisibles      : {audit['unreadable_total']}")
    print(f"Valeurs négatives (< 0)            : {audit['negative_total']}")
    print(f"Lignes conservées après nettoyage  : {audit['rows_clean']:,} / {audit['rows_raw']:,}")
    print(
        f"Doublons exacts (12 col., total)   : {audit['exact_duplicates_all_cols']:,} "
        f"({100 * audit['exact_duplicates_all_cols'] / audit['rows_raw']:.2f} %)"
    )
    print(
        f"Doublons exacts (12 col., allow)   : {audit['exact_duplicates_allow_all_cols']:,}"
    )
    print(
        f"Doublons sur 4 mesures (allow)     : {audit['feature_duplicates_allow']:,} "
        f"({audit['unique_feature_profiles_allow']:,} profils uniques)"
    )
    print("Décision doublons : CONSERVÉS (sessions distinctes sans horodatage ni IP).")

    df_clean = load_and_clean(path)
    allow_df = df_clean.loc[df_clean[ACTION_COLUMN] == ALLOW_ACTION].copy()

    print("\n=== 3. QUANTILES DU TRAFIC ALLOW (n = 37 640) ===")
    quantiles = allow_df[FEATURES].quantile([0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    print(quantiles.to_string())

    typical_row = find_typical_allow_row(allow_df)
    payload = {col: int(typical_row[col]) for col in FEATURES}

    train_df, val_df, test_df = split_allow(df_clean, random_state=RANDOM_STATE)
    print(
        f"\n=== 4. DÉCOUPAGE ALLOW (D5) ===\n"
        f"train={len(train_df):,} (60 %) | val={len(val_df):,} (20 %) | test={len(test_df):,} (20 %)"
    )

    pipeline = _fit_reference_model(train_df)
    val_scores = -pipeline.score_samples(val_df[FEATURES])
    p95 = float(np.percentile(val_scores, 95))
    p99 = float(np.percentile(val_scores, 99))

    typical_df = from_api_payload(payload)
    typical_score = float(-pipeline.score_samples(typical_df[FEATURES])[0])
    typical_pct = float((val_scores <= typical_score).mean() * 100.0)

    print("\n=== 5. LIGNE ALLOW TYPIQUE POUR LA DÉMO (D6) ===")
    print(f"Index pandas : {int(typical_row.name)} (ligne CSV {int(typical_row.name) + 2})")
    print(json.dumps(payload, indent=2))
    print(
        f"Score Isolation Forest : {typical_score:.4f} (percentile val = {typical_pct:.1f} %, "
        f"seuil P95 = {p95:.4f}, seuil P99 = {p99:.4f}, alerte P99 = {typical_score > p99})"
    )

    allow_plot = plot_allow_traffic(allow_df, typical_row, TRAFFIC_ALLOW_PLOT)
    comp_summary = compare_allow_vs_refused(
        df_clean, pipeline, val_scores, test_df, COMPARISON_PLOT
    )

    print("\n=== 6. COMPARAISON ALLOW vs DENY/DROP (D7) ===")
    for group_name, stats in comp_summary.items():
        print(
            f"  - {group_name:10s} (n={int(stats['count']):6,d}) : "
            f"score moy={stats['mean']:.4f}, méd={stats['median']:.4f}, "
            f">P95={stats['pct_above_p95']:5.2f} %, >P99={stats['pct_above_p99']:5.2f} %"
        )
    print(f"\nGraphiques générés :\n  - {allow_plot}\n  - {COMPARISON_PLOT}")


if __name__ == "__main__":
    run_exploration()
