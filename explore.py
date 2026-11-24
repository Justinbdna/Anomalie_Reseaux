"""Exploration et contrôle qualité des journaux pare-feu (D2, D4).

Exécuter avec :
    python explore.py

Produit :
- Un rapport complet sur la console (premier contrôle D2, audit nettoyage D3,
  quantiles du trafic allow, ligne typique pour la démo).
- Le graphique des volumes et durées du trafic autorisé : artifacts/trafic_allow.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from features import (
    ACTION_COLUMN,
    ALLOW_ACTION,
    DATA_PATH,
    FEATURES,
    audit_cleaning,
    find_typical_allow_row,
    load_and_clean,
    load_raw_logs,
)

ARTIFACTS_DIR = Path("artifacts")
TRAFFIC_ALLOW_PLOT = ARTIFACTS_DIR / "trafic_allow.png"


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


def run_exploration(path: Path = DATA_PATH) -> None:
    """Exécute le contrôle qualité complet et génère le graphique d'exploration."""
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
    print("\n=== 4. LIGNE ALLOW TYPIQUE POUR LA DÉMO ===")
    print(f"Index pandas : {int(typical_row.name)} (ligne CSV {int(typical_row.name) + 2})")
    print(json.dumps(payload, indent=2))

    out_path = plot_allow_traffic(allow_df, typical_row, TRAFFIC_ALLOW_PLOT)
    print(f"\n=== 5. GRAPHIQUE GÉNÉRÉ (D4) ===\n{out_path}")


if __name__ == "__main__":
    run_exploration()
