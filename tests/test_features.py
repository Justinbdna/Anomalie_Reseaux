"""Tests du nettoyage et du découpage des journaux réseau."""

import pandas as pd

from features import FEATURES, load_and_clean, split_allow


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["10", "20", "allow"],
            ["bad", "20", "allow"],
            ["30", "-1", "allow"],
            ["40", "50", " DENY "],
        ],
        columns=["Bytes Sent", "Bytes Received", "Action"],
    ).assign(
        Packets=[1, 2, 3, 4],
        **{"Elapsed Time (sec)": [5, 6, 7, 8]},
    )


def test_load_and_clean_renomme_convertit_et_rejette_lignes_invalides(tmp_path):
    path = tmp_path / "logs.csv"
    _raw_rows().to_csv(path, index=False)

    cleaned = load_and_clean(path)

    assert list(cleaned.columns) == [*FEATURES, "Action"]
    assert len(cleaned) == 2
    assert cleaned["Action"].tolist() == ["allow", "deny"]
    assert cleaned["bytes_sent"].dtype.kind in "fi"
    assert (cleaned[FEATURES] >= 0).all().all()


def test_split_allow_est_reproductible_sans_recouvrement():
    rows = pd.DataFrame(
        {
            "bytes_sent": range(30),
            "bytes_received": range(100, 130),
            "packets": range(200, 230),
            "elapsed_time_sec": range(300, 330),
            "Action": ["allow"] * 24 + ["deny"] * 6,
        }
    )

    first = split_allow(rows, random_state=42)
    second = split_allow(rows, random_state=42)

    assert [len(part) for part in first] == [14, 5, 5]
    assert all(a.equals(b) for a, b in zip(first, second))
    signatures = [
        set(map(tuple, part[FEATURES].to_numpy())) for part in first
    ]
    assert signatures[0].isdisjoint(signatures[1])
    assert signatures[0].isdisjoint(signatures[2])
    assert signatures[1].isdisjoint(signatures[2])
    assert all((part["Action"] == "allow").all() for part in first)
