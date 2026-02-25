import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_theme_cluster_map import build_theme_cluster_map, validate_map


def _make_activations():
    rows = []
    for tw in ["2024-01-31", "2024-02-29"]:
        rows += [
            {"cluster": 0, "cluster_label": "deepmind ai", "time_window": tw,
             "theme_0_score": 0.8, "theme_1_score": 0.1, "theme_2_score": 0.05},
            {"cluster": 1, "cluster_label": "protein folding", "time_window": tw,
             "theme_0_score": 0.1, "theme_1_score": 0.9, "theme_2_score": 0.05},
            {"cluster": 2, "cluster_label": "nhs data", "time_window": tw,
             "theme_0_score": 0.05, "theme_1_score": 0.05, "theme_2_score": 0.85},
        ]
    return pd.DataFrame(rows)


def test_build_theme_cluster_map_assigns_best_cluster():
    df = _make_activations()
    result = build_theme_cluster_map(df, theme_ids=[0, 1, 2])
    assert result[0]["cluster_id"] == 0
    assert result[1]["cluster_id"] == 1
    assert result[2]["cluster_id"] == 2


def test_build_theme_cluster_map_includes_avg_score():
    df = _make_activations()
    result = build_theme_cluster_map(df, theme_ids=[0])
    assert abs(result[0]["avg_score"] - 0.8) < 0.001


def test_validate_map_detects_unmatched_themes():
    mapping = {0: {"cluster_id": 5, "cluster_label": "x", "avg_score": 0.1}}
    warnings = validate_map(mapping, expected_theme_count=3)
    assert len(warnings) == 2
