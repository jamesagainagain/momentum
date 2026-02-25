import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.event_forecast import estimate_event_direction


def _make_profiles():
    # Theme 4 maps to cluster 33 via the map — cluster 4 has no entry
    return pd.DataFrame([{
        "cluster": 33, "n_windows": 10, "avg_post_count": 100,
        "avg_volatility": 0.2, "avg_market_share": 0.05,
        "recent_growth": 0.3, "recent_momentum": 10.0,
        "recent_growth_lag1": 0.25, "recent_growth_lag2": 0.2,
        "recent_momentum_lag1": 8.0, "recent_momentum_lag2": 7.0,
        "trend_slope": 2.0, "direction_score": 0.3, "direction_label": "up",
    }])


def test_uses_cluster_map_not_theme_id():
    """Without map, theme 4 looks up cluster 4 (missing) → no activation.
    With map, it correctly uses cluster 33 → activates."""
    themes = [{
        "id": 4, "name": "Protein Folding",
        "keywords": ["alphafold", "protein", "folding"],
        "synonyms": [], "phrases": [],
    }]
    profiles = _make_profiles()
    theme_cluster_map = {
        "4": {"cluster_id": 33, "cluster_label": "alphafold", "avg_score": 0.8}
    }
    result = estimate_event_direction(
        event_text="AlphaFold predicts new protein structure",
        themes=themes,
        cluster_profiles_df=profiles,
        activation_threshold=1.0,
        theme_cluster_map=theme_cluster_map,
    )
    assert result["activated_theme_count"] == 1
    assert result["activated_themes"][0]["theme_id"] == 4
    assert result["activated_themes"][0]["mapped_cluster_id"] == 33


def test_falls_back_to_theme_id_when_no_map():
    """Without a map, behaviour is unchanged (uses tid as cluster id)."""
    themes = [{
        "id": 0, "name": "Test",
        "keywords": ["test", "example"],
        "synonyms": [], "phrases": [],
    }]
    profiles = pd.DataFrame([{
        "cluster": 0, "n_windows": 5, "avg_post_count": 50,
        "avg_volatility": 0.1, "avg_market_share": 0.02,
        "recent_growth": 0.1, "recent_momentum": 2.0,
        "recent_growth_lag1": 0.1, "recent_growth_lag2": 0.1,
        "recent_momentum_lag1": 2.0, "recent_momentum_lag2": 2.0,
        "trend_slope": 0.5, "direction_score": 0.1, "direction_label": "flat",
    }])
    result = estimate_event_direction(
        event_text="test example content",
        themes=themes,
        cluster_profiles_df=profiles,
        activation_threshold=0.1,
        theme_cluster_map=None,
    )
    assert result["activated_theme_count"] == 1
    assert result["activated_themes"][0]["mapped_cluster_id"] == 0
