import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set(ENGLISH_STOP_WORDS)


def _normalize_tokens(text):
    text = "" if text is None else str(text).lower()
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t and t not in STOPWORDS]


def _normalize_text(text):
    return " ".join(_normalize_tokens(text))


def _theme_terms(theme):
    terms = []
    for key in ("keywords", "synonyms", "phrases"):
        terms.extend(theme.get(key, []) or [])
    if not terms:
        terms = [theme.get("name", "")]
    normalized = [_normalize_text(t) for t in terms]
    return sorted(set(t for t in normalized if t))


def score_event_against_lexicon(event_text, themes):
    """
    Score one event against all themes using TF-IDF-like weighting.
    - TF from event tokens
    - IDF from how many theme lexicons contain each token
    """
    event_tokens = _normalize_tokens(event_text)
    tf = Counter(event_tokens)
    n_themes = max(len(themes), 1)

    theme_token_sets = {}
    token_df = Counter()
    for theme in themes:
        tid = theme["id"]
        terms = _theme_terms(theme)
        toks = set(" ".join(terms).split())
        theme_token_sets[tid] = toks
        for tok in toks:
            token_df[tok] += 1

    def idf(tok):
        return math.log((1.0 + n_themes) / (1.0 + token_df.get(tok, 0))) + 1.0

    normalized_event = " ".join(event_tokens)
    scores = {}
    for theme in themes:
        tid = theme["id"]
        tok_set = theme_token_sets.get(tid, set())
        score = 0.0
        for tok, freq in tf.items():
            if tok in tok_set:
                score += float(freq) * idf(tok)

        # Phrase bonus helps explicit event/theme phrase matches.
        phrase_bonus = 0.0
        for phrase in _theme_terms(theme):
            if " " in phrase and phrase in normalized_event:
                phrase_bonus += 0.5
        scores[tid] = score + phrase_bonus

    return scores


def _extract_json_object(text):
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        text = text[:-3].strip() if text.endswith("```") else text
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _semantic_rerank_with_gemini(event_text, themes, base_scores, llm_config):
    provider = str(llm_config.get("provider", "")).lower()
    if provider not in {"google", "gemini", "google_gemini"}:
        return base_scores, {"enabled": False, "reason": f"provider_not_supported:{provider}"}

    try:
        from google import genai
    except Exception:
        return base_scores, {"enabled": False, "reason": "google_genai_not_installed"}

    model_name = llm_config.get("model", "gemini-2.5-flash")
    top_k = int(llm_config.get("semantic_top_k", 10))
    alpha = float(llm_config.get("semantic_blend_alpha", 0.35))
    top_themes = sorted(
        [
            {
                "theme_id": int(t["id"]),
                "name": t.get("name", str(t["id"])),
                "terms": _theme_terms(t)[:12],
                "base_score": float(base_scores.get(t["id"], 0.0)),
            }
            for t in themes
        ],
        key=lambda x: x["base_score"],
        reverse=True,
    )[:top_k]
    if not top_themes:
        return base_scores, {"enabled": True, "reason": "no_themes"}

    prompt = {
        "task": "Rate semantic relevance of each candidate theme for an event.",
        "instructions": [
            "Return valid JSON only.",
            "relevance must be in [0,1].",
            "Do not invent theme ids.",
        ],
        "event_text": event_text,
        "candidates": top_themes,
        "output_schema": {
            "theme_relevance": [
                {"theme_id": 0, "relevance": 0.0, "reason": "short phrase"}
            ]
        },
    }

    try:
        client = genai.Client()
        resp = client.models.generate_content(model=model_name, contents=json.dumps(prompt, ensure_ascii=True))
        raw_text = getattr(resp, "text", "") or ""
        parsed = _extract_json_object(raw_text)
        items = parsed.get("theme_relevance", []) if isinstance(parsed, dict) else []
        relevance = {}
        for item in items:
            try:
                tid = int(item["theme_id"])
                rel = float(item["relevance"])
                relevance[tid] = min(1.0, max(0.0, rel))
            except Exception:
                continue
    except Exception as exc:
        return base_scores, {"enabled": False, "reason": f"gemini_error:{exc}"}

    max_base = max([float(v) for v in base_scores.values()] + [1.0])
    blended = {}
    for t in themes:
        tid = int(t["id"])
        base = float(base_scores.get(tid, 0.0))
        rel = float(relevance.get(tid, 0.0))
        blended[tid] = base + (alpha * rel * max_base)

    return blended, {
        "enabled": True,
        "provider": provider,
        "model": model_name,
        "semantic_top_k": top_k,
        "semantic_blend_alpha": alpha,
        "relevance_scores": {str(k): float(v) for k, v in relevance.items()},
    }


def build_cluster_profiles(temporal_df, lookback=3):
    """
    Build one profile row per cluster from temporal stats.
    Output is the "database" used for event impact estimation.
    """
    if temporal_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "n_windows",
                "avg_post_count",
                "avg_volatility",
                "avg_market_share",
                "recent_growth",
                "recent_momentum",
                "recent_growth_lag1",
                "recent_growth_lag2",
                "recent_momentum_lag1",
                "recent_momentum_lag2",
                "trend_slope",
                "direction_score",
                "direction_label",
            ]
        )

    df = temporal_df.copy()
    df["time_window"] = pd.to_datetime(df["time_window"], errors="coerce")
    df = df.dropna(subset=["time_window"])
    rows = []

    for cluster_id, cdf in df.groupby("cluster"):
        cdf = cdf.sort_values("time_window")
        n = len(cdf)
        avg_post_count = float(cdf.get("post_count", pd.Series([0])).mean())
        avg_volatility = float(cdf.get("volume_volatility", pd.Series([0.0])).fillna(0).mean())
        avg_market_share = float(cdf.get("market_share", pd.Series([0.0])).fillna(0).mean())
        recent_growth = float(cdf.get("volume_pct_change", pd.Series([0.0])).tail(lookback).fillna(0).mean())
        recent_momentum = float(cdf.get("momentum", pd.Series([0.0])).tail(lookback).fillna(0).mean())
        growth_series = cdf.get("volume_pct_change", pd.Series([0.0])).fillna(0.0)
        momentum_series = cdf.get("momentum", pd.Series([0.0])).fillna(0.0)
        growth_lag1 = float(growth_series.iloc[-1]) if len(growth_series) >= 1 else 0.0
        growth_lag2 = float(growth_series.iloc[-2]) if len(growth_series) >= 2 else growth_lag1
        momentum_lag1 = float(momentum_series.iloc[-1]) if len(momentum_series) >= 1 else 0.0
        momentum_lag2 = float(momentum_series.iloc[-2]) if len(momentum_series) >= 2 else momentum_lag1

        y = cdf.get("post_count", pd.Series([0.0])).fillna(0).to_numpy(dtype=float)
        if len(y) > 1:
            x = np.arange(len(y), dtype=float)
            trend_slope = float(np.polyfit(x, y, 1)[0])
        else:
            trend_slope = 0.0

        momentum_norm = recent_momentum / max(avg_post_count, 1.0)
        slope_norm = trend_slope / max(avg_post_count, 1.0)
        direction_score = (0.5 * recent_growth) + (0.3 * momentum_norm) + (0.2 * slope_norm)

        if direction_score > 0.05:
            direction_label = "up"
        elif direction_score < -0.05:
            direction_label = "down"
        else:
            direction_label = "flat"

        rows.append(
            {
                "cluster": int(cluster_id),
                "n_windows": int(n),
                "avg_post_count": avg_post_count,
                "avg_volatility": avg_volatility,
                "avg_market_share": avg_market_share,
                "recent_growth": recent_growth,
                "recent_momentum": recent_momentum,
                "recent_growth_lag1": growth_lag1,
                "recent_growth_lag2": growth_lag2,
                "recent_momentum_lag1": momentum_lag1,
                "recent_momentum_lag2": momentum_lag2,
                "trend_slope": trend_slope,
                "direction_score": direction_score,
                "direction_label": direction_label,
            }
        )

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def estimate_event_direction(
    event_text,
    themes,
    cluster_profiles_df,
    activation_threshold=0.03,
    per_theme_thresholds=None,
    llm_config=None,
    theme_cluster_map=None,
):
    """
    Estimate likely direction and volatility for a new event.
    Uses theme_cluster_map to resolve theme → cluster; falls back to theme id.
    """
    base_scores = score_event_against_lexicon(event_text, themes)
    llm_meta = {"enabled": False}
    if llm_config and llm_config.get("enabled"):
        scores, llm_meta = _semantic_rerank_with_gemini(
            event_text=event_text,
            themes=themes,
            base_scores=base_scores,
            llm_config=llm_config,
        )
    else:
        scores = base_scores
    if cluster_profiles_df.empty:
        return {
            "predicted_direction": "flat",
            "predicted_direction_score": 0.0,
            "predicted_volatility": 0.0,
            "activated_theme_count": 0,
            "activated_themes": [],
            "theme_scores": {str(int(k)): float(v) for k, v in scores.items()},
            "llm_semantic": llm_meta,
        }

    profiles = cluster_profiles_df.copy()
    profiles["cluster"] = profiles["cluster"].astype(int)
    profile_by_cluster = profiles.set_index("cluster").to_dict(orient="index")

    thresholds = per_theme_thresholds or {}
    activated = []
    for theme in themes:
        tid = int(theme["id"])
        score = float(scores.get(tid, 0.0))
        threshold = float(
            thresholds.get(str(tid), thresholds.get(tid, activation_threshold))
        )
        if score < threshold:
            continue
        # Resolve cluster via explicit map, fall back to tid
        if theme_cluster_map and str(tid) in theme_cluster_map:
            mapped_cluster_id = theme_cluster_map[str(tid)].get("cluster_id", tid)
        else:
            mapped_cluster_id = tid

        profile = profile_by_cluster.get(mapped_cluster_id)
        if not profile:
            continue
        activated.append(
            {
                "theme_id": tid,
                "mapped_cluster_id": mapped_cluster_id,
                "theme_name": theme.get("name", str(tid)),
                "score": score,
                "direction_score": float(profile["direction_score"]),
                "direction_label": profile["direction_label"],
                "avg_volatility": float(profile["avg_volatility"]),
                "avg_market_share": float(profile.get("avg_market_share", 0.0)),
                "recent_momentum": float(profile.get("recent_momentum", 0.0)),
                "recent_growth_lag1": float(profile.get("recent_growth_lag1", 0.0)),
                "recent_growth_lag2": float(profile.get("recent_growth_lag2", 0.0)),
                "recent_momentum_lag1": float(profile.get("recent_momentum_lag1", 0.0)),
                "recent_momentum_lag2": float(profile.get("recent_momentum_lag2", 0.0)),
                "threshold": threshold,
            }
        )

    if not activated:
        return {
            "predicted_direction": "flat",
            "predicted_direction_score": 0.0,
            "predicted_volatility": float(profiles["avg_volatility"].mean()),
            "activated_theme_count": 0,
            "activated_themes": [],
            "theme_scores": {str(int(k)): float(v) for k, v in scores.items()},
            "llm_semantic": llm_meta,
        }

    total = sum(a["score"] for a in activated) or 1.0
    for a in activated:
        a["weight"] = a["score"] / total

    direction_score = sum(a["weight"] * a["direction_score"] for a in activated)
    predicted_volatility = sum(a["weight"] * a["avg_volatility"] for a in activated)
    if direction_score > 0.05:
        direction = "up"
    elif direction_score < -0.05:
        direction = "down"
    else:
        direction = "flat"

    activated_sorted = sorted(activated, key=lambda x: x["score"], reverse=True)
    return {
        "predicted_direction": direction,
        "predicted_direction_score": float(direction_score),
        "predicted_volatility": float(predicted_volatility),
        "activated_theme_count": len(activated_sorted),
        "activated_themes": activated_sorted,
        "theme_scores": {str(int(k)): float(v) for k, v in scores.items()},
        "llm_semantic": llm_meta,
    }
