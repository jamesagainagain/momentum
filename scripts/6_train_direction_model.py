"""
Script 6: Train directional forecast model.

Builds per-cluster/window supervised features and trains models to
predict next-window direction (down/flat/up).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.forecast_model import (
        DIRECTION_LABELS,
        build_training_frame,
        heuristic_predict_direction,
        pick_label_from_proba,
    )
    from scripts.io_utils import (
        build_temporal_fallback_from_clusters_csv,
        read_parquet_safe,
    )
else:
    from .forecast_model import (
        DIRECTION_LABELS,
        build_training_frame,
        heuristic_predict_direction,
        pick_label_from_proba,
    )
    from .io_utils import build_temporal_fallback_from_clusters_csv, read_parquet_safe


def _progress(step, total, label, start_time=None, width=30):
    filled = int(width * step / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    elapsed = f"  {time.time() - start_time:.0f}s" if start_time else ""
    print(f"\r[{bar}] {step}/{total}  {label}{elapsed}", end="", flush=True)
    if step >= total:
        print()


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_theme_windows_or_regenerate(theme_windows_path, config_path):
    if os.path.exists(theme_windows_path):
        return read_parquet_safe(theme_windows_path), False

    script_path = os.path.join(os.path.dirname(__file__), "4_theme_activation_tfidf.py")
    cmd = [sys.executable, script_path, "--config", config_path, "--mode", "windows"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and os.path.exists(theme_windows_path):
        return read_parquet_safe(theme_windows_path), True
    print("WARN: Theme activations unavailable; training without theme_*_score features.")
    if proc.returncode != 0:
        print(f"WARN: Theme regeneration failed: {proc.stderr.strip()}")
    return pd.DataFrame(), False


def model_candidates(model_names, random_state):
    out = {}
    skipped = {}
    names = {m.strip().lower() for m in model_names}
    if "logistic" in names or "logistic_regression" in names:
        out["logistic"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if "random_forest" in names or "rf" in names:
        out["random_forest"] = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=8,
            max_leaf_nodes=80,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    if "extra_trees" in names or "et" in names:
        out["extra_trees"] = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=8,
            max_leaf_nodes=80,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if "hist_gbm" in names or "hist_gradient_boosting" in names:
        out["hist_gbm"] = HistGradientBoostingClassifier(
            max_depth=4,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=random_state,
        )
    if "xgboost" in names or "xgb" in names:
        try:
            from xgboost import XGBClassifier

            out["xgboost"] = XGBClassifier(
                objective="multi:softprob",
                num_class=len(DIRECTION_LABELS),
                n_estimators=200,
                max_depth=4,
                min_child_weight=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=random_state,
                eval_metric="mlogloss",
                nthread=-1,
            )
        except Exception as exc:
            skipped["xgboost"] = str(exc)
    if "lightgbm" in names or "lgbm" in names:
        try:
            from lightgbm import LGBMClassifier

            out["lightgbm"] = LGBMClassifier(
                objective="multiclass",
                num_class=len(DIRECTION_LABELS),
                n_estimators=200,
                max_depth=4,
                num_leaves=15,
                min_child_samples=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        except Exception as exc:
            skipped["lightgbm"] = str(exc)
    if "catboost" in names or "cat" in names:
        try:
            from catboost import CatBoostClassifier

            out["catboost"] = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=500,
                learning_rate=0.05,
                depth=6,
                min_data_in_leaf=5,
                l2_leaf_reg=3.0,
                random_seed=random_state,
                verbose=False,
            )
        except Exception as exc:
            skipped["catboost"] = str(exc)
    return out, skipped


def evaluate(y_true, y_pred):
    per_class_precision = precision_score(y_true, y_pred, labels=DIRECTION_LABELS, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=DIRECTION_LABELS, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=DIRECTION_LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=DIRECTION_LABELS, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }
            for i, label in enumerate(DIRECTION_LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=DIRECTION_LABELS).tolist(),
        "labels": DIRECTION_LABELS,
    }


def choose_best_model(results):
    ranked = sorted(
        [(name, metrics) for name, metrics in results.items() if name != "heuristic"],
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]),
        reverse=True,
    )
    return ranked[0][0] if ranked else None


def generate_walk_forward_folds(times, n_splits, min_train_windows, purge_gap=1):
    folds = []
    if len(times) <= min_train_windows:
        return folds
    max_splits = max(1, int(n_splits))
    step = max(1, (len(times) - min_train_windows) // max_splits)
    train_end = min_train_windows
    while train_end < len(times):
        test_start = min(len(times), train_end + purge_gap)
        test_end = min(len(times), test_start + step)
        if test_end <= test_start:
            break
        folds.append((times[:train_end], times[test_start:test_end]))
        train_end = test_end
        if len(folds) >= max_splits:
            break
    return folds


def calibrate_if_enabled(estimator, X_train, y_train, enabled):
    if not enabled:
        return estimator, False
    y_counts = pd.Series(y_train).value_counts()
    if y_counts.empty or int(y_counts.min()) < 2:
        return estimator, False
    try:
        cv = int(min(3, y_counts.min()))
        if cv < 2:
            return estimator, False
        calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        calibrated.fit(X_train, y_train)
        return calibrated, True
    except Exception:
        return estimator, False


def predict_proba_frame(estimator, X, labels):
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(X)
        classes = list(getattr(estimator, "classes_", labels))
        rows = []
        for row in raw:
            probs = {label: 0.0 for label in labels}
            for idx, c in enumerate(classes):
                label_key = None
                try:
                    c_int = int(c)
                    if 0 <= c_int < len(labels):
                        label_key = labels[c_int]
                except Exception:
                    pass
                if label_key is None:
                    c_str = str(c)
                    if c_str in labels:
                        label_key = c_str
                if label_key is not None:
                    probs[label_key] = float(row[idx])
            rows.append(probs)
        return rows
    preds = estimator.predict(X)
    rows = []
    for p in preds:
        row = {label: 0.0 for label in labels}
        row[str(p)] = 1.0
        rows.append(row)
    return rows


def ensemble_proba_rows(member_rows, labels):
    n = len(member_rows[0]) if member_rows else 0
    out = []
    for i in range(n):
        avg = {label: 0.0 for label in labels}
        for rows in member_rows:
            for label in labels:
                avg[label] += float(rows[i].get(label, 0.0))
        m = max(1, len(member_rows))
        out.append({label: avg[label] / m for label in labels})
    return out


def proba_rows_to_preds(proba_rows, labels, threshold_multipliers):
    preds = []
    adjusted_rows = []
    for row in proba_rows:
        pred, adjusted = pick_label_from_proba(
            proba_map=row,
            threshold_multipliers=threshold_multipliers,
            labels=labels,
        )
        preds.append(pred)
        adjusted_rows.append(adjusted)
    return preds, adjusted_rows


def _unwrap_estimator(estimator):
    """Unwrap CalibratedClassifierCV, averaging importances across CV folds."""
    if hasattr(estimator, "calibrated_classifiers_"):
        inner_ests = [cc.estimator for cc in estimator.calibrated_classifiers_]
        # Average feature_importances_ across all calibration folds
        importances = [
            e.feature_importances_ for e in inner_ests
            if hasattr(e, "feature_importances_")
        ]
        if importances:
            avg = np.mean(importances, axis=0)
            # Return a lightweight object with the averaged importances
            wrapper = type("_AvgImportances", (), {"feature_importances_": avg})()
            return wrapper
        # Fall back to first fold for coef_-based models
        return inner_ests[0]
    return estimator


def extract_feature_importances(estimator, feature_cols):
    est = _unwrap_estimator(estimator)
    if hasattr(est, "feature_importances_"):
        raw = est.feature_importances_
    elif hasattr(est, "named_steps"):
        clf = est.named_steps.get("clf", est)
        clf = _unwrap_estimator(clf)
        if hasattr(clf, "feature_importances_"):
            raw = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            raw = np.abs(clf.coef_).mean(axis=0)
        else:
            return {}
    elif hasattr(est, "coef_"):
        raw = np.abs(est.coef_).mean(axis=0)
    else:
        return {}
    return {col: float(raw[i]) for i, col in enumerate(feature_cols) if i < len(raw)}


def prune_features(importances, feature_cols, drop_fraction=0.25):
    if not importances:
        return feature_cols
    sorted_feats = sorted(importances.items(), key=lambda x: x[1])
    n_drop = max(1, int(len(sorted_feats) * drop_fraction))
    drop_set = {f for f, _ in sorted_feats[:n_drop]}
    pruned = [c for c in feature_cols if c not in drop_set]
    return pruned if pruned else feature_cols


def build_stacking_meta_features(fitted_models, X, labels):
    cols = []
    arrays = []
    for name in sorted(fitted_models.keys()):
        est = fitted_models[name]
        proba_rows = predict_proba_frame(est, X, labels)
        for label in labels:
            col_name = f"{name}_prob_{label}"
            cols.append(col_name)
            arrays.append([row.get(label, 0.0) for row in proba_rows])
    meta_df = pd.DataFrame(dict(zip(cols, arrays)), index=X.index)
    return meta_df, cols


def train_stacking_ensemble(
    train_df, feature_cols, estimators, labels, numeric_target_models,
    label_to_int, calibration_enabled, random_state, n_meta_folds=3,
):
    train_times = sorted(train_df["time_window"].unique())
    if len(train_times) < n_meta_folds + 1:
        return None, None, None

    oof_meta = pd.DataFrame(index=train_df.index)
    fold_size = max(1, len(train_times) // (n_meta_folds + 1))
    folds = []
    for i in range(n_meta_folds):
        fold_train_end = fold_size * (i + 1)
        fold_test_start = fold_train_end + 1
        fold_test_end = min(len(train_times), fold_test_start + fold_size)
        if fold_test_start >= len(train_times) or fold_test_end <= fold_test_start:
            continue
        folds.append((train_times[:fold_train_end], train_times[fold_test_start:fold_test_end]))

    if not folds:
        return None, None, None

    meta_cols = []
    for label in labels:
        for name in sorted(estimators.keys()):
            meta_cols.append(f"{name}_prob_{label}")

    for col in meta_cols:
        oof_meta[col] = np.nan

    for fold_train_times, fold_test_times in folds:
        fold_train = train_df[train_df["time_window"].isin(fold_train_times)]
        fold_test = train_df[train_df["time_window"].isin(fold_test_times)]
        if fold_train.empty or fold_test.empty:
            continue
        X_ft = fold_train[feature_cols].fillna(0.0)
        X_fv = fold_test[feature_cols].fillna(0.0)
        y_ft = fold_train["target"]
        for name in sorted(estimators.keys()):
            est = copy.deepcopy(estimators[name])
            y_fit = y_ft.map(label_to_int).astype(int) if name in numeric_target_models else y_ft
            if name in numeric_target_models and len(y_fit.unique()) < len(labels):
                for label in labels:
                    oof_meta.loc[fold_test.index, f"{name}_prob_{label}"] = 1.0 / len(labels)
                continue
            est.fit(X_ft, y_fit)
            est, _ = calibrate_if_enabled(est, X_ft, y_fit, calibration_enabled)
            proba_rows = predict_proba_frame(est, X_fv, labels)
            for label in labels:
                col = f"{name}_prob_{label}"
                oof_meta.loc[fold_test.index, col] = [row.get(label, 0.0) for row in proba_rows]

    valid_rows = oof_meta.dropna()
    if len(valid_rows) < 10:
        return None, None, None

    y_meta = train_df.loc[valid_rows.index, "target"]
    X_meta = valid_rows[meta_cols].astype(float)

    meta_learner = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            C=1.0,
        )),
    ])
    meta_learner.fit(X_meta, y_meta)
    return meta_learner, meta_cols, sorted(estimators.keys())


def _evaluate_fold(fold_data):
    (train_times, test_times, frame_eps, feature_cols,
     estimators, numeric_target_models, label_to_int,
     calibration_enabled, epsilon) = fold_data
    train_df = frame_eps[frame_eps["time_window"].isin(train_times)]
    test_df = frame_eps[frame_eps["time_window"].isin(test_times)]
    if train_df.empty or test_df.empty:
        return None
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["target"]
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["target"]

    fold_metrics = {}
    member_probs = []
    for name, estimator in estimators.items():
        est = copy.deepcopy(estimator)
        if hasattr(est, "n_jobs"):
            est.set_params(n_jobs=1)
        elif hasattr(est, "nthread"):
            est.set_params(nthread=1)
        y_fit = y_train.map(label_to_int).astype(int) if name in numeric_target_models else y_train
        if name in numeric_target_models and len(y_fit.unique()) < len(DIRECTION_LABELS):
            fold_metrics[name] = evaluate(y_test, pd.Series(["flat"] * len(y_test), index=y_test.index))
            continue
        est.fit(X_train, y_fit)
        est, _ = calibrate_if_enabled(est, X_train, y_fit, calibration_enabled)
        proba_rows = predict_proba_frame(est, X_test, DIRECTION_LABELS)
        preds, _ = proba_rows_to_preds(
            proba_rows=proba_rows,
            labels=DIRECTION_LABELS,
            threshold_multipliers={"down": 1.0, "flat": 1.0, "up": 1.0},
        )
        fold_metrics[name] = evaluate(y_test, preds)
        member_probs.append(proba_rows)

    if len(member_probs) >= 2:
        ensemble_probs = ensemble_proba_rows(member_probs, DIRECTION_LABELS)
        ensemble_preds, _ = proba_rows_to_preds(
            proba_rows=ensemble_probs,
            labels=DIRECTION_LABELS,
            threshold_multipliers={"down": 1.0, "flat": 1.0, "up": 1.0},
        )
        fold_metrics["soft_voting_ensemble"] = evaluate(y_test, ensemble_preds)

    heur = test_df.apply(lambda row: heuristic_predict_direction(row, epsilon=epsilon), axis=1)
    fold_metrics["heuristic"] = evaluate(y_test, heur)
    return fold_metrics


def default_threshold_candidates():
    return [
        {"down": 1.0, "flat": 1.0, "up": 1.0},
        {"down": 1.05, "flat": 1.0, "up": 1.0},
        {"down": 1.0, "flat": 1.05, "up": 1.0},
        {"down": 1.0, "flat": 1.0, "up": 1.05},
        {"down": 0.95, "flat": 1.0, "up": 1.0},
        {"down": 1.0, "flat": 1.0, "up": 0.95},
    ]


def normalize_threshold_candidates(values):
    if not values:
        return default_threshold_candidates()
    out = []
    for row in values:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "down": float(row.get("down", 1.0)),
                "flat": float(row.get("flat", 1.0)),
                "up": float(row.get("up", 1.0)),
            }
        )
    return out or default_threshold_candidates()


def main():
    parser = argparse.ArgumentParser(description="Train direction model on historical cluster-window features.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--theme-activations-path", default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    config_dir = os.path.dirname(config_path)

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(config_dir, path)

    training_cfg = config.get("model_training", {})
    forecast_cfg = config.get("event_forecast", {})
    test_fraction = float(training_cfg.get("test_fraction", 0.2))
    base_epsilon = float(training_cfg.get("target_epsilon", forecast_cfg.get("target_epsilon", 0.05)))
    random_state = int(training_cfg.get("random_state", 42))
    calibration_enabled = bool(training_cfg.get("calibrate_probabilities", True))
    walk_forward_splits = int(training_cfg.get("walk_forward_splits", 4))
    min_train_windows = int(training_cfg.get("walk_forward_min_train_windows", 6))
    epsilon_grid = [float(x) for x in training_cfg.get("target_epsilon_grid", [0.03, base_epsilon, 0.08])]
    threshold_candidates = normalize_threshold_candidates(training_cfg.get("class_threshold_candidates", []))

    temporal_stats_path = resolve(config["output"]["temporal_stats"])
    clusters_csv_path = resolve(config["input"]["data"])
    ts_col = config["input"]["timestamp_column"]
    default_window = config.get("temporal", {}).get("default_window", "1M")

    theme_out_base = resolve(config.get("theme_activation", {}).get("output", "output/theme_activations.parquet"))
    theme_windows_path = args.theme_activations_path or theme_out_base.replace(".parquet", "_windows.parquet")

    model_path = resolve(training_cfg.get("output_model_path", "output/models/direction_model.pkl"))
    metrics_path = resolve(training_cfg.get("output_metrics_path", "output/models/direction_model_metrics.json"))
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    if os.path.exists(model_path) and not args.force:
        print(f"Model exists: {model_path} (use --force to retrain)")
        return

    if os.path.exists(temporal_stats_path):
        try:
            temporal_df = read_parquet_safe(temporal_stats_path)
        except Exception as exc:
            print(f"WARN: Could not read temporal stats parquet. Falling back to CSV-derived stats ({exc})")
            temporal_df = build_temporal_fallback_from_clusters_csv(
                clusters_csv_path,
                ts_col=ts_col,
                window=default_window,
            )
    else:
        print("WARN: temporal stats missing; deriving fallback temporal stats from CSV.")
        temporal_df = build_temporal_fallback_from_clusters_csv(
            clusters_csv_path,
            ts_col=ts_col,
            window=default_window,
        )
    theme_df, regenerated = load_theme_windows_or_regenerate(theme_windows_path, config_path)
    if regenerated:
        print(f"Regenerated theme activations at {theme_windows_path}")

    configured_models = training_cfg.get("models", ["logistic", "random_forest"])
    estimators, skipped_optional = model_candidates(configured_models, random_state=random_state)
    if not estimators:
        raise RuntimeError("No valid models configured. Use logistic and/or random_forest.")
    numeric_target_models = {"xgboost", "lightgbm"}
    label_to_int = {label: idx for idx, label in enumerate(DIRECTION_LABELS)}

    # Epsilon tuning + walk-forward model ranking
    epsilon_cv = {}
    best_epsilon = base_epsilon
    best_cv_score = -1.0
    walk_forward_summary = {"enabled": True, "fold_count": 0, "by_epsilon": {}}

    max_cores_cfg = int(config.get("max_cores", 0))
    cv_n_jobs = max_cores_cfg if max_cores_cfg > 0 else -1

    epsilon_list = sorted(set(epsilon_grid))
    n_models = len(estimators)
    total_cv_steps = len(epsilon_list) * walk_forward_splits
    cv_step = 0
    t_start = time.time()
    print(f"Step 1/3  Walk-forward CV  ({len(epsilon_list)} epsilons × {walk_forward_splits} folds × {n_models} models)")

    for epsilon in epsilon_list:
        frame_eps, feature_cols = build_training_frame(
            temporal_df=temporal_df,
            theme_df=theme_df,
            epsilon=epsilon,
            max_lag=int(training_cfg.get("max_lag", 4)),
        )
        if len(frame_eps) < 8:
            continue
        times_eps = sorted(frame_eps["time_window"].unique())
        folds = generate_walk_forward_folds(
            times=times_eps,
            n_splits=walk_forward_splits,
            min_train_windows=min_train_windows,
        )
        if not folds:
            continue

        fold_inputs = [
            (train_times, test_times, frame_eps, feature_cols,
             estimators, numeric_target_models, label_to_int,
             calibration_enabled, epsilon)
            for train_times, test_times in folds
        ]
        _progress(cv_step, total_cv_steps, f"ε={epsilon}", t_start)
        raw_results = joblib.Parallel(n_jobs=cv_n_jobs, prefer="processes")(
            joblib.delayed(_evaluate_fold)(fd) for fd in fold_inputs
        )
        fold_scores = [r for r in raw_results if r is not None]
        cv_step += len(fold_inputs)
        _progress(cv_step, total_cv_steps, f"ε={epsilon} done", t_start)

        if not fold_scores:
            continue

        agg = {}
        model_names = set()
        for fold in fold_scores:
            model_names.update(fold.keys())
        for model_name in sorted(model_names):
            vals = [f[model_name] for f in fold_scores if model_name in f]
            if not vals:
                continue
            agg[model_name] = {
                "accuracy": float(np.mean([v["accuracy"] for v in vals])),
                "macro_f1": float(np.mean([v["macro_f1"] for v in vals])),
                "folds_used": len(vals),
            }
        walk_forward_summary["by_epsilon"][str(epsilon)] = agg
        walk_forward_summary["fold_count"] = max(
            int(walk_forward_summary.get("fold_count", 0)),
            len(fold_scores),
        )

        best_name_eps = choose_best_model(agg)
        if best_name_eps:
            score = float(agg[best_name_eps]["macro_f1"])
            epsilon_cv[str(epsilon)] = {"best_model": best_name_eps, "best_macro_f1": score}
            if score > best_cv_score:
                best_cv_score = score
                best_epsilon = epsilon

    if not walk_forward_summary["by_epsilon"]:
        walk_forward_summary = {"enabled": False, "reason": "insufficient_windows_for_walk_forward"}
        epsilon_cv["fallback"] = {"best_model": "n/a", "best_macro_f1": 0.0}

    epsilon = float(best_epsilon)
    frame, feature_cols = build_training_frame(
        temporal_df=temporal_df,
        theme_df=theme_df,
        epsilon=epsilon,
        max_lag=int(training_cfg.get("max_lag", 4)),
    )
    if len(frame) < 6:
        raise RuntimeError("Not enough training rows after target shift. Need at least 6 rows.")

    times = sorted(frame["time_window"].unique())
    split_idx = max(1, min(len(times) - 1, int(len(times) * (1.0 - test_fraction))))
    cutoff = times[split_idx - 1]
    train_df = frame[frame["time_window"] <= cutoff].copy()
    test_df = frame[frame["time_window"] > cutoff].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Time split produced empty train/test set. Adjust test_fraction.")

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["target"]
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["target"]

    results = {}
    fitted = {}
    calibration_info = {}
    member_prob_rows = []
    model_names_list = list(estimators.keys())
    print(f"Step 2/3  Final model fit  ({len(model_names_list)} models, ε={epsilon})")
    for i, (name, estimator) in enumerate(estimators.items()):
        _progress(i, len(model_names_list), name, t_start)
        est = copy.deepcopy(estimator)
        y_fit = y_train.map(label_to_int).astype(int) if name in numeric_target_models else y_train
        est.fit(X_train, y_fit)
        est, calibrated = calibrate_if_enabled(est, X_train, y_fit, calibration_enabled)
        calibration_info[name] = bool(calibrated)
        fitted[name] = est
        proba_rows = predict_proba_frame(est, X_test, DIRECTION_LABELS)
        member_prob_rows.append((name, proba_rows))
    _progress(len(model_names_list), len(model_names_list), "done", t_start)

    # Threshold tuning on holdout
    threshold_eval = {}
    tuned_predictions = {}
    for name, proba_rows in member_prob_rows:
        best_metrics = None
        best_thresholds = None
        for cand in threshold_candidates:
            preds, _ = proba_rows_to_preds(
                proba_rows=proba_rows,
                labels=DIRECTION_LABELS,
                threshold_multipliers=cand,
            )
            metrics = evaluate(y_test, preds)
            if (
                best_metrics is None
                or metrics["macro_f1"] > best_metrics["macro_f1"]
                or (
                    metrics["macro_f1"] == best_metrics["macro_f1"]
                    and metrics["accuracy"] > best_metrics["accuracy"]
                )
            ):
                best_metrics = metrics
                best_thresholds = cand
                tuned_predictions[name] = preds
        results[name] = best_metrics
        threshold_eval[name] = {"best_thresholds": best_thresholds}

    print(f"Step 3/3  Ensemble, pruning, diagnostics")
    stacking_meta_learner = None
    stacking_meta_cols = None
    stacking_base_names = None
    if len(member_prob_rows) >= 2:
        ensemble_rows = ensemble_proba_rows([rows for _, rows in member_prob_rows], DIRECTION_LABELS)
        best_metrics = None
        best_thresholds = None
        best_preds = None
        for cand in threshold_candidates:
            preds, _ = proba_rows_to_preds(
                proba_rows=ensemble_rows,
                labels=DIRECTION_LABELS,
                threshold_multipliers=cand,
            )
            metrics = evaluate(y_test, preds)
            if (
                best_metrics is None
                or metrics["macro_f1"] > best_metrics["macro_f1"]
                or (
                    metrics["macro_f1"] == best_metrics["macro_f1"]
                    and metrics["accuracy"] > best_metrics["accuracy"]
                )
            ):
                best_metrics = metrics
                best_thresholds = cand
                best_preds = preds
        results["soft_voting_ensemble"] = best_metrics
        threshold_eval["soft_voting_ensemble"] = {"best_thresholds": best_thresholds}
        tuned_predictions["soft_voting_ensemble"] = best_preds

        stacking_result = train_stacking_ensemble(
            train_df=train_df,
            feature_cols=feature_cols,
            estimators=estimators,
            labels=DIRECTION_LABELS,
            numeric_target_models=numeric_target_models,
            label_to_int=label_to_int,
            calibration_enabled=calibration_enabled,
            random_state=random_state,
        )
        if stacking_result[0] is not None:
            stacking_meta_learner, stacking_meta_cols, stacking_base_names = stacking_result
            test_meta, _ = build_stacking_meta_features(fitted, X_test, DIRECTION_LABELS)
            test_meta = test_meta[stacking_meta_cols].astype(float)
            stacking_proba = stacking_meta_learner.predict_proba(test_meta)
            stacking_classes = list(stacking_meta_learner.classes_)
            stacking_proba_rows = []
            for row in stacking_proba:
                probs = {label: 0.0 for label in DIRECTION_LABELS}
                for idx, c in enumerate(stacking_classes):
                    if c in DIRECTION_LABELS:
                        probs[c] = float(row[idx])
                stacking_proba_rows.append(probs)
            best_metrics = None
            best_thresholds = None
            best_preds = None
            for cand in threshold_candidates:
                preds, _ = proba_rows_to_preds(
                    proba_rows=stacking_proba_rows,
                    labels=DIRECTION_LABELS,
                    threshold_multipliers=cand,
                )
                metrics = evaluate(y_test, preds)
                if (
                    best_metrics is None
                    or metrics["macro_f1"] > best_metrics["macro_f1"]
                    or (
                        metrics["macro_f1"] == best_metrics["macro_f1"]
                        and metrics["accuracy"] > best_metrics["accuracy"]
                    )
                ):
                    best_metrics = metrics
                    best_thresholds = cand
                    best_preds = preds
            results["stacking_ensemble"] = best_metrics
            threshold_eval["stacking_ensemble"] = {"best_thresholds": best_thresholds}
            tuned_predictions["stacking_ensemble"] = best_preds
            print(f"Stacking ensemble: accuracy={best_metrics['accuracy']:.4f}, macro_f1={best_metrics['macro_f1']:.4f}")

    heuristic_preds = test_df.apply(lambda row: heuristic_predict_direction(row, epsilon=epsilon), axis=1)
    results["heuristic"] = evaluate(y_test, heuristic_preds)

    best_name = choose_best_model(results)
    if best_name is None:
        raise RuntimeError("Failed to select a best model.")

    feature_importance_map = {}
    pruning_report = {"attempted": False}
    if best_name in fitted and best_name not in ("soft_voting_ensemble", "stacking_ensemble"):
        importances = extract_feature_importances(fitted[best_name], feature_cols)
        feature_importance_map = importances
        pruned_cols = prune_features(importances, feature_cols, drop_fraction=0.25)
        if len(pruned_cols) < len(feature_cols):
            pruning_report["attempted"] = True
            pruning_report["original_count"] = len(feature_cols)
            pruning_report["pruned_count"] = len(pruned_cols)
            pruning_report["dropped"] = [c for c in feature_cols if c not in pruned_cols]
            X_train_p = train_df[pruned_cols].fillna(0.0)
            X_test_p = test_df[pruned_cols].fillna(0.0)
            est_p = copy.deepcopy(estimators[best_name])
            y_fit_p = y_train.map(label_to_int).astype(int) if best_name in numeric_target_models else y_train
            est_p.fit(X_train_p, y_fit_p)
            est_p, cal_p = calibrate_if_enabled(est_p, X_train_p, y_fit_p, calibration_enabled)
            proba_p = predict_proba_frame(est_p, X_test_p, DIRECTION_LABELS)
            selected_thresh = threshold_eval.get(best_name, {}).get("best_thresholds", {"down": 1.0, "flat": 1.0, "up": 1.0})
            preds_p, _ = proba_rows_to_preds(proba_p, DIRECTION_LABELS, selected_thresh)
            metrics_p = evaluate(y_test, preds_p)
            pruning_report["pruned_macro_f1"] = metrics_p["macro_f1"]
            pruning_report["pruned_accuracy"] = metrics_p["accuracy"]
            pruning_report["original_macro_f1"] = results[best_name]["macro_f1"]
            force_prune = bool(training_cfg.get("force_feature_prune", False))
            max_f1_drop = float(training_cfg.get("force_prune_max_f1_drop", 0.0))
            f1_drop = results[best_name]["macro_f1"] - metrics_p["macro_f1"]
            accept_pruned = (
                metrics_p["macro_f1"] >= results[best_name]["macro_f1"]
                or (force_prune and f1_drop <= max_f1_drop)
            )
            if accept_pruned:
                pruning_report["accepted"] = True
                if f1_drop > 0:
                    pruning_report["accepted_via_force_prune"] = True
                    pruning_report["f1_drop_accepted"] = float(f1_drop)
                feature_cols = pruned_cols
                fitted[best_name] = est_p
                calibration_info[best_name] = bool(cal_p)
                results[best_name] = metrics_p
                print(f"Feature pruning accepted: {pruning_report['original_count']} -> {pruning_report['pruned_count']} features"
                      + (f" (force_prune, F1 drop {f1_drop:.2%})" if f1_drop > 0 else ""))
            else:
                pruning_report["accepted"] = False
                print(f"Feature pruning rejected: pruned F1 {metrics_p['macro_f1']:.4f} < original {results[best_name]['macro_f1']:.4f}")

    selected_thresholds = threshold_eval.get(best_name, {}).get(
        "best_thresholds",
        {"down": 1.0, "flat": 1.0, "up": 1.0},
    )
    artifact = {
        "model_name": best_name,
        "feature_columns": feature_cols,
        "labels": DIRECTION_LABELS,
        "target_epsilon": epsilon,
        "class_thresholds": selected_thresholds,
    }
    if best_name == "stacking_ensemble" and stacking_meta_learner is not None:
        artifact["stacking_meta_learner"] = stacking_meta_learner
        artifact["stacking_meta_cols"] = stacking_meta_cols
        artifact["ensemble_members"] = [
            {"name": name, "estimator": fitted[name]} for name in sorted(fitted.keys())
        ]
    elif best_name == "soft_voting_ensemble":
        artifact["ensemble_members"] = [
            {"name": name, "estimator": fitted[name]} for name in sorted(fitted.keys())
        ]
    else:
        artifact["model"] = fitted[best_name]

    low_q_cfg = float(forecast_cfg.get("risk_band_quantiles", {}).get("low", 0.33))
    high_q_cfg = float(forecast_cfg.get("risk_band_quantiles", {}).get("high", 0.66))
    vol = pd.to_numeric(temporal_df.get("volume_volatility", pd.Series([0.0])), errors="coerce").fillna(0.0)
    low_q = float(vol.quantile(low_q_cfg))
    high_q = float(vol.quantile(high_q_cfg))

    artifact["risk_bands"] = {
        "low_quantile": low_q_cfg,
        "high_quantile": high_q_cfg,
        "low_value": low_q,
        "high_value": high_q,
    }
    joblib.dump(artifact, model_path)

    # --- Overfitting diagnostics ---
    train_eval = {}
    if best_name in fitted and best_name not in ("soft_voting_ensemble", "stacking_ensemble"):
        train_proba = predict_proba_frame(fitted[best_name], X_train, DIRECTION_LABELS)
        train_preds, _ = proba_rows_to_preds(train_proba, DIRECTION_LABELS, selected_thresholds)
        train_eval = evaluate(y_train, train_preds)

    best_cv_macro_f1 = None
    if walk_forward_summary.get("enabled") and walk_forward_summary.get("by_epsilon"):
        eps_key = str(epsilon)
        cv_by_eps = walk_forward_summary["by_epsilon"].get(eps_key, {})
        if best_name in cv_by_eps:
            best_cv_macro_f1 = cv_by_eps[best_name].get("macro_f1")
        elif cv_by_eps:
            best_cv_name = max(cv_by_eps, key=lambda k: cv_by_eps[k].get("macro_f1", 0) if k != "heuristic" else 0)
            if best_cv_name != "heuristic":
                best_cv_macro_f1 = cv_by_eps[best_cv_name].get("macro_f1")

    overfit_diagnosis = {
        "train_accuracy": train_eval.get("accuracy"),
        "train_macro_f1": train_eval.get("macro_f1"),
        "test_accuracy": results[best_name]["accuracy"],
        "test_macro_f1": results[best_name]["macro_f1"],
        "walk_forward_cv_macro_f1": best_cv_macro_f1,
    }

    if train_eval:
        train_test_gap = train_eval["macro_f1"] - results[best_name]["macro_f1"]
        overfit_diagnosis["train_test_f1_gap"] = float(train_test_gap)
        if train_test_gap > 0.15:
            overfit_diagnosis["verdict"] = "likely_overfitting"
            overfit_diagnosis["detail"] = (
                f"Train F1 ({train_eval['macro_f1']:.2%}) exceeds test F1 "
                f"({results[best_name]['macro_f1']:.2%}) by {train_test_gap:.2%}. "
                "Consider reducing model complexity or adding regularization."
            )
        elif train_test_gap > 0.08:
            overfit_diagnosis["verdict"] = "mild_overfitting"
            overfit_diagnosis["detail"] = (
                f"Train F1 ({train_eval['macro_f1']:.2%}) exceeds test F1 "
                f"({results[best_name]['macro_f1']:.2%}) by {train_test_gap:.2%}. "
                "Moderate gap — monitor but may be acceptable."
            )
        else:
            overfit_diagnosis["verdict"] = "no_significant_overfitting"
            overfit_diagnosis["detail"] = (
                f"Train F1 ({train_eval['macro_f1']:.2%}) and test F1 "
                f"({results[best_name]['macro_f1']:.2%}) are within {train_test_gap:.2%}."
            )

    if best_cv_macro_f1 is not None:
        holdout_cv_gap = results[best_name]["macro_f1"] - best_cv_macro_f1
        overfit_diagnosis["holdout_cv_f1_gap"] = float(holdout_cv_gap)
        if holdout_cv_gap > 0.10:
            overfit_diagnosis["temporal_stability"] = "unstable"
            overfit_diagnosis["temporal_detail"] = (
                f"Holdout F1 ({results[best_name]['macro_f1']:.2%}) exceeds walk-forward CV F1 "
                f"({best_cv_macro_f1:.2%}) by {holdout_cv_gap:.2%}. "
                "The model may not generalise across time windows. "
                "Consider sliding-window training or more walk-forward folds."
            )
        else:
            overfit_diagnosis["temporal_stability"] = "stable"
            overfit_diagnosis["temporal_detail"] = (
                f"Holdout F1 ({results[best_name]['macro_f1']:.2%}) and walk-forward CV F1 "
                f"({best_cv_macro_f1:.2%}) are within {holdout_cv_gap:.2%}."
            )

    print(f"Overfit check: {overfit_diagnosis.get('verdict', 'n/a')} | "
          f"train F1={overfit_diagnosis.get('train_macro_f1', 'n/a')}, "
          f"test F1={overfit_diagnosis.get('test_macro_f1')}, "
          f"CV F1={best_cv_macro_f1 or 'n/a'}")

    metrics = {
        "split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "cutoff_time_window": str(cutoff),
            "test_fraction": test_fraction,
        },
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "models": results,
        "best_model": best_name,
        "walk_forward_cv": walk_forward_summary,
        "epsilon_tuning": {
            "grid": sorted(set(epsilon_grid)),
            "selected": epsilon,
            "cv_summary": epsilon_cv,
        },
        "threshold_tuning": {
            "candidates": threshold_candidates,
            "selected_by_model": threshold_eval,
            "selected_for_best": selected_thresholds,
        },
        "calibration": {
            "enabled": calibration_enabled,
            "applied_by_model": calibration_info,
        },
        "feature_importance": feature_importance_map,
        "feature_pruning": pruning_report,
        "overfitting_diagnostics": overfit_diagnosis,
        "optional_models_skipped": skipped_optional,
        "comparison_vs_heuristic": {
            "best_accuracy_delta": float(results[best_name]["accuracy"] - results["heuristic"]["accuracy"]),
            "best_macro_f1_delta": float(results[best_name]["macro_f1"] - results["heuristic"]["macro_f1"]),
            "meets_acceptance_metric": bool(
                results[best_name]["accuracy"] >= results["heuristic"]["accuracy"]
                or results[best_name]["macro_f1"] >= results["heuristic"]["macro_f1"]
            ),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    print(f"Saved model artifact to {model_path}")
    print(f"Saved metrics artifact to {metrics_path}")
    print(f"Best model: {best_name}")
    print(
        "Heuristic vs best:"
        f" accuracy {results['heuristic']['accuracy']:.4f} -> {results[best_name]['accuracy']:.4f},"
        f" macro_f1 {results['heuristic']['macro_f1']:.4f} -> {results[best_name]['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
