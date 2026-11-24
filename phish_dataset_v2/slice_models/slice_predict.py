"""
Reusable inference helpers for the five independent SLiCE score models.

Each artifact (models/slice_<score>_model.joblib) is a plain dict:
    model, feature_columns, preprocessing, sampling_strategy, threshold,
    model_name, model_version, validation_metrics, training_config, ...

The preprocessing used here is the *same* code path the training script uses,
so training, validation, test and production inputs are transformed identically.

Usage
-----
    import pandas as pd
    from slice_models.slice_predict import predict_score, predict_all_scores

    df = pd.read_excel("phishing_pot_dataset.xlsx", sheet_name="Dataset")
    sts_prob = predict_score(df, "STS")              # np.ndarray, phishing probability
    scores   = predict_all_scores(df)                # DataFrame: STS_prob, STS_flag, ...
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCORES = ("STS", "BAS", "DMS", "LOS", "RCS")
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Sentinel values defined by the ORIGINAL feature modules (not invented here).
#   sts_features.new_sender_domain_label(): returns -1 when domain age is unknown
#   (no From domain, RDAP non-200, no 'registration' event, or any exception).
#   -1 is therefore "missing", not an age bucket; it is mapped to NaN and
#   LightGBM's native missing-value handling is used.
SENTINELS = {"STS__new_sender_domain": [-1]}


def model_path(score: str, model_dir: Path | str | None = None) -> Path:
    return Path(model_dir or MODEL_DIR) / f"slice_{score.lower()}_model.joblib"


def family_columns(columns, score: str) -> list[str]:
    """Columns of one score family, in their original dataset order."""
    prefix = f"{score.upper()}__"
    return [c for c in columns if str(c).startswith(prefix)]


def preprocess(df: pd.DataFrame, feature_columns: list[str], sentinels: dict | None = None) -> pd.DataFrame:
    """Select exactly `feature_columns` (in that order), coerce to float,
    map +/-inf and documented sentinels to NaN. Stateless: identical for
    train / validation / test / inference."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required feature columns: {missing}")
    X = df.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col, values in (sentinels if sentinels is not None else SENTINELS).items():
        if col in X.columns:
            X.loc[X[col].isin(values), col] = np.nan
    return X


_CACHE: dict[str, dict] = {}


def load_slice_model(score: str, model_dir: Path | str | None = None, use_cache: bool = True) -> dict:
    score = score.upper()
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    key = f"{model_dir}|{score}"
    if use_cache and key in _CACHE:
        return _CACHE[key]
    art = joblib.load(model_path(score, model_dir))
    bad = [c for c in art["feature_columns"] if not c.startswith(f"{score}__")]
    if bad:
        raise ValueError(f"{score} artifact contains foreign-family features: {bad}")
    if use_cache:
        _CACHE[key] = art
    return art


def predict_score(df: pd.DataFrame, score: str, model_dir: Path | str | None = None,
                  return_flag: bool = False):
    """Phishing probability (model score, class 1) for one SLiCE score family.
    With return_flag=True also returns the 0/1 decision at the artifact's
    validation-selected threshold."""
    art = load_slice_model(score, model_dir)
    X = preprocess(df, art["feature_columns"], art["preprocessing"]["sentinels"])
    prob = art["model"].predict_proba(X)[:, 1]
    if return_flag:
        return prob, (prob >= art["threshold"]).astype(int)
    return prob


def predict_all_scores(df: pd.DataFrame, model_dir: Path | str | None = None) -> pd.DataFrame:
    """Run all five independent models; returns <SCORE>_prob and <SCORE>_flag columns."""
    out = pd.DataFrame(index=df.index)
    for s in SCORES:
        prob, flag = predict_score(df, s, model_dir, return_flag=True)
        out[f"{s}_prob"] = prob
        out[f"{s}_flag"] = flag
    return out
