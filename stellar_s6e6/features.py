"""
Domain feature engineering for SDSS stellar classification.

The physics that drives this problem:
  * `redshift` is by far the strongest single feature.
        STAR   -> redshift ~ 0
        GALAXY -> small/moderate redshift
        QSO    -> large redshift
  * Astronomers classify objects using COLOR INDICES, i.e. differences between
    magnitudes in adjacent bands (u-g, g-r, r-i, i-z). A magnitude is a log
    brightness, so a difference is a flux *ratio* -> it encodes the spectral
    slope, which separates the three object types and even spectral subtypes.
  * Interactions of redshift with colour are informative because the observed
    colour of a redshifted object shifts in a class-dependent way.

We therefore build:
  - all pairwise colour indices over {u,g,r,i,z},
  - redshift transforms + redshift x colour interactions,
  - simple per-row magnitude statistics,
  - cleaning of the SDSS -9999 sentinel.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from config import Config


def _clean_sentinels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Replace SDSS -9999 failed measurements with NaN (models handle NaN or
    we median-impute later)."""
    bad = cfg.schema.sentinel_bad
    mag_cols = [c for c in cfg.schema.mag_cols if c in df.columns]
    for c in mag_cols + ["redshift"]:
        if c in df.columns:
            df[c] = df[c].replace(bad, np.nan)
            # Also treat absurd magnitudes as missing.
            df.loc[df[c] < -100, c] = np.nan
    return df


def _color_indices(df: pd.DataFrame, mag_cols: list[str]) -> pd.DataFrame:
    feats = {}
    for a, b in itertools.combinations(mag_cols, 2):
        feats[f"col_{a}_{b}"] = df[a] - df[b]
    return pd.DataFrame(feats, index=df.index)


def _redshift_feats(df: pd.DataFrame, color_df: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    if "redshift" not in df.columns:
        return pd.DataFrame(index=df.index)
    z = df["redshift"]
    feats["z_raw"] = z
    feats["z_log1p"] = np.log1p(z.clip(lower=-0.999))
    feats["z_sqrt"] = np.sqrt(z.clip(lower=0))
    feats["z_is_near_zero"] = (z.abs() < 1e-3).astype("int8")  # star-like
    feats["z_gt_0_5"] = (z > 0.5).astype("int8")               # qso-like
    out = pd.DataFrame(feats, index=df.index)
    # redshift x colour interactions (strong separators).
    for c in color_df.columns:
        out[f"{c}_x_z"] = color_df[c] * z
    return out


def _row_stats(df: pd.DataFrame, mag_cols: list[str]) -> pd.DataFrame:
    m = df[mag_cols]
    feats = {
        "mag_mean": m.mean(axis=1),
        "mag_std": m.std(axis=1),
        "mag_max": m.max(axis=1),
        "mag_min": m.min(axis=1),
        "mag_range": m.max(axis=1) - m.min(axis=1),
    }
    return pd.DataFrame(feats, index=df.index)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a numeric feature matrix (no target, no id)."""
    df = df.copy()
    df = _clean_sentinels(df, cfg)

    schema = cfg.schema
    mag_cols = [c for c in schema.mag_cols if c in df.columns]

    parts: list[pd.DataFrame] = []

    # Base physical columns we always keep.
    base_keep = mag_cols + [c for c in ("alpha", "delta", "redshift")
                            if c in df.columns]
    parts.append(df[base_keep].copy())

    color_df = pd.DataFrame(index=df.index)
    if cfg.use_color_indices and len(mag_cols) >= 2:
        color_df = _color_indices(df, mag_cols)
        parts.append(color_df)

    if cfg.use_redshift_interactions:
        parts.append(_redshift_feats(df, color_df))

    if cfg.use_ratios_and_stats and len(mag_cols) >= 2:
        parts.append(_row_stats(df, mag_cols))

    # Optionally retain spectroscopic-observation metadata (potential leakage).
    if cfg.keep_spectro_meta:
        meta = [c for c in schema.spectro_meta_cols if c in df.columns]
        if meta:
            parts.append(df[meta].copy())

    X = pd.concat(parts, axis=1)
    # Drop duplicate columns if any (e.g. redshift kept twice).
    X = X.loc[:, ~X.columns.duplicated()]
    # Replace inf from interactions.
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def align_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ensure train/test share the same columns in the same order."""
    cols = [c for c in X_train.columns if c in X_test.columns]
    return X_train[cols].copy(), X_test[cols].copy(), cols
