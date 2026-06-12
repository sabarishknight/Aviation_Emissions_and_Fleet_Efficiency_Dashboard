"""
Data loading and harmonisation.

Handles:
  * loading the competition train/test,
  * optionally concatenating the original SDSS17 dataset into the train set
    (a classic Playground-Series trick: the synthetic data is generated from
    the original, so adding it almost always helps),
  * label-encoding the target consistently.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from config import Config


# Canonical class order. Keeping it fixed makes OOF arrays / submissions stable.
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"Point Config.paths at your Kaggle/local data, or run "
            f"make_mock_data.py first."
        )
    return pd.read_csv(path)


def _normalise_class_labels(s: pd.Series) -> pd.Series:
    """Uppercase + strip so 'Galaxy', 'galaxy ', 'QUASAR' all map cleanly."""
    out = s.astype(str).str.strip().str.upper()
    out = out.replace({"QUASAR": "QSO"})
    return out


def load_raw(cfg: Config):
    """Return (train_df, test_df, sample_submission_df)."""
    p = cfg.paths
    train = _read_csv(p.p(p.train_csv))
    test = _read_csv(p.p(p.test_csv))
    sample = None
    sample_path = p.p(p.sample_submission_csv)
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
    return train, test, sample


def maybe_concat_original(cfg: Config, train: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the original SDSS17 dataset onto the synthetic train set."""
    if not cfg.concat_original or not cfg.paths.original_csv:
        return train
    if not os.path.exists(cfg.paths.original_csv):
        print(f"[data] original_csv not found, skipping concat: "
              f"{cfg.paths.original_csv}")
        return train

    orig = pd.read_csv(cfg.paths.original_csv)
    tgt = cfg.schema.target_col
    if tgt not in orig.columns:
        print(f"[data] original data lacks target '{tgt}', skipping concat")
        return train

    # Keep only columns the competition train also has.
    shared = [c for c in train.columns if c in orig.columns]
    orig = orig[shared].copy()
    orig["__is_original__"] = 1
    train = train.copy()
    train["__is_original__"] = 0
    combined = pd.concat([train, orig], axis=0, ignore_index=True)
    # De-dup exact rows that may overlap.
    combined = combined.drop_duplicates(
        subset=[c for c in shared if c != tgt], keep="first"
    ).reset_index(drop=True)
    print(f"[data] concatenated original: train {len(train)} + orig {len(orig)} "
          f"-> {len(combined)} (after dedup)")
    return combined


def encode_target(cfg: Config, y_raw: pd.Series):
    """Map string classes -> integer ids using the canonical order."""
    y_norm = _normalise_class_labels(y_raw)
    present = [c for c in CLASS_ORDER if c in set(y_norm.unique())]
    # Fall back to whatever is present if labels differ from expectation.
    classes = present if present else sorted(y_norm.unique())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = y_norm.map(class_to_id).astype("int64").to_numpy()
    return y, classes, class_to_id


def decode_target(ids: np.ndarray, classes: list[str]) -> np.ndarray:
    inv = np.array(classes)
    return inv[ids]
