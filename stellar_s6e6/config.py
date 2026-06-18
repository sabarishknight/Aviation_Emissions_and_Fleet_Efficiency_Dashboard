"""
Central configuration for the S6E6 Stellar Classification pipeline.

Everything that you might want to tweak between a local smoke-test run and a
real Kaggle run lives here. The defaults are chosen so the pipeline runs
end-to-end on small mock data; bump the model params / fold count for the
real competition.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# On Kaggle the competition data lives under /kaggle/input/playground-series-s6e6/
# and the original SDSS17 dataset (if you attach it) under e.g.
# /kaggle/input/stellar-classification-dataset-sdss17/star_classification.csv
#
# Locally, point these at wherever you downloaded the CSVs. The mock-data
# generator writes to ./data by default.
# ----------------------------------------------------------------------------
@dataclass
class Paths:
    data_dir: str = os.environ.get("STELLAR_DATA_DIR", "data")
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    sample_submission_csv: str = "sample_submission.csv"
    # Optional: the original (non-synthetic) SDSS17 dataset to concatenate.
    original_csv: str | None = os.environ.get("STELLAR_ORIGINAL_CSV") or None
    out_dir: str = os.environ.get("STELLAR_OUT_DIR", "artifacts")

    def p(self, name: str) -> str:
        return os.path.join(self.data_dir, name)


# ----------------------------------------------------------------------------
# Columns / schema (SDSS DR17 stellar classification)
# ----------------------------------------------------------------------------
@dataclass
class Schema:
    id_col: str = "id"                       # competition row id (PS adds this)
    target_col: str = "class"                # GALAXY / STAR / QSO
    # Photometric magnitudes (ugriz system).
    mag_cols: tuple = ("u", "g", "r", "i", "z")
    # Core physical / positional features.
    core_cols: tuple = ("alpha", "delta", "redshift")
    # ID / observation-metadata columns. Dropped by default but the runner can
    # optionally keep the "spectroscopic" ones (plate, MJD, fiber_ID) which can
    # carry leakage in the original SDSS data.
    id_like_cols: tuple = (
        "obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID",
        "spec_obj_ID", "plate", "MJD", "fiber_ID",
    )
    spectro_meta_cols: tuple = ("plate", "MJD", "fiber_ID", "cam_col")
    # Sentinel value used in SDSS for failed measurements.
    sentinel_bad: float = -9999.0


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 2026
    # The competition metric. "accuracy" -> argmax. "balanced_accuracy" ->
    # per-class threshold / prior search is applied at the end.
    metric: str = "accuracy"


# ----------------------------------------------------------------------------
# Which base models to run. Toggle to control runtime.
# ----------------------------------------------------------------------------
@dataclass
class ModelToggles:
    lightgbm: bool = True
    xgboost: bool = True
    catboost: bool = True
    hist_gbm: bool = True
    extra_trees: bool = True
    logreg: bool = True
    # Heavier NN models (need torch). Off by default for CPU smoke tests.
    mlp_torch: bool = False


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    schema: Schema = field(default_factory=Schema)
    cv: CVConfig = field(default_factory=CVConfig)
    models: ModelToggles = field(default_factory=ModelToggles)

    # Feature-engineering switches.
    use_color_indices: bool = True
    use_redshift_interactions: bool = True
    use_ratios_and_stats: bool = True
    keep_spectro_meta: bool = False   # set True to test leakage from plate/MJD/...
    concat_original: bool = True       # concat original SDSS17 data into train

    # Reproducibility.
    seed: int = 2026

    def ensure_dirs(self) -> None:
        os.makedirs(self.paths.out_dir, exist_ok=True)


# A ready-to-use default instance.
CFG = Config()
