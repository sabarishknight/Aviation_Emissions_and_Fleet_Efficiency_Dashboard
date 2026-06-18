"""
Diverse base-model zoo.

Each entry returns a fresh estimator exposing the sklearn-style
``fit(X, y)`` / ``predict_proba(X)`` interface. Diversity (different model
families + different inductive biases) is what makes the downstream stacking
ensemble strong -- error diversity matters more than any single model.

Models that need clean numeric input (LogReg) are wrapped in a Pipeline with
imputation + scaling. Tree models receive raw (NaN-tolerant) input.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import Config

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Optional heavy libraries -- import lazily so the pipeline still runs if one
# is missing.
# ----------------------------------------------------------------------------
def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:  # pragma: no cover
        return None


_lgb = _try_import("lightgbm")
_xgb = _try_import("xgboost")
_cat = _try_import("catboost")


def _lightgbm(seed: int, n_classes: int):
    return _lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_samples=40,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _xgboost(seed: int, n_classes: int):
    return _xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=3,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )


def _catboost(seed: int, n_classes: int):
    return _cat.CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def _hist_gbm(seed: int, n_classes: int):
    return HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )


def _extra_trees(seed: int, n_classes: int):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def _logreg(seed: int, n_classes: int):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            n_jobs=-1,
            random_state=seed,
        )),
    ])


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def get_model_factories(cfg: Config) -> dict[str, Callable[[int, int], object]]:
    """Return {name: factory(seed, n_classes) -> estimator} for enabled models."""
    t = cfg.models
    reg: dict[str, Callable] = {}

    if t.lightgbm and _lgb is not None:
        reg["lightgbm"] = _lightgbm
    if t.xgboost and _xgb is not None:
        reg["xgboost"] = _xgboost
    if t.catboost and _cat is not None:
        reg["catboost"] = _catboost
    if t.hist_gbm:
        reg["hist_gbm"] = _hist_gbm
    if t.extra_trees:
        reg["extra_trees"] = _extra_trees
    if t.logreg:
        reg["logreg"] = _logreg

    if not reg:
        raise RuntimeError("No models enabled / importable. Check toggles & installs.")
    return reg


def needs_dense_clean(name: str) -> bool:
    """LogReg/ExtraTrees pipelines already impute; tree libs tolerate NaN.
    HistGBM tolerates NaN natively. So nothing extra needed, but xgboost<1.6
    needs explicit handling -- modern xgboost is fine with NaN."""
    return False
