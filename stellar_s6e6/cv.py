"""
Out-of-fold (OOF) training engine.

Trains every base model with the SAME stratified folds so their OOF
predictions are perfectly aligned -- this is the foundation for clean
stacking. For each model we produce:
  * oof_probs : (n_train, n_classes)  -- never-seen-in-training predictions
  * test_probs: (n_test,  n_classes)  -- averaged over folds

Early stopping is used for the gradient-boosting libraries when available,
with a graceful fallback to a plain fit.
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from config import Config


def _fit_one(model, name, X_tr, y_tr, X_val, y_val):
    """Fit a single model, using early stopping where supported."""
    cls = type(model).__name__.lower()
    try:
        if "lgbm" in cls:
            import lightgbm as lgb
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(0)],
            )
            return model
        if "xgb" in cls:
            try:
                model.set_params(early_stopping_rounds=100)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                model.fit(X_tr, y_tr)
            return model
        if "catboost" in cls:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=100, verbose=False)
            return model
    except Exception as e:  # pragma: no cover
        print(f"   [warn] early-stopping fit failed for {name} ({e}); plain fit")
    model.fit(X_tr, y_tr)
    return model


def _predict_proba_aligned(model, X, n_classes, model_classes):
    """predict_proba but re-ordered/padded to a fixed 0..n_classes-1 layout."""
    p = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", np.arange(p.shape[1])))
    if p.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return p
    out = np.zeros((p.shape[0], n_classes), dtype=float)
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    return out


def run_oof(cfg: Config, X, y, X_test, model_factories, n_classes,
            sample_weight=None):
    """Return (oof_dict, test_dict, scores_dict)."""
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    Xt = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y = np.asarray(y)

    skf = StratifiedKFold(
        n_splits=cfg.cv.n_splits, shuffle=cfg.cv.shuffle, random_state=cfg.cv.seed
    )
    folds = list(skf.split(Xv, y))

    oof_dict, test_dict, scores = {}, {}, {}

    for name, factory in model_factories.items():
        t0 = time.time()
        oof = np.zeros((len(y), n_classes), dtype=float)
        test_pred = np.zeros((Xt.shape[0], n_classes), dtype=float)

        for k, (tr, va) in enumerate(folds):
            model = factory(cfg.seed + k, n_classes)
            X_tr, y_tr = Xv[tr], y[tr]
            X_va, y_va = Xv[va], y[va]
            model = _fit_one(model, name, X_tr, y_tr, X_va, y_va)

            oof[va] = _predict_proba_aligned(model, X_va, n_classes, None)
            test_pred += _predict_proba_aligned(model, Xt, n_classes, None) / len(folds)

        acc = accuracy_score(y, oof.argmax(1))
        bacc = balanced_accuracy_score(y, oof.argmax(1))
        oof_dict[name] = oof
        test_dict[name] = test_pred
        scores[name] = {"accuracy": acc, "balanced_accuracy": bacc,
                        "seconds": round(time.time() - t0, 1)}
        print(f"   [{name:>11}] acc={acc:.5f}  bacc={bacc:.5f}  "
              f"({scores[name]['seconds']}s)")

    return oof_dict, test_dict, scores
