"""
Ensembling layer.

Two complementary combiners (use whichever scores best on OOF):

  1. stack_logreg  -- a LogisticRegression meta-learner trained on the
     log-probabilities of every base model (the "logits + LogisticRegressionCV"
     trick from the 1st-place S6E4 writeup). Honest meta-OOF via cross_val.

  2. hill_climb    -- greedy weighted blend of base-model probability vectors,
     selecting (with replacement) the model that most improves the OOF metric.
     Robust, hard to overfit, and naturally down-weights weak/correlated models.

Plus metric-aware final decisioning:
  * accuracy            -> plain argmax
  * balanced_accuracy   -> per-class prior/weight search on the blended probs
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from config import Config


def _metric_fn(metric: str):
    return balanced_accuracy_score if metric == "balanced_accuracy" else accuracy_score


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p)


def stack_logreg(cfg: Config, oof_dict, test_dict, y, n_classes):
    names = list(oof_dict.keys())
    Z_tr = np.hstack([_logit(oof_dict[n]) for n in names])
    Z_te = np.hstack([_logit(test_dict[n]) for n in names])

    meta = LogisticRegression(C=1.0, max_iter=3000)
    skf = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True,
                          random_state=cfg.cv.seed)
    meta_oof = cross_val_predict(meta, Z_tr, y, cv=skf, method="predict_proba",
                                 n_jobs=-1)
    meta.fit(Z_tr, y)
    meta_test = meta.predict_proba(Z_te)

    acc = accuracy_score(y, meta_oof.argmax(1))
    bacc = balanced_accuracy_score(y, meta_oof.argmax(1))
    print(f"   [stack-logreg] acc={acc:.5f}  bacc={bacc:.5f}")
    return meta_oof, meta_test, {"accuracy": acc, "balanced_accuracy": bacc}


def hill_climb(cfg: Config, oof_dict, test_dict, y, n_iter=100):
    metric = cfg.cv.metric
    score = _metric_fn(metric)
    names = list(oof_dict.keys())
    oofs = [oof_dict[n] for n in names]
    tests = [test_dict[n] for n in names]

    # Seed with the single best model.
    base_scores = [score(y, o.argmax(1)) for o in oofs]
    best_idx = int(np.argmax(base_scores))
    blend = oofs[best_idx].copy()
    blend_test = tests[best_idx].copy()
    chosen = [best_idx]
    cur = score(y, blend.argmax(1))

    for _ in range(n_iter):
        best_gain, best_j = 0.0, -1
        n = len(chosen)
        for j in range(len(oofs)):
            cand = (blend * n + oofs[j]) / (n + 1)
            s = score(y, cand.argmax(1))
            if s > cur + best_gain:
                best_gain, best_j = s - cur, j
        if best_j < 0:
            break
        chosen.append(best_j)
        n = len(chosen)
        blend = (blend * (n - 1) + oofs[best_j]) / n
        blend_test = (blend_test * (n - 1) + tests[best_j]) / n
        cur = score(y, blend.argmax(1))

    weights = {names[i]: chosen.count(i) / len(chosen) for i in set(chosen)}
    acc = accuracy_score(y, blend.argmax(1))
    bacc = balanced_accuracy_score(y, blend.argmax(1))
    print(f"   [hill-climb]  acc={acc:.5f}  bacc={bacc:.5f}  weights={weights}")
    return blend, blend_test, weights, {"accuracy": acc, "balanced_accuracy": bacc}


def search_class_weights(probs, y, n_classes, n_iter=4000, seed=2026):
    """For balanced_accuracy: find multiplicative per-class priors that maximise
    OOF balanced accuracy. argmax(probs * w)."""
    rng = np.random.default_rng(seed)
    best_w = np.ones(n_classes)
    best = balanced_accuracy_score(y, probs.argmax(1))
    for _ in range(n_iter):
        w = best_w * rng.uniform(0.85, 1.15, size=n_classes)
        s = balanced_accuracy_score(y, (probs * w).argmax(1))
        if s > best:
            best, best_w = s, w
    return best_w, best


def finalize(cfg: Config, oof_probs, test_probs, y, n_classes):
    """Apply the metric-appropriate decision rule. Returns final test labels
    (int ids) and the chosen weights (or None)."""
    if cfg.cv.metric == "balanced_accuracy":
        w, s = search_class_weights(oof_probs, y, n_classes, seed=cfg.seed)
        print(f"   [finalize] balanced-acc weight search -> {s:.5f}  w={w.round(3)}")
        return (test_probs * w).argmax(1), w
    # accuracy
    return test_probs.argmax(1), None
