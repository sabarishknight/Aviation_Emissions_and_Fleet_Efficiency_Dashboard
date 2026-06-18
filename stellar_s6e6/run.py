"""
End-to-end runner for the S6E6 stellar-classification pipeline.

    python run.py

Pipeline:
  1. load competition train/test (+ optional original SDSS17 concat)
  2. build domain features (color indices, redshift interactions, ...)
  3. train the diverse model zoo with shared stratified folds -> OOF + test
  4. ensemble: stacking (logreg-on-logits) and greedy hill-climb blend
  5. pick the better combiner on OOF, apply metric-aware decisioning
  6. write submission.csv

Swap in the real data by editing config.Paths (or env vars):
    STELLAR_DATA_DIR=/kaggle/input/playground-series-s6e6 \
    STELLAR_ORIGINAL_CSV=/kaggle/input/stellar-classification-dataset-sdss17/star_classification.csv \
    python run.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from config import CFG
import data as data_mod
import features as feat_mod
import models as model_mod
import cv as cv_mod
import ensemble as ens_mod


def main():
    cfg = CFG
    cfg.ensure_dirs()
    np.random.seed(cfg.seed)

    print("== 1. Load data ==")
    train, test, sample = data_mod.load_raw(cfg)
    train = data_mod.maybe_concat_original(cfg, train)
    print(f"   train={train.shape}  test={test.shape}")

    y, classes, class_to_id = data_mod.encode_target(cfg, train[cfg.schema.target_col])
    n_classes = len(classes)
    print(f"   classes={classes}")

    print("== 2. Feature engineering ==")
    X_train = feat_mod.build_features(train, cfg)
    X_test = feat_mod.build_features(test, cfg)
    X_train, X_test, cols = feat_mod.align_features(X_train, X_test)
    print(f"   {len(cols)} features")

    print("== 3. Base model zoo (OOF) ==")
    factories = model_mod.get_model_factories(cfg)
    oof_dict, test_dict, scores = cv_mod.run_oof(
        cfg, X_train, y, X_test, factories, n_classes)

    print("== 4. Ensemble ==")
    stack_oof, stack_test, stack_sc = ens_mod.stack_logreg(
        cfg, oof_dict, test_dict, y, n_classes)
    hc_oof, hc_test, hc_w, hc_sc = ens_mod.hill_climb(
        cfg, oof_dict, test_dict, y)

    metric = cfg.cv.metric
    if stack_sc[metric] >= hc_sc[metric]:
        print(f"   -> using STACK ({metric}={stack_sc[metric]:.5f})")
        final_oof, final_test = stack_oof, stack_test
    else:
        print(f"   -> using HILL-CLIMB ({metric}={hc_sc[metric]:.5f})")
        final_oof, final_test = hc_oof, hc_test

    print("== 5. Finalize ==")
    pred_ids, weights = ens_mod.finalize(cfg, final_oof, final_test, y, n_classes)
    pred_labels = data_mod.decode_target(pred_ids, classes)

    print("== 6. Submission ==")
    id_col = cfg.schema.id_col if cfg.schema.id_col in test.columns else test.columns[0]
    sub = pd.DataFrame({id_col: test[id_col], cfg.schema.target_col: pred_labels})
    out_path = os.path.join(cfg.paths.out_dir, "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"   wrote {out_path}  ({len(sub)} rows)")
    print(sub[cfg.schema.target_col].value_counts())

    # If running on mock data, report the held-out test score.
    truth_path = cfg.paths.p("_mock_test_truth.csv")
    if os.path.exists(truth_path):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        truth = pd.read_csv(truth_path)
        yt = data_mod._normalise_class_labels(truth[cfg.schema.target_col])
        yp = data_mod._normalise_class_labels(sub[cfg.schema.target_col])
        print(f"\n[MOCK HOLD-OUT] accuracy={accuracy_score(yt, yp):.5f}  "
              f"balanced_accuracy={balanced_accuracy_score(yt, yp):.5f}")

    return sub


if __name__ == "__main__":
    main()
