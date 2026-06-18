"""
Generate small mock data matching the SDSS17 schema so the whole pipeline can
be smoke-tested without the real Kaggle files. The class-dependent structure
(esp. redshift) is realistic enough that a good pipeline should score well
above chance -- which is what we want to verify the feature engineering and
ensembling actually work.

Run:  python make_mock_data.py
Writes train.csv / test.csv / sample_submission.csv into Config.paths.data_dir.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from config import CFG

CLASSES = ["GALAXY", "STAR", "QSO"]


def _make(n, rng):
    # class priors roughly like the real dataset (galaxy-heavy).
    probs = [0.59, 0.22, 0.19]
    y = rng.choice(CLASSES, size=n, p=probs)

    redshift = np.zeros(n)
    u = np.zeros(n); g = np.zeros(n); r = np.zeros(n); i = np.zeros(n); z = np.zeros(n)

    for cls in CLASSES:
        m = y == cls
        k = m.sum()
        if cls == "STAR":
            redshift[m] = rng.normal(0.0, 0.0005, k)
            base = rng.normal(17.5, 1.5, k)
            ug, gr, ri, iz = 1.2, 0.5, 0.2, 0.1
        elif cls == "GALAXY":
            redshift[m] = np.abs(rng.normal(0.25, 0.18, k))
            base = rng.normal(19.0, 1.2, k)
            ug, gr, ri, iz = 1.8, 1.0, 0.5, 0.35
        else:  # QSO
            redshift[m] = np.abs(rng.normal(1.6, 0.8, k)) + 0.3
            base = rng.normal(19.5, 1.0, k)
            ug, gr, ri, iz = 0.4, 0.2, 0.15, 0.1

        noise = lambda: rng.normal(0, 0.25, k)
        u[m] = base + ug + noise()
        g[m] = base + gr + noise()
        r[m] = base + ri + noise()
        i[m] = base + iz + noise()
        z[m] = base + noise()

    df = pd.DataFrame({
        "obj_ID": rng.integers(1e18, 2e18, n),
        "alpha": rng.uniform(0, 360, n),
        "delta": rng.uniform(-20, 70, n),
        "u": u, "g": g, "r": r, "i": i, "z": z,
        "run_ID": rng.integers(100, 9000, n),
        "rerun_ID": 301,
        "cam_col": rng.integers(1, 7, n),
        "field_ID": rng.integers(1, 900, n),
        "spec_obj_ID": rng.integers(1e18, 9e18, n),
        "class": y,
        "redshift": redshift,
        "plate": rng.integers(200, 12000, n),
        "MJD": rng.integers(51000, 59000, n),
        "fiber_ID": rng.integers(1, 1000, n),
    })

    # inject a few SDSS -9999 sentinels into u/g/z
    for col in ("u", "g", "z"):
        idx = rng.choice(n, size=max(1, n // 500), replace=False)
        df.loc[idx, col] = -9999.0
    return df


def main(n_train=6000, n_test=3000, seed=2026):
    rng = np.random.default_rng(seed)
    os.makedirs(CFG.paths.data_dir, exist_ok=True)

    train = _make(n_train, rng)
    test = _make(n_test, rng)

    # add competition-style row id, drop target from test
    train.insert(0, "id", np.arange(len(train)))
    test.insert(0, "id", np.arange(len(test)))
    y_test = test.pop("class")  # hidden truth (kept out of test.csv)

    train.to_csv(CFG.paths.p(CFG.paths.train_csv), index=False)
    test.to_csv(CFG.paths.p(CFG.paths.test_csv), index=False)

    sub = pd.DataFrame({"id": test["id"], "class": "GALAXY"})
    sub.to_csv(CFG.paths.p(CFG.paths.sample_submission_csv), index=False)

    # stash hidden truth for the smoke test to compute a real test score
    pd.DataFrame({"id": test["id"], "class": y_test}).to_csv(
        CFG.paths.p("_mock_test_truth.csv"), index=False)

    print(f"[mock] wrote {len(train)} train / {len(test)} test rows to "
          f"{CFG.paths.data_dir}")


if __name__ == "__main__":
    main()
