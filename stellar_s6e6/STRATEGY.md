# Playground Series S6E6 — Stellar Classification: Winning Strategy

Goal: top-of-leaderboard finish on the SDSS DR17 stellar-classification task
(predict `class` ∈ {GALAXY, STAR, QSO}). This document is the plan; the code in
this folder is a runnable implementation of it.

> Note on the data: the live competition data is JS-gated and not in this repo.
> This pipeline targets the well-known **SDSS DR17 Stellar Classification**
> schema that S6E6 is built from. The code auto-adapts to whatever columns the
> real `train.csv`/`test.csv` actually contain, and a mock-data generator lets
> you verify everything runs before you plug in the Kaggle files.

---

## 1. The problem in one paragraph (domain intuition)

Each row is a Sloan Digital Sky Survey observation. The physically meaningful
columns are five photometric magnitudes `u, g, r, i, z` (brightness in five
filters, blue→infrared), the sky position `alpha`/`delta`, and `redshift`.
The rest (`obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`,
`spec_obj_ID`, `plate`, `MJD`, `fiber_ID`) are observation bookkeeping / IDs.

Physics that decides the label:
- **redshift dominates.** Stars sit in our galaxy → `redshift ≈ 0`. Galaxies are
  low/moderate. Quasars (QSO) are far away → high redshift. This single feature
  gets you most of the way.
- **Colour indices** (differences of magnitudes such as `u-g`, `g-r`, `r-i`,
  `i-z`) encode the spectral slope and are the classical tool astronomers use to
  separate object types. A magnitude is a log-flux, so a difference is a flux
  ratio — scale-invariant and very informative.

This is why the score ceiling on this dataset is very high (~0.97–0.99 acc):
the classes are largely separable. **At that ceiling, the competition is won in
the third and fourth decimal** — i.e. by ensembling and decision calibration,
exactly the pattern in the S6E4 writeups you shared.

---

## 2. What the S6E4 winners actually did (and how we copy it)

| Writeup | Core idea | What we reuse |
|---|---|---|
| 1st — One-vs-Rest + multiclass | classes rarely confused → decompose; blend OOF with LogisticRegressionCV on **logits**; threshold search for the metric | logit-space stacking meta-learner; metric-aware final decision |
| 3rd — "Error diversity matters", 200-model stack | many *different* models, stacked | diverse model zoo + greedy hill-climb that rewards de-correlated errors |
| 4th — "more ensemblers than models" | the ensembling layer matters more than any single model | two combiners (stack + hill-climb), pick the best on OOF |
| 2nd / 5th — AI-driven large-scale experimentation + GPU logreg | run many configs fast, keep clean OOF | shared stratified folds so every OOF is comparable & stackable |

The throughline: **clean out-of-fold predictions + diverse models + a strong
ensembling layer + metric-specific decisioning.** That is precisely the
architecture implemented here.

---

## 3. The plan, end to end

### Step 0 — Validation harness (do this first, never break it)
- One `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)`, reused by
  **every** model, so OOF arrays line up and stacking is honest.
- Trust CV over the public LB (public LB is a fraction of test). Track the gap.

### Step 1 — Add the original dataset (free points)
The synthetic competition data is generated from the original SDSS17 CSV.
Concatenating the original into the train set almost always helps on Playground
comps. Attach `fedesoriano/stellar-classification-dataset-sdss17` and set
`STELLAR_ORIGINAL_CSV`. (`data.maybe_concat_original` dedups overlaps.)

### Step 2 — Clean
- Replace the SDSS `-9999` failed-measurement sentinel (and absurd magnitudes)
  with NaN; tree models handle NaN, linear models get median imputation.

### Step 3 — Domain feature engineering (`features.py`)
- **All pairwise colour indices** over `{u,g,r,i,z}` (10 features): `u-g`, `g-r`,
  `r-i`, `i-z`, `u-r`, `u-z`, `u-i`, `g-i`, `g-z`, `r-z`.
- **Redshift transforms**: raw, `log1p`, `sqrt`, `is_near_zero` (star flag),
  `>0.5` (QSO flag).
- **redshift × colour interactions** (observed colour shifts with redshift in a
  class-dependent way).
- **Per-row magnitude stats**: mean/std/min/max/range.
- Keep `alpha`, `delta`. Optionally test the spectroscopic-metadata columns
  (`plate`, `MJD`, `fiber_ID`) — they can leak label info via how spectroscopic
  targets were selected. Toggle `keep_spectro_meta=True` and **only keep it if
  CV says so and you trust it generalises to private LB.**

### Step 4 — Diverse base-model zoo (`models.py`, `cv.py`)
Different inductive biases → de-correlated errors → better stack:
- LightGBM, XGBoost, CatBoost (3 different GBDT implementations),
- HistGradientBoosting, ExtraTrees (bagged trees, different bias),
- multinomial LogisticRegression (linear; cheap diversity).
- Optional: a tabular NN (TabM / RealMLP / FT-Transformer) for extra diversity —
  enable `models.mlp_torch` and add a torch model (biggest *additional* gains in
  PS comps come from adding a strong NN to a GBDT stack).
Each produces 5-fold OOF + fold-averaged test probabilities.

### Step 5 — Ensemble (`ensemble.py`)
- **Stacking**: LogisticRegression on the concatenated **log-probabilities** of
  all base models; honest meta-OOF via `cross_val_predict`.
- **Hill-climbing**: greedy weighted average of model probabilities that
  maximises the OOF metric (robust, hard to overfit, auto down-weights weak/
  correlated models).
- Pick whichever scores higher on OOF.

### Step 6 — Metric-aware decision (`ensemble.finalize`)
- **accuracy** → plain `argmax`.
- **balanced_accuracy** → search per-class multiplicative priors on OOF, apply
  to test (`search_class_weights`). If the metric is balanced accuracy, this step
  alone can be worth several places.

### Step 7 — Squeeze the last decimals
- **Seed-averaging**: rerun the zoo over a few seeds, average OOF/test.
- **Pseudo-labeling**: add high-confidence test rows to train, refit once.
- **Per-class threshold tuning** on the final blend.
- **Feature ablation**: confirm each FE block helps CV; drop dead weight.

---

## 4. How to run

### A) Smoke test (no Kaggle data needed)
```bash
cd stellar_s6e6
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python make_mock_data.py     # writes data/*.csv
.venv/bin/python run.py                # -> artifacts/submission.csv
```
Expected: every base model trains, stack + hill-climb run, a valid
`id,class` submission is written, and a mock hold-out score is printed
(~0.99 on the mock data, which validates the FE + ensembling end-to-end).

### B) Real competition (run on a Kaggle Notebook with GPU)
1. Attach the competition data and (recommended) the original SDSS17 dataset.
2. Point the pipeline at them:
```bash
export STELLAR_DATA_DIR=/kaggle/input/playground-series-s6e6
export STELLAR_ORIGINAL_CSV=/kaggle/input/stellar-classification-dataset-sdss17/star_classification.csv
export STELLAR_OUT_DIR=/kaggle/working
python run.py
```
3. Before submitting, **set the real metric** in `config.CVConfig.metric`
   (`"accuracy"` vs `"balanced_accuracy"`) — confirm it from the competition's
   Evaluation tab. The final decision rule depends on it.
4. Submit `/kaggle/working/submission.csv`.

### Things to verify against the live competition (I could not access it)
- Exact target class names / label casing (code normalises common variants).
- The official **metric** (drives Step 6).
- Exact feature column names (code adapts, but confirm nothing important is
  dropped as "id-like").

---

## 5. Priority order if you're short on time
1. CV harness + colour indices + redshift features → strong single LightGBM.
2. Concatenate the original SDSS17 data.
3. Add XGBoost + CatBoost + one NN; hill-climb blend.
4. Metric-aware final decision (huge if balanced accuracy).
5. Seed-averaging + pseudo-labeling for the last decimals.
