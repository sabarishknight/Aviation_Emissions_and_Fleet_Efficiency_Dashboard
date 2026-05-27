# =========================================================
# /kaggle/working/final_eval.py
#
# FINAL TTA EVALUATION DRIVER
#
# Reports AUC + macro-F1 ONLY.
# Uses 3 seeds per (method x dataset) for stable, reportable
# numbers. Reloads model from checkpoint between every run
# to prevent state leakage.
#
# Methods evaluated:
#   Baseline   - no adaptation
#   TENT       - canonical (BN-only)
#   SAR        - canonical (BN+LN, model-recovery on)
#   EATA       - canonical (no Fisher; matches FunOTTA setting)
#   T3A        - canonical (init from classifier weights)
#   CF-NODE    - your method; plug in via the registry below
#
# Outputs:
#   <out_dir>/results_raw.csv      one row per (method, dataset, seed)
#   <out_dir>/results_summary.csv  mean & std per (method, dataset)
#   <out_dir>/results_table.md     paper-ready markdown table
# =========================================================

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from tta_config import TTAConfig
from tta_utils import forward_logits
from tta_tent import run_tta as run_tent
from tta_sar import run_sar
from tta_eata import run_eata
from tta_t3a import run_t3a


# =========================================================
# Baseline (no adaptation)
# =========================================================
@torch.no_grad()
def run_baseline(model, loader, cfg, return_probs=False):
    model.eval()
    device = next(model.parameters()).device
    preds_all, probs_all, labels_all = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = forward_logits(model, images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        preds_all.extend(preds.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        labels_all.extend(labels.numpy())
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    probs_all = np.concatenate(probs_all)
    if return_probs:
        return preds_all, labels_all, probs_all
    return preds_all, labels_all


# =========================================================
# Metrics: AUC + F1 ONLY
# =========================================================
def compute_auc_f1(y_true, y_pred, y_probs, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probs = np.asarray(y_probs)

    # macro one-vs-rest AUC; falls back gracefully on degenerate splits
    try:
        present = np.unique(y_true)
        if len(present) < 2:
            auc = float("nan")
        elif len(present) < num_classes:
            # restrict probs/labels to classes actually present
            cols = present.astype(int)
            sub_probs = y_probs[:, cols]
            sub_probs = sub_probs / sub_probs.sum(axis=1, keepdims=True)
            auc = roc_auc_score(
                y_true, sub_probs, multi_class="ovr", average="macro",
                labels=cols,
            )
        else:
            auc = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="macro",
            )
    except ValueError:
        auc = float("nan")

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(auc), float(f1)


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Default method registry
# =========================================================
def default_methods(cf_node_runner=None):
    """
    Returns an ordered dict {name: runner_fn(model, loader, cfg, return_probs=True)}.

    Plug your CF-NODE in by passing a callable cf_node_runner with the
    same signature as run_baseline / run_tent / etc.
    """
    methods = {
        "Baseline": run_baseline,
        "TENT":     run_tent,
        "SAR":      run_sar,
        "EATA":     run_eata,
        "T3A":      run_t3a,
    }
    if cf_node_runner is not None:
        methods["CF-NODE"] = cf_node_runner
    return methods


# =========================================================
# Main evaluator
# =========================================================
def evaluate(
    model_factory,
    target_loaders,
    methods=None,
    cfg=None,
    seeds=(42, 123, 2024),
    output_dir="/kaggle/working/tta_results",
    verbose=True,
):
    """
    Args:
        model_factory: zero-arg callable that returns a FRESH model loaded
                       from your checkpoint and moved to the right device.
                       (Re-called between every run to avoid state leakage.)
        target_loaders: dict {dataset_name: DataLoader}
        methods:        dict {method_name: runner_fn(model, loader, cfg,
                                                     return_probs=True)
                              -> (preds, labels, probs)}
                        Defaults to Baseline+TENT+SAR+EATA+T3A.
        cfg:            TTAConfig (or any object with the required attrs)
        seeds:          iterable of ints; each method x dataset is run once
                        per seed.
        output_dir:     where to write the CSVs and markdown table.
    """
    cfg = cfg or TTAConfig()
    methods = methods or default_methods()
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for dataset_name, loader in target_loaders.items():
        if verbose:
            print(f"\n{'='*60}\nDATASET: {dataset_name}\n{'='*60}")
        for method_name, runner in methods.items():
            for seed in seeds:
                set_seed(seed)
                model = model_factory()
                try:
                    preds, labels, probs = runner(
                        model, loader, cfg, return_probs=True
                    )
                except Exception as e:
                    print(f"  [!] {method_name} seed={seed} FAILED: {e}")
                    auc, f1 = float("nan"), float("nan")
                else:
                    auc, f1 = compute_auc_f1(
                        labels, preds, probs, cfg.num_classes
                    )
                if verbose:
                    print(
                        f"  {method_name:9s} seed={seed:>4}  "
                        f"AUC={auc:.4f}  F1={f1:.4f}"
                    )
                rows.append({
                    "dataset": dataset_name,
                    "method":  method_name,
                    "seed":    seed,
                    "auc":     auc,
                    "f1":      f1,
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # Long-format raw
    raw_csv = os.path.join(output_dir, "results_raw.csv")
    df.to_csv(raw_csv, index=False)

    # Mean / std summary
    summary = (
        df.groupby(["method", "dataset"], sort=False)
          .agg(auc_mean=("auc", "mean"),
               auc_std =("auc", "std"),
               f1_mean =("f1",  "mean"),
               f1_std  =("f1",  "std"))
          .reset_index()
    )
    summary_csv = os.path.join(output_dir, "results_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Markdown table
    md = _build_markdown_table(
        df,
        method_order=list(methods.keys()),
        dataset_order=list(target_loaders.keys()),
    )
    md_path = os.path.join(output_dir, "results_table.md")
    with open(md_path, "w") as fh:
        fh.write(md)
    if verbose:
        print("\n" + md)
        print(f"\nSaved:\n  {raw_csv}\n  {summary_csv}\n  {md_path}")

    return df, summary, md


# =========================================================
# Markdown table builder (mean ± std, bold best per column)
# =========================================================
def _fmt_cell(mean, std, is_best):
    """Format a metric cell as 'mean ± std' on a 0-100 scale."""
    if np.isnan(mean):
        return "—"
    cell = f"{mean*100:.2f}"
    if not np.isnan(std):
        cell = f"{cell} ± {std*100:.2f}"
    if is_best:
        cell = f"**{cell}**"
    return cell


def _build_markdown_table(df, method_order, dataset_order):
    grp = df.groupby(["method", "dataset"], sort=False)
    pivot_mean = grp[["auc", "f1"]].mean()
    pivot_std  = grp[["auc", "f1"]].std()

    # best per (dataset, metric)
    best = {}
    for d in dataset_order:
        for metric in ("auc", "f1"):
            best_m, best_v = None, -np.inf
            for m in method_order:
                key = (m, d)
                if key not in pivot_mean.index:
                    continue
                v = pivot_mean.loc[key, metric]
                if not np.isnan(v) and v > best_v:
                    best_m, best_v = m, v
            best[(d, metric)] = best_m

    # per-method averages across datasets
    avg = {}
    for m in method_order:
        aucs, f1s = [], []
        for d in dataset_order:
            key = (m, d)
            if key in pivot_mean.index:
                aucs.append(pivot_mean.loc[key, "auc"])
                f1s.append(pivot_mean.loc[key, "f1"])
        avg[m] = (
            float(np.nanmean(aucs)) if aucs else float("nan"),
            float(np.nanmean(f1s))  if f1s  else float("nan"),
        )
    best_avg_auc = max(method_order, key=lambda m: avg[m][0] if not np.isnan(avg[m][0]) else -np.inf)
    best_avg_f1  = max(method_order, key=lambda m: avg[m][1] if not np.isnan(avg[m][1]) else -np.inf)

    # build markdown
    header = ["Method"]
    for d in dataset_order:
        header += [f"{d} AUC", f"{d} F1"]
    header += ["Avg AUC", "Avg F1"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for m in method_order:
        row = [m]
        for d in dataset_order:
            key = (m, d)
            if key in pivot_mean.index:
                a_mean = pivot_mean.loc[key, "auc"]
                a_std  = pivot_std.loc[key, "auc"]
                f_mean = pivot_mean.loc[key, "f1"]
                f_std  = pivot_std.loc[key, "f1"]
            else:
                a_mean = a_std = f_mean = f_std = float("nan")
            row.append(_fmt_cell(a_mean, a_std, best[(d, "auc")] == m))
            row.append(_fmt_cell(f_mean, f_std, best[(d, "f1")]  == m))
        row.append(_fmt_cell(avg[m][0], float("nan"), m == best_avg_auc))
        row.append(_fmt_cell(avg[m][1], float("nan"), m == best_avg_f1))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# =========================================================
# Usage skeleton (drop into a Kaggle cell)
# =========================================================
"""
# --- in your Kaggle notebook ---
import torch
from model import get_model
from tta_config import TTAConfig
from final_eval import evaluate, default_methods

CKPT = "/kaggle/input/models/.../best_dsr50_balanced_ordinal_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = TTAConfig()

# 1) model factory: ALWAYS returns a fresh model from the checkpoint
def model_factory():
    m = get_model(cfg).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    if "state_dict" in state:
        state = state["state_dict"]
    # strip any DataParallel "module." prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    m.load_state_dict(state, strict=True)
    if torch.cuda.device_count() > 1:
        m = torch.nn.DataParallel(m)
    return m

# 2) build your target loaders (you already have these)
target_loaders = {
    "MESSIDOR":  build_loader("messidor"),
    "IDRID":     build_loader("idrid"),
    "DeepDRiD":  build_loader("deepdrid"),
    "APTOS":     build_loader("aptos"),
    "DDR":       build_loader("ddr"),
}

# 3) (optional) plug in CF-NODE; signature must match the others
# from cfnode import run_cfnode
# methods = default_methods(cf_node_runner=run_cfnode)
methods = default_methods()  # without CF-NODE for now

# 4) run; 3 seeds is enough for stable AUC/F1
df, summary, md = evaluate(
    model_factory   = model_factory,
    target_loaders  = target_loaders,
    methods         = methods,
    cfg             = cfg,
    seeds           = (42, 123, 2024),
    output_dir      = "/kaggle/working/tta_results",
)

print(md)
"""
