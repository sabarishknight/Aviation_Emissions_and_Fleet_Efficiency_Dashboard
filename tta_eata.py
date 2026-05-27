# =========================================================
# /kaggle/working/tta_eata.py
#
# Canonical EATA (Niu et al., ICML 2022).
#   - Two filters:
#       (1) entropy < e_margin   (reliable)
#       (2) cos-sim to running-mean prob < d_margin  (non-redundant)
#   - Per-sample re-weighting: coeff = 1/exp(ent - e_margin)
#   - Optional Fisher regularizer for anti-forgetting
#     (computed on a held-out source / training subset)
#
# Original repo: github.com/mr-eggplant/EATA
# =========================================================

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tta_utils import (
    configure_model_bn_only,
    configure_model_classifier_only,
    collect_params,
    copy_model_and_optimizer,
    load_model_and_optimizer,
    forward_logits,
    softmax_entropy,
    default_e_margin,
)


# =========================================================
# Probability EMA helper
# =========================================================
def _update_probs(current, new_probs, m=0.9):
    if current is None:
        return None if new_probs.numel() == 0 else new_probs.mean(0).detach()
    if new_probs.numel() == 0:
        return current
    with torch.no_grad():
        return m * current + (1 - m) * new_probs.mean(0)


# =========================================================
# Optional: compute Fisher info on a labeled source loader
# =========================================================
@torch.no_grad()
def _params_dict(model, mode="bn_only"):
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    out = {}
    for nm, p in base.named_parameters():
        if p.requires_grad:
            out[nm] = p
    return out


def compute_fishers(model, source_loader, num_classes, device,
                    mode="bn_only", num_steps=200):
    """
    Empirical Fisher diagonal on the BN params (or whatever mode selects),
    used by EATA's anti-forgetting regularizer.

    fishers[name] = (fisher_diag_tensor, theta_source_tensor)
    """
    # configure same way EATA will use it
    if mode == "bn_only":
        model = configure_model_bn_only(model)
    elif mode == "classifier_only":
        model = configure_model_classifier_only(model)
    params, names = collect_params(model, mode=mode)

    # snapshot source params (theta_source)
    theta_src = {n: p.detach().clone() for n, p in zip(names, params)}

    # accumulate squared grads of CE wrt selected params
    fishers = {n: torch.zeros_like(p) for n, p in zip(names, params)}
    ce = nn.CrossEntropyLoss()
    seen = 0
    model.train()
    for step, (x, y) in enumerate(source_loader):
        if step >= num_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        logits = forward_logits(model, x)
        loss = ce(logits, y)
        loss.backward()
        for n, p in zip(names, params):
            if p.grad is not None:
                fishers[n] += p.grad.detach() ** 2
        seen += 1

    if seen > 0:
        for n in fishers:
            fishers[n] /= seen

    return {n: (fishers[n], theta_src[n]) for n in fishers}


# =========================================================
# EATA module (paper-faithful)
# =========================================================
class EATA(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        params_for_ewc=None,
        param_names_for_ewc=None,
        fishers=None,
        fisher_alpha=2000.0,
        steps=1,
        e_margin=None,
        d_margin=0.05,
        grad_clip=1.0,
        num_classes=5,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.e_margin = (
            e_margin if e_margin is not None else default_e_margin(num_classes, 0.4)
        )
        self.d_margin = d_margin
        self.grad_clip = grad_clip
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.params_for_ewc = params_for_ewc or []
        self.param_names_for_ewc = param_names_for_ewc or []
        self.current_model_probs = None

        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self._forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        outputs = forward_logits(self.model, x)
        probs = outputs.softmax(1)

        # ---- filter 1: reliable (low-entropy) samples ----
        ent = softmax_entropy(outputs)
        keep1 = torch.where(ent < self.e_margin)
        ent_kept = ent[keep1]
        probs_kept = probs[keep1]

        if ent_kept.numel() == 0:
            return outputs.detach()

        # ---- filter 2: non-redundant samples (cosine sim) ----
        if self.current_model_probs is not None:
            cos_sim = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0),
                probs_kept,
                dim=1,
            )
            keep2 = torch.where(cos_sim.abs() < self.d_margin)
            ent_final = ent_kept[keep2]
            probs_final = probs_kept[keep2]
        else:
            ent_final = ent_kept
            probs_final = probs_kept

        # ---- update running prob EMA ----
        self.current_model_probs = _update_probs(
            self.current_model_probs, probs_final, m=0.9
        )

        if ent_final.numel() == 0:
            return outputs.detach()

        # ---- per-sample reweighting ----
        coeff = 1.0 / torch.exp(ent_final.detach() - self.e_margin)
        loss = (ent_final * coeff).mean(0)

        # ---- optional Fisher anti-forgetting ----
        if self.fishers is not None and len(self.params_for_ewc) > 0:
            ewc = 0.0
            for nm, p in zip(self.param_names_for_ewc, self.params_for_ewc):
                if nm in self.fishers:
                    f_diag, theta_src = self.fishers[nm]
                    ewc = ewc + (f_diag * (p - theta_src) ** 2).sum()
            loss = loss + self.fisher_alpha * ewc

        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                self.grad_clip,
            )
        self.optimizer.step()
        return outputs.detach()

    def reset(self):
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.current_model_probs = None


# =========================================================
# RUN EATA
# =========================================================
def run_eata(model, loader, cfg, return_probs=False, source_loader=None):
    """
    cfg required fields:
        cfg.num_classes
        cfg.eata_lr           (default 1e-3 BN-only / 1e-4 classifier-only)
        cfg.eata_steps        (default 1)
        cfg.eata_e_margin     (default 0.4 * ln(C))
        cfg.eata_d_margin     (default 0.05)
        cfg.eata_fisher_alpha (default 2000.0)
        cfg.eata_use_fisher   (default False; needs source_loader)
        cfg.eata_fisher_steps (default 200)
        cfg.eata_grad_clip    (default 1.0)
        cfg.eata_mode         ("bn_only" or "classifier_only")
    """
    num_classes  = getattr(cfg, "num_classes", 5)
    mode         = getattr(cfg, "eata_mode", "bn_only")
    lr           = getattr(cfg, "eata_lr", 1e-3 if mode == "bn_only" else 1e-4)
    steps        = getattr(cfg, "eata_steps", 1)
    e_margin     = getattr(cfg, "eata_e_margin", default_e_margin(num_classes, 0.4))
    d_margin     = getattr(cfg, "eata_d_margin", 0.05)
    fisher_alpha = getattr(cfg, "eata_fisher_alpha", 2000.0)
    use_fisher   = getattr(cfg, "eata_use_fisher", False)
    fisher_steps = getattr(cfg, "eata_fisher_steps", 200)
    grad_clip    = getattr(cfg, "eata_grad_clip", 1.0)

    device = next(model.parameters()).device

    fishers = None
    if use_fisher:
        if source_loader is None:
            raise ValueError("EATA Fisher requires a source_loader.")
        fishers = compute_fishers(
            model, source_loader, num_classes, device,
            mode=mode, num_steps=fisher_steps,
        )

    if mode == "bn_only":
        model = configure_model_bn_only(model)
    elif mode == "classifier_only":
        model = configure_model_classifier_only(model)
    else:
        raise ValueError(f"Unknown eata_mode: {mode}")

    params, names = collect_params(model, mode=mode)
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    eata = EATA(
        model,
        optimizer,
        params_for_ewc=params,
        param_names_for_ewc=names,
        fishers=fishers,
        fisher_alpha=fisher_alpha,
        steps=steps,
        e_margin=e_margin,
        d_margin=d_margin,
        grad_clip=grad_clip,
        num_classes=num_classes,
    )
    eata.reset()  # clean start per target domain

    preds_all, probs_all, labels_all = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = eata(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        preds_all.extend(preds.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        labels_all.extend(labels.numpy())

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    probs_all = np.concatenate(probs_all)

    if return_probs:
        return preds_all, labels_all, probs_all
    return preds_all, labels_all
