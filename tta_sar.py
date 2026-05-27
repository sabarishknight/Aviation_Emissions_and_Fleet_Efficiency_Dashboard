# =========================================================
# /kaggle/working/tta_sar.py
#
# Canonical SAR (Niu et al., ICLR 2023, Oral).
#   - SAM optimizer (rho=0.05 by default)
#   - Two-stage filtering: select reliable samples on BOTH
#     forward passes (your previous version reused the
#     first-pass mask, which is incorrect)
#   - EMA-based MODEL RECOVERY when entropy collapses
#     (the *signature* feature of SAR — was missing entirely)
#   - Skip layer4 of ResNet (paper Section 4)
#   - Adapts BN + LN (skip the LN inside the final classifier)
#
# Original repo: github.com/mr-eggplant/SAR
# =========================================================

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tta_utils import (
    configure_model_bn_and_ln,
    configure_model_classifier_only,
    collect_params,
    copy_model_and_optimizer,
    load_model_and_optimizer,
    forward_logits,
    softmax_entropy,
    default_e_margin,
)


# =========================================================
# SAM Optimizer (Foret et al., ICLR 2021)
# =========================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device)
                for g in self.param_groups
                for p in g["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm


# =========================================================
# EMA helper for model-recovery
# =========================================================
def _update_ema(ema, value, m=0.9):
    return value if ema is None else m * ema + (1 - m) * value


# =========================================================
# SAR module (paper-faithful)
# =========================================================
class SAR(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        margin_e0=None,
        reset_constant_em=0.2,
        grad_clip=1.0,
        num_classes=5,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.margin_e0 = (
            margin_e0 if margin_e0 is not None else default_e_margin(num_classes, 0.4)
        )
        self.reset_constant_em = reset_constant_em
        self.grad_clip = grad_clip
        self.ema = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self._forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        # ---- first pass ----
        self.optimizer.zero_grad(set_to_none=True)
        outputs = forward_logits(self.model, x)
        ent1 = softmax_entropy(outputs)
        keep1 = torch.where(ent1 < self.margin_e0)
        ent1_sel = ent1[keep1]
        if ent1_sel.numel() == 0:
            return outputs.detach()

        loss1 = ent1_sel.mean(0)
        loss1.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                self.grad_clip,
            )
        self.optimizer.first_step(zero_grad=True)

        # ---- second pass at perturbed weights ----
        outputs2 = forward_logits(self.model, x)
        ent2_full = softmax_entropy(outputs2)
        ent2_sel = ent2_full[keep1]              # paper uses keep1 here
        loss_second_value = ent2_sel.detach().mean(0).item() if ent2_sel.numel() else float("nan")
        keep2 = torch.where(ent2_sel < self.margin_e0)
        ent2_final = ent2_sel[keep2]

        if ent2_final.numel() > 0:
            loss2 = ent2_final.mean(0)
            loss2.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.grad_clip,
                )
        self.optimizer.second_step(zero_grad=True)

        # ---- EMA + model recovery ----
        if not math.isnan(loss_second_value):
            self.ema = _update_ema(self.ema, loss_second_value, m=0.9)
        if self.ema is not None and self.ema < self.reset_constant_em:
            self.reset()

        return outputs2.detach()

    def reset(self):
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.ema = None


# =========================================================
# RUN SAR
# =========================================================
def run_sar(model, loader, cfg, return_probs=False):
    """
    cfg required fields:
        cfg.sar_lr            (default 1e-3)
        cfg.sar_rho           (default 0.05)
        cfg.sar_steps         (default 1)
        cfg.sar_margin_e0     (default 0.4 * ln(num_classes))
        cfg.sar_reset_em      (default 0.2)
        cfg.sar_grad_clip     (default 1.0)
        cfg.sar_mode          ("bn_and_ln_sar" or "classifier_only", default "bn_and_ln_sar")
        cfg.num_classes
    """
    num_classes = getattr(cfg, "num_classes", 5)
    mode        = getattr(cfg, "sar_mode", "bn_and_ln_sar")
    lr          = getattr(cfg, "sar_lr", 1e-3)
    rho         = getattr(cfg, "sar_rho", 0.05)
    steps       = getattr(cfg, "sar_steps", 1)
    margin_e0   = getattr(cfg, "sar_margin_e0", default_e_margin(num_classes, 0.4))
    reset_em    = getattr(cfg, "sar_reset_em", 0.2)
    grad_clip   = getattr(cfg, "sar_grad_clip", 1.0)

    if mode == "bn_and_ln_sar":
        model = configure_model_bn_and_ln(
            model, skip_last_block=True, skip_classifier_ln=True
        )
    elif mode == "classifier_only":
        model = configure_model_classifier_only(model)
    else:
        raise ValueError(f"Unknown sar_mode: {mode}")

    params, _ = collect_params(model, mode=mode)
    optimizer = SAM(params, torch.optim.Adam, lr=lr, rho=rho)

    sar = SAR(
        model,
        optimizer,
        steps=steps,
        margin_e0=margin_e0,
        reset_constant_em=reset_em,
        grad_clip=grad_clip,
        num_classes=num_classes,
    )
    sar.reset()  # clean start per target domain

    preds_all, probs_all, labels_all = [], [], []
    device = next(model.parameters()).device

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = sar(images)
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
