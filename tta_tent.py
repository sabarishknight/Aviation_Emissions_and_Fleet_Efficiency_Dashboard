# =========================================================
# /kaggle/working/tta_tent.py
#
# Canonical TENT (Wang et al., ICLR 2021).
#   - BN-only adaptation OR classifier-only ("dagger" mode)
#   - BN running stats frozen
#   - Episodic reset between datasets
#   - Gradient clipping for fundus stability
#
# Original repo: github.com/DequanWang/tent
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
)


# =========================================================
# Tent module (paper-faithful)
# =========================================================
class Tent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=True, grad_clip=1.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.grad_clip = grad_clip
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
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                self.grad_clip,
            )
        self.optimizer.step()
        return outputs

    def reset(self):
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


# =========================================================
# RUN TENT
# =========================================================
def run_tta(model, loader, cfg, return_probs=False):
    """
    cfg required fields:
        cfg.tent_lr          (float, default 1e-3 for BN-only, 1e-4 for classifier-only)
        cfg.tent_steps       (int,   default 1)
        cfg.tent_mode        (str,   "bn_only" or "classifier_only", default "bn_only")
        cfg.tent_episodic    (bool,  default True; reset per call to run_tta)
        cfg.tent_grad_clip   (float, default 1.0)
    """
    mode      = getattr(cfg, "tent_mode", "bn_only")
    lr        = getattr(cfg, "tent_lr", 1e-3 if mode == "bn_only" else 1e-4)
    steps     = getattr(cfg, "tent_steps", 1)
    episodic  = getattr(cfg, "tent_episodic", True)
    grad_clip = getattr(cfg, "tent_grad_clip", 1.0)

    if mode == "bn_only":
        model = configure_model_bn_only(model)
    elif mode == "classifier_only":
        model = configure_model_classifier_only(model)
    else:
        raise ValueError(f"Unknown tent_mode: {mode}")

    params, _ = collect_params(model, mode=mode)
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    tent = Tent(model, optimizer, steps=steps, episodic=episodic, grad_clip=grad_clip)
    if episodic:
        tent.reset()  # clean start for each target domain

    preds_all, probs_all, labels_all = [], [], []
    device = next(model.parameters()).device

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = tent(images)
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
