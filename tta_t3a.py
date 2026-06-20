# =========================================================
# /kaggle/working/tta_t3a.py
#
# Canonical T3A (Iwasawa & Matsuo, NeurIPS 2021).
#
# Key fixes vs. previous version:
#   - Initialize support set from CLASSIFIER WEIGHT ROWS
#     (the trained class prototypes). This is the *defining*
#     property of T3A and was missing.
#   - Use the 64-d input to the final linear as the feature
#     space, so prototypes share geometry with the trained
#     classifier.
#   - Per-class top-K filtering by entropy, no gradient updates.
#
# Original repo: github.com/matsuolab/T3A
# =========================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tta_utils import (
    forward_logits,
    softmax_entropy,
    extract_t3a_features,
    get_final_linear,
)


class T3A(nn.Module):
    def __init__(self, model, num_classes=5, filter_K=20):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.filter_K = filter_K  # max supports kept per class; -1 means keep all

        # ----- canonical init from classifier weights -----
        final = get_final_linear(model)
        warmup_supports = final.weight.data.clone().detach()  # [C, D]
        with torch.no_grad():
            warmup_logits = final(warmup_supports)             # [C, C]
            warmup_probs = warmup_logits.softmax(dim=1)
            warmup_ent = softmax_entropy(warmup_logits)
            warmup_labels = F.one_hot(
                warmup_probs.argmax(1), num_classes=num_classes
            ).float()

        self.register_buffer("supports", warmup_supports)
        self.register_buffer("labels", warmup_labels)
        self.register_buffer("ent", warmup_ent)

        # eval mode: no parameter updates whatsoever
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        # 1) feature in the 64-d space the classifier was trained on
        z = extract_t3a_features(self.model, x)  # [B, D]
        # 2) prediction from the original classifier (used only for pseudo-labels)
        final = get_final_linear(self.model)
        p = final(z)                              # [B, C]
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        # 3) extend support set
        self.supports = torch.cat([self.supports, z], dim=0)
        self.labels   = torch.cat([self.labels, yhat], dim=0)
        self.ent      = torch.cat([self.ent, ent], dim=0)

        # 4) per-class top-K (lowest entropy) selection
        sup_sel, lab_sel, ent_sel = [], [], []
        for c in range(self.num_classes):
            mask = self.labels[:, c] > 0
            if mask.sum() == 0:
                continue
            cls_sup = self.supports[mask]
            cls_lab = self.labels[mask]
            cls_ent = self.ent[mask]
            if self.filter_K > 0 and cls_ent.numel() > self.filter_K:
                idx = torch.argsort(cls_ent)[: self.filter_K]
                cls_sup, cls_lab, cls_ent = cls_sup[idx], cls_lab[idx], cls_ent[idx]
            sup_sel.append(cls_sup)
            lab_sel.append(cls_lab)
            ent_sel.append(cls_ent)
        self.supports = torch.cat(sup_sel, dim=0)
        self.labels   = torch.cat(lab_sel, dim=0)
        self.ent      = torch.cat(ent_sel, dim=0)

        # 5) class prototypes via labels^T @ normalized supports
        sup_norm = F.normalize(self.supports, dim=1)
        proto = self.labels.t() @ sup_norm                 # [C, D]
        proto = F.normalize(proto, dim=1)

        # 6) cosine-similarity logits
        z_norm = F.normalize(z, dim=1)
        return z_norm @ proto.t()                          # [B, C]


# =========================================================
# RUN T3A
# =========================================================
def run_t3a(model, loader, cfg, return_probs=False):
    """
    cfg required fields:
        cfg.num_classes        (default 5)
        cfg.t3a_max_supports   (int K, default 20; -1 = unbounded)
    """
    num_classes = getattr(cfg, "num_classes", 5)
    filter_K    = getattr(cfg, "t3a_max_supports", 20)

    t3a = T3A(model, num_classes=num_classes, filter_K=filter_K)

    preds_all, probs_all, labels_all = [], [], []
    device = next(model.parameters()).device

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = t3a(images)
        # cosine-sim logits: temperature can help; default 1
        probs = F.softmax(logits / getattr(cfg, "t3a_temperature", 1.0), dim=1)
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
