# =========================================================
# /kaggle/working/tta_utils.py
#
# Shared utilities for TENT / SAR / EATA / T3A,
# tailored to your DeepSetModel (ResNet50 backbone +
# LayerNorm heads + ordinal head).
#
# Key fixes vs. previous version:
#   - configure_model splits into 3 explicit modes:
#       * bn_only         -> TENT / EATA canonical
#       * bn_and_ln       -> SAR canonical (skip layer4 + last LN)
#       * classifier_only -> "dagger" protocol (FunOTTA fundus)
#   - BN running stats are FROZEN (forces batch stats),
#     matching the official TENT/SAR/EATA repos.
#   - Adds save/load of model+optimizer state for episodic
#     resets (per-domain reset is critical for fair eval).
#   - T3A feature extractor returns 64-d, the *true* input
#     to the final classifier linear, so cosine-similarity
#     prototypes match the trained classifier geometry.
# =========================================================

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- model unwrap --------------------

def get_base_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


# -------------------- forward helpers --------------------

def forward_logits(model, x):
    """Robust forward: handles (logits), (logits, nodes), (logits, nodes, ord)."""
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


def _features_fused(base, x):
    """Return the 256-d fused feature (post FusionBlock)."""
    feat = base.features(x)
    global_feat = base.global_proj(base.gem_pool(feat))
    nodes = feat.flatten(2).transpose(1, 2)
    nodes = base.node_proj(nodes)
    pooled_nodes = base.attention_pool(nodes)
    fused = torch.cat([pooled_nodes, global_feat], dim=1)
    fused = base.fusion_block(fused)
    return fused


def extract_features(model, x):
    """Backwards-compatible alias: returns 256-d fused feature."""
    return _features_fused(get_base_model(model), x)


def extract_t3a_features(model, x):
    """
    Features that feed the FINAL classifier linear (64-d).
    Required for canonical T3A so that prototype cosine
    similarity matches the trained classifier geometry.
    """
    base = get_base_model(model)
    fused = _features_fused(base, x)
    h = fused
    # apply every layer of the classifier head EXCEPT the last Linear
    for layer in list(base.classifier.children())[:-1]:
        h = layer(h)
    return h  # shape [B, 64]


def get_final_linear(model):
    """Return the final nn.Linear of the classifier head (the prototype layer)."""
    base = get_base_model(model)
    last = list(base.classifier.children())[-1]
    assert isinstance(last, nn.Linear), \
        "Final layer of classifier must be nn.Linear for T3A."
    return last


# -------------------- entropy --------------------

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits) along dim=1."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# -------------------- save / restore --------------------

def copy_model_and_optimizer(model, optimizer):
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None,
    )


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)


# -------------------- BN handling --------------------

def freeze_bn_running_stats(model):
    """
    Force BN to use BATCH statistics during TTA.
    This is what the TENT/SAR/EATA papers do; without it,
    you are still using stale source-domain running stats
    and adaptation barely helps.
    """
    base = get_base_model(model)
    for m in base.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


# -------------------- configure modes --------------------

def configure_model_bn_only(model):
    """
    Canonical TENT / EATA setup:
      - model.train()
      - all params frozen
      - BN affine (gamma, beta) require_grad
      - BN running stats disabled (batch stats only)
    """
    base = get_base_model(model)
    base.train()
    base.requires_grad_(False)
    for m in base.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def configure_model_bn_and_ln(model, skip_last_block=True, skip_classifier_ln=True):
    """
    Canonical SAR setup:
      - update BN in backbone (except layer4 if skip_last_block)
      - update LN in heads (except inside classifier head if
        skip_classifier_ln; this LN is right before the prediction)
      - BN running stats disabled (batch stats only)
    """
    base = get_base_model(model)
    base.train()
    base.requires_grad_(False)

    for nm, m in base.named_modules():
        if skip_last_block and nm.startswith("features.7"):
            # ResNet50 layer4 lives at features.7 in your nn.Sequential wrap
            continue
        if skip_classifier_ln and nm.startswith("classifier"):
            # don't touch the LN feeding the final logits
            continue

        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def configure_model_classifier_only(model):
    """
    "Dagger" protocol from FunOTTA: only the classifier head
    (and ordinal head) get adapted. This is what FunOTTA found
    works on fundus when full BN-style adaptation collapses.
    """
    base = get_base_model(model)
    base.train()
    base.requires_grad_(False)
    for p in base.classifier.parameters():
        p.requires_grad_(True)
    if hasattr(base, "ordinal_head"):
        for p in base.ordinal_head.parameters():
            p.requires_grad_(True)
    # backbone stays in eval mode so BN uses source running stats
    base.features.eval()
    return model


# -------------------- collect params --------------------

def collect_bn_params(model, skip_last_block=False):
    base = get_base_model(model)
    params, names = [], []
    for nm, m in base.named_modules():
        if skip_last_block and nm.startswith("features.7"):
            continue
        if isinstance(m, nn.BatchNorm2d):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def collect_ln_params(model, skip_classifier_ln=True):
    base = get_base_model(model)
    params, names = [], []
    for nm, m in base.named_modules():
        if skip_classifier_ln and nm.startswith("classifier"):
            continue
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias") and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def collect_classifier_params(model):
    base = get_base_model(model)
    params, names = [], []
    for nm, p in base.classifier.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(f"classifier.{nm}")
    if hasattr(base, "ordinal_head"):
        for nm, p in base.ordinal_head.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(f"ordinal_head.{nm}")
    return params, names


def collect_params(model, mode="bn_only"):
    """
    Unified entry point.
      mode in {"bn_only", "bn_and_ln_sar", "classifier_only"}
    """
    if mode == "bn_only":
        return collect_bn_params(model, skip_last_block=False)
    if mode == "bn_and_ln_sar":
        bn_p, bn_n = collect_bn_params(model, skip_last_block=True)
        ln_p, ln_n = collect_ln_params(model, skip_classifier_ln=True)
        return bn_p + ln_p, bn_n + ln_n
    if mode == "classifier_only":
        return collect_classifier_params(model)
    raise ValueError(f"Unknown mode: {mode}")


# -------------------- defaults helpers --------------------

def default_e_margin(num_classes, scale=0.4):
    """E_0 = scale * ln(C). With scale=0.4 this matches the SAR paper."""
    return scale * math.log(num_classes)
