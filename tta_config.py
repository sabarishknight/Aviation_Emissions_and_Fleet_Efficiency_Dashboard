# =========================================================
# /kaggle/working/tta_config.py
#
# Recommended starting hyperparameters for the four TTA
# baselines on EyePACS -> {MESSIDOR, IDRID, DeepDRiD, APTOS, DDR}
# (DR grading, 5 classes, ResNet50 + LayerNorm-head model).
#
# These are sensible *defaults* that match the original
# papers, scaled for 5-class fundus. ALWAYS sweep lr on a
# held-out target before finalizing your table.
# =========================================================

import math


class TTAConfig:
    # ---- shared ----
    num_classes = 5
    batch_size  = 32           # >= 32 strongly recommended for BN-based methods
    image_size  = 512          # match training

    # ---- TENT (BN-only canonical) ----
    tent_mode      = "bn_only"           # or "classifier_only" (FunOTTA-style)
    tent_lr        = 1e-3                # 1e-4 if classifier_only
    tent_steps     = 1
    tent_episodic  = True                # reset per target domain
    tent_grad_clip = 1.0

    # ---- SAR (BN + LN, skip layer4) ----
    sar_mode       = "bn_and_ln_sar"     # or "classifier_only"
    sar_lr         = 1e-3
    sar_rho        = 0.05
    sar_steps      = 1
    sar_margin_e0  = 0.4 * math.log(num_classes)   # ~0.644 for C=5
    sar_reset_em   = 0.2
    sar_grad_clip  = 1.0

    # ---- EATA (BN-only canonical) ----
    eata_mode         = "bn_only"        # or "classifier_only"
    eata_lr           = 1e-3
    eata_steps        = 1
    eata_e_margin     = 0.4 * math.log(num_classes)
    eata_d_margin     = 0.05
    eata_fisher_alpha = 2000.0
    eata_use_fisher   = False            # set True + provide source_loader for full EATA
    eata_fisher_steps = 200
    eata_grad_clip    = 1.0

    # ---- T3A (no learning rate) ----
    t3a_max_supports = 20                # K per class; sweep {5, 20, 50, 100}
    t3a_temperature  = 1.0
