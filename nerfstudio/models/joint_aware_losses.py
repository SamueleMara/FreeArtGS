"""ForceArtGS joint-aware part-supervision semantics (end-to-end spec section 3.6).

For an isolated part render: pixels confirmed as this part (``positive``) supervise RGB, depth and alpha=1;
verified visible absence of the whole object (``silhouette_valid & ~positive``) supervises alpha=0; unknown,
other-part (possibly occluding) and robot pixels give exactly zero gradient. Means are normalised by the
number of contributing pixels; an image without support returns zeros (and a flag), never NaN.
"""
from typing import Dict

import torch


def joint_aware_part_losses(pred_rgb, gt_rgb, pred_alpha, positive, silhouette_valid, pred_depth=None,
                            gt_depth=None, ssim_fn=None, ssim_lambda: float = 0.2, mask_lambda: float = 1.0,
                            depth_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
    """pred_rgb/gt_rgb [H,W,3], pred_alpha [H,W,1], positive/silhouette_valid [H,W,1] bool,
    depths [H,W,1] metres (gt 0 = invalid)."""
    pos = positive.float()
    sil = (silhouette_valid | positive).float()
    zero = pred_rgb.sum() * 0.0
    n_pos = pos.sum()
    out = {}
    if n_pos > 0:
        l1 = (torch.abs(gt_rgb - pred_rgb) * pos).sum() / (n_pos * 3.0)
        if ssim_fn is not None and ssim_lambda > 0:
            # SSIM on positive-masked images (both sides zero elsewhere -> no gradient outside positives)
            s = 1 - ssim_fn((gt_rgb * pos).permute(2, 0, 1)[None], (pred_rgb * pos).permute(2, 0, 1)[None])
            out["main_loss"] = (1 - ssim_lambda) * l1 + ssim_lambda * s
        else:
            out["main_loss"] = l1
    else:
        out["main_loss"] = zero
    n_sil = sil.sum()
    out["mask_loss_l1"] = (torch.abs(pos - pred_alpha) * sil).sum() / n_sil * mask_lambda if n_sil > 0 else zero
    if pred_depth is not None and gt_depth is not None:
        dv = pos * (gt_depth > 0).float()
        n_d = dv.sum()
        out["sensor_depth_loss"] = (torch.abs(pred_depth - gt_depth) * dv).sum() / n_d * depth_lambda if n_d > 0 else zero
    out["_support_pixels"] = n_pos.detach()
    return out
