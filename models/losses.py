# =========================
# 📦 IMPORTS
# =========================
# torch: main PyTorch library for tensor operations and GPU acceleration
import torch
# torch.nn: neural network modules (here we use BCEWithLogitsLoss)
import torch.nn as nn
# torch.nn.functional: contains functions like interpolate (for resizing tensors)
import torch.nn.functional as F

# =========================
# 🔥 BCE LOSS (INSTANTIATED)
# =========================
# BCEWithLogitsLoss combines a sigmoid layer and binary cross-entropy in one class.
# It's more numerically stable than using sigmoid + BCE separately.
# Used as the pixel‑wise classification loss for binary segmentation.
bce_loss = nn.BCEWithLogitsLoss()


# =========================
# 🎲 DICE LOSS FROM LOGITS
# =========================
def dice_loss_from_logits(logits, targets, smooth=1e-6):
    """
    Computes the Dice loss from raw logits (before sigmoid).
    
    Dice loss = 1 - Dice coefficient.
    Dice coefficient measures overlap between prediction and ground truth.
    
    Args:
        logits: raw outputs from the model (B, 1, H, W)
        targets: ground truth masks (B, 1, H, W), values in {0,1}
        smooth: small constant to avoid division by zero
    
    Returns:
        scalar Dice loss
    """
    # Convert logits to probabilities via sigmoid
    probs = torch.sigmoid(logits)
    
    # Flatten each image/mask to a 1D vector per sample
    probs = probs.view(probs.size(0), -1)   # (B, H*W)
    targets = targets.view(targets.size(0), -1)  # (B, H*W)
    
    # Compute intersection and sum of both masks
    intersection = (probs * targets).sum(dim=1)   # (B,)
    denom = probs.sum(dim=1) + targets.sum(dim=1)  # (B,)
    
    # Dice coefficient per sample (with smoothing)
    dice = (2.0 * intersection + smooth) / (denom + smooth)  # (B,)
    
    # Return average Dice loss over the batch
    return 1.0 - dice.mean()


# =========================
# 🧠 DEEP SUPERVISION LOSS
# =========================
def deep_supervision_loss(outputs, targets, weights=(1.0, 0.5, 0.25, 0.125)):
    """
    Computes the total loss for UNet3+ with deep supervision.
    
    UNet3+ outputs four segmentation maps at different resolutions:
        out1: full resolution (largest)
        out2: 1/2 resolution
        out3: 1/4 resolution
        out4: 1/8 resolution (smallest)
    
    Each output is compared to the ground truth (after resizing) using
    BCE + Dice loss. The losses are combined with decreasing weights,
    giving more importance to higher‑resolution outputs.
    
    Args:
        outputs: tuple/list of 4 tensors from UNet3+ (out1, out2, out3, out4)
        targets: ground truth mask (B, 1, H, W) at full resolution
        weights: tuple of 4 weights for each output level (default: 1.0, 0.5, 0.25, 0.125)
    
    Returns:
        total_loss: scalar weighted sum of all losses
    """
    # Verify that we received exactly 4 outputs (as expected from UNet3+)
    assert len(outputs) == 4, "UNet3Plus should return 4 outputs."

    total_loss = 0.0
    
    # Iterate over each output and its corresponding weight
    for out, w in zip(outputs, weights):
        # If the output spatial size differs from target, resize output to target size
        # This typically happens for out2/out3/out4 which are smaller than full resolution
        if out.shape[2:] != targets.shape[2:]:
            out = F.interpolate(
                out,
                size=targets.shape[2:],
                mode="bilinear",
                align_corners=False
            )
        
        # Compute binary cross-entropy loss (logits vs. target)
        bce = bce_loss(out, targets)
        
        # Compute Dice loss (logits vs. target)
        dice = dice_loss_from_logits(out, targets)
        
        # Add weighted sum of (BCE + Dice) to total loss
        total_loss = total_loss + w * (bce + dice)
    
    return total_loss