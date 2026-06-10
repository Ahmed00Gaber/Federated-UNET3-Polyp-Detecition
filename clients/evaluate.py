# =========================
# 📦 IMPORTS
# =========================
# torch: main PyTorch library for tensor operations and evaluation mode
import torch
# torch.nn.functional: provides interpolation functions (for resizing outputs)
import torch.nn.functional as F
# deep_supervision_loss: combines BCE + Dice loss across all 4 UNet3+ outputs
from models.losses import deep_supervision_loss
# compute_metrics_from_logits: calculates precision, recall, dice, IoU from logits
from models.metrics import compute_metrics_from_logits


# =========================
# 📊 EVALUATE MODEL
# =========================
def evaluate_model(model, loader, device: str):
    """
    Evaluate the UNet3+ model on a given dataloader (validation or test set).
    
    Computes:
        - Deep supervision loss (BCE + Dice) averaged over all batches
        - Precision, Recall, Dice, IoU using the main (highest resolution) output
    
    Args:
        model: UNet3+ model (should be in eval mode)
        loader: DataLoader providing (images, masks) batches
        device: "cuda" or "cpu"
    
    Returns:
        dictionary with keys: loss, precision, recall, dice, iou
    """
    # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    model.eval()
    
    # Accumulators for averaging
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_steps = 0
    
    # Disable gradient computation (saves memory and speeds up evaluation)
    with torch.no_grad():
        for images, masks in loader:
            # Move batch to the correct device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass: get 4 deep supervision outputs (out1, out2, out3, out4)
            outputs = model(images)
            
            # Compute total loss (sum of weighted BCE+Dice across all 4 outputs)
            loss = deep_supervision_loss(outputs, masks)
            
            # For metrics, use only the main (full‑resolution) output = outputs[0]
            main_output = outputs[0]
            
            # If the main output size differs from ground truth (should not happen for out1,
            # but safe to handle), resize it to match the mask dimensions
            if main_output.shape[2:] != masks.shape[2:]:
                main_output = F.interpolate(
                    main_output,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            
            # Compute precision, recall, dice, iou from the main output logits
            precision, recall, dice, iou = compute_metrics_from_logits(main_output, masks)
            
            # Accumulate values
            total_loss += loss.item()
            total_precision += precision
            total_recall += recall
            total_dice += dice
            total_iou += iou
            total_steps += 1
    
    # Avoid division by zero if loader is empty
    if total_steps == 0:
        return {
            "loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "dice": 0.0,
            "iou": 0.0,
        }
    
    # Return averages over all batches
    return {
        "loss": total_loss / total_steps,
        "precision": total_precision / total_steps,
        "recall": total_recall / total_steps,
        "dice": total_dice / total_steps,
        "iou": total_iou / total_steps,
    }