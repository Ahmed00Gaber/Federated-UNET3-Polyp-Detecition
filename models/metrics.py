# =========================
# 📦 IMPORTS
# =========================
# torch: main PyTorch library for tensor operations
# We use it for sigmoid, tensor comparisons, and reshaping
import torch


# =========================
# 🎯 THRESHOLD PREDICTIONS
# =========================
def threshold_predictions(logits, threshold=0.5):
    """
    Convert raw logits to binary predictions using a threshold.
    
    Args:
        logits: raw outputs from the model (before sigmoid)
        threshold: probability threshold (default 0.5)
    
    Returns:
        binary tensor of same shape as logits, values 0 or 1
    """
    # Apply sigmoid to get probabilities in (0, 1)
    probs = torch.sigmoid(logits)
    # Convert to binary: 1 if prob > threshold else 0
    preds = (probs > threshold).float()
    return preds


# =========================
# 📊 CONFUSION MATRIX ELEMENTS
# =========================
def compute_confusion_elements(preds, targets, eps=1e-7):
    """
    Compute True Positives, False Positives, False Negatives, True Negatives.
    
    Args:
        preds: binary predictions (0 or 1)
        targets: ground truth binary masks (0 or 1)
        eps: small epsilon (unused here, kept for consistency)
    
    Returns:
        tp, fp, fn, tn as scalar tensors
    """
    # Flatten both tensors to 1D vectors for easy summation
    preds = preds.float().view(-1)
    targets = targets.float().view(-1)
    
    # True Positive: predicted 1 and actually 1
    tp = (preds * targets).sum()
    
    # False Positive: predicted 1 but actually 0
    fp = (preds * (1.0 - targets)).sum()
    
    # False Negative: predicted 0 but actually 1
    fn = ((1.0 - preds) * targets).sum()
    
    # True Negative: predicted 0 and actually 0
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    
    return tp, fp, fn, tn


# =========================
# 📈 METRICS FROM LOGITS
# =========================
def compute_metrics_from_logits(logits, targets, threshold=0.5):
    """
    Compute Precision, Recall, Dice, and IoU from model logits.
    
    Args:
        logits: raw model outputs (B, 1, H, W)
        targets: ground truth masks (B, 1, H, W) with values 0 or 1
        threshold: threshold for binarizing probabilities (default 0.5)
    
    Returns:
        precision, recall, dice, iou as Python floats
    """
    # Step 1: convert logits to binary predictions
    preds = threshold_predictions(logits, threshold=threshold)
    
    # Step 2: compute confusion matrix elements
    tp, fp, fn, tn = compute_confusion_elements(preds, targets)
    
    # Step 3: calculate metrics (add small epsilon to avoid division by zero)
    precision = tp / (tp + fp + 1e-7)   # of predicted positives, how many were correct?
    recall = tp / (tp + fn + 1e-7)      # of actual positives, how many were found?
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)   # harmonic mean of precision and recall
    iou = tp / (tp + fp + fn + 1e-7)              # intersection over union (Jaccard index)
    
    # Convert from 0‑dimensional tensors to Python numbers
    return precision.item(), recall.item(), dice.item(), iou.item()