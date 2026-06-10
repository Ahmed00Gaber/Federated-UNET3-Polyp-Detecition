# =========================
# 📦 IMPORTS
# =========================
# random: sets Python's random seed for reproducibility
import random

# numpy: used for setting random seed and potential array operations
import numpy as np

# torch: main PyTorch library for seeding, CUDA management, and device handling
import torch

# Import training configuration (device, learning_rate, local_epochs, seed)
# and model configuration (encoder_name, in_channels, num_classes, etc.)
from config import TRAINING_CONFIG, MODEL_CONFIG

# Import the UNet3+ model architecture
from models.unet3plus import UNet3Plus

# Import the deep supervision loss function (combines BCE + Dice across 4 outputs)
from models.losses import deep_supervision_loss


# =========================
# 🌱 REPRODUCIBILITY SEED
# =========================
def seed_everything(seed: int = 42):
    """
    Set all random seeds to ensure reproducible training runs.
    Affects Python's random, NumPy, PyTorch (CPU and CUDA).
    Also configures cuDNN for deterministic behavior (slower but reproducible).
    """
    random.seed(seed)                     # Python's random module
    np.random.seed(seed)                  # NumPy's random generator
    torch.manual_seed(seed)               # PyTorch CPU seed
    torch.cuda.manual_seed_all(seed)      # PyTorch all GPUs seed
    torch.backends.cudnn.deterministic = True   # Force deterministic cuDNN convolutions
    torch.backends.cudnn.benchmark = False      # Disable auto‑tuning for reproducibility


# =========================
# 🏗️ BUILD MODEL
# =========================
def build_model(device: str):
    """
    Create a UNet3+ model using parameters from MODEL_CONFIG.
    Move the model to the specified device (cuda or cpu).
    """
    model = UNet3Plus(
        encoder_name=MODEL_CONFIG["encoder_name"],       # e.g., "efficientnet-b0"
        encoder_weights=MODEL_CONFIG["encoder_weights"], # "imagenet"
        in_channels=MODEL_CONFIG["in_channels"],         # 3 for RGB
        num_classes=MODEL_CONFIG["num_classes"],         # 1 for binary segmentation
        cat_channels=MODEL_CONFIG["cat_channels"],       # 64
        decoder_channels=MODEL_CONFIG["decoder_channels"],# 256
    )
    return model.to(device)


# =========================
# 🏋️ LOCAL TRAINING (ONE ROUND)
# =========================
def train_client_one_round(model, train_loader, device: str, local_epochs: int):
    """
    Train the model on the client's local dataset for a fixed number of epochs.
    Uses Adam optimizer and the deep_supervision_loss.
    
    Args:
        model: UNet3+ model (already on some device)
        train_loader: DataLoader providing (image, mask) batches
        device: string "cuda" or "cpu"
        local_epochs: number of passes over the entire local dataset
    
    Returns:
        (trained_model, average_loss)
    """
    # Fix randomness for this training run
    seed_everything(TRAINING_CONFIG["seed"])
    
    # Convert device string to torch.device object
    device = torch.device(device)
    
    # Setup optimizer (Adam with learning rate from config)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    
    model.train()
    epoch_loss = 0.0
    total_steps = 0   # total number of batches processed (across all epochs)
    
    # Loop over local epochs
    for epoch in range(local_epochs):
        # Loop over batches in the training loader
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            
            try:
                # Forward pass: get 4 deep supervision outputs
                outputs = model(images)   # outputs = (out1, out2, out3, out4)
                
                # Compute combined loss (BCE + Dice) across all outputs
                loss = deep_supervision_loss(outputs, masks)
                
                # Backward pass and optimization step
                loss.backward()
                optimizer.step()
                
            except RuntimeError as exc:
                # Handle CUDA out-of-memory gracefully: fall back to CPU
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    print(
                        "[WARNING] CUDA out of memory during local training. "
                        "Falling back to CPU for this client."
                    )
                    torch.cuda.empty_cache()           # Free GPU memory
                    device = torch.device("cpu")       # Switch to CPU
                    model = model.to(device)           # Move model to CPU
                    # Recreate optimizer with model on CPU
                    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
                    # Move current batch to CPU and re-run forward/backward
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = deep_supervision_loss(outputs, masks)
                    loss.backward()
                    optimizer.step()
                else:
                    # Re-raise any other RuntimeError (e.g., shape mismatch)
                    raise
            
            # Accumulate loss for averaging
            epoch_loss += loss.item()
            total_steps += 1
    
    # Compute average loss per batch (across all epochs and batches)
    average_loss = epoch_loss / max(total_steps, 1)
    
    return model, average_loss