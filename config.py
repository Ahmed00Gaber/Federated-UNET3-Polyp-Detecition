"""
config.py

Purpose:
---------
This file contains all global configuration settings for the Federated UNet3+ project.
It centralizes paths, dataset structure, training parameters, and model configuration
to ensure consistency across the entire system (server, clients, and utilities).

-contains:
- Project paths for data and saved models
- Dataset mapping for federated clients
- Model architecture configuration
- Training hyperparameters (rounds, epochs, learning rate, etc.)


Why this is important:
----------------------
- Avoids hardcoding values across multiple files
- Makes experiments reproducible
- Allows easy tuning of hyperparameters
- Controls how federated learning is executed
"""

from pathlib import Path
import torch

# =========================
# 📁 PROJECT PATHS
# =========================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Main data directory
DATA_ROOT = PROJECT_ROOT / "data"

# Raw datasets (before splitting into federated clients)
RAW_DATA_ROOT = DATA_ROOT / "raw"

# Federated dataset directory (after splitting)
FEDERATED_ROOT = DATA_ROOT / "federated"

# Directory containing all client datasets (4 clients setup)
CLIENTS_ROOT = FEDERATED_ROOT / "4_clients"


# =========================
# 🧪 DATASET CONFIGURATION
# =========================

# Mapping each client ID to a specific dataset
# This simulates different hospitals having different datasets
DATASET_MAP = {
    1: RAW_DATA_ROOT / "DB1_CVC_Clinic",
    2: RAW_DATA_ROOT / "DB2_ETIS-Larib",
    3: RAW_DATA_ROOT / "DB3_kvasir",
    4: RAW_DATA_ROOT / "DB4_bkai-igh-neopolyp",
}

# Directory for each client after data splitting
# Example: client_1 → data/federated/4_clients/client_1
CLIENT_DIRS = {
    client_id: CLIENTS_ROOT / f"client_{client_id}"
    for client_id in DATASET_MAP
}

# Total number of federated clients
NUM_CLIENTS = len(CLIENT_DIRS)

# =========================
# 🌐 SERVER CONFIGURATION
# =========================

# Address where the federated server will run
SERVER_ADDRESS = "127.0.0.1:8080"

# Port used by the server
SERVER_PORT = 8080

# =========================
# 🧠 MODEL CONFIGURATION
# =========================
# Configuration for UNet3+ model
MODEL_CONFIG = {
    "encoder_name": "efficientnet-b0",   # Backbone network
    "encoder_weights": "imagenet",       # Pretrained weights
    "in_channels": 3,                    # RGB images
    "num_classes": 1,                    # Binary segmentation
    "cat_channels": 64,                  # Feature concatenation channels
    "decoder_channels": 256,             # Decoder depth size
}

# =========================
# ⚙️ TRAINING CONFIGURATION
# =========================

TRAINING_CONFIG = {
    "rounds": 42,                        # Number of federated rounds (global updates)
    "local_epochs": 2,                   # Training epochs per client per round
    "batch_size": 1,                     # Batch size (small due to medical images)
    "learning_rate": 1e-4,               # Optimizer learning rate
    "num_workers": 0,                    # DataLoader workers (0 = safer for Windows)
    
    # Use GPU if available, otherwise CPU
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,                          # Random seed for reproducibility
}

# =========================
# 💾 SAVING RESULTS
# =========================

# Directory to store trained models and metrics
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# File to store IoU results per round (used for plotting)
SAVED_METRICS_PATH = SAVED_MODELS_DIR / "round_iou.csv"

# =========================
# 🖼️ IMAGE SETTINGS
# =========================

# Input image size (images will be resized to 256x256)
IMAGE_SIZE = 256

# =========================
# 📊 DATA SPLITTING RATIOS
# =========================

# Ratio of dataset split for each client
TRAIN_RATIO = 0.70   # 70% training
VAL_RATIO = 0.15     # 15% validation
TEST_RATIO = 0.15    # 15% testing