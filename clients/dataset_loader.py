# =========================
# 🗂️ FILE PURPOSE: Loads and preprocesses polyp segmentation datasets for each federated client.
#    Handles image/mask pairing, train/val/test splitting, data augmentation,
#    and creates PyTorch DataLoaders for training, validation, and testing.
# =========================

# =========================
# 📦 IMPORTS
# =========================
# cv2 (OpenCV): reads images and masks from disk
import cv2
# numpy: array operations for image/mask processing
import numpy as np
# torch: core PyTorch for tensor conversion
import torch
# pathlib: object‑oriented filesystem paths
from pathlib import Path
# sklearn: train_test_split for splitting dataset names
from sklearn.model_selection import train_test_split
# torch.utils.data: Dataset and DataLoader classes
from torch.utils.data import Dataset, DataLoader
# albumentations: fast and flexible image augmentation library
import albumentations as A
# ToTensorV2: converts numpy image to PyTorch tensor and scales to [0,1]
from albumentations.pytorch import ToTensorV2

# Import configuration constants
from config import (
    CLIENT_DIRS,        # mapping: client_id -> path to client's data folder
    IMAGE_SIZE,         # target resize size (e.g., 256)
    TRAIN_RATIO,        # proportion of data for training (e.g., 0.7)
    VAL_RATIO,          # proportion for validation (e.g., 0.15)
    TEST_RATIO,         # proportion for testing (e.g., 0.15)
    TRAINING_CONFIG,    # contains batch_size, num_workers, seed, etc.
)

# Allowed image file extensions (case‑insensitive)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =========================
# 🎨 DATA AUGMENTATION TRANSFORMS
# =========================
def build_transforms(train: bool = True):
    """
    Returns an albumentations Compose object for image/mask preprocessing.
    
    For training: applies random flips, affine transforms, color jitter, resizing, normalization.
    For validation/test: only resizing and normalization (no random augmentation).
    """
    if train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),                    # resize to fixed square
            A.HorizontalFlip(p=0.5),                             # random horizontal flip
            A.VerticalFlip(p=0.2),                               # random vertical flip
            A.Affine(translate_percent=0.05, scale=(0.90, 1.10), rotate=15, border_mode=0, p=0.5),
            A.ColorJitter(p=0.3),                                # random brightness/contrast
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet stats
            ToTensorV2(),                                         # to tensor and scale [0,1]
        ], additional_targets={"mask": "mask"})
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


# =========================
# 🧩 CUSTOM DATASET CLASS
# =========================
class PolypSegmentationDataset(Dataset):
    """
    PyTorch Dataset for polyp segmentation.
    Expects folder structure:
        root_dir/
            images/   (image files)
            masks/    (mask files, same base name as images)
    """
    def __init__(self, root_dir: Path, sample_names=None, transform=None):
        """
        Args:
            root_dir: path to client's data folder (e.g., data/federated/4_clients/client_1)
            sample_names: optional list of image base names to include (for splitting)
            transform: albumentations transform pipeline
        """
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.transform = transform

        self.samples = self._load_samples(sample_names)

    def _load_samples(self, sample_names):
        """Build list of (image_path, mask_path) tuples."""
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder must contain images/ and masks/: {self.root_dir}"
            )

        # Collect all image files with allowed extensions
        image_paths = [
            p for p in sorted(self.image_dir.iterdir())
            if p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        # Map mask stem -> mask path
        mask_paths = {
            p.stem: p for p in sorted(self.mask_dir.iterdir())
            if p.suffix.lower() in ALLOWED_EXTENSIONS
        }

        samples = []
        for image_path in image_paths:
            key = image_path.stem
            if key not in mask_paths:
                continue          # skip images without a corresponding mask
            if sample_names is not None and key not in sample_names:
                continue          # skip if not in the allowed list (for splits)
            samples.append((image_path, mask_paths[key]))

        if len(samples) == 0:
            raise ValueError(f"No matched image/mask samples found in {self.root_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_image(image_path):
        """Read image, convert BGR to RGB, return as float32."""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)

    @staticmethod
    def _read_mask(mask_path):
        """
        Read mask, convert to binary (0/1) float32.
        Handles grayscale, RGB, RGBA, and 2D masks.
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")

        # Convert various formats to a single binary channel
        if mask.ndim == 2:
            gray_mask = mask.astype(np.uint8)
        elif mask.ndim == 3:
            if mask.shape[2] == 4:           # RGBA: drop alpha
                mask = mask[:, :, :3]
            if mask.shape[2] == 3:           # RGB: any channel >127 -> foreground
                gray_mask = np.any(mask > 127, axis=2).astype(np.uint8) * 255
            elif mask.shape[2] == 1:         # single channel
                gray_mask = mask[:, :, 0].astype(np.uint8)
            else:
                gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape} for {mask_path}")

        binary_mask = (gray_mask > 127).astype(np.float32)
        return binary_mask

    def __getitem__(self, index):
        """Return (image_tensor, mask_tensor). Image shape: (3, H, W), mask: (1, H, W)."""
        image_path, mask_path = self.samples[index]
        image = self._read_image(image_path)
        mask = self._read_mask(mask_path)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Manual normalization (no augmentation)
            image = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Ensure mask has channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# =========================
# ✂️ SPLIT NAMES (TRAIN/VAL/TEST)
# =========================
def _split_names(root_dir: Path, seed: int = 42):
    """
    Split image base names into train, validation, test sets.
    Uses ratios from config (TRAIN_RATIO, VAL_RATIO, TEST_RATIO).
    """
    # Create a temporary dataset just to get the list of sample names
    dataset = PolypSegmentationDataset(root_dir, transform=None)
    names = [image_path.stem for image_path, _ in dataset.samples]

    # First split: separate train from the rest (val+test)
    train_names, temp_names = train_test_split(
        names,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=seed,
        shuffle=True,
    )

    # Split the remaining (temp) into val and test proportionally
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_names, test_names = train_test_split(
        temp_names,
        test_size=1.0 - val_size,
        random_state=seed,
        shuffle=True,
    )

    return train_names, val_names, test_names


# =========================
# 🚀 BUILD CLIENT DATA LOADERS
# =========================
def build_client_loaders(client_id: int):
    """
    Create and return train_loader, val_loader, test_loader for a given client.
    
    Args:
        client_id: integer (1..NUM_CLIENTS)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Validate client ID
    if client_id not in CLIENT_DIRS:
        raise ValueError(f"Unknown client id: {client_id}")

    root_dir = CLIENT_DIRS[client_id]
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Client dataset directory not found: {root_dir}. "
            "Run main.py --role prepare-data or verify the federated data folder."
        )

    # Get train/val/test split of sample names (deterministic based on seed)
    train_names, val_names, test_names = _split_names(root_dir, seed=TRAINING_CONFIG["seed"])

    # Create datasets with appropriate transforms
    train_dataset = PolypSegmentationDataset(
        root_dir, sample_names=train_names, transform=build_transforms(train=True)
    )
    val_dataset = PolypSegmentationDataset(
        root_dir, sample_names=val_names, transform=build_transforms(train=False)
    )
    test_dataset = PolypSegmentationDataset(
        root_dir, sample_names=test_names, transform=build_transforms(train=False)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,                # evaluate one image at a time
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader