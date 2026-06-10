# =========================
# 📦 IMPORTS
# =========================
# shutil: provides high-level file operations (copy, move, remove)
#         here we use shutil.copy2() to copy files while preserving metadata
import shutil

# pathlib: object-oriented interface for filesystem paths (Path objects)
#          easier and safer than os.path
from pathlib import Path

# typing: provides type hints for better code readability and IDE support
#         Dict[int, Path] means a dictionary with integer keys and Path values
from typing import Dict

# Import from our config folder import CLIENT_DIRS and DATASET_MAP
from config import CLIENT_DIRS, DATASET_MAP


# =========================
# 🗂️ CREATE CLIENT DIRECTORIES
# =========================
def create_client_directories() -> Dict[int, Path]:
    """
    Creates the folder structure for each federated client:
        client_N/
            images/
            masks/
    Returns a dictionary mapping client_id -> client_dir path.
    """
    result = {}
    for client_id, client_dir in CLIENT_DIRS.items():
        images_dir = client_dir / "images"
        masks_dir = client_dir / "masks"
        # Create directories (and parents if missing); no error if already exist
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        result[client_id] = client_dir
    return result


# =========================
# 📄 COPY DATASET DIRECTORY
# =========================
def copy_dataset_directory(source: Path, destination: Path) -> None:
    """
    Copies all files from source directory to destination directory.
    Skips files that already exist in the destination.
    """
    destination.mkdir(parents=True, exist_ok=True)
    # Iterate through all items in the source folder
    for path in sorted(source.iterdir()):
        if path.is_file():                     # Only copy files, not subfolders
            target = destination / path.name
            if not target.exists():            # Avoid overwriting existing files
                shutil.copy2(path, target)     # copy2 preserves metadata


# =========================
# 🚀 PREPARE FEDERATED DATA (MAIN FUNCTION)
# =========================
def prepare_federated_data() -> None:
    """
    Create the federated client folder layout under data/federated/4_clients.
    If raw data is available in data/raw/DB1..DB4, this function copies files into the client folders.
    This function is called by `main.py --role prepare-data`.
    """
    # First, create all client images/ and masks/ folders
    create_client_directories()

    # Loop over each client (1,2,3,4) and its corresponding raw dataset path
    for client_id, raw_path in DATASET_MAP.items():
        # Check if the raw dataset folder actually exists
        if not raw_path.exists():
            print(f"[WARN] raw dataset missing: {raw_path}")
            continue

        # Expect raw dataset structure: raw_path/images/ and raw_path/masks/
        images_src = raw_path / "images"
        masks_src = raw_path / "masks"
        if not images_src.exists() or not masks_src.exists():
            print(f"[WARN] raw dataset {raw_path} must contain 'images/' and 'masks/' subfolders.")
            continue

        # Destination is the federated client folder (e.g., data/federated/4_clients/client_1)
        client_dir = CLIENT_DIRS[client_id]
        
        # Copy all image files
        copy_dataset_directory(images_src, client_dir / "images")
        # Copy all mask files
        copy_dataset_directory(masks_src, client_dir / "masks")
        
        print(f"[INFO] prepared client {client_id} from raw dataset {raw_path}")