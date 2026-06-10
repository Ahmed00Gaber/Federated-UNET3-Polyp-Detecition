from pathlib import Path
import csv
import matplotlib.pyplot as plt

# visualization.py helps read federated training history from disk and
# generate plot images for average IoU and average loss per round.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
HISTORY_CSV = SAVED_MODELS_DIR / "round_iou.csv"


def read_training_history(path: Path = HISTORY_CSV):
    """Read the training history CSV and return round, IoU, and loss lists."""
    if not path.exists():
        raise FileNotFoundError(f"Training history file not found: {path}")

    rounds = []
    avg_iou = []
    avg_loss = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rounds.append(int(row["round"]))
            avg_iou.append(float(row["avg_iou"]))
            avg_loss.append(float(row["avg_loss"]))

    return rounds, avg_iou, avg_loss


def plot_round_iou(path: Path = HISTORY_CSV, output_path: Path | None = None):
    rounds, avg_iou, _ = read_training_history(path)

    # Use a default output path when none is provided
    output_path = output_path or SAVED_MODELS_DIR / "round_iou.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, avg_iou, marker="o")
    plt.title("Federated Training: Round vs. Average IoU")
    plt.xlabel("Round")
    plt.ylabel("Average IoU")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"IoU plot saved at: {output_path}")
    return output_path


def plot_round_loss(path: Path = HISTORY_CSV, output_path: Path | None = None):
    rounds, _, avg_loss = read_training_history(path)

    # Use a default output path when none is provided
    output_path = output_path or SAVED_MODELS_DIR / "round_loss.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, avg_loss, marker="o")
    plt.title("Federated Training: Round vs. Average Loss")
    plt.xlabel("Round")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Loss plot saved at: {output_path}")
    return output_path


# ✅ MAIN RUN
if __name__ == "__main__":
    print("Running visualization...")

    iou_path = plot_round_iou()
    loss_path = plot_round_loss()

    print("IoU saved at:", iou_path)
    print("Loss saved at:", loss_path)