# =========================
# 📦 IMPORTS
# =========================
# argparse: parses command-line arguments (e.g., --client-id, --server-address)
import argparse
# torch: main PyTorch library for tensors and converting between CPU/GPU
import torch
# flwr (Flower): federated learning framework.
# NumPyClient: base class for a client that communicates via NumPy arrays.
# start_numpy_client: starts a client that connects to a Flower server.
import flwr as fl

# Import configuration values from config.py
from config import SERVER_ADDRESS, TRAINING_CONFIG, NUM_CLIENTS, CLIENT_DIRS

# Import client‑side helper functions
# build_model: creates a UNet3+ model
# train_client_one_round: performs local training for a single round
from clients.train import build_model, train_client_one_round

# evaluate_model: computes loss and metrics (IoU, Dice, etc.) on validation set
from clients.evaluate import evaluate_model

# build_client_loaders: creates DataLoaders for train/val/test splits for a given client
from clients.dataset_loader import build_client_loaders


# =========================
# 🔧 UTILITY: GET MODEL PARAMETERS
# =========================
def _get_parameters(model):
    """
    Extract model parameters (weights and biases) as a list of NumPy arrays.
    This format is required by Flower's NumPyClient.
    """
    # state_dict() returns an OrderedDict of parameter tensors.
    # .detach() removes gradient tracking, .cpu() moves to CPU, .numpy() converts to NumPy.
    return [param.detach().cpu().numpy() for param in model.state_dict().values()]


# =========================
# 🔧 UTILITY: SET MODEL PARAMETERS
# =========================
def _set_parameters(model, parameters):
    """
    Load a list of NumPy arrays back into the model's state_dict.
    This is the inverse of _get_parameters.
    """
    # Get the current state_dict (which knows the parameter names and shapes)
    state_dict = model.state_dict()
    # Iterate over parameter names and the incoming NumPy arrays
    for key, array in zip(state_dict.keys(), parameters):
        # Convert NumPy array back to a PyTorch tensor with the correct dtype
        state_dict[key] = torch.tensor(array, dtype=state_dict[key].dtype)
    # Load the updated state_dict into the model
    model.load_state_dict(state_dict)


# =========================
# 👤 FEDERATED CLIENT (FLOWER NUMPY CLIENT)
# =========================
class PolypClient(fl.client.NumPyClient):
    """
    A Flower client that participates in federated learning.
    Each client holds its own local data, model, and implements:
        - get_parameters: send current model weights to the server
        - fit: train on local data (using server‑provided weights)
        - evaluate: evaluate on local validation set
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device = TRAINING_CONFIG["device"]   # "cuda" or "cpu"
        
        # Build the UNet3+ model and move it to the device
        self.model = build_model(self.device)
        
        # Create data loaders for this client (train, validation, test)
        self.train_loader, self.val_loader, self.test_loader = build_client_loaders(client_id)
        
        print(
            f"[CLIENT {client_id}] initialized on {self.device} "
            f"with {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val, "
            f"{len(self.test_loader.dataset)} test samples"
        )

    def get_parameters(self, config=None):
        """
        Called by the server to get the client's current model weights.
        Returns: list of NumPy arrays (weights + biases)
        """
        return _get_parameters(self.model)

    def fit(self, parameters, config):
        """
        Called by the server to train the client on its local data.
        
        Args:
            parameters: global model weights sent by the server
            config: dictionary with training hyperparameters (e.g., local_epochs)
        
        Returns:
            (updated_parameters, num_examples, metrics_dict)
        """
        # Overwrite local model with the global parameters
        _set_parameters(self.model, parameters)
        
        # Determine number of local epochs (can be overridden by server config)
        local_epochs = TRAINING_CONFIG["local_epochs"]
        if config and "local_epochs" in config:
            local_epochs = int(config["local_epochs"])
        
        print(f"[CLIENT {self.client_id}] starting fit for {local_epochs} epochs")
        
        # Train the model for one round (local_epochs passes over the local dataset)
        self.model, loss = train_client_one_round(
            self.model,
            self.train_loader,
            self.device,
            local_epochs,
        )
        
        num_examples = len(self.train_loader.dataset)
        print(f"[CLIENT {self.client_id}] fit finished, loss={loss:.4f}, examples={num_examples}")
        
        # Return updated weights, number of training samples, and the loss
        return self.get_parameters(), num_examples, {"loss": float(loss)}

    def evaluate(self, parameters, config):
        """
        Called by the server to evaluate the global model on the client's validation set.
        
        Args:
            parameters: global model weights (or current client weights)
            config: optional configuration (may contain round number)
        
        Returns:
            (loss, num_examples, metrics_dict)
        """
        # Load the provided parameters into the model
        _set_parameters(self.model, parameters)
        
        # Evaluate on validation loader (returns dict with loss, precision, recall, dice, iou)
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        num_examples = len(self.val_loader.dataset)
        
        round_id = config.get("rnd", "N/A") if config else "N/A"
        print(
            f"[CLIENT {self.client_id}] evaluation round={round_id}, "
            f"loss={metrics['loss']:.4f}, iou={metrics['iou']:.4f}"
        )
        
        # Return loss (required), number of validation samples, and extra metrics
        return float(metrics["loss"]), num_examples, {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
        }


# =========================
# 🚀 START CLIENT
# =========================
def start_client(client_id: int, server_address: str = SERVER_ADDRESS):
    """
    Starts a Flower NumPy client that connects to the given server address.
    """
    # Validate client ID
    if client_id < 1 or client_id > NUM_CLIENTS:
        raise ValueError(f"Unknown client id {client_id}. Valid client ids are 1..{NUM_CLIENTS}.")
    
    print(f"[CLIENT {client_id}] connecting to server at {server_address}")
    
    # Start the client (this call blocks until the client finishes)
    fl.client.start_numpy_client(server_address=server_address, client=PolypClient(client_id))


# =========================
# 🔑 ENTRY POINT (COMMAND LINE)
# =========================
def main() -> None:
    """
    Parse command-line arguments and launch the client.
    Example: python -m clients.client --client-id 2 --server-address localhost:8080
    """
    parser = argparse.ArgumentParser(description="Flower federated UNet3+ client")
    parser.add_argument("--client-id", type=int, default=1, choices=list(CLIENT_DIRS.keys()))
    parser.add_argument("--server-address", type=str, default=SERVER_ADDRESS)
    args = parser.parse_args()
    
    start_client(args.client_id, args.server_address)


# =========================
# 📌 SCRIPT EXECUTION
# =========================
if __name__ == "__main__":
    main()