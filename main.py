# =========================
# 📦 IMPORTS
# =========================
import argparse
from config import CLIENT_DIRS


# =========================
# 🚀 MAIN FUNCTION
# =========================
def main():
    # =========================
    # ⚙️ ARGPARSE SETUP
    # =========================
    # argparse is Python's built-in library for parsing command-line arguments.
    # It lets you pass options like --role server or --client-id 2 when running the script.
    
    parser = argparse.ArgumentParser(description="Federated UNet3+ training with Flower")
    
    # --role is required, must be one of four choices
    parser.add_argument(
        "--role",
        choices=["server", "client", "prepare-data", "visualize"],
        required=True,
    )
    # --client-id is optional (defaults to 1), expects an integer, only used when role=client
    parser.add_argument("--client-id", type=int, default=1)
    
    # Parse the actual command line (e.g., python main.py --role client --client-id 2)
    args = parser.parse_args()
    
    # =========================
    # 🧠 HOW THE SCRIPT USES args
    # =========================
    # After args = parser.parse_args(), the code checks args.role:
    #
    # --role prepare-data → runs data preparation
    # --role server       → starts the federated server
    # --role client       → starts a client (needs --client-id)
    # --role visualize    → creates plots
    #
    # Example: If you run `python main.py --role client --client-id 2`, then:
    #   args.role == "client"   → enters the client block
    #   args.client_id == 2     → launches client 2

    # =========================
    # 📂 PREPARE DATA
    # =========================
    if args.role == "prepare-data":
        from utils.data_split import prepare_federated_data

        prepare_federated_data()
        return

    # =========================
    # 🖥️ SERVER
    # =========================
    if args.role == "server":
        from server.server import start_server

        start_server()
        return

    # =========================
    # 👤 CLIENT
    # =========================
    if args.role == "client":
        # Validate that the provided client-id exists in CLIENT_DIRS
        if args.client_id not in CLIENT_DIRS:
            raise ValueError(
                f"client_id must be one of {list(CLIENT_DIRS.keys())}, got {args.client_id}"
            )
        from clients.client import start_client

        start_client(args.client_id)
        return
    # =========================
    # 📊 VISUALIZE
    # =========================
    if args.role == "visualize":
        from utils.visualization import plot_round_iou, plot_round_loss

        iou_path = plot_round_iou()
        loss_path = plot_round_loss()
        print(f"[VISUALIZATION] saved IoU plot to {iou_path}")
        print(f"[VISUALIZATION] saved loss plot to {loss_path}")
        return


# =========================
# 🔑 ENTRY POINT
# =========================
if __name__ == "__main__":
    main()