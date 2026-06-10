import argparse
import flwr as fl
from server.strategy import FlowerFedAvgStrategy
from config import SERVER_ADDRESS, TRAINING_CONFIG

#start the server and specify the number of rounds of federated learning
def start_server(server_address: str = SERVER_ADDRESS, num_rounds: int = TRAINING_CONFIG["rounds"]) -> None:
    strategy = FlowerFedAvgStrategy()
    print(f"[SERVER] starting Flower server at {server_address} for {num_rounds} rounds")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.server_config.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower federated UNet3+ server")
    parser.add_argument("--server-address", type=str, default=SERVER_ADDRESS)
    parser.add_argument("--num-rounds", type=int, default=TRAINING_CONFIG["rounds"])
    args = parser.parse_args()
    start_server(server_address=args.server_address, num_rounds=args.num_rounds)


if __name__ == "__main__":
    main()
