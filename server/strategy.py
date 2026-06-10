from pathlib import Path
import csv
import torch
from flwr.server.strategy import FedAvg
from config import NUM_CLIENTS, TRAINING_CONFIG, SAVED_MODELS_DIR, SAVED_METRICS_PATH
from clients.train import build_model


class FlowerFedAvgStrategy(FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        self.global_model = build_model("cpu")
        self.history = []
        self._prepare_storage()

    def _prepare_storage(self):
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if not SAVED_METRICS_PATH.exists():
            with open(SAVED_METRICS_PATH, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["round", "avg_iou", "avg_loss"])

    def _set_global_model_parameters(self, parameters):
        state_dict = self.global_model.state_dict()
        for key, array in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(array, dtype=state_dict[key].dtype)
        self.global_model.load_state_dict(state_dict)

    def _save_global_model(self, rnd: int):
        file_path = SAVED_MODELS_DIR / f"global_model_round_{rnd}.pth"
        torch.save(self.global_model.state_dict(), file_path)
        latest_path = SAVED_MODELS_DIR / "global_model_latest.pth"
        torch.save(self.global_model.state_dict(), latest_path)
        print(f"[SERVER] saved global model to {file_path}")

    def _extract_evaluate_stats(self, results):
        total_examples = 0
        sum_iou = 0.0
        sum_loss = 0.0

        for result in results:
            evaluate_res = result[1] if isinstance(result, tuple) and len(result) == 2 else result
            num_examples = getattr(evaluate_res, "num_examples", None)
            if num_examples is None:
                num_examples = getattr(evaluate_res, "num_examples", 0)
            loss = getattr(evaluate_res, "loss", None)
            metrics = getattr(evaluate_res, "metrics", {})
            if loss is None and isinstance(metrics, dict):
                loss = metrics.get("loss", 0.0)
            iou = 0.0
            if isinstance(metrics, dict):
                iou = float(metrics.get("iou", 0.0))
            elif hasattr(metrics, "get"):
                iou = float(metrics.get("iou", 0.0))

            if num_examples is None or num_examples <= 0:
                continue

            sum_iou += iou * num_examples
            sum_loss += float(loss) * num_examples
            total_examples += num_examples

        if total_examples == 0:
            return 0.0, 0.0

        return sum_iou / total_examples, sum_loss / total_examples

    def fit_config(self, rnd: int):
        return {"local_epochs": TRAINING_CONFIG["local_epochs"]}

    def evaluate_config(self, rnd: int):
        return {}

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        print(f"[SERVER] Round {rnd}: aggregated {len(results)} clients, failures={len(failures)}")

        if aggregated is not None:
            try:
                self._set_global_model_parameters(aggregated)
                self._save_global_model(rnd)
            except Exception as exc:
                print(f"[SERVER] failed to save global model at round {rnd}: {exc}")
        return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        avg_iou, avg_loss = self._extract_evaluate_stats(results)
        self.history.append({"round": rnd, "avg_iou": avg_iou, "avg_loss": avg_loss})
        with open(SAVED_METRICS_PATH, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([rnd, avg_iou, avg_loss])
        print(f"[SERVER] Round {rnd}: avg_iou={avg_iou:.4f}, avg_loss={avg_loss:.4f}")
        return super().aggregate_evaluate(rnd, results, failures)
