from flwr.server.strategy import FedAvg
from flwr.server import start_server
from flwr.server.server import ServerConfig
from task import Net, get_weights, set_weights
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import torch 
import os
class FedAvgWithSave(FedAvg):
    def __init__(self, *args, model_save_path="global_model.pt", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.net = Net()  # initialize a model instance to load weights into

    def aggregate_fit(self, rnd, results, failures):
        # Call base method to get aggregated parameters
        aggregated_parameters, agg_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            # Convert aggregated parameters (List[np.ndarray]) to model weights
            weights = parameters_to_ndarrays(aggregated_parameters)
            # Set the weights to the model
            set_weights(self.net, weights)
            # Save the PyTorch model
            torch.save(self.net.state_dict(), self.model_save_path)
            print(f" Saved aggregated model to {self.model_save_path} after round {rnd}")

        return aggregated_parameters, agg_metrics

def main():
    print("Starting Flower server")
    model = Net()
    if os.path.exists("global_model.pt"):
        print("📥 Loading existing global_model.pt for initialization")
        model.load_state_dict(torch.load("global_model.pt", map_location="cpu"))
    else:
        print("📦 No global_model.pt found, using base model")
    
    strategy = FedAvgWithSave(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
        model_save_path="global_model.pt",  # your desired path
    )

    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=2),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
