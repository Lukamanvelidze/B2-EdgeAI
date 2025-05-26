from flwr.server.strategy import FedAvg
from flwr.server import start_server
from flwr.server.server import ServerConfig
from mva.task import Net, get_weights
from flwr.common import ndarrays_to_parameters

class FedAvgWithSave(FedAvg):
    def __init__(self, *args, model_save_path="global_model.pt", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.net = Net()  # initialize a model instance to load weights into

    def aggregate_fit(self, rnd, results, failures):
        # Call base method to get aggregated parameters
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            # Convert aggregated parameters (List[np.ndarray]) to model weights
            weights = parameters_to_ndarrays(aggregated_parameters)
            # Set the weights to the model
            set_weights(self.net, weights)
            # Save the PyTorch model
            torch.save(self.net.state_dict(), self.model_save_path)
            print(f" Saved aggregated model to {self.model_save_path} after round {rnd}")

        return aggregated_parameters

def main():
    print("Starting Flower server")

    strategy = FedAvgWithSave(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
        model_save_path="global_model.pt",  # your desired path
    )

    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

"""
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from mva.task import Net, get_weights
import flwr as fl


    
def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
"""

