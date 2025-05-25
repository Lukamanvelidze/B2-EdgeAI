from flwr.server.strategy import FedAvg
from flwr.server import start_server
from flwr.server.server import ServerConfig
from mva.task import Net, get_weights
from flwr.common import ndarrays_to_parameters

def main():
    print("🚀 Starting Flower server")

    # Load initial model parameters
    net = Net()
    parameters = ndarrays_to_parameters(get_weights(net))

    strategy = FedAvg(
        initial_parameters=parameters,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1,
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

