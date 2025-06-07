from flwr.server.strategy import FedAvg
from flwr.server import start_server
from flwr.server.server import ServerConfig
from task import Net, get_weights, set_weights
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import torch 
import os
import argparse

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

            self.net.save(self.model_save_path)
            print(f" Saved aggregated model to {self.model_save_path} after round {rnd}")

        return aggregated_parameters, agg_metrics

def main():

    # parameters
    """
    - for connection:
        + server_address *port default 0.0.0.0.8080
    - saving the model:
        + model_save_path default ./global_model.pt or global_model.pt
    - strategy:
        + fraction_fit d:1.0
        + min_fit_clients d:1
        + min evaluate client d1
        + min_available_clients d:1
        + rounds

    """

    # change the default here:
    d_port = 8080
    d_save_path = "global_model.pt"
    d_frac_fit = 1.0
    d_min_fit_client = 1
    d_min_eval_client = 1
    d_min_available_client = 1
    d_rounds = 1
    #--------------------------------

    parser = argparse.ArgumentParser(description="Federated Learning Server")

    parser.add_argument("--port", type=int, default=d_port, help="Port number (int, default: " + str(d_port) + ")" )

    parser.add_argument("--model_save_path", type=str, default=d_save_path, help="Where to save the trained model (str, default: " + str(d_save_path) + ")")

    parser.add_argument("--fraction_fit", type=float, default=d_frac_fit, help="The fraction of the total client that is taken for fitting (float, default: " + str(d_frac_fit) + ")")
    parser.add_argument("--min-fit-clients", type=int, default=d_min_fit_client, help="Minimum number of clients to use for training (int, default: " + str(d_min_fit_client) + ")")
    parser.add_argument("--min-eval-clients", type=int, default=d_min_eval_client, help="Minimum number of clients must be selected to evaluate the global model at the end of training round (int, default: " + str(d_min_eval_client) + ")")
    parser.add_argument("--min-available-clients", type=int, default=d_min_available_client, help="Minimum number of clients needed to start training (int, default: " + str(d_min_available_client) + ")")
    parser.add_argument("--rounds", type=int, default=d_rounds, help="Number of federated learning rounds (int, default: " + str(d_rounds) + ")")
    args = parser.parse_args()


    port = args.port 

    model_save_path = args.model_save_path

    fraction_fit = args.fraction_fit
    min_fit_clients = args.min_fit_clients
    min_evaluate_clients = args.min_eval_clients
    min_available_clients = args.min_available_clients
    num_rounds = args.rounds



    model = Net()
    if os.path.exists(model_save_path):
        print("📥 Loading existing global_model.pt for initialization")
        model.YOLO(model_save_path)
    else:
        print("📦 No global_model.pt found, using base model")
    
    strategy = FedAvgWithSave(
        fraction_fit= fraction_fit,
        min_fit_clients= min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=ndarrays_to_parameters(get_weights(model)),
        model_save_path= model_save_path,  
    )

    start_server(
        server_address="0.0.0.0:"+str(port),
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()