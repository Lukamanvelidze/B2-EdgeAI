"""MVA: A Flower / PyTorch app."""

import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from mva.task import Net, get_weights, load_data, set_weights, test, train
import hashlib
import numpy as np


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
changes from here
"""

        self.last_weights_hash = None  # For comparison
    def _hash_parameters(self, parameters):
        flat_array = np.concatenate([p.flatten() for p in parameters])
        return hashlib.md5(flat_array.tobytes()).hexdigest()

    def fit(self, parameters, config):
        # Log hash of incoming weights
        current_hash = self._hash_parameters(parameters)
        print(f"[Client] Received model hash: {current_hash}")

        if self.last_weights_hash is not None:
            if current_hash == self.last_weights_hash:
                print("[Client] ⚠️ Model weights unchanged since last round.")
            else:
                print("[Client] ✅ Model weights updated.")
        else:
            print("[Client] First round. No previous weights to compare.")

        # Store current hash for next round comparison
        self.last_weights_hash = current_hash

        # Set weights and train
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)

        return (
            get_weights(self.net),
            self.net.dataset_size,
            {"train_loss": train_loss},
        )

"""
to here
"""
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, self.net.dataset_size, {"accuracy": accuracy}



def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
#app = ClientApp(
#    client_fn,
#)

def main():
    net = Net()
    trainloader, valloader = None, None  # or real loaders if available
    fl.client.start_client(
            server_address="34.32.103.57:8080", #but the pub ip server address
        client=FlowerClient(net, trainloader, valloader, local_epochs=1).to_client(),
    )
if __name__ == "__main__":
    main()

#server_address="149.233.55.34:9092",
