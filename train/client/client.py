import os
import hashlib
import numpy as np
import torch
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context
import argparse
from task import Net, get_weights, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, local_epochs):
        self.net = net
        self.local_epochs = local_epochs
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            print("[Client] Using Apple MPS backend for GPU acceleration.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("[Client] Using NVIDIA CUDA GPU.")
        else:
            self.device = torch.device("cpu")
            print("[Client] Using CPU.")

        
        self.last_weights_hash = None
        self.first_round = True

    def _hash_parameters(self, parameters):
        flat_array = np.concatenate([p.flatten() for p in parameters])
        return hashlib.md5(flat_array.tobytes()).hexdigest()


    def fit(self, parameters, config):
        if self.first_round:
            self.first_round = False

        current_hash = self._hash_parameters(parameters)
        print(f"[Client] Received model hash: {current_hash}")

        if self.last_weights_hash:
            if current_hash == self.last_weights_hash:
                print("[Client] Model weights unchanged since last round.")
            else:
                print("[Client] Model weights changed from last round.")
        else:
            print("[Client] First round — no previous hash to compare.")

        self.last_weights_hash = current_hash

        set_weights(self.net, parameters)
        train_loss = train(self.net, self.local_epochs, self.device)

        # Save weights in deterministic order
        state_dict = self.net.state_dict()
        ordered_keys = sorted(state_dict.keys())
        ordered_weights = {k: state_dict[k] for k in ordered_keys}
        torch.save({"model": ordered_weights}, "client_prev_global.pt")

        print("[Client] Saved updated model to client_prev_global.pt")

        return (
            get_weights(self.net),
            self.net.dataset_size,
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.device)
        return loss, self.net.dataset_size, {"accuracy": accuracy}



def main():

    # default server address
    d_server_address = "0.0.0.0:8080"

    parser = argparse.ArgumentParser(description="Federated Learning Client")

    parser.add_argument("--server-address", type=str, default=d_server_address, help="Server IP and port (str, format: IP:port, example: 23.0.24.264:8080)")
    args = parser.parse_args()

    server_address = args.server_address
  

    net = Net()
    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient(net, local_epochs=5).to_client(),
    )


if __name__ == "__main__":
    main()

