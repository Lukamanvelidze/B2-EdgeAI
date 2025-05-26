"""MVA: A Flower / PyTorch app."""

import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from mva.task import Net, get_weights, load_data, set_weights, test, train
import hashlib
import numpy as np
from collections import OrderedDict
import os


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _save_current_weights(self, path="client_prev_global.pt"):
        torch.save(self.net.model.model.state_dict(), path)
        print(f"[Client] ✅ Saved current global model to {path}")

    def _load_previous_weights(self, path="client_prev_global.pt"):
        if not os.path.exists(path):
            print("[Client] ❌ No previous global model found. Skipping comparison.")
            return None

        prev_model = self.net.__class__()  # reinstantiate the Net class
        prev_model = prev_model.model.model  # extract nn.Module
        prev_state = torch.load(path, map_location="cpu")

        # Filter: Load only matching keys and shapes
        compatible_weights = {
            k: v for k, v in prev_state.items()
            if k in prev_model.state_dict() and v.shape == prev_model.state_dict()[k].shape
        }

        prev_model.load_state_dict(compatible_weights, strict=False)
        print(f"[Client] 🔁 Loaded {len(compatible_weights)} layers for comparison")
        return prev_model

    def _compare_with_previous_model(self, new_params):
        prev_model = self._load_previous_weights()
        if prev_model is None:
            print("[Client] 🆕 First round or missing model — assuming this is a new global model.")
            return

        # Convert new global model to tensor state_dict
        keys = self.net.model.model.state_dict().keys()
        new_state = dict(zip(keys, [torch.tensor(p) for p in new_params]))

        # Compare
        diff_sum = 0.0
        for k in new_state:
            if k in prev_model.state_dict():
                prev_tensor = prev_model.state_dict()[k]
                new_tensor = new_state[k]
                if prev_tensor.shape == new_tensor.shape:
                    diff_sum += torch.norm(prev_tensor - new_tensor).item()

        print(f"[Client] 🔍 Total weight difference from last round: {diff_sum:.4f}")

    def fit(self, parameters, config):
        print("\n[Client] 🚀 Starting local training")
        self._compare_with_previous_model(parameters)

        # Load the new global model
        set_weights(self.net, parameters)

        # Train locally
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)

        # Save current model for next round diff
        self._save_current_weights()

        print("[Client] ✅ Finished local training\n")
        return get_weights(self.net), self.net.dataset_size, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, self.net.dataset_size, {"accuracy": accuracy}



# Flower ClientApp
#app = ClientApp(
#    client_fn,
#)

def main():
    net = Net()
    trainloader, valloader = None, None  # or real loaders if available
    fl.client.start_client(
            server_address="34.32.102.222:8080", #but the pub ip server address
        client=FlowerClient(net, trainloader, valloader, local_epochs=1).to_client(),
    )
if __name__ == "__main__":
    main()

#server_address="149.233.55.34:9092",
