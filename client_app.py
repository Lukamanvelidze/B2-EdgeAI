import os
import hashlib
import numpy as np
import torch
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context

from mva.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_weights_hash = None
        self.first_round = True

    def _hash_parameters(self, parameters):
        flat_array = np.concatenate([p.flatten() for p in parameters])
        return hashlib.md5(flat_array.tobytes()).hexdigest()

    def _compare_with_previous_model(self, parameters):
        """Compare incoming model weights with previously saved model."""
        if not os.path.exists("client_prev_global.pt"):
            print("[Client] 🟡 No previous saved global model found. Assuming fresh start.")
            return

        print("[Client] 📥 Loading weights for comparison...")
        ckpt = torch.load("client_prev_global.pt", map_location="cpu")
        state_dict = ckpt["model"]

        prev_model = Net()
        prev_weights_np = [val.cpu().numpy() for val in state_dict.values()]
        set_weights(prev_model, prev_weights_np)

        prev_weights = get_weights(prev_model)
        current_weights = [p for p in parameters]  # from server

        try:
            same = all(np.allclose(p1, p2) for p1, p2 in zip(current_weights, prev_weights))
            if same:
                print("[Client] ❌ Still using the old global weights (model not updated since last run).")
            else:
                print("[Client] ✅ Using new global weights (updated since last run).")
        except ValueError as e:
            print(f"[Client] 🚨 Shape mismatch during comparison: {e}")

    def fit(self, parameters, config):
        if self.first_round:
            self._compare_with_previous_model(parameters)
            self.first_round = False

        current_hash = self._hash_parameters(parameters)
        print(f"[Client] 🔍 Received model hash: {current_hash}")

        if self.last_weights_hash:
            if current_hash == self.last_weights_hash:
                print("[Client] ⚠️ Model weights unchanged since last round.")
            else:
                print("[Client] ✅ Model weights changed from last round.")
        else:
            print("[Client] ℹ️ First round — no previous hash to compare.")

        self.last_weights_hash = current_hash

        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)

        # Save weights in deterministic order
        state_dict = self.net.state_dict()
        ordered_keys = sorted(state_dict.keys())
        ordered_weights = {k: state_dict[k] for k in ordered_keys}
        torch.save({"model": ordered_weights}, "client_prev_global.pt")

        print("[Client] 💾 Saved updated model to client_prev_global.pt")

        return (
            get_weights(self.net),
            self.net.dataset_size,
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, self.net.dataset_size, {"accuracy": accuracy}


def client_fn(context: Context):
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


def main():
    net = Net()
    trainloader, valloader = None, None  # Replace with actual loaders if needed
    fl.client.start_client(
        server_address="34.32.104.227:8080",
        client=FlowerClient(net, trainloader, valloader, local_epochs=1).to_client(),
    )


if __name__ == "__main__":
    main()


#server_address="149.233.55.34:9092",
