"""MVA: A Flower / PyTorch app."""

import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from mva.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def fit(self, parameters, config):
        print("🚀 Starting fit()")
        set_weights(self.net, parameters)
        print("📦 Weights set, starting training")

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        print("✅ Training done, returning updated weights")
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
