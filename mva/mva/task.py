"""MVA: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from flwr_datasets import FederatedDataset
#from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from ultralytics import YOLO
import os

class Net():
    def __init__(self):
        # Use absolute path if necessary
        self.model = YOLO("yolo11n.pt")
        self.config_path = "/home/lemonanaquis/Documents/mva/mva/mva/data/mvadata"


        # Optional: simulate a dataset size for return
        self.dataset_size = 100
        
    def to(self, device):
        # Move the underlying model to device
        self.model.model.to(device)
        return self  # allow chaining

    def state_dict(self):
        return self.model.model.state_dict()



def load_data(partition_id: int, num_partitions: int):
    # You don't need to manually load DataLoader if using Ultralytics API
    return None, None



def train(net, trainloader, epochs, device):
    """
    Train YOLOv8 model using Ultralytics native training.
    Note: Flower client does not train using batches directly, it calls YOLO's train method.
    """
    net.model.train(data=net.config_path, epochs=epochs, device=device)
    # Simulated loss return; Ultralytics does not return loss directly in this mode
    return 0.0  # Replace with parsed log or real value if needed


def test(net, testloader, device):
    """
    Evaluate YOLOv8 model using Ultralytics native validation.
    """
    results = net.model.val(data=net.config_path, device=device)
    accuracy = results.box.map50  # mAP@0.5
    loss = results.loss.box  # or another appropriate metric
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.model.model.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.model.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.model.model.load_state_dict(state_dict, strict=True)
