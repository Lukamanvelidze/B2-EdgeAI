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

class Net:
    def __init__(self):
        self.config_path = "/home/catmub/Documents/project/mva/mva/mva/data/mvadata/data.yaml"

        # Load model architecture from YAML (custom with nc=16)
        self.model = YOLO("yolo11n.yaml")

        # Load weights manually from .pt file
        ckpt = torch.load("yolo11n.pt", map_location="cpu")
        self.model.model.load_state_dict(ckpt["model"].state_dict(), strict=False)

        self.dataset_size = 100


    def to(self, device):
        self.model.model.to(device)
        return self

    def state_dict(self):
        return self.model.model.state_dict()



def load_data(partition_id: int, num_partitions: int):
    # You don't need to manually load DataLoader if using Ultralytics API
    return None, None



def train(net, trainloader, epochs, device):
    net.model.train(
        data=net.config_path,
        epochs=epochs,
        device=device,
        imgsz=416,     # ↓ Reduce image size
        batch=2,       # ↓ Reduce batch size
        workers=0,     # ↓ Avoid parallel data loading
        save=False,    # ↓ Reduce file I/O
        plots=False,
        val=False,
    )
    return 0.0



def test(net, testloader, device):
    results = net.model.val(data=net.config_path, device=device)
    accuracy = results.box.map50  # mean average precision
    loss = 0.0  # no loss available, placeholder
    return loss, accuracy



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.model.model.state_dict().items()]

def set_weights(net, parameters):
    model_keys = list(net.model.model.state_dict().keys())
    params_dict = dict(zip(model_keys, parameters))

    # Filtered state_dict: only load weights with matching shapes
    filtered_state_dict = OrderedDict()
    for k, v in params_dict.items():
        try:
            tensor = torch.tensor(v)
            if net.model.model.state_dict()[k].shape == tensor.shape:
                filtered_state_dict[k] = tensor
            else:
                print(f"Skipping {k}: shape mismatch {tensor.shape} != {net.model.model.state_dict()[k].shape}")
        except Exception as e:
            print(f"Skipping {k}: error converting or checking shape. Error: {e}")

    net.model.model.load_state_dict(filtered_state_dict, strict=False)

