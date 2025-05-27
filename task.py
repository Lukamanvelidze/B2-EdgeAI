"""MVA: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from flwr_datasets import FederatedDataset
#from flwr_datasets.partitioner import IidPartitioner
#from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, Normalize, ToTensor
from ultralytics import YOLO
import os

class Net:
    def __init__(self):
        self.config_path = "./data/mvadata/data.yaml"

        # Load model architecture from YAML (custom with nc=16)
        self.model = YOLO("yolo11n.yaml")

        # Load weights manually from .pt file
        # Inside Net.__init__()
        ckpt = torch.load("yolo11n.pt", map_location="cpu")
        loaded_state_dict = ckpt["model"].state_dict()

        # Filter: Only load weights that match in shape and name
        current_state_dict = self.model.model.state_dict()
        compatible_weights = {
            k: v for k, v in loaded_state_dict.items()
            if k in current_state_dict and v.shape == current_state_dict[k].shape
        }

        # Load only compatible weights
        self.model.model.load_state_dict(compatible_weights, strict=False)
        print(f"[Init] ✅ Loaded {len(compatible_weights)} compatible layers from yolo11n.pt")

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
    model_state = net.model.model.state_dict()
    model_keys = list(model_state.keys())
    params_dict = dict(zip(model_keys, parameters))

    with torch.no_grad():  # Ensure we're not in inference mode
        for k, v in params_dict.items():
            try:
                if k in model_state and model_state[k].shape == torch.tensor(v).shape:
                    model_state[k].copy_(torch.tensor(v))
                else:
                    print(f"⚠️ Skipping {k}: shape mismatch or missing")
            except Exception as e:
                print(f"⚠️ Skipping {k}: {e}")
