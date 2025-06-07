from collections import OrderedDict
import torch
from format_classes import handle_model_size_mismatch
from ultralytics import YOLO
import os

class Net:
    def __init__(self):
        self.config_path = "./data/mvadata/data.yaml"
        self.model = YOLO("yolo11n.pt")
        self.dataset_size = 1

    def to(self, device):
        self.model.model.to(device)
        return self

    def state_dict(self):
        return self.model.model.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        return self.model.model.load_state_dict(state_dict, strict=strict)
    


def load_data(partition_id: int, num_partitions: int):
    return None, None  # handled by YOLO internally


def train(net, trainloader, epochs, device):
    print("Type of net.model:", type(net.model))
    net.model.train(
        data=net.config_path,
        epochs=epochs,
        device=device,
        imgsz=416,
        batch=2,
        workers=0,
        save=True,
        plots=True,
        val=True,
    )
    return 0.0


def test(net, testloader, device):
    results = net.model.val(data=net.config_path, device=device)
    accuracy = results.box.map50
    return 0.0, accuracy


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
