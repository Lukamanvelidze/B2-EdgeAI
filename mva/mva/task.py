from collections import OrderedDict
import torch
from ultralytics import YOLO
import os

class Net:
    def __init__(self):
        self.config_path = "./data/mvadata/data.yaml"
        self.model = YOLO("yolo11n.yaml")  # Should define nc: 16

        num_classes = self.model.model.model[-1].nc
        if num_classes != 16:
            print(f"[Net] ⚠️ WARNING: Model initialized with {num_classes} classes, expected 16!")

        if os.path.exists("client_prev_global.pt"):
            print("[Net] 📥 Loading weights from client_prev_global.pt...")
            try:
                ckpt = torch.load("client_prev_global.pt", map_location="cpu")
                loaded_state_dict = ckpt["model"]

                current_state_dict = self.model.model.state_dict()
                compatible_weights = {
                    k: v for k, v in loaded_state_dict.items()
                    if k in current_state_dict and v.shape == current_state_dict[k].shape
                }

                self.model.model.load_state_dict(compatible_weights, strict=False)
                print(f"[Net] ✅ Loaded {len(compatible_weights)} compatible layers.")
            except Exception as e:
                print(f"[Net] 🚨 Failed to load previous weights: {e}")
        else:
            print("[Net] 🟡 No client_prev_global.pt found. Starting with fresh model.")
            if os.path.exists("yolo11n.pt"):
                print("[Net] 🔄 Loading base weights from yolo11n.pt...")
                weights = torch.load("yolo11n.pt", map_location="cpu")
                self.model.model.load_state_dict(weights["model"], strict=False)

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
