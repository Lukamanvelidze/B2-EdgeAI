from ultralytics.nn.tasks import DetectionModel
import torch

# Build your 16-class model
model = DetectionModel('yolo11n_16.yaml')

# Load original pretrained model checkpoint
ckpt = torch.load('yolo11n.pt', map_location='cpu')
pretrained_model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

# Load matching weights only
pretrained_state = pretrained_model.state_dict()
model_state = model.state_dict()
loaded_state = {}

for k, v in pretrained_state.items():
    if k in model_state and v.shape == model_state[k].shape:
        loaded_state[k] = v
    else:
        print(f"Skipping: {k} -- shape mismatch")

model.load_state_dict(loaded_state, strict=False)

# Optional: attach class names
model.names = [
    'bike', 'bus', 'caravan', 'coupe', 'crossover', 'hatchback',
    'jeep', 'mpv', 'pickup-truck', 'sedan', 'suv', 'taxi',
    'truck', 'van', 'vehicle', 'wagon'
]

# Save the model for FL training
torch.save({'model': model}, 'yolo11n_16nc.pt')
print("✅ Converted and saved model with 16 classes.")
