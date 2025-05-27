"""MVA: A Flower / PyTorch app with YOLO."""
from collections import OrderedDict
import torch
import torch.nn as nn
from ultralytics import YOLO
import os
import yaml
import tempfile
import shutil


class Net:
    def __init__(self, num_classes=16):
        self.num_classes = num_classes
        self.config_path = "./data/mvadata/data.yaml"
        
        # Ensure the data.yaml has correct number of classes
        self._ensure_correct_yaml_config()
        
        # Create YOLO model with custom configuration
        self.model = self._create_yolo_model()
        
        # Load compatible weights from pretrained model
        self._load_pretrained_weights()
        
        self.dataset_size = 1
        
    def _ensure_correct_yaml_config(self):
        """Ensure the data.yaml file has the correct number of classes."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update number of classes if needed
            if config.get('nc') != self.num_classes:
                print(f"[Net] 🔧 Updating data.yaml: nc from {config.get('nc')} to {self.num_classes}")
                config['nc'] = self.num_classes
                
                # Backup original file
                backup_path = self.config_path + '.backup'
                if not os.path.exists(backup_path):
                    shutil.copy2(self.config_path, backup_path)
                
                # Write updated config
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                    
        except Exception as e:
            print(f"[Net] ⚠️ Could not update data.yaml: {e}")
    
    def _create_yolo_model(self):
        """Create YOLO model with correct number of classes."""
        try:
            # Try to use existing YAML file, modify for correct classes
            if os.path.exists("yolo11n.yaml"):
                # Load and modify the YAML config
                with open("yolo11n.yaml", 'r') as f:
                    model_config = yaml.safe_load(f)
                
                # Update number of classes in the model config
                if 'nc' in model_config:
                    model_config['nc'] = self.num_classes
                    
                # Create temporary config file
                temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.dump(model_config, temp_config, default_flow_style=False)
                temp_config.close()
                
                model = YOLO(temp_config.name)
                
                # Clean up temp file
                os.unlink(temp_config.name)
                
            else:
                # Create model from scratch with correct classes
                model = YOLO("yolo11n.pt")  # Start with pretrained
                # The model will be modified for correct classes during weight loading
                
            print(f"[Net] ✅ Created YOLO model for {self.num_classes} classes")
            return model
            
        except Exception as e:
            print(f"[Net] ⚠️ Error creating custom YOLO model: {e}")
            print("[Net] 🔄 Falling back to standard yolo11n.pt")
            return YOLO("yolo11n.pt")
    
    def _load_pretrained_weights(self):
        """Load compatible weights from pretrained YOLO model."""
        try:
            if os.path.exists("yolo11n.pt"):
                print("[Net] 🔄 Loading compatible weights from yolo11n.pt...")
                
                # Load pretrained checkpoint
                ckpt = torch.load("yolo11n.pt", map_location="cpu")
                
                if "model" in ckpt:
                    loaded_state_dict = ckpt["model"].state_dict()
                else:
                    loaded_state_dict = ckpt
                
                # Get current model state
                current_state_dict = self.model.model.state_dict()
                
                # Filter compatible weights (same name and shape)
                compatible_weights = {}
                incompatible_layers = []
                
                for k, v in loaded_state_dict.items():
                    if k in current_state_dict:
                        if v.shape == current_state_dict[k].shape:
                            compatible_weights[k] = v
                        else:
                            incompatible_layers.append(f"{k}: {v.shape} vs {current_state_dict[k].shape}")
                    else:
                        incompatible_layers.append(f"{k}: not found in current model")
                
                # Load compatible weights
                missing_keys, unexpected_keys = self.model.model.load_state_dict(compatible_weights, strict=False)
                
                print(f"[Net] ✅ Loaded {len(compatible_weights)} compatible layers")
                if incompatible_layers:
                    print(f"[Net] ⚠️ Skipped {len(incompatible_layers)} incompatible layers")
                    # Print first few incompatible layers for debugging
                    for layer in incompatible_layers[:3]:
                        print(f"[Net]    - {layer}")
                    if len(incompatible_layers) > 3:
                        print(f"[Net]    - ... and {len(incompatible_layers) - 3} more")
                        
            else:
                print("[Net] ⚠️ yolo11n.pt not found, using random initialization")
                
        except Exception as e:
            print(f"[Net] ❌ Error loading pretrained weights: {e}")
            print("[Net] 🔄 Continuing with random initialization")
    
    def to(self, device):
        """Move model to device."""
        self.model.model.to(device)
        return self
    
    def state_dict(self):
        """Get model state dictionary."""
        return self.model.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=False):
        """Load model state dictionary."""
        return self.model.model.load_state_dict(state_dict, strict=strict)
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.model.eval()


def load_data(partition_id: int, num_partitions: int):
    """
    Load data partitions. For YOLO, this is handled internally,
    but you might want to implement custom data splitting here.
    """
    # For YOLO, data loading is handled by the framework
    # You could implement custom data partitioning here if needed
    print(f"[Data] 📊 Partition {partition_id}/{num_partitions} - Using YOLO internal data loading")
    return None, None


def train(net, trainloader, epochs, device):
    """Train the YOLO model."""
    try:
        print(f"[Train] 🏋️ Starting training for {epochs} epochs on {device}")
        
        # Set model to training mode
        net.train_mode()
        
        # Configure training parameters
        train_args = {
            'data': net.config_path,
            'epochs': epochs,
            'device': device if isinstance(device, str) else str(device),
            'imgsz': 416,      # Reduced image size for federated learning
            'batch': 4,        # Small batch size
            'workers': 0,      # Avoid multiprocessing issues
            'save': False,     # Don't save checkpoints during FL
            'plots': False,    # Disable plotting
            'val': False,      # Disable validation during training
            'verbose': True,   # Enable verbose output
            'exist_ok': True,  # Allow overwriting
        }
        
        # Start training
        results = net.model.train(**train_args)
        
        # Extract training loss if available
        if hasattr(results, 'results_dict') and 'train/box_loss' in results.results_dict:
            train_loss = float(results.results_dict['train/box_loss'])
        else:
            train_loss = 0.0  # Fallback if loss not available
            
        print(f"[Train] ✅ Training completed. Loss: {train_loss:.4f}")
        return train_loss
        
    except Exception as e:
        print(f"[Train] ❌ Training error: {e}")
        return 0.0


def test(net, testloader, device):
    """Test/validate the YOLO model."""
    try:
        print(f"[Test] 🧪 Starting validation on {device}")
        
        # Set model to evaluation mode
        net.eval_mode()
        
        # Run validation
        results = net.model.val(
            data=net.config_path, 
            device=device if isinstance(device, str) else str(device),
            verbose=False
        )
        
        # Extract metrics
        if hasattr(results, 'box') and hasattr(results.box, 'map50'):
            accuracy = float(results.box.map50)  # mAP@0.5
        else:
            accuracy = 0.0
            
        # Try to get validation loss if available
        if hasattr(results, 'box') and hasattr(results.box, 'loss'):
            loss = float(results.box.loss)
        else:
            loss = 0.0
            
        print(f"[Test] ✅ Validation completed. Loss: {loss:.4f}, mAP@0.5: {accuracy:.4f}")
        return loss, accuracy
        
    except Exception as e:
        print(f"[Test] ❌ Validation error: {e}")
        return 0.0, 0.0


def get_weights(net):
    """Extract model weights as numpy arrays."""
    try:
        state_dict = net.model.model.state_dict()
        weights = []
        
        for name, param in state_dict.items():
            # Convert to numpy and add to list
            weight_np = param.detach().cpu().numpy()
            weights.append(weight_np)
            
        print(f"[Weights] 📤 Extracted {len(weights)} weight tensors")
        return weights
        
    except Exception as e:
        print(f"[Weights] ❌ Error extracting weights: {e}")
        return []


def set_weights(net, parameters):
    """Set model weights from numpy arrays."""
    try:
        if not parameters:
            print("[Weights] ⚠️ No parameters provided")
            return
            
        state_dict = net.model.model.state_dict()
        param_keys = list(state_dict.keys())
        
        if len(parameters) != len(param_keys):
            print(f"[Weights] ⚠️ Parameter count mismatch: {len(parameters)} vs {len(param_keys)}")
            return
        
        # Create new state dict with updated weights
        updated_state_dict = OrderedDict()
        skipped_layers = []
        
        for i, (key, param_array) in enumerate(zip(param_keys, parameters)):
            try:
                # Convert numpy array to tensor
                new_tensor = torch.from_numpy(param_array)
                
                # Check shape compatibility
                if new_tensor.shape == state_dict[key].shape:
                    updated_state_dict[key] = new_tensor
                else:
                    # Keep original weight if shapes don't match
                    updated_state_dict[key] = state_dict[key]
                    skipped_layers.append(f"{key}: {new_tensor.shape} vs {state_dict[key].shape}")
                    
            except Exception as e:
                # Keep original weight if conversion fails
                updated_state_dict[key] = state_dict[key]
                skipped_layers.append(f"{key}: {str(e)}")
        
        # Load the updated state dict
        net.model.model.load_state_dict(updated_state_dict, strict=True)
        
        successful_layers = len(param_keys) - len(skipped_layers)
        print(f"[Weights] ✅ Updated {successful_layers}/{len(param_keys)} layers")
        
        if skipped_layers:
            print(f"[Weights] ⚠️ Skipped {len(skipped_layers)} layers due to shape mismatch")
            # Print first few for debugging
            for layer in skipped_layers[:3]:
                print(f"[Weights]    - {layer}")
            if len(skipped_layers) > 3:
                print(f"[Weights]    - ... and {len(skipped_layers) - 3} more")
        
    except Exception as e:
        print(f"[Weights] ❌ Error setting weights: {e}")
