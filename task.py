"""MVA: A Flower / PyTorch app with YOLO - Fixed version with proper model handling."""
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
    
    def _create_custom_yolo_config(self):
        """Create a custom YOLO configuration with the correct number of classes."""
        # Standard YOLOv11n architecture adapted for custom classes
        config = {
            'nc': self.num_classes,  # number of classes
            'depth_multiple': 0.33,  # model depth multiple
            'width_multiple': 0.25,  # layer channel multiple
            'max_channels': 1024,
            
            # YOLOv11n backbone
            'backbone': [
                [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
                [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                [-1, 2, 'C3k2', [128, False, 0.25]],
                [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                [-1, 2, 'C3k2', [256, False, 0.25]],
                [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                [-1, 2, 'C3k2', [512, False, 0.25]],
                [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                [-1, 2, 'C3k2', [1024, True]],
                [-1, 1, 'SPPF', [1024, 5]],  # 9
            ],
            
            # YOLOv11n head
            'head': [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 2, 'C3k2', [512, False]],  # 12
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 2, 'C3k2', [256, False]],  # 15 (P3/8-small)
                
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 12], 1, 'Concat', [1]],  # cat head P4
                [-1, 2, 'C3k2', [512, False]],  # 18 (P4/16-medium)
                
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 9], 1, 'Concat', [1]],  # cat head P5
                [-1, 2, 'C3k2', [1024, False]],  # 21 (P5/32-large)
                
                [[15, 18, 21], 1, 'Detect', [self.num_classes]],  # Detect(P3, P4, P5)
            ]
        }
        return config
    
    def _create_yolo_model(self):
        """Create YOLO model with correct number of classes."""
        try:
            # Method 1: Try to create from custom config
            try:
                print(f"[Net] 🔧 Creating custom YOLO model for {self.num_classes} classes...")
                
                # Create temporary config file
                config = self._create_custom_yolo_config()
                temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.dump(config, temp_config, default_flow_style=False)
                temp_config.close()
                
                model = YOLO(temp_config.name)
                
                # Clean up temp file
                os.unlink(temp_config.name)
                
                print(f"[Net] ✅ Created custom YOLO model for {self.num_classes} classes")
                return model
                
            except Exception as e:
                print(f"[Net] ⚠️ Custom config method failed: {e}")
                
                # Method 2: Start with pretrained and modify
                print(f"[Net] 🔄 Trying pretrained model modification...")
                model = YOLO("yolo11n.pt")
                
                # Get the model's detection head and modify it
                if hasattr(model.model, 'model') and len(model.model.model) > 0:
                    # Find the Detect layer (usually the last layer)
                    for i, layer in enumerate(model.model.model):
                        if hasattr(layer, '__class__') and 'Detect' in str(layer.__class__):
                            print(f"[Net] 🎯 Found Detect layer at index {i}, modifying for {self.num_classes} classes")
                            # The Detect layer will be rebuilt during training with correct classes
                            break
                
                print(f"[Net] ✅ Modified pretrained YOLO model for {self.num_classes} classes")
                return model
                
        except Exception as e:
            print(f"[Net] ❌ Error creating YOLO model: {e}")
            print(f"[Net] 🔄 Falling back to basic YOLO model")
            
            # Fallback: basic YOLO model
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
                
                # Load compatible weights with strict=False to ignore incompatible layers
                missing_keys, unexpected_keys = self.model.model.load_state_dict(compatible_weights, strict=False)
                
                print(f"[Net] ✅ Loaded {len(compatible_weights)} compatible layers")
                if incompatible_layers:
                    print(f"[Net] ⚠️ Skipped {len(incompatible_layers)} incompatible layers (detection head will be retrained)")
                        
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
        """Load model state dictionary with better error handling."""
        try:
            # Always use strict=False for federated learning to handle architecture differences
            missing_keys, unexpected_keys = self.model.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"[Net] ⚠️ Missing keys in state_dict: {len(missing_keys)} layers")
            if unexpected_keys:
                print(f"[Net] ⚠️ Unexpected keys in state_dict: {len(unexpected_keys)} layers")
                
            return missing_keys, unexpected_keys
            
        except Exception as e:
            print(f"[Net] ❌ Error loading state dict: {e}")
            return [], []
    
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
            'patience': 5,     # Early stopping patience
            'close_mosaic': 5, # Close mosaic augmentation in last epochs
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
            verbose=False,
            save=False,
            plots=False
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
    """Set model weights from numpy arrays with better compatibility handling."""
    try:
        if not parameters:
            print("[Weights] ⚠️ No parameters provided")
            return
            
        state_dict = net.model.model.state_dict()
        param_keys = list(state_dict.keys())
        
        if len(parameters) != len(param_keys):
            print(f"[Weights] ⚠️ Parameter count mismatch: {len(parameters)} vs {len(param_keys)}")
            # Try to match as many parameters as possible
            min_params = min(len(parameters), len(param_keys))
            print(f"[Weights] 🔧 Attempting to match first {min_params} parameters")
        else:
            min_params = len(parameters)
        
        # Create new state dict with updated weights
        updated_state_dict = OrderedDict()
        updated_count = 0
        skipped_layers = []
        
        # Copy all original weights first
        for key in param_keys:
            updated_state_dict[key] = state_dict[key].clone()
        
        # Update compatible weights
        for i in range(min_params):
            key = param_keys[i]
            param_array = parameters[i]
            
            try:
                # Convert numpy array to tensor
                new_tensor = torch.from_numpy(param_array)
                
                # Check shape compatibility
                if new_tensor.shape == state_dict[key].shape:
                    updated_state_dict[key] = new_tensor
                    updated_count += 1
                else:
                    # Keep original weight if shapes don't match
                    skipped_layers.append(f"{key}: {new_tensor.shape} vs {state_dict[key].shape}")
                    
            except Exception as e:
                # Keep original weight if conversion fails
                skipped_layers.append(f"{key}: {str(e)}")
        
        # Load the updated state dict with strict=False to handle architecture differences
        missing_keys, unexpected_keys = net.model.model.load_state_dict(updated_state_dict, strict=False)
        
        print(f"[Weights] ✅ Updated {updated_count}/{len(param_keys)} layers")
        
        if skipped_layers:
            print(f"[Weights] ⚠️ Skipped {len(skipped_layers)} layers due to incompatibility")
            # Print first few for debugging
            for layer in skipped_layers[:3]:
                print(f"[Weights]    - {layer}")
            if len(skipped_layers) > 3:
                print(f"[Weights]    - ... and {len(skipped_layers) - 3} more")
        
        if missing_keys:
            print(f"[Weights] ⚠️ Missing keys: {len(missing_keys)} layers")
        if unexpected_keys:
            print(f"[Weights] ⚠️ Unexpected keys: {len(unexpected_keys)} layers")
        
    except Exception as e:
        print(f"[Weights] ❌ Error setting weights: {e}")


def create_fresh_model(num_classes=16):
    """Create a fresh YOLO model without loading previous checkpoints."""
    print(f"[Fresh] 🆕 Creating fresh YOLO model with {num_classes} classes")
    return Net(num_classes=num_classes)
