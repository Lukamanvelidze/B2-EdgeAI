"""MVA: A Flower / PyTorch YOLO client with proper model persistence."""

import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from mva.task import Net, get_weights, load_data, set_weights, test, train
import hashlib
import numpy as np
import os
import json
from datetime import datetime


class FlowerYOLOClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id=0):
        self.net = net
        self.trainloader = trainloader  # Not used by YOLO but kept for compatibility
        self.valloader = valloader      # Not used by YOLO but kept for compatibility
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Model persistence
        self.global_model_path = f"client_{partition_id}_global_model.pt"
        self.training_log_path = f"client_{partition_id}_training_log.json"
        self.last_weights_hash = None
        
        # Training state
        self.current_round = 0
        self.training_history = self._load_training_history()
        
        # Load previous global model if it exists
        self._load_previous_global_model()
        
        print(f"[Client-{partition_id}] 🚀 YOLO Federated Learning Client initialized")
        print(f"[Client-{partition_id}] 🎯 Model: {net.num_classes} classes")
        print(f"[Client-{partition_id}] 💻 Device: {self.device}")

    def _hash_parameters(self, parameters):
        """Create hash of model parameters for comparison."""
        try:
            flat_array = np.concatenate([p.flatten() for p in parameters if p is not None])
            return hashlib.md5(flat_array.tobytes()).hexdigest()
        except Exception as e:
            print(f"[Client-{self.partition_id}] ⚠️ Error hashing parameters: {e}")
            return "error_hash"

    def _load_training_history(self):
        """Load training history from previous sessions."""
        if os.path.exists(self.training_log_path):
            try:
                with open(self.training_log_path, 'r') as f:
                    history = json.load(f)
                print(f"[Client-{self.partition_id}] 📊 Loaded training history: {len(history)} previous rounds")
                return history
            except Exception as e:
                print(f"[Client-{self.partition_id}] ⚠️ Error loading training history: {e}")
        
        return []

    def _save_training_history(self, round_data):
        """Save training round data to history."""
        try:
            self.training_history.append(round_data)
            with open(self.training_log_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            print(f"[Client-{self.partition_id}] ⚠️ Error saving training history: {e}")

    def _load_previous_global_model(self):
        """Load the last saved global model if it exists."""
        if os.path.exists(self.global_model_path):
            try:
                print(f"[Client-{self.partition_id}] 🔄 Loading previous global model from {self.global_model_path}")
                
                # Load the saved state
                saved_state = torch.load(self.global_model_path, map_location=self.device)
                
                # Handle different save formats
                if isinstance(saved_state, dict) and 'state_dict' in saved_state:
                    state_dict = saved_state['state_dict']
                    if 'round' in saved_state:
                        self.current_round = saved_state['round'] + 1
                        print(f"[Client-{self.partition_id}] 📊 Resuming from round: {self.current_round}")
                elif isinstance(saved_state, dict):
                    state_dict = saved_state
                else:
                    state_dict = saved_state.state_dict() if hasattr(saved_state, 'state_dict') else saved_state
                
                # Load state dict into the YOLO model
                missing_keys, unexpected_keys = self.net.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"[Client-{self.partition_id}] ⚠️ Missing keys: {len(missing_keys)} layers")
                if unexpected_keys:
                    print(f"[Client-{self.partition_id}] ⚠️ Unexpected keys: {len(unexpected_keys)} layers")
                
                print(f"[Client-{self.partition_id}] ✅ Successfully loaded previous global model")
                
                # Store hash of loaded weights
                current_weights = get_weights(self.net)
                if current_weights:
                    self.last_weights_hash = self._hash_parameters(current_weights)
                
            except Exception as e:
                print(f"[Client-{self.partition_id}] ❌ Error loading previous global model: {e}")
                print(f"[Client-{self.partition_id}] 🔄 Starting with fresh model")
        else:
            print(f"[Client-{self.partition_id}] 🆕 No previous global model found")
            print(f"[Client-{self.partition_id}] 🔄 Starting with fresh YOLO model")

    def _save_global_model(self, round_num):
        """Save current global model state."""
        try:
            save_dict = {
                'state_dict': self.net.state_dict(),
                'round': round_num,
                'model_hash': self.last_weights_hash,
                'timestamp': datetime.now().isoformat(),
                'partition_id': self.partition_id,
                'num_classes': self.net.num_classes
            }
            torch.save(save_dict, self.global_model_path)
            print(f"[Client-{self.partition_id}] 💾 Global model saved (Round {round_num})")
        except Exception as e:
            print(f"[Client-{self.partition_id}] ❌ Error saving global model: {e}")

    def fit(self, parameters, config):
        """Training phase with YOLO model."""
        # Get current round number
        server_round = config.get("server_round", self.current_round)
        self.current_round = server_round
        
        print(f"\n[Client-{self.partition_id}] 🏋️ === ROUND {server_round} TRAINING ===")
        
        # Check if we received valid parameters
        if not parameters or len(parameters) == 0:
            print(f"[Client-{self.partition_id}] ⚠️ No parameters received, using current model")
            current_weights = get_weights(self.net)
        else:
            # Log hash of received weights
            current_hash = self._hash_parameters(parameters)
            print(f"[Client-{self.partition_id}] 📥 Received model hash: {current_hash[:10]}...")

            # Check if weights have changed
            if self.last_weights_hash is not None:
                if current_hash == self.last_weights_hash:
                    print(f"[Client-{self.partition_id}] ⚠️ Model weights unchanged since last round")
                else:
                    print(f"[Client-{self.partition_id}] ✅ Model weights updated since last round")
            else:
                print(f"[Client-{self.partition_id}] 🆕 First round or fresh start")

            # Update model with new global weights
            print(f"[Client-{self.partition_id}] 🔄 Setting new weights...")
            set_weights(self.net, parameters)
            self.last_weights_hash = current_hash
            current_weights = parameters

        # Move model to correct device
        self.net.to(self.device)

        # Train the YOLO model
        print(f"[Client-{self.partition_id}] 🏋️ Starting YOLO training for {self.local_epochs} epochs...")
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        
        # Get updated weights after training
        updated_weights = get_weights(self.net)
        
        # Calculate dataset size (mock for YOLO)
        num_examples = 100  # YOLO handles data internally, this is a placeholder
        
        # Save training results
        round_data = {
            'round': server_round,
            'train_loss': train_loss,
            'num_examples': num_examples,
            'timestamp': datetime.now().isoformat(),
            'local_epochs': self.local_epochs
        }
        self._save_training_history(round_data)

        # Save the updated global model
        self._save_global_model(server_round)

        print(f"[Client-{self.partition_id}] ✅ Training completed - Loss: {train_loss:.4f}")
        print(f"[Client-{self.partition_id}] 📤 Sending updated weights to server")
        
        return (
            updated_weights,
            num_examples,
            {"train_loss": train_loss, "round": server_round, "partition_id": self.partition_id},
        )

    def evaluate(self, parameters, config):
        """Evaluation phase with YOLO model."""
        server_round = config.get("server_round", self.current_round)
        
        print(f"\n[Client-{self.partition_id}] 🧪 === ROUND {server_round} EVALUATION ===")
        
        # Set weights for evaluation if provided
        if parameters and len(parameters) > 0:
            print(f"[Client-{self.partition_id}] 🔄 Setting evaluation weights...")
            set_weights(self.net, parameters)
        
        # Move model to correct device
        self.net.to(self.device)
        
        # Evaluate YOLO model
        print(f"[Client-{self.partition_id}] 🧪 Starting YOLO evaluation...")
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        # Mock dataset size for evaluation
        num_examples = 50  # YOLO handles data internally
        
        print(f"[Client-{self.partition_id}] ✅ Evaluation completed")
        print(f"[Client-{self.partition_id}] 📊 Loss: {loss:.4f}, mAP@0.5: {accuracy:.4f}")
        
        return (
            loss, 
            num_examples, 
            {
                "accuracy": accuracy, 
                "mAP@0.5": accuracy,
                "round": server_round,
                "partition_id": self.partition_id
            }
        )


def client_fn(context: Context):
    """Create YOLO client instance with proper configuration."""
    # Get configuration from context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    
    print(f"\n[Flower] 🌸 Initializing YOLO client {partition_id}/{num_partitions}")
    
    # Create YOLO model with 16 classes
    net = Net(num_classes=16)
    
    # Load data (YOLO handles this internally via data.yaml)
    trainloader, valloader = load_data(partition_id, num_partitions)
    
    print(f"[Flower] 📚 YOLO data config: {net.config_path}")
    print(f"[Flower] 🏋️ Local epochs per round: {local_epochs}")
    print(f"[Flower] 🎯 Number of classes: {net.num_classes}")

    # Return YOLO Client instance
    return FlowerYOLOClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


# Flower ClientApp
app = ClientApp(client_fn)


def main():
    """Main function for direct client execution."""
    print("🌸 Starting Flower YOLO Federated Learning Client")
    
    # Initialize YOLO model with 16 classes
    net = Net(num_classes=16)
    
    # Load data (YOLO uses internal data loading)
    trainloader, valloader = load_data(0, 1)  # Single partition for direct execution
    
    print(f"🎯 YOLO Model: {net.num_classes} classes")
    print(f"📁 Data config: {net.config_path}")
    print(f"🌐 Connecting to server at: 34.32.97.252:8080")
    
    # Start Flower client
    try:
        fl.client.start_client(
            server_address="34.32.97.252:8080",
            client=FlowerYOLOClient(net, trainloader, valloader, local_epochs=1, partition_id=0).to_client(),
        )
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user")
    except Exception as e:
        print(f"❌ Client error: {e}")


if __name__ == "__main__":
    main()
