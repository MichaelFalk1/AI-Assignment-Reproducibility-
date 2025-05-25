import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import copy
from scipy.stats import norm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sliding_puzzle import SlidingPuzzle

class WUNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=20, dropout_rate=0.025):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Output: mean and log variance
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights with He Normal initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        self.train()  # Enable dropout for MC sampling
        with torch.no_grad():
            outputs = torch.stack([self(x) for _ in range(n_samples)])  # [n_samples, batch_size, 2]
            means = outputs[:, :, 0]       # [n_samples, batch_size]
            log_vars = outputs[:, :, 1]    # [n_samples, batch_size]

            mean = means.mean(dim=0)  # [batch_size]
            aleatoric_std = torch.sqrt(torch.exp(log_vars).mean(dim=0))  # [batch_size]
            epistemic_std = means.std(dim=0)  # [batch_size]

            return mean, aleatoric_std, epistemic_std

    

class FFNNPlanner(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class FFNNTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.memory_buffer = []
        self.max_records = 25000
        
    
    def prepare_features(self, wunn, states, targets):
        try:
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)  # Convert to numpy array first
            targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)
            
            if wunn is None:
                return states_tensor.to(self.device), targets_tensor.to(self.device)
                
            features = []
            batch_size = min(128, len(states_tensor))
            wunn.eval()
            with torch.no_grad():
                for i in range(0, len(states_tensor), batch_size):
                    batch = states_tensor[i:i+batch_size].to(self.device)
                    mean, sigma_a, _ = wunn.predict_with_uncertainty(batch)
                    
                    # Stack features properly
                    batch_features = torch.stack([mean, sigma_a], dim=1)
                    features.append(batch_features)
            
            features = torch.cat(features, dim=0)
            features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
            return features.to(self.device), targets_tensor.to(self.device)
        except Exception as e:
            print(f"Feature preparation error: {str(e)}")
            return states_tensor.to(self.device), targets_tensor.to(self.device)

    def train(self, wunn, ffnn, dataset, epochs=1000, lr=1e-4):
        try:
            states, targets = zip(*dataset)
            
            # Convert to tensors and ensure proper shapes
            targets = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)
            
            # Prepare features and scale targets properly
            features, targets = self.prepare_features(wunn, states, targets)
            
            # Debug prints for verification
            print("\n=== Training Debug Info ===")
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Sample features: {features[0]}")
            print(f"Sample target: {targets[0].item()}")
            print(f"Target stats - Min: {targets.min().item():.2f}, Max: {targets.max().item():.2f}, Mean: {targets.mean().item():.2f}")
            
            # Normalize targets to [0,1] range
            target_min = targets.min()
            target_max = targets.max()
            targets = (targets - target_min) / (target_max - target_min + 1e-6)
            
            dataset = TensorDataset(features, targets)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            optimizer = optim.Adam(ffnn.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            
            # Add learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            
            ffnn.train()
            best_loss = float('inf')
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for x_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = ffnn(x_batch)
                    
                    # Scale outputs to match normalized targets
                    outputs = torch.sigmoid(outputs)  # Constrain to [0,1]
                    
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(ffnn.parameters(), 1.0)
                
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                scheduler.step(avg_loss)
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(ffnn.state_dict())
                
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                if torch.isnan(torch.tensor(avg_loss)):
                    print("Loss became NaN - stopping training")
                    break
            
            # Load best model
            ffnn.load_state_dict(best_model_state)
            
            # Store scaling parameters for inference
            ffnn.target_min = target_min
            ffnn.target_max = target_max
            
            return ffnn
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            traceback.print_exc()
            return FFNNPlanner(input_dim=features.shape[1] if 'features' in locals() else 2, 
                            hidden_dim=20).to(self.device)

    def create_heuristic_fn(self, wunn, ffnn, y_q=None, epsilon=1.0):
        def heuristic(state):
            x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Get predictions
                mean, sigma_a, sigma_e = wunn.predict_with_uncertainty(x)
                
                # Apply sigmoid to ensure positive outputs
                mean = torch.sigmoid(mean) * 50  # Scale to reasonable puzzle range (0-50)
                sigma_a = torch.sigmoid(sigma_a) * 5  # Scale uncertainty appropriately
                
                features = torch.cat([mean.unsqueeze(1), sigma_a.unsqueeze(1)], dim=1)
                
                # Debug prints
                md = state.manhattan_distance()
                print(f"\nState MD: {md}")
                print(f"NN pred - mean: {mean.item():.2f}, σ_a: {sigma_a.item():.2f}, σ_e: {sigma_e.item():.2f}")
                
                # Decision logic
                if y_q is not None and mean.item() < y_q:
                    # Scale FFNN output back to original range
                    if hasattr(ffnn, 'target_min'):
                        h = ffnn(features).item() * (ffnn.target_max - ffnn.target_min) + ffnn.target_min
                    else:
                        h = ffnn(features).item()
                    print(f"Using FFNN prediction: {h:.2f}")
                else:
                    # Use uncertainty-adjusted estimate
                    z = norm.ppf(min(self.alpha, 0.95))  # Cap alpha for numerical stability
                    h = mean.item() + z * (sigma_a.item() + sigma_e.item())
                    print(f"Using uncertainty-adjusted estimate: {h:.2f} (z={z:.2f})")
                
                # Ensure heuristic is reasonable compared to Manhattan distance
                h = max(h, md * 0.9)  # Never less than 90% of MD
                h = min(h, md * 2.0)  # Never more than 2x MD
                
                print(f"Final heuristic: {h:.2f} (MD: {md})")
                
                return h
                
        return heuristic