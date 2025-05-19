import torch
import torch.nn as nn
import torch.optim as optim
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
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class FFNNTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.memory_buffer = []
        self.max_records = 25000
        
    
    def prepare_features(self, wunn, states, targets):
        """Robust feature preparation with shape validation"""
        try:
            # Convert to tensors
            states_tensor = torch.tensor(states, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
            
            if wunn is None:
                return states_tensor.to(self.device), targets_tensor.to(self.device)
                
            # Get features from WUNN
            features = []
            batch_size = min(128, len(states_tensor))
            wunn.eval()
            with torch.no_grad():
                for i in range(0, len(states_tensor), batch_size):
                    batch = states_tensor[i:i+batch_size].to(self.device)
                    mean, sigma_a, _ = wunn.predict_with_uncertainty(batch)
                    
                    # Ensure proper dimensions
                    if mean.dim() == 1:
                        mean = mean.unsqueeze(1)
                    if sigma_a.dim() == 1:
                        sigma_a = sigma_a.unsqueeze(1)
                        
                    features.append(torch.cat([mean, sigma_a], dim=1))
            
            return torch.cat(features).to(self.device), targets_tensor.to(self.device)
            
        except Exception as e:
            print(f"Feature preparation error: {str(e)}")
            # Fallback to direct state features
            return states_tensor.to(self.device), targets_tensor.to(self.device)

    def train(self, wunn, ffnn, dataset, epochs=1000, lr=0.001):
        """Robust training with error handling"""
        try:
            # Prepare features
            states, targets = zip(*dataset)
            features, targets = self.prepare_features(wunn, states, targets)
            
            # Create DataLoader
            dataset = TensorDataset(features, targets)
            dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
            
            # Training setup
            optimizer = optim.Adam(ffnn.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            best_loss = float('inf')
            
            ffnn.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for x_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = ffnn(x_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ffnn.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
                
                # Early stopping if loss becomes NaN
                if torch.isnan(torch.tensor(avg_loss)):
                    print("Loss became NaN - stopping training")
                    break
                    
            return ffnn
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            # Return untrained model if error occurs
            return FFNNPlanner(input_dim=features.shape[1], hidden_dim=20).to(self.device)

    def create_heuristic_fn(self, wunn, ffnn, y_q=None, epsilon=1.0):
        def heuristic(state):
            x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Debug prints
                print(f"Input shape: {x.shape}")  # Should be [1, 128]
                
                mean, sigma_a, sigma_e = wunn.predict_with_uncertainty(x)
                print(f"Mean: {mean.item():.2f}, Sigma_a: {sigma_a.item():.2f}, Sigma_e: {sigma_e.item():.2f}")
                
                features = torch.cat([mean.unsqueeze(1), sigma_a.unsqueeze(1)], dim=1)
                print(f"Features shape: {features.shape}")  # Should be [1, 2]
                
                if y_q is not None:
                    print(f"Comparison: mean={mean.item():.2f} vs y_q={y_q:.2f}")
                
                if y_q is not None and mean < y_q:
                    h = ffnn(features).item()
                    print(f"Using FFNN prediction: {h:.2f}")
                else:
                    h = max(mean.item() - 1.0 * (sigma_a.item() + epsilon), 0)
                    print(f"Using conservative estimate: {h:.2f}")
                    
                print(f"Final heuristic: {h:.2f} vs Manhattan: {state.manhattan_distance()}")
                return h
        return heuristic