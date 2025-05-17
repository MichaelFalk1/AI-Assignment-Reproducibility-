# heuristic_learner.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # <- added

class HeuristicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_model(dataset, epochs=20, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeuristicNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Use NumPy for efficient conversion
    states_np = np.array([x for x, _ in dataset], dtype=np.float32)
    costs_np = np.array([y for _, y in dataset], dtype=np.float32).reshape(-1, 1)

    states = torch.from_numpy(states_np)
    costs = torch.from_numpy(costs_np)

    dataloader = DataLoader(TensorDataset(states, costs), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_costs in dataloader:
            batch_states, batch_costs = batch_states.to(device), batch_costs.to(device)
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = loss_fn(outputs, batch_costs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
