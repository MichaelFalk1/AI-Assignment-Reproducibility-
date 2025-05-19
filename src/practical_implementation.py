import torch
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sliding_puzzle import SlidingPuzzle
from model import WUNN, FFNNPlanner, FFNNTrainer


class UncertaintyDrivenTaskGenerator:
    def __init__(self, wunn, device, epsilon=1.0, kappa=0.64, max_steps=1000, min_depth=30):
        self.wunn = wunn
        self.device = device
        self.epsilon = epsilon
        self.kappa = kappa
        self.max_steps = max_steps
        self.min_depth = min_depth

    def generate_task(self, puzzle_class):
        """Generate tasks using epistemic uncertainty with improved exploration"""
        current = puzzle_class()  # Start at goal state
        prev_move = None
        visited = set()
        best_state = None
        max_uncertainty = -1
        
        for step in range(self.max_steps):
            legal = current.get_legal_moves()
            
            # Exclude the move that would take us back to previous state
            if prev_move:
                undo = {'up':'down', 'down':'up', 'left':'right', 'right':'left'}[prev_move]
                if undo in legal:
                    legal.remove(undo)
            
            if not legal:
                break

            uncertainties = []
            successors = []
            for mv in legal:
                child = current.move(mv)
                if child.to_hashable() in visited:
                    continue
                    
                x = torch.tensor(child.to_one_hot(), dtype=torch.float32, device=self.device).unsqueeze(0)
                _, _, sigma_e = self.wunn.predict_with_uncertainty(x, n_samples=100)
                uncertainty = sigma_e.item()
                uncertainties.append(uncertainty)
                successors.append((mv, child, uncertainty))
                
                # Track the state with maximum uncertainty seen so far
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    best_state = child

            current_depth = step + 1
            
            # Return if we find a state meeting our criteria
            if uncertainties and max(uncertainties) >= self.epsilon and (current_depth >= self.min_depth):
                print(f"Found state at depth {current_depth} with σₑ = {max(uncertainties):.2f} ≥ {self.epsilon}")
                return current
            self.epsilon *= 0.98  # Decay per iteration

            # If we have successors, choose one based on uncertainty
            if successors:
                # Use softmax with temperature to balance exploration/exploitation
                temp = 0.5  # Temperature parameter
                uncertainties = np.array([u[2] for u in successors])
                probs = np.exp(uncertainties/temp) / np.sum(np.exp(uncertainties/temp))
                idx = np.random.choice(len(successors), p=probs)
                prev_move, current, _ = successors[idx]
                visited.add(current.to_hashable())
        
        # Fallback: Return the best state we found (even if below threshold)
        if best_state is not None and best_state.manhattan_distance() >= 10:
            print(f"Using fallback state with σₑ = {max_uncertainty:.2f} (target ≥ {self.epsilon})")
            return best_state
        
        # Final fallback: return a randomly shuffled puzzle
        print("Using random state as final fallback")
        return puzzle_class().shuffle(self.min_depth)

class LikelyAdmissibleHeuristicLearner:
    def __init__(self, device, num_tasks_per_iter=10, epsilon=1.0, 
                 alpha=0.99, kappa=0.64, beta_0=0.05, gamma=0.9, 
                 memory_buffer_max=25000, t_max=60):
        self.device = device
        self.num_tasks_per_iter = num_tasks_per_iter
        self.epsilon = epsilon
        self.alpha = alpha
        self.kappa = kappa
        self.beta_0 = beta_0
        self.gamma = gamma
        self.memory_buffer_max = memory_buffer_max
        self.t_max = t_max
        
        # Initialize models
        self.wunn = WUNN(input_dim=128, hidden_dim=20, dropout_rate=0.025).to(device)
        self.ffnn = FFNNPlanner(input_dim=2, hidden_dim=20).to(device)
        
        # Training parameters
        self.beta = beta_0
        self.memory_buffer = []
        self.solved_history = []
        
        # Initialize optimizers
        self.wunn_optimizer = optim.Adam(self.wunn.parameters(), lr=0.01)
        self.ffnn_trainer = FFNNTrainer(device)

     # Add this before starting curriculum
    def warmup_wunn(self, device, n_samples=1000):
        """Pretrain WUNN on random states"""
        optimizer = optim.Adam(self.wunn.parameters(), lr=0.01)
        for _ in range(100):  # 100 warmup iterations
            states = torch.randn(n_samples, 128).to(device)  # Random states
            targets = torch.randn(n_samples, 1).to(device) * 50  # Random costs
            optimizer.zero_grad()
            outputs = self.wunn(states)
            loss = nn.MSELoss()(outputs[:,0], targets.squeeze())
            loss.backward()
            optimizer.step()
    
    def run_curriculum(self, puzzle_class, num_iter=50):
        """Main learning loop"""
        for iteration in range(num_iter):
            print(f"\n=== Curriculum Iteration {iteration+1}/{num_iter} ===")
            print(f"Current α: {self.alpha:.2f}, β: {self.beta:.4f}")
            task_generator = UncertaintyDrivenTaskGenerator(
                self.wunn, self.device,
                epsilon=self.epsilon,
                kappa=self.kappa,
                max_steps=1000,
                min_depth=30
            )
            
            tasks_solved = 0
            current_tasks = []
            
            while len(current_tasks) < self.num_tasks_per_iter:
                task = task_generator.generate_task(puzzle_class)
                if task.manhattan_distance() >= 10:
                    current_tasks.append(task)
            
            y_q = np.quantile([y for _, y in self.memory_buffer], 0.95) if self.memory_buffer else 0
            heuristic_fn = lambda s: self._rho_likely_admissible_heuristic(s, y_q)
            manhatten = SlidingPuzzle.manhattan_distance          
            
            
            
            for task in current_tasks:
                print("Solving Tasks...")
                h_fn = heuristic_fn(task)
                print(f"MD: {task.manhattan_distance():2d} -> H: {h_fn:.1f} (α={self.alpha:.2f})")
                # NOTE remeber to change this back just doing this to test ffnn
                path, cost = self._solve_task(task, manhatten)
                if path:
                    tasks_solved += 1
                    self._update_memory_buffer(path, cost)


            print(f"Solved {tasks_solved}/{self.num_tasks_per_iter} tasks this iteration")
            self._adjust_parameters(tasks_solved)
            self.solved_history.append(tasks_solved)
            self._train_wunn()
        
        self._train_ffnn()
        return self.wunn, self.ffnn, self.memory_buffer, self.solved_history
    
    # def _rho_likely_admissible_heuristic(self, state, y_q):
    #     """Likely-admissible heuristic calculation"""
    #     x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(self.device)
    #     mean, sigma_a, sigma_e = self.wunn.predict_with_uncertainty(x)
        
    #     if y_q is not None and mean.item() < y_q:
    #         sigma_t = sigma_a
    #     else:
    #         sigma_t = sigma_a + self.epsilon
            
    #     z_score = norm.ppf(self.alpha)
    #     quantile = mean + z_score * sigma_t
    #     return max(quantile.item(), 0)

    def _rho_likely_admissible_heuristic(self, state, y_q, alpha=None):
        """Improved heuristic with Manhattan distance blending and better scaling"""
        alpha = alpha or self.alpha
        
        # Get Manhattan distance as fallback
        manhattan = state.manhattan_distance()
        
        # Get neural network prediction
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, sigma_a, sigma_e = self.wunn.predict_with_uncertainty(x)
            
            # Dynamic blending with Manhattan based on training progress
            training_progress = min(1.0, len(self.memory_buffer) / 5000)
            blend_factor = max(0, 1 - training_progress**2)  # Faster decay
            
            # Combine uncertainties
            sigma_t = sigma_a + self.epsilon * min(1.0, training_progress*2)
            
            # Compute quantile
            z_score = norm.ppf(alpha)
            nn_estimate = mean + z_score * sigma_t
            nn_estimate = max(nn_estimate.item(), 0)
            
            # Blend with Manhattan
            final_h = (1-blend_factor) * nn_estimate + blend_factor * manhattan
            
            # Ensure admissibility during early training
            if len(self.memory_buffer) < 1000:
                final_h = min(final_h, manhattan)
                
            return final_h
        
    def _log_heuristic_stats(self):
        if len(self.memory_buffer) % 1000 == 0:
            test_states = [SlidingPuzzle().shuffle(i) for i in [10,20,30,40]]
            for s in test_states:
                h = self._rho_likely_admissible_heuristic(s, 0)
                print(f"MD: {s.manhattan_distance():2d} -> H: {h:.1f} (α={self.alpha:.2f})")

    def _solve_task(self, task, heuristic_fn):
        """Solve task with time limit"""
        start_time = time.time()
        path, cost = ida_star_search(task, heuristic_fn, timeout=self.t_max)
        return path, cost

    def _update_memory_buffer(self, path, cost):
        """Update memory buffer with shape validation"""
        for i, state in enumerate(path):
            if not state.is_solved():
                try:
                    features = state.to_one_hot()
                    cost_to_go = len(path) - i - 1
                    
                    # Ensure features is a flat numpy array
                    if isinstance(features, torch.Tensor):
                        features = features.cpu().numpy()
                    features = np.array(features).flatten()
                    
                    # Validate shapes
                    if features.shape != (128,):  # For 15-puzzle one-hot encoding
                        print(f"Invalid feature shape: {features.shape} - skipping")
                        continue
                        
                    if len(self.memory_buffer) >= self.memory_buffer_max:
                        self.memory_buffer.pop(0)
                        
                    self.memory_buffer.append((features, float(cost_to_go)))
                    
                except Exception as e:
                    print(f"Error updating memory buffer: {str(e)}")

    def _adjust_parameters(self, tasks_solved):
        """Better alpha adaptation based on multiple factors"""
        progress = len(self.memory_buffer) / self.memory_buffer_max
        
        # Base adjustment
        if tasks_solved < self.num_tasks_per_iter * 0.6:
            self.alpha = max(0.7, self.alpha - 0.05)
        elif tasks_solved == self.num_tasks_per_iter and progress > 0.5:
            self.alpha = min(0.99, self.alpha + 0.01)
            
        # Progressive tightening
        if progress > 0.8 and self.alpha < 0.95:
            self.alpha = min(0.99, self.alpha + 0.01)

   

    def _train_wunn(self):
        """Enhanced WUNN training for better uncertainty estimates"""
        if len(self.memory_buffer) < 100:
            return  # Wait until we have enough data
            
        states = torch.tensor(np.array([x for x, _ in self.memory_buffer]), dtype=torch.float32)
        targets = torch.tensor(np.array([y for _, y in self.memory_buffer]), dtype=torch.float32).view(-1,1)
        
        # Normalize targets
        target_mean, target_std = targets.mean(), targets.std()
        normalized_targets = (targets - target_mean) / (target_std + 1e-6)
        
        for epoch in range(100):
            # Sample based on both uncertainty and recentness
            weights = torch.ones(len(states))
            if epoch > 10:  # After initial exploration
                _, _, sigma_e = self.wunn.predict_with_uncertainty(states.to(self.device))
                weights = 0.7*sigma_e.squeeze() + 0.3*torch.linspace(0,1,len(states))  # Combine
            
            # Train batch
            idx = torch.multinomial(weights, len(states), replacement=True)
            dataloader = DataLoader(TensorDataset(states[idx], normalized_targets[idx]), 
                                batch_size=100, shuffle=True)
            
            for x_batch, y_batch in dataloader:
                self.wunn_optimizer.zero_grad()
                outputs = self.wunn(x_batch.to(self.device))
                
                # Split outputs
                mean = outputs[:,0] * target_std + target_mean  # Denormalize
                log_var = outputs[:,1]
                
                # Loss components
                var = torch.exp(log_var).clamp(min=1e-4)
                nll_loss = ((mean - targets[idx])**2 / var + log_var).mean()
                kl_div = sum(p.pow(2).sum() for p in self.wunn.parameters()) / (2 * 10)
                
                loss = nll_loss + self.beta * kl_div
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wunn.parameters(), 1.0)
                self.wunn_optimizer.step()

    def _train_ffnn(self):
        """Train planning network with robust memory buffer handling"""
        if not self.memory_buffer or len(self.memory_buffer) < 100:
            print("Not enough training data - skipping FFNN training")
            return
        # Convert memory buffer to numpy arrays with proper shape checking
        states = []
        targets = []
        for x, y in self.memory_buffer:
            # Ensure each state is properly shaped
            if isinstance(x, (np.ndarray, list)):
                states.append(np.array(x).flatten())  # Flatten to 1D array
            else:
                states.append(x.numpy().flatten() if hasattr(x, 'numpy') else np.array(x).flatten())
            targets.append(float(y))
        
        states = np.array(states)
        targets = np.array(targets)

        print("\nTraining Data Summary:")
        print(f"Number of samples: {len(states)}")
        print(f"Targets - Min: {targets.min():.2f}, Max: {targets.max():.2f}, Mean: {targets.mean():.2f}")

        if len(states) > 0:
            sample_idx = np.random.randint(0, len(states))
            print(f"\nSample input: {states[sample_idx][:10]}... (length {len(states[sample_idx])})")
            print(f"Sample target: {targets[sample_idx]}")
        
        # Check for invalid values
        if np.isnan(states).any() or np.isinf(states).any() or \
        np.isnan(targets).any() or np.isinf(targets).any():
            print("Invalid values detected in training data - resetting memory buffer")
            self.memory_buffer = []
            return
            
        print(f"Training FFNN on {len(states)} samples (target range: {targets.min():.1f}-{targets.max():.1f})")
        self.ffnn = self.ffnn_trainer.train(self.wunn, self.ffnn, list(zip(states, targets)))

    


def ida_star_search(start_puzzle, heuristic_fn, max_depth=100, timeout=60):
    """Iterative Deepening A* Search with heuristic function"""
    threshold = heuristic_fn(start_puzzle)
    start_time = time.time()
    
    while True:
        result, path, cost = depth_limited_search(
            current_path=[start_puzzle],
            g=0,
            threshold=threshold,
            heuristic_fn=heuristic_fn,
            max_depth=max_depth,
            start_time=start_time,
            timeout=timeout
        )
        
        if result == "FOUND":
            return path, cost
        elif result == float('inf'):
            return None, None
        elif time.time() - start_time > timeout:
            print("IDA* timeout reached")
            return None, None
        
        threshold = result

def depth_limited_search(current_path, g, threshold, heuristic_fn, max_depth, start_time, timeout):
    """Helper function for IDA*"""
    current_state = current_path[-1]
    f = g + heuristic_fn(current_state)
    
    if time.time() - start_time > timeout:
        return float('inf'), None, None
        
    if f > threshold:
        return f, None, None
        
    if current_state.is_solved():
        return "FOUND", current_path, len(current_path)-1
        
    if g >= max_depth:
        return float('inf'), None, None
        
    min_threshold = float('inf')
    for move in current_state.get_legal_moves():
        neighbor = current_state.move(move)
        
        if any(np.array_equal(neighbor.to_array(), p.to_array()) for p in current_path):
            continue
            
        result, solution_path, cost = depth_limited_search(
            current_path + [neighbor],
            g + 1,
            threshold,
            heuristic_fn,
            max_depth,
            start_time,
            timeout
        )
        
        if result == "FOUND":
            return "FOUND", solution_path, cost
        if result < min_threshold:
            min_threshold = result
            
    return min_threshold, None, None

# Add this before starting curriculum
def warmup_wunn(wunn, device, n_samples=1000):
    """Pretrain WUNN on random states"""
    optimizer = optim.Adam(wunn.parameters(), lr=0.01)
    for _ in range(100):  # 100 warmup iterations
        states = torch.randn(n_samples, 128).to(device)  # Random states
        targets = torch.randn(n_samples, 1).to(device) * 50  # Random costs
        optimizer.zero_grad()
        outputs = wunn(states)
        loss = nn.MSELoss()(outputs[:,0], targets.squeeze())
        loss.backward()
        optimizer.step()
    