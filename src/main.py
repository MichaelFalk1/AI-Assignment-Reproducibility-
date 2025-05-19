import torch
import numpy as np
from sliding_puzzle import SlidingPuzzle
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import WUNN, FFNNPlanner, FFNNTrainer

class UncertaintyDrivenTaskGenerator:
    def __init__(self, wunn, device, epsilon=1.0, kappa=0.64, max_steps=1000, min_depth=30):
        self.wunn = wunn
        self.device = device
        self.epsilon = epsilon
        self.kappa = kappa
        self.max_steps = max_steps
        self.min_depth = min_depth

    
    def generate_task(self):
        """Implementation of GenerateTaskPrac using one-hot encoding"""
        current = SlidingPuzzle()  # Start at goal state
        prev_move = None
        visited = set()
        
        for step in range(self.max_steps):
            legal = current.get_legal_moves()
            
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
                    
                # Use one-hot encoding instead of to_array()
                x = torch.tensor(child.to_one_hot(), dtype=torch.float32, device=self.device).unsqueeze(0)
                _, _, sigma_e = self.wunn.predict_with_uncertainty(x, n_samples=100)
                uncertainties.append(sigma_e.item())
                successors.append((mv, child))

            # Check stopping condition (with kappa adjustment)
            current_depth = step + 1
            if uncertainties and max(uncertainties) >= self.epsilon and (current_depth >= self.min_depth):
                print(f"Found state at depth {current_depth} with σₑ = {max(uncertainties):.2f} ≥ {self.epsilon}")
                return current
            
            # Softmax sampling to explore uncertain states
            if successors:
                probs = np.exp(uncertainties) / np.sum(np.exp(uncertainties))
                idx = np.random.choice(len(successors), p=probs)
                prev_move, current = successors[idx]
                visited.add(current.to_hashable())

        print(f"Warning: No state found with σₑ ≥ {self.epsilon} and depth ≥ {self.min_depth}")
        return current

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
        self.wunn = WUNN(dropout_rate=0.025).to(device)  # 2.5% dropout as per paper
        self.ffnn = FFNNPlanner(input_dim=2, hidden_dim=20).to(device)
        
        # Training parameters
        self.beta = beta_0
        self.memory_buffer = []
        self.solved_history = []
        
        # Initialize optimizers
        self.wunn_optimizer = optim.Adam(self.wunn.parameters(), lr=0.01)  # Paper's learning rate
        self.ffnn_trainer = FFNNTrainer(device)
        
    def run_curriculum(self, num_iter=50):
        """Main learning loop implementing LearnHeuristicPrac"""
        for iteration in range(num_iter):
            print(f"\n=== Curriculum Iteration {iteration+1}/{num_iter} ===")
            print(f"Current α: {self.alpha:.2f}, β: {self.beta:.4f}")
            
            # Generate and solve tasks
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
                task = task_generator.generate_task()
                if task.manhattan_distance() >= 10:  # Filter trivial tasks
                    current_tasks.append(task)
            
            # Solve tasks with current heuristic
            y_q = np.quantile([y for _, y in self.memory_buffer], 0.95) if self.memory_buffer else 0
            heuristic_fn = lambda s: self._rho_likely_admissible_heuristic(s, y_q)
            
            for task in current_tasks:
                path, cost = self._solve_task(task, heuristic_fn)
                if path:
                    tasks_solved += 1
                    self._update_memory_buffer(path, cost)
            
            # Adjust alpha based on performance
            self._adjust_parameters(tasks_solved)
            self.solved_history.append(tasks_solved)
            
            print(f"Solved {tasks_solved}/{self.num_tasks_per_iter} tasks this iteration")
            
            # Train WUNN with the paper's convergence criteria
            self._train_wunn()
        
        # Final FFNN training
        print("\nTraining final FFNN...")
        self._train_ffnn()
        
        return self.wunn, self.ffnn, self.memory_buffer, self.solved_history
    
    def _rho_likely_admissible_heuristic(self, state, y_q):
        """Equation 3 with Section 4 approximation from the paper"""
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, sigma_a, sigma_e = self.wunn.predict_with_uncertainty(x)
        
        # Paper's approximation (Section 4)
        if y_q is not None and mean.item() < y_q:
            sigma_t = sigma_a  # Use only aleatoric uncertainty
        else:
            sigma_t = sigma_a + self.epsilon  # Lower-bound epistemic uncertainty
        
        # Compute quantile using inverse Gaussian CDF
        z_score = norm.ppf(self.alpha)
        quantile = mean + z_score * sigma_t
        
        return max(quantile.item(), 0)  # Floor at 0
    
    def _solve_task(self, task, heuristic_fn):
        """Solve task with time limit using IDA*"""
        start_time = time.time()
        path, cost = ida_star_search(
            task, 
            heuristic_fn,
            max_depth=100,
            timeout=self.t_max
        )
        return path, cost
    
    def _update_memory_buffer(self, path, cost):
        """Update memory buffer with new training data"""
        for i, state in enumerate(path):
            if not state.is_solved():
                cost_to_go = len(path) - i - 1  # Each move has cost 1
                features = state.to_one_hot()
                
                # Maintain memory buffer size (FIFO)
                if len(self.memory_buffer) >= self.memory_buffer_max:
                    self.memory_buffer.pop(0)
                self.memory_buffer.append((features, cost_to_go))
    
    def _adjust_parameters(self, tasks_solved):
        """Adjust α and β according to paper's strategy"""
        # Adjust α (admissibility probability)
        if tasks_solved < 6:  # NumTasksPerIterThresh=6
            self.alpha = max(self.alpha - 0.05, 0.5)  # Δ=0.05
        
        # β is adjusted during WUNN training if not converged
    
    def _train_wunn(self):
        """Train WUNN with paper's convergence criteria"""
        if not self.memory_buffer:
            return
            
        states = torch.tensor(np.array([x for x, _ in self.memory_buffer]), dtype=torch.float32)
        targets = torch.tensor(np.array([y for _, y in self.memory_buffer]), dtype=torch.float32).view(-1, 1)
        
        # Train until epistemic uncertainty is below threshold or max iterations
        converged = False
        for train_iter in range(100):  # MaxTrainIter=100
            # Sample batch according to epistemic uncertainty
            with torch.no_grad():
                _, _, sigma_e = self.wunn.predict_with_uncertainty(states.to(self.device))
                weights = torch.where(sigma_e >= self.kappa*self.epsilon,
                                   torch.exp(sigma_e),
                                   torch.exp(torch.ones_like(sigma_e)*-1))
                weights = weights / weights.sum()
            
            # Train epoch
            epoch_loss = 0
            idx = torch.multinomial(weights.flatten(), len(states), replacement=True)
            dataloader = DataLoader(TensorDataset(states[idx], targets[idx]), 
                                  batch_size=100, shuffle=True)
            
            self.wunn.train()
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.wunn_optimizer.zero_grad()
                
                output = self.wunn(x_batch)
                mean = output[:,0:1]
                log_var = output[:,1:2]
                
                # Negative log likelihood
                var = torch.exp(log_var).clamp(min=1e-4)
                nll_loss = ((mean - y_batch)**2 / var + log_var).mean()
                
                # KL divergence term
                kl_div = sum(torch.sum(p**2) for p in self.wunn.parameters() if p.dim() > 1) / (2 * 10)  # σ₀²=10
                
                # Total loss
                loss = nll_loss + self.beta * kl_div
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wunn.parameters(), 1.0)
                self.wunn_optimizer.step()
                epoch_loss += loss.item()
            
            # Check convergence
            self.wunn.eval()
            with torch.no_grad():
                _, _, sigma_e = self.wunn.predict_with_uncertainty(states.to(self.device))
                max_unc = sigma_e.max().item()
                mean_unc = sigma_e.mean().item()
                
                print(f"Train Iter {train_iter}: Loss={epoch_loss/len(dataloader):.3f} | "
                      f"Max σₑ={max_unc:.3f} | Mean σₑ={mean_unc:.3f}")
                
                if torch.all(sigma_e < self.kappa*self.epsilon):
                    print(f"Epistemic uncertainty converged below {self.kappa*self.epsilon:.2f}")
                    converged = True
                    break
        
        # Adjust β if not converged
        if not converged:
            self.beta = self.gamma * self.beta
            print(f"Reducing β to {self.beta:.6f} for next iteration")
    
    def _train_ffnn(self):
        """Train FFNN for planning"""
        states = np.array([x for x, _ in self.memory_buffer])
        targets = np.array([y for _, y in self.memory_buffer])
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


def generate_benchmark_tasks(n=100):
    """Generate standard 15-puzzle benchmark tasks"""
    puzzles = []
    for _ in range(n):
        p = SlidingPuzzle().shuffle(50)  # Heavier shuffle for benchmark
        while p.manhattan_distance() < 30:  # Ensure non-trivial puzzles
            p = SlidingPuzzle().shuffle(50)
        puzzles.append(p)
    return puzzles

def main():
    # Set device (use GPU if available)
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize models with paper's architecture specifications
    print("\nInitializing models...")
    wunn = WUNN(
        input_dim=128,  # 16 tiles * 2 coordinates (row,col) with 4-bit one-hot encoding
        hidden_dim=20,  # As specified in paper for 15-puzzle
        dropout_rate=0.025  # 2.5% dropout for aleatoric uncertainty
    ).to(device)
    
    ffnn = FFNNPlanner(
        input_dim=128,  # Same as WUNN
        hidden_dim=20   # Same as WUNN
    ).to(device)

    # Initialize learner with paper's exact parameters
    print("\nInitializing curriculum learner...")
    learner = LikelyAdmissibleHeuristicLearner(
        device=device,
        num_tasks_per_iter=10,       # NumTasksPerIter
        epsilon=0.8,                  # ε threshold
        alpha=0.99,                   # Initial α (admissibility probability)
        kappa=0.64,                   # κ*ε convergence threshold
        beta_0=0.05,                  # Initial β for KL term
        gamma=0.9,                    # β decay factor
        memory_buffer_max=25000,      # MemoryBufferMaxRecords
        t_max=60                      # Planning time limit (seconds)
    )

    # Run curriculum learning for 50 iterations as in paper's experiments
    print("\nStarting curriculum learning...")
    start_time = time.time()
    wunn, ffnn, memory_buffer, solved_history = learner.run_curriculum(num_iter=50)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")

    # Save final models
    print("\nSaving models...")
    torch.save({
        'wunn_state_dict': wunn.state_dict(),
        'ffnn_state_dict': ffnn.state_dict(),
        'memory_buffer': memory_buffer,
        'solved_history': solved_history,
        'training_time': training_time
    }, 'final_models.pth')

    # Evaluation on test tasks
    print("\nEvaluating on test tasks...")
    
    # Compute y_q as 95th percentile of training costs (q=0.95 from paper)
    y_q = np.quantile([y for _, y in memory_buffer], 0.95) if memory_buffer else 0
    
    # Create heuristic function using both networks as per paper's implementation
    heuristic_fn = lambda state: learner._rho_likely_admissible_heuristic(state, y_q)
    
    # Test on standard 100 15-puzzle benchmark tasks
    benchmark_tasks = generate_benchmark_tasks(100)  # You'll need to implement this
    results = evaluate_heuristic(benchmark_tasks, heuristic_fn)
    
    print("\nEvaluation Results:")
    print(f"- Tasks solved: {results['solved']}/100")
    print(f"- Average suboptimality: {results['suboptimality']:.2f}%")
    print(f"- Average nodes generated: {results['avg_nodes']:,}")
    print(f"- Average planning time: {results['avg_time']:.2f}s")

    # Additional analysis suggested by paper
    print("\nAdditional Analysis:")
    analyze_heuristic_quality(ffnn, memory_buffer, device)
    plot_training_progress(solved_history)

def evaluate_heuristic(tasks, heuristic_fn):
    """Evaluate heuristic on test tasks"""
    results = {
        'solved': 0,
        'suboptimality': 0,
        'avg_nodes': 0,
        'avg_time': 0
    }
    
    for task in tasks:
        start_time = time.time()
        path, cost = ida_star_search(task, heuristic_fn)
        elapsed = time.time() - start_time
        
        if path:
            results['solved'] += 1
            optimal_cost = len(path) - 1  # Assuming we know optimal cost
            subopt = (cost / optimal_cost - 1) * 100
            results['suboptimality'] += subopt
            results['avg_nodes'] += ...  # Track nodes generated
        results['avg_time'] += elapsed
    
    # Compute averages
    if results['solved'] > 0:
        results['suboptimality'] /= results['solved']
        results['avg_nodes'] /= results['solved']
    results['avg_time'] /= len(tasks)
    
    return results

def analyze_heuristic_quality(ffnn, memory_buffer, device):
    """Additional analysis of heuristic quality"""
    # Feature importance analysis
    print("\nFeature Importance Analysis:")
    # analyze_feature_importance(ffnn, device)
    
    # Heuristic error analysis
    print("\nHeuristic Error Analysis:")
    # analyze_heuristic_error(ffnn, memory_buffer, device)

def plot_training_progress(solved_history):
    """Plot training progress as in paper's figures"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(solved_history, marker='o')
    plt.xlabel("Curriculum Iteration")
    plt.ylabel("Tasks Solved per Iteration")
    plt.title("Learning Progress")
    plt.grid(True)
    plt.savefig("learning_progress.png")
    print("\nSaved learning progress plot to learning_progress.png")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure more detailed logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    main()