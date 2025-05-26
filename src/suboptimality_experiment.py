import torch
from scipy.stats import norm
import numpy as np
from practical_implementation import LikelyAdmissibleHeuristicLearner, ida_star_search
from sliding_puzzle import SlidingPuzzle
import time
from model import WUNN, FFNNPlanner, FFNNTrainer

def generate_benchmark_tasks(n=100):
    """Generate standard 15-puzzle benchmark tasks"""
    puzzles = []
    for a in range(n):
        p = SlidingPuzzle().shuffle(100)  # Heavier shuffle for benchmark
        while p.manhattan_distance() < 20:  # Ensure non-trivial puzzles
            p = SlidingPuzzle().shuffle(50)
        puzzles.append(p)
        print(f"Puzzle generated {a+1}")
    return puzzles

def evaluate_heuristic(tasks, heuristic_fn):
    """Evaluate heuristic on test tasks"""
    results = {
        'solved': 0,
        'suboptimality': 0,
        'avg_time': 0,
        'optimal': 0,
        'avg_nodes': 0
    }
    for task in tasks:
        start_time = time.time()
        path, cost, nodes_generated = ida_star_search(task, heuristic_fn)
        elapsed = time.time() - start_time
        if path:
            results['solved'] += 1
            optimal_cost = len(path) - 1
            subopt = (cost / optimal_cost - 1) * 100
            results['suboptimality'] += subopt
            if subopt < 1e-6:
                results['optimal'] += 1
            results['avg_nodes'] += nodes_generated
        results['avg_time'] += elapsed
    n_tasks = len(tasks)
    if results['solved'] > 0:
        results['suboptimality'] /= results['solved']
        results['optimal'] = (results['optimal'] / n_tasks) * 100
        results['avg_nodes'] /= results['solved']
    results['avg_time'] /= n_tasks
    return results
def run_suboptimality_experiment():
    device = torch.device('cpu')
    
    # Experiment parameters from paper
    params = {
        'num_tasks_per_iter': 10,
        'epsilon': 1.0,
        'alpha': 0.99,
        'kappa': 0.64,
        'beta_0': 0.05,
        'gamma': 0.9,
        'memory_buffer_max': 25000,
        't_max': 60,
        'num_iter': 50
    }
    
    # Run curriculum learning
    learner = LikelyAdmissibleHeuristicLearner(
        device,
        alpha=0.99,
        num_tasks_per_iter=10,
        epsilon=1,
        kappa=0.64,
        beta_0=0.05,
        gamma=0.9,
        memory_buffer_max=25000,
        t_max=60
    )
    
    # Warmup and train
    learner.warmup_wunn(device='cpu',n_samples=500)
    print("Checking heuristic stats")
    learner._log_heuristic_stats()
    
    wunn, ffnn, memory_buffer, solved_history = learner.run_curriculum(
        puzzle_class=SlidingPuzzle,
        num_iter=5
    )
    
    # Prepare for evaluation
    targets = [t for _, t in memory_buffer]
    y_q = np.quantile(targets, q=0.99) if memory_buffer else 0
    print("Generate benchmark tasks...")
    benchmark_tasks = generate_benchmark_tasks(10)
    print("Starting Evaluation")
    
    trainer = FFNNTrainer(device=device)
    heuristic_fn = trainer.create_heuristic_fn(wunn=wunn, ffnn=ffnn, y_q=y_q)
    
    # Evaluate main configuration
    main_results = evaluate_heuristic(benchmark_tasks, heuristic_fn)
    print("Main results")
    print(f"{main_results['avg_time']:.1f}  {main_results['suboptimality']:.1f}%  {main_results['optimal']:.1f}%  {main_results['avg_nodes']:.1f} nodes")

        
    
    # Print results table
    print("\nSuboptimality Results for 15-puzzle:")
    print("α    Time    Nodes   Subopt  Optimal")
    print("------------------------------------------")
    
    # Evaluate different alpha values
    alpha_results = {}
    for alpha in [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]:
        results = evaluate_for_alpha(alpha, benchmark_tasks, ffnn, wunn, device, y_q)
        print(f"{alpha:<6} {results['avg_time']:.1f}  {results['avg_nodes']:.1f}    {results['suboptimality']:.1f}%  {results['optimal']:.1f}%")
        alpha_results[alpha] = results
    
    # Compare with single output FFNN if needed
    # single_output_results = evaluate_single_output(benchmark_tasks, device)
    # print(f"N/A    {single_output_results['avg_time']:.1f}  {single_output_results['avg_nodes']:,}  {single_output_results['suboptimality']:.1f}%  {single_output_results['optimal']:.1f}%")

def evaluate_for_alpha(alpha, tasks, ffnn, wunn, device, y_q):
    """Evaluate specific alpha configuration"""
    def heuristic_fn(state):
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, sigma_a, sigma_e = wunn.predict_with_uncertainty(x)
            
            # Apply sigmoid to ensure positive outputs
            mean = torch.sigmoid(mean) * 50  # Scale to reasonable puzzle range
            sigma_a = torch.sigmoid(sigma_a)
            sigma_e = torch.sigmoid(sigma_e)
            
            sigma_t = sigma_a + sigma_e
            z = norm.ppf(min(alpha, 0.95))  # Cap alpha for numerical stability
            
            # Use FFNN prediction when appropriate
            if y_q is not None and mean.item() < y_q:
                features = torch.cat([mean.unsqueeze(1), sigma_a.unsqueeze(1)], dim=1)
                h = ffnn(features).item()
            else:
                h = mean.item() + z * sigma_t.item()
            
            # Ensure reasonable bounds
            md = state.manhattan_distance()
            h = max(h, md * 0.9)  # Never less than 90% of MD
            h = min(h, md * 2.0)  # Never more than 2x MD
            
            return h
    
    return evaluate_heuristic(tasks, heuristic_fn)

def evaluate_single_output(tasks, device):
    """Baseline FFNN without uncertainty"""
    model = FFNNPlanner(input_dim=128, hidden_dim=20).to(device)  # Match input dim to state encoding
    trainer = FFNNTrainer(device)
    
    # Generate realistic training data
    dummy_puzzles = [SlidingPuzzle().shuffle(i) for i in range(10, 30) for _ in range(50)]
    dummy_data = [(p.to_one_hot(), p.manhattan_distance()) for p in dummy_puzzles]
    
    model = trainer.train(None, model, dummy_data)  # Bypass WUNN
    
    def heuristic_fn(state):
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return max(model(x).item(), state.manhattan_distance())  # Ensure admissibility
    
    return evaluate_heuristic(tasks, heuristic_fn)

if __name__ == "__main__":
    run_suboptimality_experiment()