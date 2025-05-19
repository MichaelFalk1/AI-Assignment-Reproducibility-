import torch
from scipy.stats import norm
import numpy as np
from practical_implementation import LikelyAdmissibleHeuristicLearner, ida_star_search
from sliding_puzzle import SlidingPuzzle
import time
from model import WUNN,FFNNPlanner,FFNNTrainer

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
        'avg_nodes': 0,
        'avg_time': 0
    }
    
    for task in tasks:
        start_time = time.time()
        path, cost = ida_star_search(task, heuristic_fn)
        elapsed = time.time() - start_time
        print(f"heursitc {heuristic_fn(task)} compared to manhatten {task.manhattan_distance()}")
        
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
    learner = LikelyAdmissibleHeuristicLearner(device,alpha=0.99,num_tasks_per_iter=10,epsilon=1,kappa=0.64,beta_0=0.05,gamma=0.9,memory_buffer_max=25000,t_max=60)
    learner.warmup_wunn(device='cpu',n_samples=500)
    print("Checking heuristc stats")
    learner._log_heuristic_stats()
    wunn, ffnn, memory_buffer, solved_history = learner.run_curriculum(
        puzzle_class=SlidingPuzzle,
        num_iter=5
    )
    
    targets = [t for _, t in memory_buffer]
    y_q = np.quantile(targets, q=0.99)
    print("Genrate Bench mark task...")
    # Evaluate on benchmark tasks
    benchmark_tasks = generate_benchmark_tasks(10)
    print("Starting Evaluation")
    trainer = FFNNTrainer(device=device)  # or 'cpu' if unspecified
    heuristic_fn = trainer.create_heuristic_fn(wunn=wunn, ffnn=ffnn, epsilon=1, y_q=y_q)
    results = evaluate_heuristic(benchmark_tasks, heuristic_fn)

    
    # Print results table as in paper
    print("\nSuboptimality Results for 15-puzzle:")
    print("α      Time     Generated  Subopt  Optimal")
    print("------------------------------------------")
    for alpha in [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]:
        results = evaluate_for_alpha(alpha, benchmark_tasks, ffnn, wunn, device)
        print(f"{alpha:<6} {results['avg_time']:.1f}  {results['avg_nodes']:,}  {results['suboptimality']:.1f}%  {results['optimal']:.1f}%")
    
    # Compare with single output FFNN
    single_output_results = evaluate_single_output(benchmark_tasks, device)
    print(f"N/A    {single_output_results['avg_time']:.1f}  {single_output_results['avg_nodes']:,}  {single_output_results['suboptimality']:.1f}%  {single_output_results['optimal']:.1f}%")

def evaluate_for_alpha(alpha, tasks, ffnn, wunn, device):
    """Evaluate specific alpha configuration"""
    def heuristic_fn(state):
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, sigma_a, sigma_e = wunn.predict_with_uncertainty(x)
            sigma_t = sigma_a + sigma_e
            z = norm.ppf(alpha)
            return max((mean + z * sigma_t).item(), 0)
    
    return evaluate_heuristic(tasks, heuristic_fn)

def evaluate_single_output(tasks, device):
    """Baseline FFNN without uncertainty"""
    # Train single-output model
    model = FFNNPlanner(input_dim=2, hidden_dim=20).to(device)
    trainer = FFNNTrainer(device)
    
    # Dummy training data (would need proper curriculum)
    dummy_data = [(np.random.randn(2), 10) for _ in range(1000)]
    model = trainer.train(None, model, dummy_data)  # Bypass WUNN
    
    def heuristic_fn(state):
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(x).item()
    
    return evaluate_heuristic(tasks, heuristic_fn)

if __name__ == "__main__":
    run_suboptimality_experiment()