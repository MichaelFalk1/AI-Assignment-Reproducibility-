import torch
import numpy as np
from practical_implementation import LikelyAdmissibleHeuristicLearner,ida_star_search
from sliding_puzzle import SlidingPuzzle
import time
import random


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



def run_efficiency_experiment():
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
        't_max': 1,  # Short time limit for efficiency experiment
        'num_iter': 20  # Fewer iterations for efficiency experiment
    }
    
    # Our approach (GTP)
    learner = LikelyAdmissibleHeuristicLearner(device,alpha=0.99,num_tasks_per_iter=10,epsilon=1,kappa=0.64,beta_0=0.05,gamma=0.9,memory_buffer_max=25000,t_max=60,)
    learner.bootstrap_training()
    _, _, _, _ = learner.run_curriculum(
        puzzle_class=SlidingPuzzle,
        num_iter=20
    )
    gtp_results = evaluate_efficiency(learner, device)
    
    # Compare with fixed step approaches
    fixed_step_results = {}
    for length_inc in [1, 2, 4, 6, 8, 10]:
        results = evaluate_fixed_step(length_inc, device)
        fixed_step_results[length_inc] = results
    
    # Print results table as in paper
    print("\nEfficiency Results for 15-puzzle:")
    print("Method      Solved Train  Solved Test")
    print("------------------------------------")
    for length_inc, results in fixed_step_results.items():
        print(f"Inc {length_inc:<3}     {results['solved_train']:.1f}%       {results['solved_test']:.1f}%")
    print(f"GTP         {gtp_results['solved_train']:.1f}%       {gtp_results['solved_test']:.1f}%")

def evaluate_efficiency(learner, device):
    """Evaluate GTP approach efficiency"""
    # Training performance
    solved_train = np.mean(learner.solved_history) * 10  # Convert to percentage
    
    # Test performance
    test_tasks = [SlidingPuzzle().shuffle(k+1) for k in range(100)]
    results = evaluate_hefficient_test(test_tasks, learner.ffnn, learner.wunn, device)
    
    return {
        'solved_train': solved_train,
        'solved_test': results['solved']
    }

def evaluate_fixed_step(length_inc, device):
    """Evaluate fixed-step curriculum"""
    class FixedStepLearner(LikelyAdmissibleHeuristicLearner):
        def generate_tasks(self):
            tasks = []
            for _ in range(self.num_tasks_per_iter):
                task = SlidingPuzzle()
                for _ in range(length_inc):
                    task = task.move(random.choice(task.get_legal_moves()))
                tasks.append(task)
            return tasks
    
    learner = FixedStepLearner(device, num_tasks_per_iter=10, t_max=1)
    _, _, _, solved_history = learner.run_curriculum(SlidingPuzzle, num_iter=20)
    
    test_tasks = [SlidingPuzzle().shuffle(k+1) for k in range(100)]
    results = evaluate_hefficient_test(test_tasks, learner.ffnn, learner.wunn, device)
    
    return {
        'solved_train': np.mean(solved_history) * 10,
        'solved_test': results['solved']
    }

def evaluate_hefficient_test(tasks, ffnn, wunn, device):
    """Efficient test evaluation"""
    def heuristic_fn(state):
        x = torch.tensor(state.to_one_hot(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return ffnn(x).item()
    
    return evaluate_heuristic(tasks, heuristic_fn)

if __name__ == "__main__":
    run_efficiency_experiment()