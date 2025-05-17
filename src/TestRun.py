# Example usage
import torch
import learned_a_star as LearnedAStar
import sliding_puzzle as SlidingPuzzle

# Load your trained model here, e.g.:
model = torch.load('heuristic_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

planner = LearnedAStar(model, device)
puzzle = SlidingPuzzle().shuffle(steps=20)
solution, cost = planner.search(puzzle)
print(f"Solution found in {cost} moves")
