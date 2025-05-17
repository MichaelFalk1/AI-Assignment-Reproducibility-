import os
import torch
from sliding_puzzle import SlidingPuzzle
from a_star_data_generator import generate_dataset
from heuristic_learner import train_model, HeuristicNet
from learned_a_star import LearnedAStar

DATA_PATH = "dataset.pt"
MODEL_PATH = "heuristic_model.pth"
DEVICE = torch.device("cpu")

def main():
    # Step 1: Generate or load dataset
    if os.path.exists(DATA_PATH):
        print("Loading dataset...")
        dataset = torch.load(DATA_PATH)
    else:
        print("Generating dataset...")
        dataset = generate_dataset(n=1000)  # tweak number of samples
        torch.save(dataset, DATA_PATH)

    # Step 2: Train or load model
    if os.path.exists(MODEL_PATH):
        print("Loading trained model...")
        model = HeuristicNet()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(DEVICE)
    else:
        print("Training model...")
        model = train_model(dataset, epochs=20)
        torch.save(model.state_dict(), MODEL_PATH)
        model.to(DEVICE)

    # Step 3: Solve a shuffled puzzle using learned heuristic
    planner = LearnedAStar(model, DEVICE)
    puzzle = SlidingPuzzle()
    puzzle.shuffle(steps=20)
    print(f"Solving puzzle:\n{puzzle}")
    solution, cost = planner.search(puzzle)

    if solution is not None:
        print(f"Solution found in {cost} moves: {solution}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
