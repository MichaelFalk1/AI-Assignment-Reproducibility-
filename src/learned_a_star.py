# learned_a_star.py
import heapq
import torch
from sliding_puzzle import SlidingPuzzle

class LearnedAStar:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.eval()

    def heuristic(self, puzzle):
        state = torch.tensor(puzzle.to_array(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h_value = self.model(state).item()
        return max(h_value, 0)  # Ensure heuristic is non-negative

    def search(self, start_puzzle):
        visited = set()
        frontier = []
        start_h = self.heuristic(start_puzzle)
        heapq.heappush(frontier, (start_h, 0, start_puzzle, []))
        visited.add(tuple(start_puzzle.to_array()))

        while frontier:
            est_total_cost, cost_so_far, current, path = heapq.heappop(frontier)
            if current.is_solved():
                return path, cost_so_far

            for move in current.get_legal_moves():
                next_state = current.move(move)
                state_key = tuple(next_state.to_array())
                if state_key not in visited:
                    new_cost = cost_so_far + 1
                    est_cost = new_cost + self.heuristic(next_state)
                    heapq.heappush(frontier, (est_cost, new_cost, next_state, path + [move]))
                    visited.add(state_key)

        return None, float('inf')
