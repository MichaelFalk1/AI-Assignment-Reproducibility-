# a_star_data_generator.py
import heapq
from sliding_puzzle import SlidingPuzzle

def a_star(start_puzzle):
    visited = set()
    frontier = []
    heapq.heappush(frontier, (start_puzzle.manhattan_distance(), 0, start_puzzle, []))
    visited.add(tuple(start_puzzle.to_array()))
    
    while frontier:
        est_total_cost, cost_so_far, current, path = heapq.heappop(frontier)
        if current.is_solved():
            return path, cost_so_far  # Return solution path and optimal cost

        for move in current.get_legal_moves():
            next_state = current.move(move)
            state_key = tuple(next_state.to_array())
            if state_key not in visited:
                new_cost = cost_so_far + 1
                est_cost = new_cost + next_state.manhattan_distance()
                heapq.heappush(frontier, (est_cost, new_cost, next_state, path + [move]))
                visited.add(state_key)

    return None, float('inf')  # No solution found

def generate_dataset(n=100):
    dataset = []
    for i in range(n):
        puzzle = SlidingPuzzle().shuffle(steps=20)
        _, cost = a_star(puzzle)
        dataset.append((puzzle.to_array(), cost))
        print(f"Sample {i+1}/{n}: Cost-to-go = {cost}")
    return dataset
