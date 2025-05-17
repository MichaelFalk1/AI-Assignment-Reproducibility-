import numpy as np
import random

class SlidingPuzzle:
    def __init__(self, board=None):
        if board is None:
            self.board = self._create_solved_board()
        else:
            self.board = np.array(board).reshape((4, 4))
        self.goal = self._create_solved_board()

    def _create_solved_board(self):
        return np.array(list(range(1, 16)) + [0]).reshape((4, 4))

    def is_solved(self):
        return np.array_equal(self.board, self.goal)

    def find_blank(self):
        pos = np.argwhere(self.board == 0)
        return tuple(pos[0])

    def get_legal_moves(self):
        x, y = self.find_blank()
        moves = []
        if x > 0: moves.append('up')
        if x < 3: moves.append('down')
        if y > 0: moves.append('left')
        if y < 3: moves.append('right')
        return moves

    def move(self, direction):
        x, y = self.find_blank()
        new_board = self.board.copy()

        if direction == 'up': new_x, new_y = x - 1, y
        elif direction == 'down': new_x, new_y = x + 1, y
        elif direction == 'left': new_x, new_y = x, y - 1
        elif direction == 'right': new_x, new_y = x, y + 1
        else: raise ValueError("Invalid move direction.")

        new_board[x, y], new_board[new_x, new_y] = new_board[new_x, new_y], new_board[x, y]
        return SlidingPuzzle(new_board)

    def shuffle(self, steps=100):
        current = self
        for _ in range(steps):
            move = random.choice(current.get_legal_moves())
            current = current.move(move)
        return current

    def manhattan_distance(self):
        distance = 0
        for i in range(4):
            for j in range(4):
                val = self.board[i, j]
                if val == 0: continue
                goal_x, goal_y = (val - 1) // 4, (val - 1) % 4
                distance += abs(goal_x - i) + abs(goal_y - j)
        return distance

    def __str__(self):
        return str(self.board)

    def to_array(self):
        return self.board.flatten()

    def clone(self):
        return SlidingPuzzle(self.board.copy())
