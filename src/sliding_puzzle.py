import numpy as np
import random
import heapq

class SlidingPuzzle:
    def __init__(self, board=None):
        self.size = 4
        if board is None:
            self.board = self._create_solved_board()
        else:
            self.board = np.array(board).reshape((self.size, self.size))
        self.goal = self._create_solved_board()

    def _create_solved_board(self):
        return np.array(list(range(1, self.size**2)) + [0]).reshape((self.size, self.size))

    def is_solved(self):
        return np.array_equal(self.board, self.goal)

    def find_blank(self):
        pos = np.argwhere(self.board == 0)
        return tuple(pos[0])

    def get_legal_moves(self):
        x, y = self.find_blank()
        moves = []
        if x > 0: moves.append('up')
        if x < self.size - 1: moves.append('down')
        if y > 0: moves.append('left')
        if y < self.size - 1: moves.append('right')
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
        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i, j]
                if val == 0: continue
                gx, gy = (val-1) // self.size, (val-1) % self.size
                distance += abs(gx - i) + abs(gy - j)
        return distance

    def to_array(self):
        return self.board.flatten()

    def to_hashable(self):
        return tuple(self.to_array())

    def __str__(self):
        return str(self.board)
    
    def to_one_hot(self):
        one_hot = np.zeros(16 * 2 * 4, dtype=np.float32)  # 128-dimensional vector
        
        for number in range(16):  # All numbers from 0 to 15
            # Find current position of this number
            pos = np.where(self.board == number)
            row, col = pos[0][0], pos[1][0]
            
            # Encode row (4 bits one-hot)
            row_start = number * 8
            one_hot[row_start + row] = 1.0
            
            # Encode column (4 bits one-hot)
            col_start = number * 8 + 4
            one_hot[col_start + col] = 1.0
            
        return one_hot