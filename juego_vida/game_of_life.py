import random

import numpy as np

try:
    from numba import njit, prange
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False


if NUMBA_OK:

    @njit(parallel=True, cache=True)
    def _step_numba(board):

        rows, cols = board.shape
        nuevo = np.zeros((rows, cols), dtype=np.uint8)

        for i in prange(rows):
            for j in range(cols):

                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors += board[ni, nj]

                if board[i, j] == 1:
                    if neighbors == 2 or neighbors == 3:
                        nuevo[i, j] = 1
                else:
                    if neighbors == 3:
                        nuevo[i, j] = 1

        return nuevo


class GameOfLife:

    def __init__(self, rows, cols, initial_state=None, backend="python"):

        self.rows = rows
        self.cols = cols

        if backend == "numba" and not NUMBA_OK:
            print("[aviso] Numba no esta instalado, uso backend numpy.")
            backend = "numpy"
        self.backend = backend

        if initial_state is not None:

            if backend == "python":
                self.board = initial_state
            else:
                self.board = np.array(initial_state, dtype=np.uint8)

        else:

            if backend == "python":
                self.board = [
                    [random.choice([0, 1]) for col in range(cols)]
                    for row in range(rows)
                ]
            else:
                rng = np.random.default_rng()
                self.board = (rng.random((rows, cols)) < 0.25).astype(np.uint8)


    def get_state(self):

        return self.board


    def step(self):

        if self.backend == "numpy":
            self._step_numpy()
        elif self.backend == "numba":
            self.board = _step_numba(self.board)
        else:
            self._step_python()

    def _step_python(self):

        next_board = [
            [0 for col in range(self.cols)]
            for row in range(self.rows)
        ]

        for row in range(self.rows):

            for col in range(self.cols):

                alive = self.board[row][col]

                neighbors = 0

                for dx in [-1, 0, 1]:

                    for dy in [-1, 0, 1]:

                        if dx == 0 and dy == 0:
                            continue

                        x = row + dx
                        y = col + dy

                        if 0 <= x < self.rows and 0 <= y < self.cols:

                            neighbors += self.board[x][y]


                if alive == 1:

                    if neighbors == 2 or neighbors == 3:

                        next_board[row][col] = 1

                else:

                    if neighbors == 3:

                        next_board[row][col] = 1

        self.board = next_board

    def _step_numpy(self):

        b = self.board
        p = np.pad(b, 1)

        neighbors = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] +                p[1:-1, 2:] +
            p[2:, :-2]  + p[2:, 1:-1]  + p[2:, 2:]
        )

        alive = b == 1
        sobrevive = alive & ((neighbors == 2) | (neighbors == 3))
        nace = (~alive) & (neighbors == 3)

        self.board = (sobrevive | nace).astype(np.uint8)


    def run(self, steps):

        def copia(b):
            if self.backend == "python":
                return [fila[:] for fila in b]
            else:
                return b.copy()

        history = [copia(self.board)]

        for i in range(steps):

            self.step()

            history.append(copia(self.board))

        return history
