

def tablero_vacio(size):

    return [ [0 for c in range(size)] for r in range(size) ]


def colocar(board, celdas, fila, col):

    for (f, c) in celdas:

        board[fila + f][col + c] = 1

    return board


def create_glider(size=32):

    board = tablero_vacio(size)

    celdas = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    return colocar(board, celdas, 1, 1)


def create_blinker(size=16):

    board = tablero_vacio(size)

    centro = size // 2

    celdas = [(0, 0), (0, 1), (0, 2)]

    return colocar(board, celdas, centro, centro - 1)


def create_toad(size=16):

    board = tablero_vacio(size)

    centro = size // 2

    celdas = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

    return colocar(board, celdas, centro, centro - 2)


def create_block(size=8):

    board = tablero_vacio(size)

    centro = size // 2

    celdas = [(0, 0), (0, 1), (1, 0), (1, 1)]

    return colocar(board, celdas, centro, centro)
