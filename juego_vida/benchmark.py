import time
import matplotlib.pyplot as plt

from game_of_life import GameOfLife, NUMBA_OK


def medir(backend, size, repeticiones, calentar=False):

    game = GameOfLife(size, size, backend=backend)

    if calentar:
        game.step()

    start = time.perf_counter()

    for i in range(repeticiones):
        game.step()

    end = time.perf_counter()

    return (end - start) / repeticiones


def run_benchmark():

    test_sizes = [32, 64, 128, 256, 512, 1024]

    datos_python = []
    datos_numpy = []
    datos_numba = []

    print()
    print("Running benchmark...")
    print()

    for size in test_sizes:

        t_numpy = medir("numpy", size, 30)
        datos_numpy.append(t_numpy)

        t_numba = medir("numba", size, 30, calentar=True) if NUMBA_OK else None
        datos_numba.append(t_numba)

        if size <= 128:
            t_python = medir("python", size, 5)
        else:
            t_python = None
        datos_python.append(t_python)

        def fmt(x):
            return ("%.6f" % x) if x is not None else "   -   "

        print("Grid:", size, "x", size)
        print("  python:", fmt(t_python), "| numpy:", fmt(t_numpy), "| numba:", fmt(t_numba))
        print()


    cells = [size * size for size in test_sizes]

    memoria_mb = [ (size * size) / (1024 * 1024) for size in test_sizes ]

    n0 = cells[0]
    t0 = datos_numpy[0]
    teorica_n = [ t0 * (c / n0) for c in cells ]
    teorica_n2 = [ t0 * (c / n0) ** 2 for c in cells ]

    def limpiar(xs, ys):
        cx, cy = [], []
        for x, y in zip(xs, ys):
            if y is not None:
                cx.append(x)
                cy.append(y)
        return cx, cy

    plt.figure(figsize=(8, 5))

    cx, cy = limpiar(cells, datos_python)
    plt.plot(cx, cy, marker='o', label="Python puro")
    cx, cy = limpiar(cells, datos_numpy)
    plt.plot(cx, cy, marker='s', label="NumPy")
    if NUMBA_OK:
        cx, cy = limpiar(cells, datos_numba)
        plt.plot(cx, cy, marker='^', label="Numba paralelo")

    plt.xlabel("Number of Cells")
    plt.ylabel("Execution Time (s)")
    plt.title("Performance Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance.png", dpi=120)
    plt.show()

    plt.figure(figsize=(8, 5))

    cx, cy = limpiar(cells, datos_python)
    plt.loglog(cx, cy, marker='o', label="Python puro")
    cx, cy = limpiar(cells, datos_numpy)
    plt.loglog(cx, cy, marker='s', label="NumPy")
    if NUMBA_OK:
        cx, cy = limpiar(cells, datos_numba)
        plt.loglog(cx, cy, marker='^', label="Numba paralelo")

    plt.loglog(cells, teorica_n, '--', label="O(n) teorico")
    plt.loglog(cells, teorica_n2, ':', label="O(n^2) teorico")

    plt.xlabel("Cells (log)")
    plt.ylabel("Time (log)")
    plt.title("Log-Log Performance vs Complejidad Teorica")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("performance_loglog.png", dpi=120)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(cells, memoria_mb, marker='d', color='purple')
    plt.xlabel("Number of Cells")
    plt.ylabel("Memory (MB)")
    plt.title("Memory Scaling (tablero uint8)")
    plt.grid(True)
    plt.savefig("memory.png", dpi=120)
    plt.show()

    if NUMBA_OK:
        print("Comparacion paralela (Numba) vs secuencial (NumPy):")
        for size, tn, tb in zip(test_sizes, datos_numpy, datos_numba):
            if tn and tb:
                speed = tn / tb
                print("  %4dx%-4d  speedup numba/numpy: %.2fx" % (size, size, speed))
        print()

    print("Graficas guardadas: performance.png, performance_loglog.png, memory.png")
