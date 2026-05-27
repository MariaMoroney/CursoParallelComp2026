import time
import numpy as np
from numba import njit


# --- Python puro ---
def estimar_pi_python(N):
    dx = 1.0 / N
    total = 0.0
    for i in range(N):
        x = (i + 0.5) * dx
        total += 4.0 / (1.0 + x * x)
    return total * dx


# --- NumPy ---
def estimar_pi_numpy(N):
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    return np.sum(4.0 / (1.0 + x ** 2)) * dx


# --- Numba ---
@njit
def estimar_pi_numba(N):
    dx = 1.0 / N
    total = 0.0
    for i in range(N):
        x = (i + 0.5) * dx
        total += 4.0 / (1.0 + x * x)
    return total * dx


def main():
    N = 10_000_000

    # "calentamos" Numba: la primera llamada compila la función
    estimar_pi_numba(10)

    for nombre, funcion in [("Python", estimar_pi_python),
                            ("NumPy", estimar_pi_numpy),
                            ("Numba", estimar_pi_numba)]:
        inicio = time.perf_counter()
        resultado = funcion(N)
        fin = time.perf_counter()
        print(f"{nombre:8} -> pi ≈ {resultado:.10f}   tiempo: {fin - inicio:.4f} s")


if __name__ == "__main__":
    main()