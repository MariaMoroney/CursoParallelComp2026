import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUTA = "housing_csv.csv"

df = pd.read_csv(RUTA)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

y_raw = df["median_house_value"].values.astype(np.float64)
X_raw = df.drop(columns=["median_house_value"]).values.astype(np.float64)

X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y = (y_raw - y_raw.mean()) / y_raw.std()
X = np.hstack([np.ones((X.shape[0], 1)), X]).astype(np.float64)

n, d = X.shape
print(f"Filas: {n}   |   Caracteristicas (con bias): {d}\n")

ITERS = 1000
LR = 0.1

w_exacta = np.linalg.solve(X.T @ X, X.T @ y)


def r2(w):
    pred = X @ w
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print(f"R2 con ecuacion normal (referencia): {r2(w_exacta):.4f}\n")


def gd_numpy(X, y, lr, iters):
    w = np.zeros(X.shape[1])
    m = len(y)
    for _ in range(iters):
        grad = (X.T @ (X @ w - y)) / m
        w -= lr * grad
    return w


from numba import njit


@njit(cache=True, fastmath=True)
def gd_numba(X, y, lr, iters):
    w = np.zeros(X.shape[1])
    m = len(y)
    for _ in range(iters):
        grad = (X.T @ (X @ w - y)) / m
        w -= lr * grad
    return w


import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(3,))
def gd_jax(X, y, lr, iters):
    def paso(w, _):
        grad = (X.T @ (X @ w - y)) / X.shape[0]
        return w - lr * grad, None
    w0 = jnp.zeros(X.shape[1])
    w, _ = jax.lax.scan(paso, w0, None, length=iters)
    return w


def cronometrar(fn, *args, warmup=False):
    if warmup:
        r = fn(*args)
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    r = fn(*args)
    if hasattr(r, "block_until_ready"):
        r.block_until_ready()
    return (time.perf_counter() - t0), np.asarray(r)


Xj, yj = jnp.asarray(X), jnp.asarray(y)

t_np, w_np = cronometrar(gd_numpy, X, y, LR, ITERS)
t_nb_c, _ = cronometrar(gd_numba, X, y, LR, ITERS)
t_nb, w_nb = cronometrar(gd_numba, X, y, LR, ITERS, warmup=True)
t_jx_c, _ = cronometrar(gd_jax, Xj, yj, LR, ITERS)
t_jx, w_jx = cronometrar(gd_jax, Xj, yj, LR, ITERS, warmup=True)

print("=== VERSION VECTORIZADA (algebra matricial / BLAS) ===")
print(f"NumPy           {t_np*1000:9.1f} ms   R2={r2(w_np):.4f}")
print(f"Numba (1ra)     {t_nb_c*1000:9.1f} ms")
print(f"Numba           {t_nb*1000:9.1f} ms   R2={r2(w_nb):.4f}")
print(f"JAX   (1ra)     {t_jx_c*1000:9.1f} ms")
print(f"JAX             {t_jx*1000:9.1f} ms   R2={r2(w_jx):.4f}\n")


ITERS_B = 200


def gd_python(X, y, lr, iters):
    n, d = X.shape
    w = [0.0] * d
    for _ in range(iters):
        grad = [0.0] * d
        for i in range(n):
            pred = 0.0
            for j in range(d):
                pred += X[i, j] * w[j]
            err = pred - y[i]
            for j in range(d):
                grad[j] += err * X[i, j]
        for j in range(d):
            w[j] -= lr * grad[j] / n
    return np.array(w)


@njit(cache=True, fastmath=True)
def gd_numba_bucle(X, y, lr, iters):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        grad = np.zeros(d)
        for i in range(n):
            pred = 0.0
            for j in range(d):
                pred += X[i, j] * w[j]
            err = pred - y[i]
            for j in range(d):
                grad[j] += err * X[i, j]
        w -= lr * grad / n
    return w


t_py, w_py = cronometrar(gd_python, X, y, LR, ITERS_B)
t_nbb, w_nbb = cronometrar(gd_numba_bucle, X, y, LR, ITERS_B, warmup=True)

print("=== VERSION CON BUCLES (CPU normal vs CPU acelerado) ===")
print(f"Python puro     {t_py*1000:9.1f} ms")
print(f"Numba           {t_nbb*1000:9.1f} ms   ->  x{t_py/t_nbb:.0f} mas rapido")
print(f"Mismos pesos?   {np.allclose(w_py, w_nbb, atol=1e-6)}\n")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.bar(["NumPy", "Numba", "JAX"],
        [t_np*1000, t_nb*1000, t_jx*1000],
        color=["#4C72B0", "#DD8452", "#55A868"])
ax1.set_title("Version vectorizada (ya compilada)")
ax1.set_ylabel("Tiempo (ms)")
ax2.bar(["Python puro", "Numba"],
        [t_py*1000, t_nbb*1000],
        color=["#C44E52", "#8172B3"])
ax2.set_title("Version con bucles")
ax2.set_ylabel("Tiempo (ms)")
ax2.set_yscale("log")
plt.tight_layout()
plt.savefig("comparacion_tiempos.png", dpi=120)
print("Grafico guardado en comparacion_tiempos.png\n")


def medir(fn, *args, reps=7):
    r = fn(*args)
    if hasattr(r, "block_until_ready"):
        r.block_until_ready()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
        tiempos.append(time.perf_counter() - t0)
    return np.mean(tiempos) * 1000, np.std(tiempos) * 1000


print("=== TIEMPO PROMEDIO (7 repeticiones, version vectorizada) ===")
metodos = [("NumPy", gd_numpy, (X, y, LR, ITERS)),
           ("Numba", gd_numba, (X, y, LR, ITERS)),
           ("JAX",   gd_jax,   (Xj, yj, LR, ITERS))]
for nombre, fn, args in metodos:
    media, desv = medir(fn, *args)
    print(f"{nombre:6} {media:8.2f} +/- {desv:5.2f} ms")
print()

lista_iters = [200, 500, 1000, 2000, 4000]
tiempos_esc = {}
print("=== ESCALADO: tiempo (ms) vs numero de iteraciones ===")
print(f"{'iters':>6} | " + " | ".join(f"{n:>8}" for n, _, _ in metodos))
for nombre, fn, base_args in [("NumPy", gd_numpy, (X, y)),
                              ("Numba", gd_numba, (X, y)),
                              ("JAX",   gd_jax,   (Xj, yj))]:
    tiempos_esc[nombre] = [medir(fn, *base_args, LR, it, reps=5)[0]
                           for it in lista_iters]
for k, it in enumerate(lista_iters):
    print(f"{it:>6} | " + " | ".join(f"{tiempos_esc[n][k]:8.1f}"
                                     for n, _, _ in metodos))

plt.figure(figsize=(7, 4.5))
colores = {"NumPy": "#4C72B0", "Numba": "#DD8452", "JAX": "#55A868"}
for nombre in tiempos_esc:
    plt.plot(lista_iters, tiempos_esc[nombre], "o-",
             label=nombre, color=colores[nombre])
plt.xlabel("Numero de iteraciones")
plt.ylabel("Tiempo (ms)")
plt.title("Escalado del tiempo segun las iteraciones")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("escalado_tiempo.png", dpi=120)
print("\nGrafico guardado en escalado_tiempo.png")
plt.show()