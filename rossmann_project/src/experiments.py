import os
import json
import time
import psutil
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polars_pipeline import run_polars_pipeline
from pandas_pipeline import run_pandas_pipeline

OVERRIDES = {"StateHoliday": pl.Utf8}


def scalability(train_path, store_path):
    fracs = [0.25, 0.5, 0.75, 1.0]
    rows = []
    for fr in fracs:
        _, tp = run_polars_pipeline(train_path, store_path, sample_frac=fr)
        _, tpd = run_pandas_pipeline(train_path, store_path, sample_frac=fr)
        rows.append({"fraction": fr, "polars_total": round(tp["total"], 4),
                     "pandas_total": round(tpd["total"], 4),
                     "speedup": round(tpd["total"] / tp["total"], 2)})
    return rows


def lazy_vs_eager(train_path):
    proc = psutil.Process(os.getpid())

    m0 = proc.memory_info().rss
    s = time.perf_counter()
    df_eager = pl.read_csv(train_path, schema_overrides=OVERRIDES).filter(
        (pl.col("Open") == 1) & (pl.col("Sales") > 0)).group_by("Store").agg(
        pl.col("Sales").mean())
    eager_time = time.perf_counter() - s
    eager_mem = (proc.memory_info().rss - m0) / 1e6

    m1 = proc.memory_info().rss
    s = time.perf_counter()
    df_lazy = pl.scan_csv(train_path, schema_overrides=OVERRIDES).filter(
        (pl.col("Open") == 1) & (pl.col("Sales") > 0)).group_by("Store").agg(
        pl.col("Sales").mean()).collect()
    lazy_time = time.perf_counter() - s
    lazy_mem = (proc.memory_info().rss - m1) / 1e6

    return {
        "eager_time_s": round(eager_time, 4), "lazy_time_s": round(lazy_time, 4),
        "eager_mem_delta_mb": round(eager_mem, 2), "lazy_mem_delta_mb": round(lazy_mem, 2),
    }


def plot_scalability(rows, out_path):
    fr = [r["fraction"] * 100 for r in rows]
    pol = [r["polars_total"] for r in rows]
    pan = [r["pandas_total"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(fr, pol, marker="o", label="Polars")
    plt.plot(fr, pan, marker="o", label="Pandas")
    plt.xlabel("Porcentaje del dataset (%)")
    plt.ylabel("Tiempo total del pipeline (s)")
    plt.title("Escalabilidad con tamano de datos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_speedup(rows, out_path):
    fr = [r["fraction"] * 100 for r in rows]
    sp = [r["speedup"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(fr, sp, marker="o", color="green")
    plt.xlabel("Porcentaje del dataset (%)")
    plt.ylabel("Speedup (Pandas / Polars)")
    plt.title("Speedup observado segun tamano")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def run(train_path, store_path, results_dir, figures_dir):
    scal = scalability(train_path, store_path)
    lazy = lazy_vs_eager(train_path)
    with open(os.path.join(results_dir, "scalability.json"), "w") as f:
        json.dump(scal, f, indent=2)
    with open(os.path.join(results_dir, "lazy_vs_eager.json"), "w") as f:
        json.dump(lazy, f, indent=2)
    plot_scalability(scal, os.path.join(figures_dir, "scalability.png"))
    plot_speedup(scal, os.path.join(figures_dir, "speedup.png"))
    return scal, lazy
