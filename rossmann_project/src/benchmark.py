import os
import json
import psutil
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polars_pipeline import run_polars_pipeline
from pandas_pipeline import run_pandas_pipeline


def system_info(train_path):
    return {
        "logical_cores": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "dataset_mb": round(os.path.getsize(train_path) / 1e6, 2),
        "n_rows": pl.read_csv(train_path, schema_overrides={"StateHoliday": pl.Utf8}).height,
    }


def compare_stages(train_path, store_path):
    _, tp = run_polars_pipeline(train_path, store_path)
    _, tpd = run_pandas_pipeline(train_path, store_path)
    stages = ["read", "filter", "join", "feature_engineering", "aggregation", "total"]
    table = []
    for st in stages:
        po, pa = tp[st], tpd[st]
        table.append({"stage": st, "polars_s": round(po, 4),
                      "pandas_s": round(pa, 4),
                      "speedup": round(pa / po, 2) if po > 0 else None})
    return table


def plot_stage_times(table, out_path):
    stages = [r["stage"] for r in table if r["stage"] != "total"]
    pol = [r["polars_s"] for r in table if r["stage"] != "total"]
    pan = [r["pandas_s"] for r in table if r["stage"] != "total"]
    x = range(len(stages))
    plt.figure(figsize=(9, 5))
    plt.bar([i - 0.2 for i in x], pol, width=0.4, label="Polars")
    plt.bar([i + 0.2 for i in x], pan, width=0.4, label="Pandas")
    plt.xticks(list(x), stages, rotation=30, ha="right")
    plt.ylabel("Tiempo (s)")
    plt.title("Tiempo por etapa: Polars vs Pandas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def run(train_path, store_path, results_dir, figures_dir):
    info = system_info(train_path)
    table = compare_stages(train_path, store_path)
    with open(os.path.join(results_dir, "system_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(results_dir, "benchmark_stages.json"), "w") as f:
        json.dump(table, f, indent=2)
    plot_stage_times(table, os.path.join(figures_dir, "stage_times.png"))
    return info, table
