import os
import json
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing import load_polars


def describe(train, store):
    return {
        "train_describe": train.describe().to_dicts(),
        "train_nulls": {c: int(train[c].null_count()) for c in train.columns},
        "store_nulls": {c: int(store[c].null_count()) for c in store.columns},
        "target_summary": {
            "mean": float(train["Sales"].mean()),
            "median": float(train["Sales"].median()),
            "min": float(train["Sales"].min()),
            "max": float(train["Sales"].max()),
            "std": float(train["Sales"].std()),
        },
    }


def plot_target(train, out_path):
    pos = train.filter(pl.col("Sales") > 0)["Sales"].to_list()
    plt.figure(figsize=(8, 5))
    plt.hist(pos, bins=60, color="steelblue", edgecolor="white")
    plt.xlabel("Sales")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion de la variable objetivo (Sales > 0)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_correlation(train, out_path):
    num = train.select(["Sales", "Customers", "DayOfWeek", "Promo", "SchoolHoliday"])
    corr = num.to_pandas().corr()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Matriz de correlacion")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def run(train_path, store_path, results_dir, figures_dir):
    train, store = load_polars(train_path, store_path)
    summary = describe(train, store)
    with open(os.path.join(results_dir, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    plot_target(train, os.path.join(figures_dir, "target_distribution.png"))
    plot_correlation(train, os.path.join(figures_dir, "correlation.png"))
    return summary
