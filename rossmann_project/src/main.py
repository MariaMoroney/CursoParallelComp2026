import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

import eda
import benchmark
import experiments
from polars_pipeline import run_polars_pipeline
from train_models import train_all

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "data", "raw", "train.csv")
STORE = os.path.join(BASE, "data", "raw", "store.csv")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    print(">> Parte 1: EDA")
    eda.run(TRAIN, STORE, RESULTS, FIGURES)

    print(">> Parte 2-3: pipeline Polars + Machine Learning")
    df, _ = run_polars_pipeline(TRAIN, STORE)
    ml = train_all(df.to_pandas())
    with open(os.path.join(RESULTS, "ml_results.json"), "w") as f:
        json.dump(ml, f, indent=2)

    print(">> Benchmark Polars vs Pandas")
    info, table = benchmark.run(TRAIN, STORE, RESULTS, FIGURES)

    print(">> Experimentos: escalabilidad y lazy execution")
    scal, lazy = experiments.run(TRAIN, STORE, RESULTS, FIGURES)

    print(json.dumps({"system": info, "ml": ml, "lazy": lazy}, indent=2))
    print("Listo. Revisa results/ y figures/")


if __name__ == "__main__":
    main()
