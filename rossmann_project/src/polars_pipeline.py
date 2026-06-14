import time
import polars as pl
from preprocessing import load_polars, scan_polars
import feature_engineering as fe


def run_polars_pipeline(train_path, store_path, sample_frac=1.0):
    t = {}
    s = time.perf_counter()
    train, store = load_polars(train_path, store_path)
    if sample_frac < 1.0:
        train = train.sample(fraction=sample_frac, seed=42)
    t["read"] = time.perf_counter() - s

    s = time.perf_counter()
    df = fe.filter_polars(train)
    t["filter"] = time.perf_counter() - s

    s = time.perf_counter()
    df = fe.join_polars(df, store)
    t["join"] = time.perf_counter() - s

    s = time.perf_counter()
    df = fe.handle_missing_polars(df)
    df = fe.transform_polars(df)
    df = fe.build_new_features_polars(df)
    t["feature_engineering"] = time.perf_counter() - s

    s = time.perf_counter()
    df = fe.aggregate_polars(df)
    t["aggregation"] = time.perf_counter() - s

    t["total"] = sum(t.values())
    return df, t


def run_polars_lazy(train_path, store_path):
    train, store = scan_polars(train_path, store_path)
    plan = (
        train.filter((pl.col("Open") == 1) & (pl.col("Sales") > 0))
        .join(store, on="Store", how="left")
        .with_columns([
            pl.col("CompetitionDistance").fill_null(
                pl.col("CompetitionDistance").median()),
            pl.col("PromoInterval").fill_null("None"),
            pl.col("Date").dt.year().alias("Year"),
            pl.col("Date").dt.month().alias("Month"),
        ])
        .group_by("Store")
        .agg(pl.col("Sales").mean().alias("AvgStoreSales"))
    )
    return plan.collect()
