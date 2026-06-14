import polars as pl
import pandas as pd


def load_polars(train_path, store_path):
    train = pl.read_csv(train_path, try_parse_dates=True,
                        schema_overrides={"StateHoliday": pl.Utf8})
    store = pl.read_csv(store_path)
    return train, store


def scan_polars(train_path, store_path):
    train = pl.scan_csv(train_path, try_parse_dates=True,
                        schema_overrides={"StateHoliday": pl.Utf8})
    store = pl.scan_csv(store_path)
    return train, store


def load_pandas(train_path, store_path):
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False,
                        dtype={"StateHoliday": str})
    store = pd.read_csv(store_path)
    return train, store