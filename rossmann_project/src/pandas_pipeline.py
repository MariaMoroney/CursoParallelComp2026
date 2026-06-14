import time
import numpy as np
import pandas as pd
from preprocessing import load_pandas

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
HOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}
TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORT_MAP = {"a": 0, "b": 1, "c": 2}


def run_pandas_pipeline(train_path, store_path, sample_frac=1.0):
    t = {}
    s = time.perf_counter()
    train, store = load_pandas(train_path, store_path)
    if sample_frac < 1.0:
        train = train.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    t["read"] = time.perf_counter() - s

    s = time.perf_counter()
    df = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    t["filter"] = time.perf_counter() - s

    s = time.perf_counter()
    df = df.merge(store, on="Store", how="left")
    t["join"] = time.perf_counter() - s

    s = time.perf_counter()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    for c in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
              "Promo2SinceWeek", "Promo2SinceYear"]:
        df[c] = df[c].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["StateHolidayEnc"] = df["StateHoliday"].astype(str).map(HOLIDAY_MAP).fillna(0).astype(int)
    df["StoreTypeEnc"] = df["StoreType"].map(TYPE_MAP).fillna(0).astype(int)
    df["AssortmentEnc"] = df["Assortment"].map(ASSORT_MAP).fillna(0).astype(int)

    df["MonthAbbr"] = df["Month"].map(lambda m: MONTH_ABBR[m - 1])
    df["CompetitionOpenMonths"] = (12 * (df["Year"] - df["CompetitionOpenSinceYear"])
                                   + (df["Month"] - df["CompetitionOpenSinceMonth"])).clip(lower=0)
    df["IsPromoMonth"] = [int(m in str(p).split(",")) for m, p in
                          zip(df["MonthAbbr"], df["PromoInterval"])]
    t["feature_engineering"] = time.perf_counter() - s

    s = time.perf_counter()
    agg = df.groupby("Store").agg(
        AvgStoreSales=("Sales", "mean"),
        AvgStoreCustomers=("Customers", "mean")).reset_index()
    df = df.merge(agg, on="Store", how="left")
    t["aggregation"] = time.perf_counter() - s

    t["total"] = sum(t.values())
    return df, t
