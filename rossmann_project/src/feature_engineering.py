import polars as pl

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
HOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}
TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORT_MAP = {"a": 0, "b": 1, "c": 2}


def filter_polars(train):
    return train.filter((pl.col("Open") == 1) & (pl.col("Sales") > 0))


def join_polars(train, store):
    return train.join(store, on="Store", how="left")


def handle_missing_polars(df):
    return df.with_columns([
        pl.col("CompetitionDistance").fill_null(pl.col("CompetitionDistance").median()),
        pl.col("CompetitionOpenSinceMonth").fill_null(0),
        pl.col("CompetitionOpenSinceYear").fill_null(0),
        pl.col("Promo2SinceWeek").fill_null(0),
        pl.col("Promo2SinceYear").fill_null(0),
        pl.col("PromoInterval").fill_null("None"),
    ])


def transform_polars(df):
    month_expr = pl.col("Date").dt.month()
    return df.with_columns([
        pl.col("Date").dt.year().alias("Year"),
        month_expr.alias("Month"),
        pl.col("Date").dt.day().alias("Day"),
        pl.col("Date").dt.week().alias("WeekOfYear"),
        pl.col("StateHoliday").cast(pl.Utf8).replace_strict(HOLIDAY_MAP, default=0).alias("StateHolidayEnc"),
        pl.col("StoreType").replace_strict(TYPE_MAP, default=0).alias("StoreTypeEnc"),
        pl.col("Assortment").replace_strict(ASSORT_MAP, default=0).alias("AssortmentEnc"),
    ])


def build_new_features_polars(df):
    abbr = pl.element().replace_strict(
        {i + 1: MONTH_ABBR[i] for i in range(12)}, default="None")
    df = df.with_columns(
        pl.col("Month").map_elements(
            lambda m: MONTH_ABBR[m - 1], return_dtype=pl.Utf8).alias("MonthAbbr"))
    df = df.with_columns([
        (12 * (pl.col("Year") - pl.col("CompetitionOpenSinceYear"))
         + (pl.col("Month") - pl.col("CompetitionOpenSinceMonth")))
        .clip(lower_bound=0).alias("CompetitionOpenMonths"),
        pl.col("PromoInterval").str.split(",").list.contains(pl.col("MonthAbbr"))
        .cast(pl.Int8).alias("IsPromoMonth"),
    ])
    return df


def aggregate_polars(df):
    agg = df.group_by("Store").agg([
        pl.col("Sales").mean().alias("AvgStoreSales"),
        pl.col("Customers").mean().alias("AvgStoreCustomers"),
    ])
    return df.join(agg, on="Store", how="left")


def feature_pipeline_polars(train, store):
    df = filter_polars(train)
    df = join_polars(df, store)
    df = handle_missing_polars(df)
    df = transform_polars(df)
    df = build_new_features_polars(df)
    df = aggregate_polars(df)
    return df
