import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n_stores = 1115
n_days = 300
dates = pd.date_range("2013-01-01", periods=n_days, freq="D")

rows = []
for s in range(1, n_stores + 1):
    promo = rng.integers(0, 2, n_days)
    openf = rng.integers(0, 2, n_days, endpoint=False)
    openf = np.where(rng.random(n_days) < 0.85, 1, 0)
    base = rng.normal(6000, 1500)
    sales = np.clip(base + 2000 * promo + rng.normal(0, 800, n_days), 0, None) * openf
    customers = np.clip(sales / rng.normal(9.5, 1.0) + rng.normal(0, 50, n_days), 0, None)
    sh = rng.choice(["0", "a", "b", "c"], n_days, p=[0.93, 0.03, 0.02, 0.02])
    for i, d in enumerate(dates):
        rows.append((s, d.dayofweek + 1, d.strftime("%Y-%m-%d"),
                     round(float(sales[i])), int(customers[i]), int(openf[i]),
                     int(promo[i]), sh[i], int(rng.integers(0, 2))))

train = pd.DataFrame(rows, columns=["Store", "DayOfWeek", "Date", "Sales", "Customers",
                                    "Open", "Promo", "StateHoliday", "SchoolHoliday"])

store_types = rng.choice(["a", "b", "c", "d"], n_stores)
assort = rng.choice(["a", "b", "c"], n_stores)
comp_dist = rng.normal(5000, 3000, n_stores)
comp_dist[rng.random(n_stores) < 0.03] = np.nan
comp_month = rng.integers(1, 13, n_stores).astype(float)
comp_year = rng.integers(2000, 2014, n_stores).astype(float)
miss = rng.random(n_stores) < 0.3
comp_month[miss] = np.nan
comp_year[miss] = np.nan
promo2 = rng.integers(0, 2, n_stores)
p2week = rng.integers(1, 52, n_stores).astype(float)
p2year = rng.integers(2009, 2014, n_stores).astype(float)
p2week[promo2 == 0] = np.nan
p2year[promo2 == 0] = np.nan
intervals = np.array(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"])
pinterval = np.where(promo2 == 1, rng.choice(intervals, n_stores), None)

store = pd.DataFrame({
    "Store": np.arange(1, n_stores + 1),
    "StoreType": store_types, "Assortment": assort,
    "CompetitionDistance": np.round(comp_dist, 1),
    "CompetitionOpenSinceMonth": comp_month, "CompetitionOpenSinceYear": comp_year,
    "Promo2": promo2, "Promo2SinceWeek": p2week, "Promo2SinceYear": p2year,
    "PromoInterval": pinterval,
})

train.to_csv("data/raw/train.csv", index=False)
store.to_csv("data/raw/store.csv", index=False)
print("train:", train.shape, "store:", store.shape)
