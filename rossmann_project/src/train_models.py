import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURES = ["DayOfWeek", "Promo", "SchoolHoliday", "StateHolidayEnc",
            "StoreTypeEnc", "AssortmentEnc", "CompetitionDistance", "Promo2",
            "Year", "Month", "Day", "WeekOfYear", "CompetitionOpenMonths",
            "IsPromoMonth", "AvgStoreSales", "AvgStoreCustomers"]


def prepare_xy(pdf):
    X = pdf[FEATURES].astype(float).values
    y = pdf["Sales"].astype(float).values
    return X, y


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def train_all(pdf):
    X, y = prepare_xy(pdf)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(
            n_estimators=80, max_depth=18, n_jobs=-1, random_state=42),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        s = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - s
        pred = model.predict(X_te)
        rmse, mae = evaluate(y_te, pred)
        results[name] = {"train_time": train_time, "rmse": rmse, "mae": mae}
    return results
