"""Entrena y evalúa los modelos de demanda sobre la muestra del dashboard.

Replica la metodología del notebook (notebooks/ride_hailing_demand_pipeline.ipynb)
a escala de la muestra incluida en el repo (data/final_dataset.csv):

  1. Features de retardo por zona (lag_1h, lag_24h), sin leakage.
  2. Split temporal: el último día completo se reserva como test.
  3. Tres modelos (Linear Regression, Random Forest, Gradient Boosting),
     cada uno en dos escenarios: con y sin features de lag.
  4. Métricas MAE, RMSE y MAPE sobre el conjunto de test.

El resultado se guarda en data/model_metrics.csv, que es lo que el
dashboard (app.py) muestra en las pestañas de Modelos y Feature Engineering.

Uso:
    python src/train_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final_dataset.csv"
METRICS_PATH = ROOT / "data" / "model_metrics.csv"

TARGET = "demand"
BASE_FEATURES = ["hour", "day_of_week", "pickup_community_area", "temperature", "is_holiday"]
# Variables de tráfico (V3), presentes solo en el export completo del pipeline
TRAFFIC_FEATURES = ["n_crashes", "congestion_speed"]
LAG_FEATURES = ["lag_1h", "lag_24h"]

MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE y MAPE. El MAPE usa y+1 en el denominador (como en el
    notebook) para evitar divisiones por cero en horas-zona sin demanda."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade los lags de demanda calculados zona a zona (sin leakage)."""
    df = df.sort_values(["pickup_community_area", "date", "hour"]).reset_index(drop=True)
    grouped = df.groupby("pickup_community_area")["demand"]
    df["lag_1h"] = grouped.shift(1)
    df["lag_24h"] = grouped.shift(24)
    return df.sort_values(["date", "hour", "pickup_community_area"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal sin shuffle. Con suficientes días (>=21) reserva la
    última semana completa como test, como en el notebook; con la muestra
    corta reserva el último día."""
    days = sorted(df["date"].unique())
    n_test = 7 if len(days) >= 21 else 1
    test_days = days[-n_test:]
    df = df.dropna(subset=LAG_FEATURES)
    train = df[~df["date"].isin(test_days)]
    test = df[df["date"].isin(test_days)]
    return train, test


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = build_features(df)
    train, test = temporal_split(df)

    base = BASE_FEATURES + [c for c in TRAFFIC_FEATURES if c in df.columns]

    print(f"Train: {len(train):,} filas ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"Test:  {len(test):,} filas ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"Features base: {base}")

    rows = []
    for scenario, features in [("con_lags", base + LAG_FEATURES), ("sin_lags", base)]:
        for name, factory in MODELS.items():
            model = factory()
            model.fit(train[features], train[TARGET])
            pred = model.predict(test[features])
            metrics = evaluate(test[TARGET].to_numpy(), pred)
            rows.append({"Modelo": name, "Escenario": scenario, **metrics})
            print(f"{name:20s} [{scenario}]  MAE={metrics['MAE']:.3f}  "
                  f"RMSE={metrics['RMSE']:.3f}  MAPE={metrics['MAPE']:.1f}%")

    pd.DataFrame(rows).round(3).to_csv(METRICS_PATH, index=False)
    print(f"\nMétricas guardadas en {METRICS_PATH}")


if __name__ == "__main__":
    main()
