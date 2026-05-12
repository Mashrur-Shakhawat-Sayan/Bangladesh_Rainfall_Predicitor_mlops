import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib
import os

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/Temp_and_rain.csv")

df["ds"] = pd.to_datetime(
    df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
)

prophet_df = df[["ds", "rain"]].rename(columns={"rain": "y"})
prophet_df = prophet_df.sort_values("ds")

train_size = int(len(prophet_df) * 0.8)
train = prophet_df.iloc[:train_size]
test = prophet_df.iloc[train_size:]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Bangladesh Climate Forecasting with Prophet")

with mlflow.start_run():
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(train)

    future = test[["ds"]]
    forecast = model.predict(future)

    y_true = test["y"].values
    y_pred = forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mlflow.log_param("model_name", "Prophet")
    mlflow.log_param("yearly_seasonality", True)
    mlflow.log_param("weekly_seasonality", False)
    mlflow.log_param("daily_seasonality", False)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    joblib.dump(model, "models/prophet_rainfall_model.pkl")

    print("Prophet model trained successfully")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")