from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("models/prophet_rainfall_model.pkl")

climate_df = pd.read_csv("data/sorted_temp_and_rain_dataset.csv")
climate_df.columns = climate_df.columns.str.strip().str.lower()

climate_df = climate_df.rename(columns={
    "tem": "temperature",
    "rain": "rainfall"
})

climate_df["year"] = climate_df["year"].astype(int)
climate_df["month"] = climate_df["month"].astype(int)
climate_df["rainfall"] = pd.to_numeric(climate_df["rainfall"], errors="coerce")
climate_df["temperature"] = pd.to_numeric(climate_df["temperature"], errors="coerce")
climate_df = climate_df.dropna()

DATA_START_YEAR = 1901
DATA_END_YEAR = int(climate_df["year"].max())


def predict_temperature_for_years(same_month_data, years):
    x = same_month_data["year"].values
    y = same_month_data["temperature"].values

    if len(x) < 2:
        avg_temp = float(np.mean(y)) if len(y) > 0 else 0
        return [avg_temp for _ in years]

    slope, intercept = np.polyfit(x, y, 1)
    predictions = [(slope * year) + intercept for year in years]
    return predictions


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    month = int(request.form["month"])
    year = int(request.form["year"])
    interval = int(request.form.get("interval", 10))

    if interval not in [5, 10]:
        interval = 10

    month_name = pd.to_datetime(f"2026-{month}-01").strftime("%B")
    prediction_date = pd.to_datetime(f"{year}-{month}-01")

    forecast = model.predict(pd.DataFrame({
        "ds": [prediction_date]
    }))

    prediction = max(float(forecast["yhat"].values[0]), 0)
    yhat_lower = max(float(forecast["yhat_lower"].values[0]), 0)
    yhat_upper = max(float(forecast["yhat_upper"].values[0]), 0)

    uncertainty_range = max(yhat_upper - yhat_lower, 1)
    confidence = 100 - min((uncertainty_range / max(prediction, 1)) * 25, 70)
    confidence = round(max(confidence, 30), 2)

    same_month_data = climate_df[
        (climate_df["month"] == month) &
        (climate_df["year"] >= DATA_START_YEAR)
    ].copy()

    same_month_data = same_month_data.sort_values("year")

    historical_rain_once = (
        same_month_data["rainfall"].gt(1).mean() * 100
        if len(same_month_data) > 0
        else 0
    )

    model_rain_once = min(max((prediction / 120) * 100, 0), 100)
    rain_chance = round((0.6 * model_rain_once) + (0.4 * historical_rain_once), 2)
    rain_chance = min(max(rain_chance, 0), 100)
    no_rain_chance = round(100 - rain_chance, 2)

    monthly_rain_values = same_month_data["rainfall"]
    low_rain_cutoff = monthly_rain_values.quantile(0.35)
    high_rain_cutoff = monthly_rain_values.quantile(0.70)

    if prediction >= high_rain_cutoff:
        weather_mode = "rainy"
    elif prediction <= low_rain_cutoff:
        weather_mode = "sunny"
    else:
        weather_mode = "normal"

    historical_same_month_temp = same_month_data["temperature"].mean()

    if historical_same_month_temp >= 28:
        temperature_mode = "hot"
    elif historical_same_month_temp <= 20:
        temperature_mode = "cool"
    else:
        temperature_mode = "mild"

    final_graph_year = max(year, DATA_END_YEAR)

    period_labels = []
    historical_avg_rainfall = []
    forecast_avg_rainfall = []
    historical_avg_temperature = []
    forecast_avg_temperature = []

    current_start = DATA_START_YEAR

    while current_start <= final_graph_year:
        current_end = min(current_start + interval - 1, final_graph_year)
        period_labels.append(f"{current_start}-{current_end}")

        historical_period_data = same_month_data[
            (same_month_data["year"] >= current_start) &
            (same_month_data["year"] <= current_end) &
            (same_month_data["year"] <= DATA_END_YEAR)
        ]

        if len(historical_period_data) > 0:
            historical_avg_rainfall.append(
                round(historical_period_data["rainfall"].mean(), 2)
            )
            historical_avg_temperature.append(
                round(historical_period_data["temperature"].mean(), 2)
            )
        else:
            historical_avg_rainfall.append(None)
            historical_avg_temperature.append(None)

        future_start = max(current_start, DATA_END_YEAR + 1)
        future_end = current_end

        if future_start <= future_end:
            future_years = list(range(future_start, future_end + 1))

            future_dates = pd.to_datetime([
                f"{future_year}-{month}-01"
                for future_year in future_years
            ])

            future_forecast = model.predict(pd.DataFrame({
                "ds": future_dates
            }))

            future_rainfall_predictions = future_forecast["yhat"].clip(lower=0)

            forecast_avg_rainfall.append(
                round(float(future_rainfall_predictions.mean()), 2)
            )

            future_temp_predictions = predict_temperature_for_years(
                same_month_data,
                future_years
            )

            forecast_avg_temperature.append(
                round(float(np.mean(future_temp_predictions)), 2)
            )
        else:
            forecast_avg_rainfall.append(None)
            forecast_avg_temperature.append(None)

        current_start += interval

    first_temp = same_month_data[
        same_month_data["year"] <= DATA_START_YEAR + 20
    ]["temperature"].mean()

    recent_temp = same_month_data[
        same_month_data["year"] >= DATA_END_YEAR - 20
    ]["temperature"].mean()

    temp_change = round(recent_temp - first_temp, 2)

    first_rain = same_month_data[
        same_month_data["year"] <= DATA_START_YEAR + 20
    ]["rainfall"].mean()

    recent_rain = same_month_data[
        same_month_data["year"] >= DATA_END_YEAR - 20
    ]["rainfall"].mean()

    rain_change = round(recent_rain - first_rain, 2)

    if temp_change > 0.2:
        temp_summary = f"For {month_name}, the temperature trend appears to be rising by about {temp_change}°C when comparing the early historical period with the recent period."
    elif temp_change < -0.2:
        temp_summary = f"For {month_name}, the temperature trend appears to be falling by about {abs(temp_change)}°C when comparing the early historical period with the recent period."
    else:
        temp_summary = f"For {month_name}, the temperature trend looks mostly stable in the available historical data."

    if rain_change > 10:
        rain_summary = f"Rainfall for {month_name} appears to be increasing by about {rain_change} mm compared with the early historical period."
    elif rain_change < -10:
        rain_summary = f"Rainfall for {month_name} appears to be reducing by about {abs(rain_change)} mm compared with the early historical period."
    else:
        rain_summary = f"Rainfall for {month_name} does not show a strong increase or decrease in this comparison."

    climate_note = (
        "This pattern may be linked with long-term climate variability, but this app alone cannot prove climate change. "
        "It only summarizes the trend found in the dataset and model forecast."
    )

    return render_template(
        "index.html",
        prediction_text=f"Expected Rainfall for {month_name} {year}: {prediction:.2f} mm",
        prediction_value=round(prediction, 2),
        yhat_lower=round(yhat_lower, 2),
        yhat_upper=round(yhat_upper, 2),
        confidence=confidence,
        rain_chance=rain_chance,
        no_rain_chance=no_rain_chance,
        weather_mode=weather_mode,
        temperature_mode=temperature_mode,
        month=month,
        month_name=month_name,
        year=year,
        interval=interval,
        historical_avg_rainfall=historical_avg_rainfall,
        forecast_avg_rainfall=forecast_avg_rainfall,
        historical_avg_temperature=historical_avg_temperature,
        forecast_avg_temperature=forecast_avg_temperature,
        period_labels=period_labels,
        temp_summary=temp_summary,
        rain_summary=rain_summary,
        climate_note=climate_note,
        data_end_year=DATA_END_YEAR
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)