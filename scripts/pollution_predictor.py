import pandas as pd
import numpy as np
import os
import joblib

def predict_pollution(df, column='PM2.5_ground', days=7):
    df = df.dropna(subset=[column])
    df = df.sort_values('date')

    y = df[column].values
    last_date = df['date'].max()

    model_path = f"models/{column}.pkl"

    # Case 1: Use trained ML model if available and enough data (7+)
    if os.path.exists(model_path) and len(y) >= 7:
        try:
            model = joblib.load(model_path)

            # Use last 7 values to create input
            recent_data = y[-7:].reshape(-1, 1)
            predictions = []

            for i in range(days):
                input_data = recent_data[-7:].reshape(1, -1)
                pred = model.predict(input_data)[0]
                predictions.append({
                    "date": last_date + pd.Timedelta(days=i + 1),
                    f"predicted_{column}": pred
                })
                # Add prediction to rolling window
                recent_data = np.append(recent_data, [[pred]], axis=0)

            return pd.DataFrame(predictions)

        except Exception as e:
            print(f"❌ Error using model `{model_path}`: {e}")
            # Fallback to smoothing if model fails

    # Case 2: Fallback to smoothing if model not found or not enough data
    if len(y) >= 3:
        predictions = []
        for i in range(days):
            avg = np.mean(y[-min(len(y), 7):])  # Use last available values (up to 7)
            predictions.append({
                "date": last_date + pd.Timedelta(days=i + 1),
                f"predicted_{column}": avg
            })
            y = np.append(y, avg)

        return pd.DataFrame(predictions)

    # Case 3: Not enough data
    raise ValueError(f"❌ Not enough data to predict `{column}`. Minimum 3 points required.")

# Pollution level classification (PM2.5_ground used here — adjust if needed)
def classify_pollution_level(value):
    if value <= 50:
        return "Good ✅"
    elif value <= 100:
        return "Moderate ⚠️"
    elif value <= 150:
        return "Unhealthy 😷"
    elif value <= 200:
        return "Very Unhealthy 🛑"
    else:
        return "Hazardous ☠️"
