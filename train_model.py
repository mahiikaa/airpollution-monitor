import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("./data/merged_pollution_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Rename columns to unify satellite and ground naming
rename_map = {
    'PM2.5_x': 'PM2.5_sat', 'PM2.5_y': 'PM2.5_ground',
    'PM10_x': 'PM10_sat', 'PM10_y': 'PM10_ground',
    'NO2_level_x': 'NO2_level_sat', 'NO2_level_y': 'NO2_level_ground',
    'O3_x': 'O3_sat', 'O3_y': 'O3_ground',
    'CO_x': 'CO_sat', 'CO_y': 'CO_ground',
    'VOCs_x': 'VOCs_sat', 'VOCs_y': 'VOCs_ground',
    'SO2_x': 'SO2_sat', 'SO2_y': 'SO2_ground'
}
df.rename(columns=rename_map, inplace=True)

# Combine all satellite and ground pollutant columns
target_columns = [col for col in df.columns if col.endswith('_sat') or col.endswith('_ground')]

# Create output folder
os.makedirs("models", exist_ok=True)

for col in target_columns:
    if col not in df.columns:
        print(f"❌ Skipping {col}: not found in dataset.")
        continue

    series = df[["date", col]].dropna().sort_values("date")

    if len(series) < 14:
        print(f"⚠️ Not enough data to train: {col}")
        continue

    # Prepare training data: past 7 days ➝ next day
    X, y = [], []
    values = series[col].values
    for i in range(7, len(values)):
        X.append(values[i-7:i])
        y.append(values[i])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{col}.pkl")
    print(f"✅ Saved model for {col}")
