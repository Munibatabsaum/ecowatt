import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/energy.csv", parse_dates=["time"]).sort_values("time")

FEATURES = ["temperature", "humidity", "hour", "is_weekend"]
TARGET = "kwh"

# Time-based split: train on the past, test on the most recent 20%
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
model.fit(train[FEATURES], train[TARGET])

pred = model.predict(test[FEATURES])
print(f"MAE: {mean_absolute_error(test[TARGET], pred):.2f} kWh")
print(f"R2 : {r2_score(test[TARGET], pred):.3f}")

joblib.dump(model, "models/model.pkl", compress=3)
print("Saved models/model.pkl")