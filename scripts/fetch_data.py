"""
Downloads real historical weather from Open-Meteo, then builds a
SIMULATED building energy load that depends on weather + time.
"""
import numpy as np
import pandas as pd
import requests

# ---- CHANGE THESE to your city (find coordinates on Google Maps) ----
LAT, LON = 31.5204, 74.3587   # example: Lahore
START, END = "2023-01-01", "2025-12-31"

URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START,
    "end_date": END,
    "hourly": "temperature_2m,relative_humidity_2m",
    "timezone": "auto",
}

print("Downloading weather...")
r = requests.get(URL, params=params, timeout=60)
r.raise_for_status()
h = r.json()["hourly"]

df = pd.DataFrame({
    "time": pd.to_datetime(h["time"]),
    "temperature": h["temperature_2m"],
    "humidity": h["relative_humidity_2m"],
})
df = df.dropna()
df["hour"] = df["time"].dt.hour
df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)


def occupancy(hour, weekend):
    """How busy the building is (0 to 1)."""
    if weekend:
        return 0.15
    if 8 <= hour <= 18:
        return 1.0
    if 6 <= hour < 8 or 18 < hour <= 21:
        return 0.5
    return 0.2


rng = np.random.default_rng(42)
occ = np.array([occupancy(hr, wk) for hr, wk in zip(df["hour"], df["is_weekend"])])

base = 80                                                 # always-on load (servers, security)
people_and_lights = 120 * occ                             # lighting + equipment
cooling = np.maximum(0, df["temperature"] - 24) * (6 + 10 * occ)
heating = np.maximum(0, 16 - df["temperature"]) * (4 + 5 * occ)
humidity_load = np.maximum(0, df["humidity"] - 60) * 0.3 * occ
noise = rng.normal(0, 6, len(df))

df["kwh"] = (base + people_and_lights + cooling + heating + humidity_load + noise).clip(lower=20)

df.to_csv("data/energy.csv", index=False)
print(f"Saved {len(df)} rows to data/energy.csv")
print(df.head())
