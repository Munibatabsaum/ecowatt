from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EcoWatt API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/model.pkl")
FEATURES = ["temperature", "humidity", "hour", "is_weekend"]

# Assumption: solar makes midday/afternoon cleaner; evening peak is dirtiest.
# Replace with real carbon-intensity data later.
CARBON_KG_PER_KWH = {
    h: (0.50 if 18 <= h <= 22 else 0.38 if 7 <= h <= 17 else 0.42)
    for h in range(24)
}


def get_weather(lat: float, lon: float, mode: str) -> tuple[pd.DataFrame, dict]:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m",
            "current": "temperature_2m,relative_humidity_2m",
            "forecast_days": 2,
            "timezone": "auto",
        },
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    h = j["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
    })

    cur = j["current"]
    current = {
        "time": cur["time"],
        "temperature": cur["temperature_2m"],
        "humidity": cur["relative_humidity_2m"],
    }

    if mode == "next24":
        now_hour = pd.to_datetime(cur["time"]).floor("h")
        df = df[df["time"] >= now_hour].head(24).copy()
    else:
        tomorrow = sorted(df["time"].dt.date.unique())[1]   # second day = tomorrow
        df = df[df["time"].dt.date == tomorrow].copy()

    df["hour"] = df["time"].dt.hour
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    return df.reset_index(drop=True), current


def build_plan(df: pd.DataFrame) -> dict:
    load = df["predicted_kwh"].to_numpy()
    hours = df["hour"].tolist()
    n = len(load)

    peak_idx = sorted(np.argsort(load)[-3:])            # 3 highest-demand hours
    first_peak = peak_idx[0]

    # Eco-mode: cut 12% of load in peak hours
    cut = 0.12
    optimized = load.copy()
    shifted = float((load[peak_idx] * cut).sum())
    optimized[peak_idx] *= (1 - cut)

    # Pre-cool candidates: the 4 hours before the peak, but only those
    # with load below the original peak, so we never create a new peak.
    window = list(range(max(0, first_peak - 4), first_peak))
    cap = load.max() * 0.97
    pre_idx = [i for i in window if load[i] < cap] or [max(0, first_peak - 1)]

    # Add shifted energy evenly, but never above the cap
    extra = shifted / len(pre_idx)
    for i in pre_idx:
        optimized[i] = max(optimized[i], min(optimized[i] + extra, cap))

    # Carbon accounting
    co2_base = sum(load[i] * CARBON_KG_PER_KWH[hours[i]] for i in range(n))
    co2_opt = sum(optimized[i] * CARBON_KG_PER_KWH[hours[i]] for i in range(n))

    peak_hour = hours[int(np.argmax(load))]
    precool_hour = hours[pre_idx[0]]
    max_temp = float(df["temperature"].max())

    recs = [
        f"Peak demand expected at {peak_hour:02d}:00 ({load.max():.0f} kWh).",
        f"Start pre-cooling at {precool_hour:02d}:00 and relax the setpoint "
        f"during {hours[peak_idx[0]]:02d}:00-{hours[peak_idx[-1]]:02d}:59.",
    ]
    if max_temp >= 35:
        recs.append(f"Very hot day ({max_temp:.0f}°C): close blinds and shade west-facing windows.")
    recs.append("Dim lighting in unoccupied zones after 18:00.")

    return {
        "optimized": optimized,
        "peak_hours": [hours[i] for i in peak_idx],
        "recommendations": recs,
        "savings": {
            "peak_demand_reduction_pct": round(max(0.0, 100 * (1 - optimized.max() / load.max())), 1),
            "co2_saved_kg": round(max(0.0, co2_base - co2_opt), 1),
            "kwh_shifted_off_peak": round(shifted, 1),
        },
    }


@app.get("/")
def root():
    return {"service": "EcoWatt", "status": "ok"}


@app.get("/predict-efficiency")
def predict_efficiency(
    lat: float = Query(31.5204, ge=-90, le=90),
    lon: float = Query(74.3587, ge=-180, le=180),
    mode: str = Query("tomorrow", pattern="^(tomorrow|next24)$"),
):
    try:
        df, current = get_weather(lat, lon, mode)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {e}")

    df["predicted_kwh"] = model.predict(df[FEATURES])
    plan = build_plan(df)
    df["optimized_kwh"] = plan["optimized"]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "date": str(df["time"].dt.date.iloc[0]),
        "mode": mode,
        "current": current,
        "peak_hours": plan["peak_hours"],
        "recommendations": plan["recommendations"],
        "savings": plan["savings"],
        "hourly": [
            {
                "time": r.time.isoformat(),
                "hour": int(r.hour),
                "temperature": round(float(r.temperature), 1),
                "humidity": round(float(r.humidity), 1),
                "predicted_kwh": round(float(r.predicted_kwh), 1),
                "optimized_kwh": round(float(r.optimized_kwh), 1),
            }
            for r in df.itertuples()
        ],
    }