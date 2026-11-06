"""
EcoWatt - single-file Streamlit app.
Loads the trained model directly and calls Open-Meteo, so no separate API is needed.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EcoWatt", page_icon="🌱", layout="wide")

# models/model.pkl lives one folder above dashboard/
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
FEATURES = ["temperature", "humidity", "hour", "is_weekend"]

# Assumption: solar makes midday/afternoon cleaner; evening peak is dirtiest.
CARBON_KG_PER_KWH = {
    h: (0.50 if 18 <= h <= 22 else 0.38 if 7 <= h <= 17 else 0.42)
    for h in range(24)
}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=50, show_spinner=False)
def get_weather(lat: float, lon: float, mode: str):
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
        tomorrow = sorted(df["time"].dt.date.unique())[1]
        df = df[df["time"].dt.date == tomorrow].copy()

    df["hour"] = df["time"].dt.hour
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    return df.reset_index(drop=True), current


def build_plan(df: pd.DataFrame) -> dict:
    load = df["predicted_kwh"].to_numpy()
    hours = df["hour"].tolist()
    n = len(load)

    peak_idx = sorted(np.argsort(load)[-3:])          # 3 highest-demand hours
    first_peak = peak_idx[0]

    # Eco-mode: cut 12% of load in peak hours
    cut = 0.12
    optimized = load.copy()
    shifted = float((load[peak_idx] * cut).sum())
    optimized[peak_idx] *= (1 - cut)

    # Pre-cool in up to 4 hours before the peak, never creating a new peak
    window = list(range(max(0, first_peak - 4), first_peak))
    cap = load.max() * 0.97
    pre_idx = [i for i in window if load[i] < cap] or [max(0, first_peak - 1)]

    extra = shifted / len(pre_idx)
    for i in pre_idx:
        optimized[i] = max(optimized[i], min(optimized[i] + extra, cap))

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
        "recommendations": recs,
        "peak_reduction": round(max(0.0, 100 * (1 - optimized.max() / load.max())), 1),
        "co2_saved": round(max(0.0, co2_base - co2_opt), 1),
        "shifted": round(shifted, 1),
    }


# ---------------------------- UI ----------------------------
st.title("🌱 EcoWatt: AI Smart Building Energy Forecaster")
st.caption("Predicts hourly energy demand from the weather forecast.")

try:
    model = load_model()
except Exception as e:
    st.error(
        "Could not load models/model.pkl. Check that the file is in the repo and that "
        f"scikit-learn in requirements.txt matches the version used for training.\n\n{e}"
    )
    st.stop()

with st.sidebar:
    st.header("Location")
    lat = st.number_input("Latitude", value=31.5204, format="%.4f")
    lon = st.number_input("Longitude", value=74.3587, format="%.4f")

    st.header("Mode")
    mode_label = st.radio("Forecast window", ["Next 24 hours (live)", "Tomorrow"], index=0)
    mode = "next24" if mode_label.startswith("Next") else "tomorrow"

    auto = st.checkbox("Auto-refresh every 60 seconds", value=True)
    refresh = st.button("Refresh now", type="primary")

if refresh:
    get_weather.clear()


@st.fragment(run_every=60 if auto else None)
def live_view():
    try:
        with st.spinner("Fetching weather..."):
            df, cur = get_weather(lat, lon, mode)
    except Exception as e:
        st.error(f"Could not fetch weather: {e}")
        return

    df = df.copy()
    df["predicted_kwh"] = model.predict(df[FEATURES])
    plan = build_plan(df)
    df["optimized_kwh"] = plan["optimized"]

    st.subheader(
        "Live view: next 24 hours" if mode == "next24"
        else f"Forecast for {df['time'].dt.date.iloc[0]}"
    )

    w1, w2, w3 = st.columns(3)
    w1.metric("Temperature now", f"{cur['temperature']} °C")
    w2.metric("Humidity now", f"{cur['humidity']} %")
    w3.metric("Weather reading time", cur["time"].replace("T", " "))

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak demand reduction", f"{plan['peak_reduction']}%")
    c2.metric("CO₂ saved (est.)", f"{plan['co2_saved']} kg")
    c3.metric("kWh shifted off-peak", f"{plan['shifted']}")

    chart_df = df.set_index("time")[["predicted_kwh", "optimized_kwh"]]
    chart_df.columns = ["Business as usual", "With EcoWatt plan"]
    st.line_chart(chart_df)

    left, right = st.columns(2)
    with left:
        st.subheader("Eco-mode recommendations")
        for rec in plan["recommendations"]:
            st.success(rec)
    with right:
        st.subheader("Weather used")
        st.line_chart(df.set_index("time")[["temperature"]])

    with st.expander("Raw hourly data"):
        st.dataframe(
            df[["time", "temperature", "humidity", "predicted_kwh", "optimized_kwh"]].round(1),
            use_container_width=True,
        )

    st.caption(
        f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
        "Energy values are model predictions; savings are simulated estimates."
    )


live_view()
