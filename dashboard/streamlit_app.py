import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EcoWatt", page_icon="🌱", layout="wide")

# On Streamlit Cloud, API_URL comes from Secrets. Locally it falls back.
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "http://127.0.0.1:8000"

st.title("🌱 EcoWatt: AI Smart Building Energy Forecaster")
st.caption("Predicts hourly energy demand from the weather forecast.")

with st.sidebar:
    st.header("Location")
    lat = st.number_input("Latitude", value=31.5204, format="%.4f")
    lon = st.number_input("Longitude", value=74.3587, format="%.4f")

    st.header("Mode")
    mode_label = st.radio(
        "Forecast window",
        ["Next 24 hours (live)", "Tomorrow"],
        index=0,
    )
    mode = "next24" if mode_label.startswith("Next") else "tomorrow"

    auto = st.checkbox("Auto-refresh every 60 seconds", value=True)
    refresh = st.button("Refresh now", type="primary")


@st.cache_data(ttl=50, show_spinner=False)
def fetch(lat, lon, mode):
    r = requests.get(
        f"{API_URL}/predict-efficiency",
        params={"lat": lat, "lon": lon, "mode": mode},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


if refresh:
    fetch.clear()


@st.fragment(run_every=60 if auto else None)
def live_view():
    try:
        with st.spinner("Contacting API (first call can take ~50 seconds on free hosting)..."):
            data = fetch(lat, lon, mode)
    except Exception as e:
        st.error(f"Could not reach the API: {e}")
        return

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    cur = data["current"]
    st.subheader(
        "Live view: next 24 hours" if data["mode"] == "next24"
        else f"Forecast for {data['date']}"
    )

    w1, w2, w3 = st.columns(3)
    w1.metric("Temperature now", f"{cur['temperature']} °C")
    w2.metric("Humidity now", f"{cur['humidity']} %")
    w3.metric("Weather reading time", cur["time"].replace("T", " "))

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak demand reduction", f"{data['savings']['peak_demand_reduction_pct']}%")
    c2.metric("CO₂ saved (est.)", f"{data['savings']['co2_saved_kg']} kg")
    c3.metric("kWh shifted off-peak", f"{data['savings']['kwh_shifted_off_peak']}")

    chart_df = df.set_index("time")[["predicted_kwh", "optimized_kwh"]]
    chart_df.columns = ["Business as usual", "With EcoWatt plan"]
    st.line_chart(chart_df)

    left, right = st.columns(2)
    with left:
        st.subheader("Eco-mode recommendations")
        for rec in data["recommendations"]:
            st.success(rec)
    with right:
        st.subheader("Weather used")
        st.line_chart(df.set_index("time")[["temperature"]])

    with st.expander("Raw hourly data"):
        st.dataframe(df.drop(columns=["hour"]), use_container_width=True)

    st.caption(
        f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
        "Energy values are model predictions; savings are simulated estimates."
    )


live_view()
