import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import tensorflow as tf
import plotly.express as px

from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ====================================
# CONFIG
# ====================================

st.set_page_config(
    page_title="Prediksi Harga Rumah Sleman",
    page_icon="🏠",
    layout="wide"
)

# ====================================
# TITIK REFERENSI
# ====================================

UGM = (-7.7705, 110.3772)
MALIOBORO = (-7.7926, 110.3658)
CITY_CENTER = (-7.7972, 110.3705)

# ====================================
# HAVERSINE
# ====================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c

# ====================================
# FORMAT RUPIAH
# ====================================

def format_rupiah_adaptive(x):

    try:

        x = float(x)

        if x >= 1_000_000_000:
            return f"Rp {x/1_000_000_000:.2f} Miliar"

        elif x >= 1_000_000:
            return f"Rp {x/1_000_000:.2f} Juta"

        else:
            return f"Rp {x:,.0f}"

    except:
        return str(x)

# ====================================
# LOAD MODEL
# ====================================

@st.cache_resource
def load_assets():

    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("best_model.pkl")

    return preprocessor, model

preprocessor, best_model = load_assets()

best_model_type = type(best_model).__name__

# ====================================
# LOAD EVALUATION CSV
# ====================================

if os.path.exists("model_results.csv"):

    metrics_df = pd.read_csv("model_results.csv")

    if "Unnamed: 0" in metrics_df.columns:
        metrics_df.rename(
            columns={"Unnamed: 0": "Model"},
            inplace=True
        )

    if "R2" in metrics_df.columns:
        metrics_df.rename(
            columns={"R2": "R²"},
            inplace=True
        )

else:

    metrics_df = pd.DataFrame()

# ====================================
# KOORDINAT LOKASI
# ====================================

location_coords = {

    "Ngaglik, Sleman": (-7.7060, 110.4010),
    "Depok, Sleman": (-7.7680, 110.4010),
    "Kalasan, Sleman": (-7.7670, 110.4720),
    "Mlati, Sleman": (-7.7420, 110.3600),
    "Sleman, Sleman": (-7.7150, 110.3550),
    "Ngemplak, Sleman": (-7.6940, 110.4300),
    "Gamping, Sleman": (-7.7990, 110.3210),
    "Godean, Sleman": (-7.7690, 110.2950),
    "Purwomartani, Sleman": (-7.7470, 110.4580),
    "Condong Catur, Sleman": (-7.7570, 110.4010),
    "Berbah, Sleman": (-7.8170, 110.4380),
    "Prambanan, Sleman": (-7.7520, 110.4920),
    "Kaliurang, Sleman": (-7.6000, 110.4200),
    "Sayegan, Sleman": (-7.7230, 110.2890),
    "Caturtunggal, Sleman": (-7.7560, 110.3850),
    "Pakem, Sleman": (-7.6640, 110.4210),
    "Moyudan, Sleman": (-7.7680, 110.2490),
    "Cebongan, Sleman": (-7.7310, 110.3420),
    "Tempel, Sleman": (-7.6500, 110.3230),
    "Turi, Sleman": (-7.6530, 110.3690),
    "Jombor, Sleman": (-7.7470, 110.3620),
    "Sidoarum, Sleman": (-7.7630, 110.3320)
}

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("🏠 House Prediction App")

menu = st.sidebar.radio(
    "Menu",
    [
        "Prediksi Harga",
        "Evaluasi Model"
    ]
)
