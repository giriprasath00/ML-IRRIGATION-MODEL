# ============================================================
#   AquaSense v4 — Real Data + Interactive CLI
#   Trains on API data, predicts on user input
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import requests
import time
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. REGIONS, SOILS & CROPS MAPPING ───────────────────────
REGION_DATA = {
    "Punjab": {"soils": ["Alluvial", "Loamy", "Silt"], "crops": ["Wheat", "Rice", "Sugarcane", "Vegetables"]},
    "Tamil Nadu": {"soils": ["Clay", "Red", "Alluvial", "Silt"], "crops": ["Rice", "Sugarcane", "Cotton", "Vegetables"]},
    "Kerala": {"soils": ["Clay", "Loamy", "Red"], "crops": ["Rice", "Vegetables"]},
    "Rajasthan": {"soils": ["Sandy", "Silt", "Loamy"], "crops": ["Cotton", "Wheat", "Vegetables"]},
    "Maharashtra": {"soils": ["Black", "Loamy", "Red"], "crops": ["Cotton", "Sugarcane", "Wheat", "Vegetables"]},
    "Uttar Pradesh": {"soils": ["Alluvial", "Loamy", "Clay", "Silt"], "crops": ["Wheat", "Rice", "Sugarcane", "Vegetables"]},
    "West Bengal": {"soils": ["Alluvial", "Clay", "Silt"], "crops": ["Rice", "Vegetables", "Sugarcane"]},
    "Gujarat": {"soils": ["Black", "Sandy", "Loamy", "Alluvial"], "crops": ["Cotton", "Wheat", "Vegetables", "Sugarcane"]},
    "Andhra Pradesh": {"soils": ["Red", "Black", "Alluvial", "Clay"], "crops": ["Rice", "Cotton", "Sugarcane", "Vegetables"]},
    "Madhya Pradesh": {"soils": ["Black", "Red", "Alluvial", "Loamy"], "crops": ["Wheat", "Cotton", "Vegetables", "Sugarcane"]},
    "Karnataka": {"soils": ["Red", "Black", "Loamy", "Clay"], "crops": ["Rice", "Cotton", "Sugarcane", "Vegetables"]},
    "Bihar": {"soils": ["Alluvial", "Clay", "Silt", "Loamy"], "crops": ["Rice", "Wheat", "Vegetables", "Sugarcane"]},
    "Haryana": {"soils": ["Alluvial", "Loamy", "Sandy", "Silt"], "crops": ["Wheat", "Rice", "Cotton", "Vegetables"]},
    "Odisha": {"soils": ["Alluvial", "Red", "Clay", "Silt"], "crops": ["Rice", "Vegetables", "Sugarcane"]},
    "Himachal Pradesh": {"soils": ["Loamy", "Silt", "Clay"], "crops": ["Wheat", "Vegetables"]},
    "Telangana": {"soils": ["Red", "Black", "Loamy"], "crops": ["Cotton", "Rice", "Sugarcane", "Vegetables"]},
    "Chhattisgarh": {"soils": ["Red", "Alluvial", "Clay", "Loamy"], "crops": ["Rice", "Wheat", "Vegetables", "Sugarcane"]},
    "Uttarakhand": {"soils": ["Loamy", "Silt", "Clay", "Alluvial"], "crops": ["Wheat", "Rice", "Vegetables"]},
    "Jharkhand": {"soils": ["Red", "Alluvial", "Clay", "Loamy"], "crops": ["Rice", "Wheat", "Vegetables"]},
    "Assam": {"soils": ["Alluvial", "Clay", "Loamy", "Silt"], "crops": ["Rice", "Vegetables", "Sugarcane"]}
}

STATE_COORDS = {
    "Punjab": (31.1471, 75.3412), "Tamil Nadu": (11.1271, 78.6569),
    "Kerala": (10.8505, 76.2711), "Rajasthan": (27.0238, 74.2179),
    "Maharashtra": (19.7515, 75.7139), "Uttar Pradesh": (26.8467, 80.9462),
    "West Bengal": (22.9868, 87.8550), "Gujarat": (22.2587, 71.1924),
    "Andhra Pradesh": (15.9129, 79.7400), "Madhya Pradesh": (22.9734, 78.6569),
    "Karnataka": (15.3173, 75.7139), "Bihar": (25.0961, 85.3131),
    "Haryana": (29.0588, 76.0856), "Odisha": (20.9517, 85.0985),
    "Himachal Pradesh": (31.1048, 77.1666), "Telangana": (18.1124, 79.0193),
    "Chhattisgarh": (21.2787, 81.8661), "Uttarakhand": (30.0668, 79.0193),
    "Jharkhand": (23.6102, 85.2799), "Assam": (26.2006, 92.9376)
}

SOIL_INFO = {
    "Clay": "🟤 Holds water well, drains slowly", "Loamy": "🌱 Best for most crops, balanced",
    "Sandy": "🏜️ Drains very fast, needs water", "Silt": "💧 Smooth texture, moderate retention",
    "Black": "⚫ Rich in minerals, excellent for cotton", "Red": "🔴 Low moisture retention",
    "Alluvial": "🌾 Highly fertile, found near rivers"
}

SOIL_RETENTION = {"Clay": 0.90, "Loamy": 0.70, "Sandy": 0.30, "Silt": 0.75, "Black": 0.85, "Red": 0.45, "Alluvial": 0.72}
SOIL_DRAINAGE = {"Clay": 0.20, "Loamy": 0.50, "Sandy": 0.90, "Silt": 0.45, "Black": 0.25, "Red": 0.65, "Alluvial": 0.48}
CROP_WATER = {"Rice": 9, "Wheat": 5, "Vegetables": 6, "Cotton": 7, "Sugarcane": 10}

def get_season(month=date.today().month):
    if 6 <= month <= 10: return "Kharif"
    elif month in [11, 12, 1, 2, 3]: return "Rabi"
    else: return "Zaid"

# ── 2. DATA FETCHING & TRAINING ENGINE ──────────────────────
def fetch_data_and_train():
    print("🌍 Contacting Open-Meteo API for historical data (This takes ~20 seconds)...")
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=365) 
    rows = []
    
    for region, (lat, lon) in STATE_COORDS.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={start_date}&end_date={end_date}"
               f"&daily=temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration,wind_speed_10m_max"
               f"&timezone=auto")
        try:
            response = requests.get(url).json()
            daily = response.get("daily", {})
            for i in range(len(daily.get("time", []))):
                temp = daily["temperature_2m_mean"][i]
                rain = daily["precipitation_sum"][i]
                et = daily["et0_fao_evapotranspiration"][i]
                wind = daily["wind_speed_10m_max"][i]
                
                if any(v is None for v in [temp, rain, et, wind]): continue
                
                soil = np.random.choice(REGION_DATA[region]["soils"])
                crop = np.random.choice(REGION_DATA[region]["crops"])
                season = get_season(pd.to_datetime(daily["time"][i]).month)
                
                # Synthetic humidity for training variance
                hum = float(np.clip(np.random.normal(60, 15) + (rain * 2), 10, 100))
                
                sm_base = 40 + SOIL_RETENTION[soil]*30 - SOIL_DRAINAGE[soil]*20
                sm = float(np.clip(sm_base - et*1.5 + rain*2, 5, 95))
                thresh = 35 - SOIL_RETENTION[soil]*10
                irrigate = int(sm < thresh or (CROP_WATER[crop] - sm/10) > 2)

                rows.append({
                    "region": region, "soil_type": soil, "crop_type": crop, "season": season,
                    "temperature_c": round(temp, 1), "humidity_pct": round(hum, 1),
                    "rainfall_mm": round(rain, 2), "wind_speed_kmh": round(wind, 1),
                    "evapotranspiration": round(et, 2), "soil_moisture_pct": round(sm, 1),
                    "should_irrigate": irrigate
                })
        except Exception as e:
            print(f"   ⚠️ Failed for {region}")
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    print(f"\n🧠 Training model on {len(df):,} days of real-world weather...")
    
    le_r = LabelEncoder(); le_s = LabelEncoder()
    le_c = LabelEncoder(); le_se = LabelEncoder()
    df["region_enc"] = le_r.fit_transform(df["region"])
    df["soil_enc"]   = le_s.fit_transform(df["soil_type"])
    df["crop_enc"]   = le_c.fit_transform(df["crop_type"])
    df["season_enc"] = le_se.fit_transform(df["season"])

    FEAT = ["region_enc", "soil_enc", "crop_enc", "season_enc", "temperature_c", 
            "humidity_pct", "rainfall_mm", "wind_speed_kmh", "evapotranspiration", "soil_moisture_pct"]

    X = df[FEAT]; y = df["should_irrigate"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mdl = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    mdl.fit(Xtr, ytr)
    
    os.makedirs("saved_model_v4", exist_ok=True)
    joblib.dump(mdl, "saved_model_v4/irrigation_model.pkl")
    joblib.dump(le_r, "saved_model_v4/le_region.pkl")
    joblib.dump(le_s, "saved_model_v4/le_soil.pkl")
    joblib.dump(le_c, "saved_model_v4/le_crop.pkl")
    joblib.dump(le_se, "saved_model_v4/le_season.pkl")
    joblib.dump(FEAT, "saved_model_v4/features.pkl")
    df.to_csv("saved_model_v4/training_dataset.csv", index=False)
    print(f"✅ Model trained! Accuracy: {accuracy_score(yte, mdl.predict(Xte))*100:.2f}%\n")

# ── 3. LOAD MODEL ───────────────────────────────────────────
if not os.path.exists("saved_model_v4/irrigation_model.pkl"):
    fetch_data_and_train()

model     = joblib.load("saved_model_v4/irrigation_model.pkl")
le_region = joblib.load("saved_model_v4/le_region.pkl")
le_soil   = joblib.load("saved_model_v4/le_soil.pkl")
le_crop   = joblib.load("saved_model_v4/le_crop.pkl")
le_season = joblib.load("saved_model_v4/le_season.pkl")
FEATURES  = joblib.load("saved_model_v4/features.pkl")

# ── 4. HELPER FUNCTIONS FOR CLI ─────────────────────────────
def pick(prompt, options):
    while True:
        print(f"\n  {prompt}")
        for i, o in enumerate(options, 1): print(f"    {i}. {o}")
        try:
            c = int(input("  👉 Enter number: "))
            if 1 <= c <= len(options): return options[c - 1]
        except ValueError: pass
        print("  ⚠️  Invalid choice.")

def get_float(prompt, lo, hi):
    while True:
        try:
            val = float(input(f"  👉 {prompt}: "))
            if lo <= val <= hi: return val
        except ValueError: pass
        print(f"  ⚠️  Enter between {lo} and {hi}.")

def enc(le, val): return int(le.transform([val])[0]) if val in le.classes_ else 0

def predict_interactive(region, soil, crop, temp, humidity, rainfall):
    season = get_season()
    # Estimate Evapotranspiration based on inputs
    et = float(np.clip(0.3 * temp + 0.1 * 10 - 0.05 * humidity + 0.5, 1, 15))
    # Estimate current soil moisture
    sm = float(np.clip(40 + SOIL_RETENTION[soil]*30 - SOIL_DRAINAGE[soil]*20 - et*1.5 + rainfall*2, 5, 95))

    X = pd.DataFrame([[
        enc(le_region, region), enc(le_soil, soil), enc(le_crop, crop), enc(le_season, season),
        temp, humidity, rainfall, 10.0, et, sm
    ]], columns=FEATURES)

    return int(model.predict(X)[0]), float(model.predict_proba(X)[0][1]) * 100, season, round(sm, 1)

# ── 5. INTERACTIVE MENU ─────────────────────────────────────
while True:
    print("=" * 58)
    print("   🌱 AquaSense v4 — Smart Irrigation Predictor")
    print("=" * 58)

    # State Selection
    regions = sorted(REGION_DATA.keys())
    region  = pick("Select your State:", regions)

    # Soil & Crop Selection (Locked to Region)
    soil = pick(f"Select Soil Type for {region}:", REGION_DATA[region]["soils"])
    print(f"      → {SOIL_INFO[soil]}")
    crop = pick(f"Select Crop Type for {region}:", REGION_DATA[region]["crops"])

    # Manual Weather Inputs
    print()
    temp     = get_float("Enter today's Temperature (°C)  [10–48]", 10, 48)
    humidity = get_float("Enter today's Humidity (%)       [10–100]", 10, 100)
    rainfall = get_float("Enter today's Rainfall (mm)      [0–50]", 0, 50)

    # Prediction
    pred, prob, season, sm = predict_interactive(region, soil, crop, temp, humidity, rainfall)

    # Output Results
    print("\n" + "=" * 58)
    print("            🌾 PREDICTION RESULT")
    print("=" * 58)
    print(f"  📍 State         : {region}")
    print(f"  🪨  Soil Type     : {soil}")
    print(f"  🌿 Crop          : {crop} ({season} season)")
    print(f"  💧 Est. Moisture : {sm}%")
    print(f"  🤖 AI Confidence : {prob:.1f}%")
    print("-" * 58)

    if pred:
        print("  🚿 DECISION  :  YES — IRRIGATE NOW!")
        if soil == "Sandy": print("  ⚠️ Sandy soil drains fast — use drip irrigation.")
    else:
        print("  ✅ DECISION  :  NO — DO NOT IRRIGATE")
        if rainfall > 5: print("  🌧️ Good rainfall detected — skip watering.")

    print("=" * 58)
    if input("\n🔄 Predict another? (y/n): ").strip().lower() != "y":
        print("\n👋 Thank you for using AquaSense v4!\n")
        break