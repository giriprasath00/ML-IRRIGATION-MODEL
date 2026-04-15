# 🌱 AquaSense v4: Smart Irrigation Predictor

**Author:** Giri Prasath V (24BCS079)

## 📌 Overview
AquaSense v4 is an advanced, terminal-based AI application designed to predict whether a specific agricultural field requires irrigation. 

Unlike static machine learning models, AquaSense v4 dynamically fetches real-world, historical weather data via the Open-Meteo API to train its `RandomForestClassifier` on the fly. It calculates complex environmental factors like Evapotranspiration (ET) and soil moisture retention specific to 20 Indian states and 7 distinct soil types, providing farmers with a highly accurate, data-driven watering decision.

## ✨ Key Features
* **Live Data Ingestion:** Contacts the Open-Meteo API to pull 365 days of historical weather data based on exact state coordinates.
* **Dynamic Model Training:** Automatically builds, trains, and saves a custom `RandomForestClassifier` locally the first time it runs. 
* **Smart Feature Engineering:** Calculates synthetic soil moisture based on real-world physics (soil retention vs. drainage rates).
* **Interactive CLI Menu:** A user-friendly terminal interface to select region, soil, crop, and input current weather conditions.
