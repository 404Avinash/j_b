# 🛡️ JATAYU - Predictive Intelligence Fusion System

**Operation Silent Watch** | Hackathon 2025

> Anticipate IED threats before they materialize using multi-source intelligence fusion and machine learning.

---

## 🎯 Mission

Transform raw intelligence into actionable predictions. JATAYU analyzes historical IED incident data from India's Red Corridor to predict future attacks, enabling security forces to preemptively deploy resources.

---

## 📊 Real Data Statistics

| Metric | Value |
|--------|-------|
| **Total Incidents** | 187 (2020-2026) |
| **States Covered** | 7 (CG, JH, OR, MH, TS, AP, Multi-border) |
| **Total Killed** | 87 |
| **Total Injured** | 225 |
| **Top Hotspot** | Bijapur (63 incidents, 27 killed) |
| **Attack Clusters** | 39 detected |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Demo

```bash
python run_real_demo.py
```

### 3. Launch Dashboard

```bash
streamlit run src/visualization/real_dashboard.py
```

---

## 📂 Project Structure

```
jatayu_beta/
├── data/
│   ├── raw_incidents.csv      # Real IED incident data (187 incidents)
│   └── processed_incidents.csv
├── src/
│   ├── data/
│   │   ├── data_generator.py  # Synthetic data generator
│   │   └── real_data_loader.py # Real data loader & preprocessing
│   ├── features/
│   │   └── feature_engineer.py # Location-agnostic feature extraction
│   ├── models/
│   │   ├── models.py          # XGBoost + LSTM ensemble
│   │   ├── real_trainer.py    # Real data ML pipeline
│   │   └── explainer.py       # SHAP-based explanations
│   └── visualization/
│       └── real_dashboard.py  # Streamlit dashboard
├── run_real_demo.py           # Complete demo script
├── requirements.txt
└── README.md
```

---

## 🤖 ML Architecture

### Features (23 engineered)

- **Temporal**: days_since_last, attack_velocity, attacks_last_7/30/90
- **Spatial**: district_historical_rate, is_high_risk_district
- **Casualty**: casualties_last_30, district_casualty_rate
- **Seasonal**: month_sin/cos, day_of_week_sin/cos

### Model Performance

- **Training**: 139 samples (2020-2024)
- **Testing**: 48 samples (2025-2026)
- **Accuracy**: 58.3%
- **Precision**: 66.7%
- **F1 Score**: 0.667

### Key Insights

1. **Bijapur** is the highest-risk district (63 incidents, 27 killed)
2. Attack **tempo increases** before major clusters
3. **Casualty momentum** is a strong predictive signal
4. **Seasonal patterns** detected (month_sin is #2 feature)

---

## 📈 Key Findings

### Top 5 Hotspot Districts

1. **Bijapur** (Chhattisgarh): 63 incidents, 27 killed
2. **West Singhbhum** (Jharkhand): 33 incidents, 14 killed
3. **Narayanpur** (Chhattisgarh): 24 incidents, 13 killed
4. **Dantewada** (Chhattisgarh): 11 incidents, 12 killed
5. **Sukma** (Chhattisgarh): 11 incidents, 8 killed

### Most Severe Attack Cluster

- **April 2023**: 4 attacks in 11 days, 11 killed, 1 injured
- Districts: Dantewada, Bijapur, Not specified

### January 2025 Cluster (Validation Target)

- **Jan 6-17, 2025**: 6 attacks, 10 killed, 6 injured
- Districts: Bijapur, Sukma, Narayanpur, West Singhbhum

---

## 🔮 Current Risk Assessment

**As of January 18, 2026:**

- Attack Probability (7 days): **37.1%**
- Risk Level: **MEDIUM**
- Recommended: Standard patrol protocols, continue monitoring

---

## 📚 Data Sources

The incident data was compiled from:

- South Asia Terrorism Portal (SATP)
- News archives
- Official reports

---

## 🛠️ Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
streamlit>=1.20.0
plotly>=5.0.0
folium>=0.14.0
streamlit-folium>=0.15.0
```

---

## 📝 License

For hackathon demonstration purposes only.

---

## 👥 Team

Built for Hackathon 2026 - Operation Silent Watch

---

*Stay Vigilant. Predict. Protect.*
