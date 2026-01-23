# -*- coding: utf-8 -*-
"""
JATAYU - Complete Demo Script

Runs the entire pipeline:
1. Generate data
2. Extract features
3. Train model
4. Predict Attack #4
5. Generate explainable alert

Run: python run_demo.py
"""

import os
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime

# Set project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

print("=" * 70)
print("[*] JATAYU - Bastar Intelligence Fusion System")
print("    Operation Silent Watch - Predictive Intelligence Demo")
print("=" * 70)

# Create directories
(project_root / "data").mkdir(exist_ok=True)
(project_root / "results").mkdir(exist_ok=True)

# ============================================
# STEP 1: Generate Data
# ============================================
print("\n" + "=" * 70)
print("STEP 1: GENERATING INTELLIGENCE DATA")
print("=" * 70)

from src.data.data_generator import BastarIntelligenceGenerator

generator = BastarIntelligenceGenerator(
    start_date=datetime(2024, 12, 20),
    end_date=datetime(2025, 1, 20),
    daily_volume=500,
    random_seed=42
)

intel_df = generator.generate_full_dataset(
    save_path=str(project_root / "data" / "bastar_intelligence_15k.csv")
)

attacks_df = generator.get_attacks_df()
attacks_df.to_csv(project_root / "data" / "attacks_ground_truth.csv", index=False)

# ============================================
# STEP 2: Extract Features
# ============================================
print("\n" + "=" * 70)
print("STEP 2: EXTRACTING LOCATION-AGNOSTIC FEATURES")
print("=" * 70)

from src.features.feature_engineer import FeatureEngineer

attacks = [
    {'id': 0, 'date': datetime(2025, 1, 6, 14, 20), 'location': {'lat': 18.50, 'lon': 81.00, 'district': 'Bijapur'}},
    {'id': 1, 'date': datetime(2025, 1, 12, 19, 0), 'location': {'lat': 18.15, 'lon': 81.25, 'district': 'Sukma'}},
    {'id': 2, 'date': datetime(2025, 1, 16, 10, 30), 'location': {'lat': 18.62, 'lon': 80.88, 'district': 'Bijapur'}},
    {'id': 3, 'date': datetime(2025, 1, 17, 7, 15), 'location': {'lat': 18.45, 'lon': 80.95, 'district': 'Narayanpur'}},
]

fe = FeatureEngineer(grid_size_km=5.0)
feature_matrix = fe.create_feature_matrix(
    intel_df,
    start_date=datetime(2024, 12, 25),
    end_date=datetime(2025, 1, 18),
    attacks=attacks
)
feature_matrix.to_csv(project_root / "data" / "feature_matrix.csv", index=False)

print(f"\n✓ Feature matrix created: {feature_matrix.shape[0]} days × {feature_matrix.shape[1]} features")

# ============================================
# STEP 3: Train Model
# ============================================
print("\n" + "=" * 70)
print("STEP 3: TRAINING PREDICTION MODEL")
print("=" * 70)
print("(Training on Attacks 1-3, Holdout: Attack #4)")

from src.models.models import AttackPredictor

predictor = AttackPredictor()
feature_matrix = predictor.load_features(str(project_root / "data" / "feature_matrix.csv"))
results = predictor.train_with_temporal_split(feature_matrix, holdout_attack_id=3)

# Save model
predictor.model.save(str(project_root / "results" / "ensemble_model.pkl"))

# ============================================
# STEP 4: Generate Explainable Alert
# ============================================
print("\n" + "=" * 70)
print("STEP 4: GENERATING EXPLAINABLE ALERT FOR JAN 15")
print("=" * 70)

from src.models.explainer import SHAPExplainer
import pandas as pd

# Load model and features
from src.models.models import EnsembleModel
model = EnsembleModel.load(str(project_root / "results" / "ensemble_model.pkl"))

features = pd.read_csv(project_root / "data" / "feature_matrix.csv")
features['date'] = pd.to_datetime(features['date'])

feature_cols = [c for c in features.columns 
               if c not in ['date', 'target_attack_imminent', 'target_attack_tomorrow']]

# Get Jan 15 features
jan15_mask = features['date'].dt.date == datetime(2025, 1, 15).date()
X = features.loc[jan15_mask, feature_cols]

# Predict
proba = model.predict_proba(X)[0]

# Create explainer
explainer = SHAPExplainer(model)
train_mask = features['date'].dt.date <= datetime(2025, 1, 13).date()
X_train = features.loc[train_mask, feature_cols]
explainer.fit(model, X_train)

# Explain
explanation = explainer.explain(X, proba)

# Generate alert
alert = explainer.generate_alert(
    date=datetime(2025, 1, 15),
    probability=proba,
    reasons=explanation['reasons'],
    location="Narayanpur / Garpa sector"
)

print(alert)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("🎯 DEMONSTRATION COMPLETE")
print("=" * 70)
print(f"""
KEY RESULTS:
├─ Dataset: {len(intel_df):,} intelligence records generated
├─ Features: {feature_matrix.shape[1]} location-agnostic features
├─ Model: XGBoost + LSTM ensemble trained
├─ Holdout: Attack #4 (Jan 17, 2025 - Garpa, Narayanpur)
│
├─ PREDICTION FOR JAN 15 (2 days before Attack #4):
│   └─ Attack Probability: {proba*100:.1f}%
│   └─ Risk Level: {'CRITICAL' if proba >= 0.7 else 'HIGH' if proba >= 0.5 else 'MEDIUM' if proba >= 0.3 else 'LOW'}
│
└─ FILES GENERATED:
    ├─ data/bastar_intelligence_15k.csv
    ├─ data/feature_matrix.csv
    ├─ data/attacks_ground_truth.csv
    └─ results/ensemble_model.pkl

To launch the dashboard:
    streamlit run src/visualization/dashboard.py
""")
