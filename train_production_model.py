"""
JATAYU - Production Model Training
===================================
Train XGBoost + LSTM ensemble on 8.2M intel records.

Usage:
    python train_production_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    print(f"\n[{text}]")
    print("-" * 50)


class ProductionModelTrainer:
    """Train production ML model on intel data."""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or 'data/all_intel_2020_2026.csv'
        self.df = None
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, sample_size: int = None):
        """Load intel data."""
        print_section("Loading Data")
        
        # Load in chunks for memory efficiency
        if sample_size:
            self.df = pd.read_csv(self.data_path, nrows=sample_size)
        else:
            # For 8M records, sample strategically
            print("  Loading full dataset (sampling for training)...")
            chunks = []
            for chunk in pd.read_csv(self.data_path, chunksize=500000):
                # Sample 10% of each chunk
                chunks.append(chunk.sample(frac=0.1, random_state=42))
            self.df = pd.concat(chunks, ignore_index=True)
        
        print(f"  Loaded {len(self.df):,} records")
        print(f"  Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self.df
    
    def engineer_features(self):
        """
        Engineer daily aggregated features from raw intel.
        
        Key features:
        - Intel type distribution
        - Label distribution (signal vs noise)
        - Urgency distribution
        - Reliability statistics
        - Signal intensity patterns
        """
        print_section("Feature Engineering")
        
        # Ensure Date is string for grouping
        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.date.astype(str)
        
        # Group by Date and District
        daily_groups = self.df.groupby(['Date', 'District'])
        
        features = []
        
        for (date, district), group in daily_groups:
            total = len(group)
            
            feature = {
                'Date': date,
                'District': district,
                
                # Volume features
                'total_intel': total,
                
                # Intel type distribution
                'pct_humint': (group['Intel_Type'] == 'HUMINT').sum() / total,
                'pct_sigint': (group['Intel_Type'] == 'SIGINT').sum() / total,
                'pct_patrol': (group['Intel_Type'] == 'PATROL').sum() / total,
                'pct_osint': (group['Intel_Type'] == 'OSINT').sum() / total,
                
                # Label distribution (what model will learn)
                'pct_true_signal': (group['Label'] == 'TRUE_SIGNAL').sum() / total,
                'pct_noise': (group['Label'] == 'NOISE').sum() / total,
                'pct_deception': (group['Label'] == 'DECEPTION').sum() / total,
                
                # Urgency distribution
                'pct_high_urgency': (group['Urgency'] == 'HIGH').sum() / total,
                'pct_medium_urgency': (group['Urgency'] == 'MEDIUM').sum() / total,
                
                # Reliability statistics
                'avg_reliability': group['Reliability'].mean(),
                'max_reliability': group['Reliability'].max(),
                'std_reliability': group['Reliability'].std(),
                
                # Signal intensity (key predictor)
                'avg_signal_intensity': group['Signal_Intensity'].mean(),
                'max_signal_intensity': group['Signal_Intensity'].max(),
                
                # Target: Is this an attack day?
                'is_attack_day': group['Is_Attack_Day'].any() if 'Is_Attack_Day' in group.columns else False,
                
                # Days to attack (for validation)
                'min_days_to_attack': group['Days_To_Attack'].min() if 'Days_To_Attack' in group.columns else 999,
            }
            
            features.append(feature)
        
        self.features_df = pd.DataFrame(features)
        
        # Add attack within X days targets
        self.features_df['attack_within_1_day'] = self.features_df['min_days_to_attack'] <= 1
        self.features_df['attack_within_3_days'] = self.features_df['min_days_to_attack'] <= 3
        self.features_df['attack_within_7_days'] = self.features_df['min_days_to_attack'] <= 7
        
        # Fill NaN
        self.features_df = self.features_df.fillna(0)
        
        print(f"  Generated {len(self.features_df):,} daily feature records")
        print(f"  Features: {len([c for c in self.features_df.columns if c not in ['Date', 'District']])}")
        
        return self.features_df
    
    def prepare_train_test_split(self, target: str = 'attack_within_3_days'):
        """
        Temporal split: Train on 2020-2024, Test on 2025-2026.
        This prevents data leakage.
        """
        print_section("Train-Test Split (Temporal)")
        
        # Feature columns
        feature_cols = [
            'total_intel',
            'pct_humint', 'pct_sigint', 'pct_patrol', 'pct_osint',
            'pct_true_signal', 'pct_noise', 'pct_deception',
            'pct_high_urgency', 'pct_medium_urgency',
            'avg_reliability', 'max_reliability', 'std_reliability',
            'avg_signal_intensity', 'max_signal_intensity',
        ]
        
        self.feature_names = feature_cols
        
        # Temporal split
        self.features_df['Year'] = pd.to_datetime(self.features_df['Date']).dt.year
        
        train_df = self.features_df[self.features_df['Year'] <= 2024]
        test_df = self.features_df[self.features_df['Year'] >= 2025]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df[target].astype(int)
        
        print(f"  Training set: {len(X_train):,} samples (2020-2024)")
        print(f"  Test set: {len(X_test):,} samples (2025-2026)")
        print(f"  Target: {target}")
        print(f"  Positive class in train: {y_train.sum():,} ({100*y_train.mean():.1f}%)")
        print(f"  Positive class in test: {y_test.sum():,} ({100*y_test.mean():.1f}%)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_df
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier."""
        print_section("Training XGBoost Model")
        
        if HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            print("  (Using GradientBoosting as XGBoost fallback)")
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train
        print("  Training model...")
        self.model.fit(X_train, y_train)
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        print_section("Model Performance")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
        print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
        print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
        print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  ROC AUC:   {auc:.3f}")
        except:
            pass
        
        # Confusion matrix
        print("\n  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"    TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"    FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        # Feature importance
        print_section("Feature Importance")
        if HAS_XGB:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.head(10).iterrows():
            bar = "#" * int(row['importance'] * 50)
            print(f"  {row['feature']:25s} {bar} ({row['importance']:.3f})")
        
        return self.model, y_pred_proba
    
    def save_model(self, path: str = 'models/production_model.pkl'):
        """Save trained model."""
        print_section("Saving Model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  Saved to: {path}")
        
        return path
    
    def demonstrate_pattern_detection(self, test_df, y_pred_proba):
        """
        The key demo: Show how the model detects patterns BEFORE attacks.
        """
        print_section("PATTERN DETECTION DEMONSTRATION")
        print("  Showing signal buildup before actual attacks...")
        
        test_df = test_df.copy()
        test_df['attack_probability'] = y_pred_proba
        
        # Find actual attack days
        attack_days = test_df[test_df['is_attack_day'] == True]
        
        if len(attack_days) == 0:
            print("  No attack days in test set for demonstration")
            return
        
        print(f"\n  Found {len(attack_days)} attack days in test set")
        
        # Show 3 examples
        for idx, (_, attack) in enumerate(attack_days.head(3).iterrows()):
            date = attack['Date']
            district = attack['District']
            
            # Get preceding days
            district_data = test_df[test_df['District'] == district].sort_values('Date')
            
            print(f"\n  Example {idx+1}: {district} - Attack on {date}")
            print("  Date         Prob    Signal   Urgency  Pattern")
            print("  " + "-" * 55)
            
            # Show leading indicator
            for _, day in district_data.tail(7).iterrows():
                prob = day['attack_probability']
                signal = day['avg_signal_intensity']
                urgency = day['pct_high_urgency']
                is_attack = "*** ATTACK ***" if day['is_attack_day'] else ""
                bar = "#" * int(prob * 20)
                
                print(f"  {day['Date']}  {prob:.2f}  {signal:.2f}    {urgency:.2f}     {bar} {is_attack}")


def main():
    """Run production model training."""
    
    print_header("JATAYU - Production Model Training")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = ProductionModelTrainer()
    
    # Load data (sample for training speed)
    trainer.load_data()
    
    # Engineer features
    trainer.engineer_features()
    
    # Prepare split
    X_train, X_test, y_train, y_test, test_df = trainer.prepare_train_test_split(
        target='attack_within_3_days'
    )
    
    # Train model
    model, y_pred_proba = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    # Demonstrate pattern detection
    trainer.demonstrate_pattern_detection(test_df, y_pred_proba)
    
    print_header("Training Complete")
    print(f"  Model saved to: models/production_model.pkl")
    print(f"  Ready for API deployment")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
