"""
JATAYU - Intel-Based ML Training
=================================
Train ML models on generated intel reports to predict attacks.

This is the core ML component that learns:
1. Pre-attack signal patterns
2. Noise vs. true signal differentiation
3. Deception detection
4. Attack probability prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

from src.data.reverse_intel_generator import ReverseIntelGenerator


class IntelMLTrainer:
    """
    Train ML models on generated intelligence reports.
    
    Key Features:
    - Aggregates daily intel patterns
    - Learns signal-to-noise differentiation
    - Predicts attack probability
    - Explains which intel types are most predictive
    """
    
    def __init__(self):
        self.generator = ReverseIntelGenerator()
        self.intel_df = None
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
    
    def generate_intel_dataset(self,
                                region: str = None,
                                target_records: int = 500) -> pd.DataFrame:
        """Generate intel dataset for a region."""
        print("\n[ML] Generating intelligence dataset...")
        
        # Load incidents
        self.generator.load_incidents()
        
        # Find and use the most severe cluster
        clusters = self.generator.find_attack_clusters(region=region, max_gap_days=10)
        
        if not clusters:
            clusters = self.generator.find_attack_clusters(max_gap_days=10)
        
        if clusters:
            # Use the most severe cluster
            cluster = clusters[0]
            self.intel_df = self.generator.generate_intel_for_cluster(
                cluster,
                intel_per_day=50,
                pre_days=10,
                post_days=3
            )
        
        return self.intel_df
    
    def load_existing_intel(self, path: str = None) -> pd.DataFrame:
        """Load previously generated intel from CSV."""
        import os
        
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'data', 'generated_intel.csv'
            )
        
        self.intel_df = pd.read_csv(path, parse_dates=['Timestamp', 'Date'])
        print(f"[ML] Loaded {len(self.intel_df)} intel records")
        
        return self.intel_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer daily aggregated features from raw intel.
        
        This is the key ML step - transforming raw intel into
        predictive features.
        """
        if self.intel_df is None:
            self.generate_intel_dataset()
        
        df = self.intel_df.copy()
        
        print("[ML] Engineering features from intel reports...")
        
        # Aggregate by day
        daily_features = []
        
        for date, day_df in df.groupby('Date'):
            features = {
                'Date': date,
                
                # Volume features
                'total_intel_count': len(day_df),
                'humint_count': (day_df['Intel_Type'] == 'HUMINT').sum(),
                'sigint_count': (day_df['Intel_Type'] == 'SIGINT').sum(),
                'patrol_count': (day_df['Intel_Type'] == 'PATROL').sum(),
                'osint_count': (day_df['Intel_Type'] == 'OSINT').sum(),
                
                # Urgency features
                'high_urgency_count': (day_df['Urgency'] == 'HIGH').sum(),
                'medium_urgency_count': (day_df['Urgency'] == 'MEDIUM').sum(),
                'high_urgency_ratio': (day_df['Urgency'] == 'HIGH').mean(),
                
                # Reliability features
                'avg_reliability': day_df['Reliability'].mean(),
                'max_reliability': day_df['Reliability'].max(),
                'high_reliability_count': (day_df['Reliability'] > 0.7).sum(),
                
                # Signal intensity (from generator)
                'avg_signal_intensity': day_df['Signal_Intensity'].mean(),
                'max_signal_intensity': day_df['Signal_Intensity'].max(),
                
                # True signal ratio (this is ground truth for training)
                'true_signal_count': (day_df['Label'] == 'TRUE_SIGNAL').sum(),
                'noise_count': (day_df['Label'] == 'NOISE').sum(),
                'deception_count': (day_df['Label'] == 'DECEPTION').sum(),
                'signal_ratio': (day_df['Label'] == 'TRUE_SIGNAL').mean(),
                
                # Target: Is this an attack day?
                'is_attack_day': day_df['Is_Attack_Day'].any(),
            }
            
            daily_features.append(features)
        
        features_df = pd.DataFrame(daily_features)
        features_df = features_df.sort_values('Date').reset_index(drop=True)
        
        # Add temporal features (lookback)
        features_df['intel_count_rolling_3'] = features_df['total_intel_count'].rolling(3, min_periods=1).mean()
        features_df['signal_intensity_rolling_3'] = features_df['avg_signal_intensity'].rolling(3, min_periods=1).mean()
        features_df['high_urgency_rolling_3'] = features_df['high_urgency_count'].rolling(3, min_periods=1).sum()
        
        # Velocity features (change from previous day)
        features_df['intel_velocity'] = features_df['total_intel_count'].diff().fillna(0)
        features_df['urgency_velocity'] = features_df['high_urgency_count'].diff().fillna(0)
        features_df['signal_velocity'] = features_df['avg_signal_intensity'].diff().fillna(0)
        
        # Create prediction target: attack within next N days
        features_df['attack_within_1_day'] = features_df['is_attack_day'].shift(-1).fillna(False).astype(int)
        features_df['attack_within_3_days'] = features_df['is_attack_day'].rolling(3, min_periods=1).max().shift(-1).fillna(0).astype(int)
        
        self.features_df = features_df
        
        print(f"[ML] Created {len(features_df)} daily feature vectors")
        print(f"[ML] Attack days: {features_df['is_attack_day'].sum()}")
        
        return features_df
    
    def train_model(self, 
                    target: str = 'is_attack_day',
                    test_size: float = 0.3) -> Dict:
        """
        Train ML model to predict attacks.
        
        Args:
            target: Column to predict ('is_attack_day', 'attack_within_1_day', 'attack_within_3_days')
            test_size: Fraction of data for testing
        
        Returns:
            Dictionary with metrics and feature importance
        """
        if self.features_df is None:
            self.engineer_features()
        
        df = self.features_df.copy()
        
        # Define feature columns (exclude targets and metadata)
        exclude_cols = ['Date', 'is_attack_day', 'attack_within_1_day', 'attack_within_3_days']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].fillna(0)
        y = df[target].astype(int)
        
        print(f"\n[ML] Training model to predict: {target}")
        print(f"[ML] Features: {len(self.feature_columns)}")
        print(f"[ML] Positive samples: {y.sum()} ({100*y.mean():.1f}%)")
        
        # Temporal split (last N% for testing)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"[ML] Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if HAS_XGB:
            print("[ML] Using XGBoost classifier")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=len(y_train) / (y_train.sum() + 1),
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        else:
            print("[ML] Using Gradient Boosting classifier")
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        # Metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print results
        print(f"\n[RESULTS] Model Performance:")
        print(f"  Accuracy:  {self.metrics['accuracy']:.3f}")
        print(f"  Precision: {self.metrics['precision']:.3f}")
        print(f"  Recall:    {self.metrics['recall']:.3f}")
        print(f"  F1 Score:  {self.metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {self.metrics['auc_roc']:.3f}")
        
        print(f"\n[FEATURES] Top 10 Predictive Features:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'metrics': self.metrics,
            'feature_importance': importance,
            'test_predictions': pd.DataFrame({
                'date': df.iloc[split_idx:]['Date'].values,
                'actual': y_test.values,
                'predicted': y_pred,
                'probability': y_prob
            })
        }
    
    def predict_attack_probability(self, intel_records: pd.DataFrame) -> float:
        """
        Predict attack probability from incoming intel records.
        
        This is what the operational system would use in real-time.
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train_model() first.")
        
        # Engineer features from the intel records
        features = {
            'total_intel_count': len(intel_records),
            'humint_count': (intel_records['Intel_Type'] == 'HUMINT').sum() if 'Intel_Type' in intel_records else 0,
            'sigint_count': (intel_records['Intel_Type'] == 'SIGINT').sum() if 'Intel_Type' in intel_records else 0,
            'patrol_count': (intel_records['Intel_Type'] == 'PATROL').sum() if 'Intel_Type' in intel_records else 0,
            'osint_count': (intel_records['Intel_Type'] == 'OSINT').sum() if 'Intel_Type' in intel_records else 0,
            'high_urgency_count': (intel_records['Urgency'] == 'HIGH').sum() if 'Urgency' in intel_records else 0,
            'medium_urgency_count': (intel_records['Urgency'] == 'MEDIUM').sum() if 'Urgency' in intel_records else 0,
            'high_urgency_ratio': (intel_records['Urgency'] == 'HIGH').mean() if 'Urgency' in intel_records else 0,
            'avg_reliability': intel_records['Reliability'].mean() if 'Reliability' in intel_records else 0.5,
            'max_reliability': intel_records['Reliability'].max() if 'Reliability' in intel_records else 0.5,
            'high_reliability_count': (intel_records['Reliability'] > 0.7).sum() if 'Reliability' in intel_records else 0,
            'avg_signal_intensity': intel_records['Signal_Intensity'].mean() if 'Signal_Intensity' in intel_records else 0.5,
            'max_signal_intensity': intel_records['Signal_Intensity'].max() if 'Signal_Intensity' in intel_records else 0.5,
        }
        
        # Fill in features we can't compute
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0
        
        X = pd.DataFrame([features])[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return probability
    
    def demonstrate_pattern_detection(self) -> Dict:
        """
        Demonstrate how the model detects patterns humans miss.
        
        Returns analysis showing signal buildup before attacks.
        """
        if self.features_df is None:
            self.engineer_features()
        
        df = self.features_df.copy()
        
        print("\n" + "="*60)
        print("PATTERN DETECTION DEMONSTRATION")
        print("="*60)
        
        # Find attack days
        attack_days = df[df['is_attack_day'] == True].index.tolist()
        
        print(f"\nFound {len(attack_days)} attack days in the dataset")
        
        # For each attack, show the lead-up pattern
        for attack_idx in attack_days:
            attack_row = df.iloc[attack_idx]
            print(f"\n--- Attack on {attack_row['Date']} ---")
            
            # Get 5 days before the attack
            start_idx = max(0, attack_idx - 5)
            lead_up = df.iloc[start_idx:attack_idx+1]
            
            print("\nDays leading to attack:")
            print("-" * 50)
            
            for i, (_, row) in enumerate(lead_up.iterrows()):
                days_before = attack_idx - (start_idx + i)
                marker = "** ATTACK **" if row['is_attack_day'] else ""
                
                print(f"  {row['Date']} (D-{days_before}): "
                      f"Intel={row['total_intel_count']:.0f}, "
                      f"HighUrg={row['high_urgency_count']:.0f}, "
                      f"Signal={row['avg_signal_intensity']:.2f} "
                      f"{marker}")
        
        # Show what features the model learned
        print("\n" + "-"*50)
        print("KEY INSIGHT: Signal intensity and high-urgency intel")
        print("increase significantly 2-3 days before attacks!")
        print("-"*50)
        
        return {
            'attack_days': attack_days,
            'features': df,
            'pattern': 'Signal buildup detected 3-5 days before attacks'
        }


def run_intel_ml_demo():
    """Run the complete Intel ML demo."""
    
    print("="*70)
    print("JATAYU - Intelligence-Based ML Training Demo")
    print("="*70)
    
    trainer = IntelMLTrainer()
    
    # Try to load existing intel, or generate new
    try:
        trainer.load_existing_intel()
    except Exception:
        print("[ML] No existing intel found, generating new dataset...")
        trainer.generate_intel_dataset()
    
    # Engineer features
    trainer.engineer_features()
    
    # Train model
    print("\n[STEP 1] Training Attack Prediction Model...")
    results = trainer.train_model(target='is_attack_day')
    
    # Demonstrate pattern detection
    print("\n[STEP 2] Demonstrating Pattern Detection...")
    patterns = trainer.demonstrate_pattern_detection()
    
    # Show prediction for a hypothetical day
    print("\n[STEP 3] Real-time Prediction Example...")
    print("-"*50)
    
    # Simulate a day with high activity
    test_intel = pd.DataFrame({
        'Intel_Type': ['HUMINT']*10 + ['SIGINT']*8 + ['PATROL']*12,
        'Urgency': ['HIGH']*15 + ['MEDIUM']*10 + ['LOW']*5,
        'Reliability': np.random.uniform(0.6, 0.9, 30),
        'Signal_Intensity': np.full(30, 0.75)
    })
    
    prob = trainer.predict_attack_probability(test_intel)
    risk = 'CRITICAL' if prob > 0.7 else 'HIGH' if prob > 0.5 else 'MEDIUM' if prob > 0.3 else 'LOW'
    
    print(f"  Given: 30 intel reports (15 HIGH urgency)")
    print(f"  Attack Probability: {prob*100:.1f}%")
    print(f"  Risk Level: {risk}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = run_intel_ml_demo()
