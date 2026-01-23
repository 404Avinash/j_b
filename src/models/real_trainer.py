"""
Real ML Training Pipeline for JATAYU
=====================================
Train models on actual IED incident data with proper temporal splitting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit
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

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    HAS_SKLEARN_ENSEMBLE = True
except ImportError:
    HAS_SKLEARN_ENSEMBLE = False

from src.data.real_data_loader import RealDataLoader


class RealMLTrainer:
    """
    Train ML models on real IED incident data.
    
    Key Features:
    - Proper temporal train-test split (no data leakage)
    - Time-series features (attack velocity, clustering)
    - District-level risk prediction
    - XGBoost + ensemble methods
    """
    
    def __init__(self):
        self.loader = RealDataLoader()
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
    
    def prepare_data(self) -> pd.DataFrame:
        """Load and prepare data with features."""
        print("\n[PREP] Loading and preparing real data...")
        
        df = self.loader.clean_data()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add spatial clustering features
        df = self._add_spatial_features(df)
        
        self.df = df
        print(f"[PREP] Prepared {len(df)} incidents with {len(self.feature_columns)} features")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-series features."""
        
        # Days since last attack
        df['days_since_last'] = df['Date'].diff().dt.days.fillna(30).clip(upper=90)
        
        # Days since last attack in same district
        df['days_since_last_district'] = df.groupby('District')['Date'].diff().dt.days.fillna(60).clip(upper=180)
        
        # Rolling attack counts (use index-based rolling for speed)
        df['attacks_last_7'] = df.set_index('Date')['Killed'].rolling('7D', min_periods=1).count().values
        df['attacks_last_30'] = df.set_index('Date')['Killed'].rolling('30D', min_periods=1).count().values
        df['attacks_last_90'] = df.set_index('Date')['Killed'].rolling('90D', min_periods=1).count().values
        
        # Attack velocity (is the tempo increasing?)
        df['attack_velocity'] = (df['attacks_last_7'] / df['attacks_last_30'].replace(0, 1)).fillna(0)
        
        # Casualty momentum
        df['casualties_last_30'] = df.set_index('Date')['Total_Casualties'].rolling('30D', min_periods=1).sum().values
        
        # Seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        
        # Is it a "hot" period (multiple recent attacks)?
        df['is_hot_period'] = (df['attacks_last_7'] >= 2).astype(int)
        
        return df
    
    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial clustering features."""
        
        # District-level historical attack rate
        district_counts = df.groupby('District').size()
        df['district_historical_rate'] = df['District'].map(district_counts)
        
        # State-level historical attack rate
        state_counts = df.groupby('State').size()
        df['state_historical_rate'] = df['State'].map(state_counts)
        
        # Is it a high-risk district? (top 5)
        high_risk_districts = district_counts.nlargest(5).index.tolist()
        df['is_high_risk_district'] = df['District'].isin(high_risk_districts).astype(int)
        
        # District casualty rate
        district_casualties = df.groupby('District')['Total_Casualties'].mean()
        df['district_casualty_rate'] = df['District'].map(district_casualties).fillna(0)
        
        return df
    
    def create_prediction_dataset(self, 
                                   prediction_window_days: int = 7,
                                   target_column: str = 'will_attack') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create dataset for predicting attacks in the next N days.
        
        For each day in the dataset, we predict whether there will be
        an attack in the next `prediction_window_days` days.
        """
        if self.df is None:
            self.prepare_data()
        
        df = self.df.copy()
        
        # For each incident date, check if there's another attack within the window
        df['next_attack_date'] = df['Date'].shift(-1)
        df['days_to_next_attack'] = (df['next_attack_date'] - df['Date']).dt.days
        
        # Target: will there be an attack within the prediction window?
        df[target_column] = (df['days_to_next_attack'] <= prediction_window_days).astype(int)
        
        # Define feature columns (excluding target and metadata)
        exclude_cols = [
            'Date', 'Time', 'State', 'District', 'Location', 'Description',
            'Killed_Details', 'Injured_Details', 'Other_Details',
            'next_attack_date', 'days_to_next_attack', target_column
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"[DATA] Created prediction dataset with {len(self.feature_columns)} features")
        print(f"[DATA] Positive samples: {df[target_column].sum()} ({100*df[target_column].mean():.1f}%)")
        
        X = df[self.feature_columns].fillna(0)
        y = df[target_column]
        
        return X, y, df
    
    def train_model(self,
                    train_end_date: str,
                    prediction_window_days: int = 7) -> Dict:
        """
        Train model with proper temporal split.
        
        Args:
            train_end_date: Last date to include in training
            prediction_window_days: Predict attacks within this many days
        
        Returns:
            Dictionary with model metrics and feature importance
        """
        print(f"\n[TRAIN] Training model with data up to {train_end_date}")
        print(f"[TRAIN] Prediction window: {prediction_window_days} days")
        
        X, y, df = self.create_prediction_dataset(prediction_window_days)
        
        # Temporal split
        train_end = pd.to_datetime(train_end_date)
        train_mask = df['Date'] <= train_end
        test_mask = df['Date'] > train_end
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"[TRAIN] Training samples: {len(X_train)}")
        print(f"[TRAIN] Test samples: {len(X_test)}")
        
        if len(X_test) == 0:
            print("[WARNING] No test samples! Adjust train_end_date.")
            return {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if HAS_XGB:
            print("[TRAIN] Using XGBoost classifier")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=len(y_train) / (y_train.sum() + 1),  # Handle imbalance
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        elif HAS_SKLEARN_ENSEMBLE:
            print("[TRAIN] Using Gradient Boosting classifier")
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ImportError("No suitable ML library found!")
        
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            importance = pd.DataFrame()
        
        # Print results
        print(f"\n[RESULTS] Model Performance:")
        print(f"  Accuracy:  {self.metrics['accuracy']:.3f}")
        print(f"  Precision: {self.metrics['precision']:.3f}")
        print(f"  Recall:    {self.metrics['recall']:.3f}")
        print(f"  F1 Score:  {self.metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {self.metrics['auc_roc']:.3f}")
        
        if len(importance) > 0:
            print(f"\n[FEATURES] Top 10 Most Important:")
            for _, row in importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'metrics': self.metrics,
            'feature_importance': importance,
            'test_predictions': pd.DataFrame({
                'date': df[test_mask]['Date'].values,
                'district': df[test_mask]['District'].values,
                'actual': y_test.values,
                'predicted': y_pred,
                'probability': y_prob
            })
        }
    
    def predict_next_attack(self, 
                            as_of_date: str = None,
                            top_n: int = 5) -> pd.DataFrame:
        """
        Predict risk of attack in the next 7 days.
        
        Args:
            as_of_date: Date to make prediction from (default: latest in data)
            top_n: Number of top risk areas to return
        """
        if self.model is None:
            print("[ERROR] Model not trained! Call train_model() first.")
            return pd.DataFrame()
        
        if as_of_date is None:
            as_of_date = self.df['Date'].max()
        else:
            as_of_date = pd.to_datetime(as_of_date)
        
        # Get recent data
        recent_df = self.df[self.df['Date'] <= as_of_date].tail(30).copy()
        
        if len(recent_df) == 0:
            print("[ERROR] No data available for prediction date!")
            return pd.DataFrame()
        
        # Use the most recent record as the "current state"
        current_state = recent_df.iloc[-1:].copy()
        
        # Scale and predict
        X = current_state[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X_scaled)[0]
        
        # Generate risk assessment
        risk_level = 'CRITICAL' if prob > 0.7 else 'HIGH' if prob > 0.5 else 'MEDIUM' if prob > 0.3 else 'LOW'
        
        # Get high-risk districts based on historical patterns
        district_risk = self.df.groupby('District').agg({
            'Killed': 'sum',
            'Injured': 'sum',
            'Is_Major_Attack': 'sum'
        }).reset_index()
        district_risk['total_incidents'] = self.df.groupby('District').size().values
        district_risk['risk_score'] = (
            district_risk['total_incidents'] * 0.4 +
            district_risk['Killed'] * 0.3 +
            district_risk['Is_Major_Attack'] * 0.3
        )
        district_risk = district_risk.sort_values('risk_score', ascending=False).head(top_n)
        
        print(f"\n[PREDICTION] As of {as_of_date.strftime('%Y-%m-%d')}:")
        print(f"  Attack probability (next 7 days): {prob*100:.1f}%")
        print(f"  Risk Level: {risk_level}")
        print(f"\n  High-Risk Districts:")
        for _, row in district_risk.iterrows():
            print(f"    - {row['District']}: {row['total_incidents']} incidents, {row['Killed']} killed")
        
        return {
            'prediction_date': as_of_date,
            'probability': prob,
            'risk_level': risk_level,
            'high_risk_districts': district_risk
        }
    
    def validate_on_known_cluster(self,
                                   cluster_start: str,
                                   cluster_end: str,
                                   train_days_before: int = 30) -> Dict:
        """
        Validate model by training on data before a known attack cluster
        and seeing if it predicts the cluster.
        
        Args:
            cluster_start: First attack date in the cluster
            cluster_end: Last attack date in the cluster
            train_days_before: Days of data to use before cluster for training
        """
        cluster_start_dt = pd.to_datetime(cluster_start)
        train_end_dt = cluster_start_dt - timedelta(days=1)
        
        print(f"\n[VALIDATE] Testing prediction of cluster: {cluster_start} to {cluster_end}")
        print(f"[VALIDATE] Training on data up to: {train_end_dt.strftime('%Y-%m-%d')}")
        
        # Train model
        results = self.train_model(
            train_end_date=train_end_dt.strftime('%Y-%m-%d'),
            prediction_window_days=7
        )
        
        if not results:
            return {}
        
        # Make prediction for the day before the cluster
        prediction = self.predict_next_attack(as_of_date=train_end_dt)
        
        return {
            'cluster': {'start': cluster_start, 'end': cluster_end},
            'train_end': train_end_dt.strftime('%Y-%m-%d'),
            'prediction': prediction,
            'model_metrics': results['metrics']
        }


def run_real_training():
    """Run the real ML training pipeline."""
    trainer = RealMLTrainer()
    
    # Prepare data
    trainer.prepare_data()
    
    # Train on data up to end of 2024, test on 2025
    print("\n" + "="*60)
    print("TRAINING: Using 2020-2024 data to predict 2025 attacks")
    print("="*60)
    
    results = trainer.train_model(
        train_end_date='2024-12-31',
        prediction_window_days=7
    )
    
    # Validate on the January 2025 cluster (Jan 6-17)
    print("\n" + "="*60)
    print("VALIDATION: January 2025 Attack Cluster")
    print("="*60)
    
    validation = trainer.validate_on_known_cluster(
        cluster_start='2025-01-06',
        cluster_end='2025-01-17'
    )
    
    return trainer, results, validation


if __name__ == "__main__":
    trainer, results, validation = run_real_training()
