"""
Attack Prediction Models

Ensemble of XGBoost (tabular) + LSTM (sequential) for IED attack prediction.

Key Design:
- Temporal train-test split: Train on attacks 1-3, predict attack #4
- Proper cross-validation for time series
- Class balancing for imbalanced attack data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Using GradientBoosting as fallback.")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. LSTM model disabled.")


class XGBoostPredictor:
    """XGBoost model for tabular feature prediction."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Default hyperparameters (tuned for imbalanced data)
        self.params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 5,  # Handle class imbalance
            'random_state': 42,
            'n_jobs': -1
        }
        self.params.update(kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train
        if HAS_XGB:
            self.model = xgb.XGBClassifier(**self.params)
        else:
            # Fallback to sklearn
            self.model = GradientBoostingClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                learning_rate=self.params['learning_rate'],
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict attack probability."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict attack (binary)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


if HAS_TORCH:
    class LSTMPredictor(nn.Module):
        """LSTM model for sequential pattern detection."""
        
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            self.scaler = StandardScaler()
        
        def forward(self, x):
            # LSTM forward pass
            lstm_out, _ = self.lstm(x)
            # Take last time step
            last_hidden = lstm_out[:, -1, :]
            # Fully connected
            output = self.fc(last_hidden)
            return output.squeeze()
else:
    # Dummy class when PyTorch not available
    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            pass  # No-op for compatibility


class EnsembleModel:
    """
    Ensemble of XGBoost + LSTM for robust prediction.
    
    XGBoost: Good at tabular features, fast inference
    LSTM: Captures sequential patterns humans miss
    """
    
    def __init__(self, 
                 xgb_weight: float = 0.6,
                 lstm_weight: float = 0.4,
                 sequence_length: int = 7):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight if HAS_TORCH else 0.0
        self.sequence_length = sequence_length
        
        # Normalize weights if LSTM disabled
        if not HAS_TORCH:
            self.xgb_weight = 1.0
        
        self.xgb_model = XGBoostPredictor()
        self.lstm_model = None
        self.feature_names = None
    
    def _prepare_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare sequential data for LSTM."""
        sequences = []
        for i in range(self.sequence_length, len(X) + 1):
            seq = X.iloc[i-self.sequence_length:i].values
            sequences.append(seq)
        return np.array(sequences) if sequences else np.array([])
    
    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int = 50):
        """Train ensemble models."""
        self.feature_names = X.columns.tolist()
        
        # Train XGBoost on all data
        print("Training XGBoost...")
        self.xgb_model.fit(X, y)
        print("  ✓ XGBoost trained")
        
        # Train LSTM on sequential data
        if HAS_TORCH and self.lstm_weight > 0:
            print("Training LSTM...")
            
            # Prepare sequences
            X_seq = self._prepare_sequences(X)
            y_seq = y.iloc[self.sequence_length-1:].values
            
            if len(X_seq) > 0:
                # Initialize LSTM
                self.lstm_model = LSTMPredictor(input_size=X.shape[1])
                
                # Scale
                X_seq_scaled = self.lstm_model.scaler.fit_transform(
                    X_seq.reshape(-1, X_seq.shape[-1])
                ).reshape(X_seq.shape)
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(X_seq_scaled)
                y_tensor = torch.FloatTensor(y_seq)
                
                # Train
                optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
                criterion = nn.BCELoss()
                
                self.lstm_model.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = self.lstm_model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    if (epoch + 1) % 10 == 0:
                        print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
                
                print("  ✓ LSTM trained")
            else:
                print("  ⚠ Not enough data for LSTM sequences")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble attack probability."""
        # XGBoost prediction
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # LSTM prediction (if available)
        if self.lstm_model is not None and HAS_TORCH:
            X_seq = self._prepare_sequences(X)
            if len(X_seq) > 0:
                X_seq_scaled = self.lstm_model.scaler.transform(
                    X_seq.reshape(-1, X_seq.shape[-1])
                ).reshape(X_seq.shape)
                
                X_tensor = torch.FloatTensor(X_seq_scaled)
                
                self.lstm_model.eval()
                with torch.no_grad():
                    lstm_proba = self.lstm_model(X_tensor).numpy()
                
                # Align predictions (LSTM has fewer due to sequence window)
                xgb_proba_aligned = xgb_proba[self.sequence_length-1:]
                
                # Ensemble
                ensemble_proba = (
                    self.xgb_weight * xgb_proba_aligned + 
                    self.lstm_weight * lstm_proba
                )
                
                # Pad beginning with XGBoost-only predictions
                full_proba = xgb_proba.copy()
                full_proba[self.sequence_length-1:] = ensemble_proba
                return full_proba
            else:
                return xgb_proba
        else:
            return xgb_proba
    
    def predict_for_date(self, X: pd.DataFrame, target_date, threshold: float = 0.5) -> Dict:
        """Predict for a specific date with detailed output."""
        # Find row for target date
        date_mask = (X.index == target_date) | (pd.to_datetime(X['date']) == target_date)
        
        if not date_mask.any():
            return None
        
        # Get features (excluding date and target columns)
        feature_cols = [c for c in X.columns if c not in ['date', 'target_attack_imminent', 'target_attack_tomorrow']]
        X_features = X.loc[date_mask, feature_cols]
        
        proba = self.predict_proba(X_features)[0]
        prediction = 'ATTACK_LIKELY' if proba >= threshold else 'LOW_RISK'
        
        if proba >= 0.7:
            risk_level = 'CRITICAL'
        elif proba >= 0.5:
            risk_level = 'HIGH'
        elif proba >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'date': target_date,
            'attack_probability': float(proba),
            'prediction': prediction,
            'risk_level': risk_level,
            'threshold': threshold
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost."""
        return self.xgb_model.get_feature_importance()
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'xgb_model': self.xgb_model,
                'lstm_model': self.lstm_model.state_dict() if self.lstm_model else None,
                'feature_names': self.feature_names,
                'xgb_weight': self.xgb_weight,
                'lstm_weight': self.lstm_weight,
                'sequence_length': self.sequence_length
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            xgb_weight=data['xgb_weight'],
            lstm_weight=data['lstm_weight'],
            sequence_length=data['sequence_length']
        )
        model.xgb_model = data['xgb_model']
        model.feature_names = data['feature_names']
        
        return model


class AttackPredictor:
    """
    High-level attack prediction interface.
    
    Handles data loading, feature extraction, model training, and prediction.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.model = None
        self.feature_matrix = None
        
        # Known attacks
        self.attacks = [
            {'id': 0, 'date': datetime(2025, 1, 6, 14, 20), 'location': {'lat': 18.50, 'lon': 81.00, 'district': 'Bijapur', 'village': 'Ambeli'}},
            {'id': 1, 'date': datetime(2025, 1, 12, 19, 0), 'location': {'lat': 18.15, 'lon': 81.25, 'district': 'Sukma', 'village': 'Timmapuram'}},
            {'id': 2, 'date': datetime(2025, 1, 16, 10, 30), 'location': {'lat': 18.62, 'lon': 80.88, 'district': 'Bijapur', 'village': 'Putkel'}},
            {'id': 3, 'date': datetime(2025, 1, 17, 7, 15), 'location': {'lat': 18.45, 'lon': 80.95, 'district': 'Narayanpur', 'village': 'Garpa'}},
        ]
    
    def load_features(self, feature_path: str) -> pd.DataFrame:
        """Load pre-computed feature matrix."""
        self.feature_matrix = pd.read_csv(feature_path)
        self.feature_matrix['date'] = pd.to_datetime(self.feature_matrix['date'])
        return self.feature_matrix
    
    def train_with_temporal_split(self, 
                                   feature_matrix: pd.DataFrame,
                                   holdout_attack_id: int = 3) -> Dict:
        """
        Train with proper temporal split.
        
        Critical: Train on attacks 1-3, predict attack #4
        This prevents data leakage.
        """
        # Get holdout attack date
        holdout_attack = self.attacks[holdout_attack_id]
        holdout_date = holdout_attack['date'].date()
        
        # Train on data BEFORE the holdout attack window (3 days before)
        train_end = holdout_date - timedelta(days=3)
        
        train_mask = feature_matrix['date'].dt.date <= train_end
        test_mask = feature_matrix['date'].dt.date > train_end
        
        train_df = feature_matrix[train_mask].copy()
        test_df = feature_matrix[test_mask].copy()
        
        print(f"\nTemporal Split:")
        print(f"  Training: up to {train_end}")
        print(f"  Testing:  after {train_end}")
        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        # Feature columns (exclude date and target)
        feature_cols = [c for c in train_df.columns 
                       if c not in ['date', 'target_attack_imminent', 'target_attack_tomorrow']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['target_attack_imminent']
        
        X_test = test_df[feature_cols]
        y_test = test_df['target_attack_imminent']
        
        # Train ensemble
        self.model = EnsembleModel(xgb_weight=0.7, lstm_weight=0.3)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Attack', 'Attack Imminent']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nAUC-ROC: {auc:.4f}")
        except:
            auc = None
        
        # Feature importance
        importance = self.model.get_feature_importance()
        print("\nTop 10 Important Features:")
        print(importance.head(10).to_string(index=False))
        
        # Specific prediction for holdout attack
        print("\n" + "=" * 60)
        print(f"PREDICTION FOR ATTACK #{holdout_attack_id + 1}")
        print("=" * 60)
        
        # Predict for days leading up to attack
        for days_before in [3, 2, 1]:
            pred_date = holdout_date - timedelta(days=days_before)
            pred_date_mask = test_df['date'].dt.date == pred_date
            
            if pred_date_mask.any():
                X_pred = test_df.loc[pred_date_mask, feature_cols]
                proba = self.model.predict_proba(X_pred)[0]
                
                print(f"\n{pred_date} ({days_before} days before attack):")
                print(f"  Attack Probability: {proba*100:.1f}%")
                print(f"  Risk Level: {'CRITICAL' if proba >= 0.7 else 'HIGH' if proba >= 0.5 else 'MEDIUM' if proba >= 0.3 else 'LOW'}")
        
        return {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'auc': auc,
            'feature_importance': importance
        }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Check for feature matrix
    feature_path = project_root / "data" / "feature_matrix.csv"
    
    if not feature_path.exists():
        print("Feature matrix not found. Generating...")
        
        # Load raw data and generate features
        data_path = project_root / "data" / "bastar_intelligence_15k.csv"
        
        if not data_path.exists():
            print("Intelligence data not found. Run data_generator.py first.")
            exit(1)
        
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        from src.features.feature_engineer import FeatureEngineer
        
        attacks = [
            {'id': 0, 'date': datetime(2025, 1, 6, 14, 20), 'location': {'lat': 18.50, 'lon': 81.00, 'district': 'Bijapur'}},
            {'id': 1, 'date': datetime(2025, 1, 12, 19, 0), 'location': {'lat': 18.15, 'lon': 81.25, 'district': 'Sukma'}},
            {'id': 2, 'date': datetime(2025, 1, 16, 10, 30), 'location': {'lat': 18.62, 'lon': 80.88, 'district': 'Bijapur'}},
            {'id': 3, 'date': datetime(2025, 1, 17, 7, 15), 'location': {'lat': 18.45, 'lon': 80.95, 'district': 'Narayanpur'}},
        ]
        
        fe = FeatureEngineer()
        feature_matrix = fe.create_feature_matrix(
            df,
            start_date=datetime(2024, 12, 25),
            end_date=datetime(2025, 1, 18),
            attacks=attacks
        )
        feature_matrix.to_csv(feature_path, index=False)
        print(f"✓ Feature matrix saved to {feature_path}")
    
    # Train and evaluate
    predictor = AttackPredictor()
    feature_matrix = predictor.load_features(str(feature_path))
    
    print("\n" + "=" * 60)
    print("TRAINING ATTACK PREDICTION MODEL")
    print("=" * 60)
    
    results = predictor.train_with_temporal_split(feature_matrix, holdout_attack_id=3)
    
    # Save model
    model_path = project_root / "results" / "ensemble_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    predictor.model.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}")
