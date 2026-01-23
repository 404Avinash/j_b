"""
JATAYU - LSTM Deep Learning Model
==================================
LSTM (Long Short-Term Memory) neural network for sequence-based
attack prediction from intel patterns.

Why LSTM?
- Learns temporal dependencies in intel sequences
- Remembers patterns over long time windows (7-14 days)
- Captures complex non-linear relationships
- Better at detecting gradual buildup patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Check for TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARNING] TensorFlow not installed. Run: pip install tensorflow")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LSTMAttackPredictor:
    """
    LSTM-based attack prediction model.
    
    Architecture:
    - Input: Sequence of daily intel features (e.g., last 7 days)
    - LSTM layers: Learn temporal patterns
    - Dense layers: Classification
    - Output: Attack probability for next N days
    
    Key Features:
    - Bidirectional LSTM for better context
    - Dropout for regularization
    - Early stopping to prevent overfitting
    """
    
    def __init__(self, sequence_length: int = 7, n_features: int = None):
        """
        Args:
            sequence_length: Number of days to look back (default: 7)
            n_features: Number of input features (auto-detected if None)
        """
        if not HAS_TF:
            raise ImportError("TensorFlow required! Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.history = None
        self.metrics = {}
    
    def build_model(self, n_features: int) -> Sequential:
        """
        Build LSTM architecture.
        
        Architecture:
        - Bidirectional LSTM (64 units) - learns forward and backward patterns
        - Dropout (0.3) - prevents overfitting
        - LSTM (32 units) - refines temporal features
        - Dense (16) - feature compression
        - Output (1, sigmoid) - attack probability
        """
        self.n_features = n_features
        
        model = Sequential([
            # Bidirectional LSTM - captures patterns from both directions
            Bidirectional(
                LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                name='bidirectional_lstm'
            ),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False, name='lstm_layer'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense classification layers
            Dense(16, activation='relu', name='dense_1'),
            Dropout(0.2),
            
            # Output layer - probability of attack
            Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'is_attack_day') -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert daily feature dataframe to sequences for LSTM.
        
        For each day, creates a sequence of the previous N days' features.
        Target is whether an attack occurs in the next 1-3 days.
        
        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,) - binary target
        """
        # Define feature columns
        exclude_cols = ['Date', 'is_attack_day', 'attack_within_1_day', 'attack_within_3_days', target_col]
        self.feature_columns = [col for col in df.columns 
                                if col not in exclude_cols 
                                and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        print(f"[LSTM] Using {len(self.feature_columns)} features")
        
        # Extract features and scale
        features = df[self.feature_columns].fillna(0).values
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        target = df[target_col].values
        
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"[LSTM] Created {len(X)} sequences of shape {X.shape}")
        print(f"[LSTM] Positive samples: {y.sum()} ({100*y.mean():.1f}%)")
        
        return X, y
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = 50,
              batch_size: int = 32) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional, uses 20% split if None)
            epochs: Maximum training epochs
            batch_size: Batch size
        
        Returns:
            Training history and metrics
        """
        # Build model if not built
        if self.model is None:
            self.build_model(n_features=X_train.shape[2])
        
        print(f"\n[LSTM] Training model...")
        print(f"[LSTM] Input shape: {X_train.shape}")
        print(self.model.summary())
        
        # Create validation split if not provided
        if X_val is None:
            split = int(0.8 * len(X_train))
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]
        
        # Class weights for imbalanced data
        pos_weight = len(y_train) / (2 * y_train.sum() + 1)
        neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()) + 1)
        class_weights = {0: neg_weight, 1: pos_weight}
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        y_pred_prob = self.model.predict(X_val).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_val, y_pred_prob) if len(np.unique(y_val)) > 1 else 0.5
        }
        
        print(f"\n[LSTM RESULTS] Model Performance:")
        print(f"  Accuracy:  {self.metrics['accuracy']:.3f}")
        print(f"  Precision: {self.metrics['precision']:.3f}")
        print(f"  Recall:    {self.metrics['recall']:.3f}")
        print(f"  F1 Score:  {self.metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {self.metrics['auc_roc']:.3f}")
        
        return {
            'history': self.history.history,
            'metrics': self.metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict attack probability."""
        if self.model is None:
            raise ValueError("Model not trained!")
        return self.model.predict(X).flatten()
    
    def predict_from_intel(self, recent_intel_df: pd.DataFrame) -> Dict:
        """
        Predict attack probability from recent intel records.
        
        Args:
            recent_intel_df: DataFrame with last `sequence_length` days of intel
        
        Returns:
            Dictionary with probability and risk level
        """
        if len(recent_intel_df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data!")
        
        # Extract and scale features
        features = recent_intel_df[self.feature_columns].fillna(0).values
        features_scaled = self.scaler.transform(features)
        
        # Take last sequence_length days
        sequence = features_scaled[-self.sequence_length:]
        X = np.array([sequence])
        
        # Predict
        probability = self.predict(X)[0]
        
        # Determine risk level
        if probability > 0.8:
            risk_level = 'CRITICAL'
            recommendation = 'IMMEDIATE: Deploy maximum security, hold all patrols!'
        elif probability > 0.6:
            risk_level = 'HIGH'
            recommendation = 'Deploy mine-protected vehicles, increase surveillance.'
        elif probability > 0.4:
            risk_level = 'ELEVATED'
            recommendation = 'Enhanced route clearance, increased area domination.'
        elif probability > 0.2:
            risk_level = 'MEDIUM'
            recommendation = 'Standard precautions with heightened awareness.'
        else:
            risk_level = 'LOW'
            recommendation = 'Normal operations with routine security.'
        
        return {
            'probability': float(probability),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'confidence': 'HIGH' if probability > 0.7 or probability < 0.3 else 'MEDIUM'
        }
    
    def save_model(self, path: str = 'models/lstm_attack_predictor.h5'):
        """Save trained model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[LSTM] Model saved to {path}")
    
    def load_model(self, path: str = 'models/lstm_attack_predictor.h5'):
        """Load trained model."""
        self.model = tf.keras.models.load_model(path)
        print(f"[LSTM] Model loaded from {path}")


class EnsemblePredictor:
    """
    Ensemble of XGBoost + LSTM for robust predictions.
    
    Combines:
    - XGBoost: Good at feature-based tabular data patterns
    - LSTM: Captures temporal sequence patterns
    
    Final prediction is weighted average based on validation performance.
    """
    
    def __init__(self):
        self.lstm_model = None
        self.xgb_model = None
        self.lstm_weight = 0.5
        self.xgb_weight = 0.5
    
    def set_models(self, lstm_model: LSTMAttackPredictor, xgb_model):
        """Set pre-trained models and compute weights based on AUC."""
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        
        # Weight based on AUC scores
        lstm_auc = lstm_model.metrics.get('auc_roc', 0.5)
        xgb_auc = getattr(xgb_model, 'metrics', {}).get('auc_roc', 0.5)
        
        total = lstm_auc + xgb_auc
        self.lstm_weight = lstm_auc / total if total > 0 else 0.5
        self.xgb_weight = xgb_auc / total if total > 0 else 0.5
        
        print(f"[ENSEMBLE] LSTM weight: {self.lstm_weight:.2f}, XGBoost weight: {self.xgb_weight:.2f}")
    
    def predict_ensemble(self, X_lstm: np.ndarray, X_xgb: np.ndarray) -> np.ndarray:
        """
        Combined prediction from both models.
        
        Returns weighted average of probabilities.
        """
        lstm_prob = self.lstm_model.predict(X_lstm)
        xgb_prob = self.xgb_model.predict_proba(X_xgb)[:, 1]
        
        ensemble_prob = self.lstm_weight * lstm_prob + self.xgb_weight * xgb_prob
        return ensemble_prob


def run_lstm_demo():
    """Run LSTM training demo on intel data."""
    
    print("="*70)
    print("JATAYU - LSTM Deep Learning Demo")
    print("="*70)
    
    if not HAS_TF:
        print("\n[ERROR] TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        return None, None
    
    # Try to load existing intel features
    try:
        from src.models.intel_trainer import IntelMLTrainer
        
        trainer = IntelMLTrainer()
        
        # Load or generate intel
        try:
            trainer.load_existing_intel()
        except:
            trainer.generate_intel_dataset()
        
        # Engineer features
        features_df = trainer.engineer_features()
        
        # Create LSTM model
        lstm = LSTMAttackPredictor(sequence_length=7)
        
        # Prepare sequences
        X, y = lstm.prepare_sequences(features_df, target_col='is_attack_day')
        
        # Temporal split (last 20% for testing)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train
        results = lstm.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=16)
        
        print("\n" + "="*50)
        print("LSTM vs XGBoost Comparison")
        print("="*50)
        print(f"LSTM AUC-ROC: {results['metrics']['auc_roc']:.3f}")
        
        return lstm, results
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None


if __name__ == "__main__":
    lstm, results = run_lstm_demo()
