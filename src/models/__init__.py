# Models module
from .models import AttackPredictor, EnsembleModel
from .explainer import SHAPExplainer

# LSTM Deep Learning
try:
    from .lstm_predictor import LSTMAttackPredictor, EnsemblePredictor
except ImportError:
    pass  # TensorFlow not installed
