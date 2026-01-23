"""
SHAP-based Explainability Module

Generates human-readable explanations for attack predictions.
Critical for commander trust in the system.

Example output:
    ALERT: Garpa-BSF Route - HIGH RISK (78%)
    ├─ Reason 1: 3 HUMINT sources report unusual movement
    ├─ Reason 2: Last patrol was 48 hours ago
    └─ Reason 3: 12km from Jan 6 attack site
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed. Falling back to feature importance.")


class SHAPExplainer:
    """
    Generate explanations for attack predictions using SHAP values.
    
    Makes ML predictions interpretable for military commanders.
    """
    
    def __init__(self, model=None):
        self.model = model
        self.explainer = None
        self.feature_descriptions = self._load_feature_descriptions()
    
    def _load_feature_descriptions(self) -> Dict[str, str]:
        """Human-readable descriptions for features."""
        return {
            # Temporal features
            'temporal_intel_velocity': 'Intelligence activity rate compared to 7-day average',
            'temporal_urgency_velocity': 'High-urgency report rate increase',
            'temporal_days_since_last_attack': 'Days since previous IED attack',
            'temporal_attack_cycle_phase': 'Position in typical attack cycle',
            'temporal_silence_indicator': 'Unusual drop in communications (silence before storm)',
            'temporal_day_of_week': 'Day of week pattern',
            'temporal_is_weekend': 'Weekend indicator',
            
            # Semantic features
            'semantic_high_threat_keyword_count': 'High-threat keywords (IED, blast, explosive)',
            'semantic_medium_threat_keyword_count': 'Medium-threat keywords (movement, suspicious)',
            'semantic_ied_mention_count': 'Direct IED/explosive mentions',
            'semantic_deception_indicator_count': 'Potential deception indicators',
            'semantic_avg_report_urgency_score': 'Average urgency level of reports',
            
            # Network features
            'network_avg_source_reliability': 'Average informant reliability score',
            'network_high_reliability_ratio': 'Proportion of reports from trusted sources',
            'network_source_corroboration_score': 'Multiple sources confirming same area',
            'network_deception_risk_score': 'Risk of planted false intelligence',
            'network_unique_source_count': 'Number of unique sources reporting',
            
            # Spatial features
            'spatial_total_intel_density': 'Total intelligence reports today',
            'spatial_max_grid_density': 'Highest concentration in single area',
            'spatial_high_density_grid_count': 'Number of hotspot areas',
            'spatial_total_humint': 'HUMINT reports received',
            'spatial_total_sigint': 'SIGINT intercepts received',
            'spatial_total_patrol': 'Patrol reports received',
            'spatial_total_geoint': 'Satellite/drone observations',
            'spatial_total_high_reliability': 'Reports from reliable sources',
            'spatial_total_high_urgency': 'High-urgency reports',
            'spatial_max_threat_score': 'Maximum threat score in any area',
            'spatial_min_distance_to_attack': 'Distance to nearest past attack (km)',
            'spatial_grids_near_attack': 'Areas near previous attack sites',
        }
    
    def fit(self, model, X_train: pd.DataFrame):
        """Fit the explainer on training data."""
        self.model = model
        
        if HAS_SHAP:
            # Use TreeExplainer for XGBoost/tree models
            try:
                self.explainer = shap.TreeExplainer(model.xgb_model.model)
            except:
                # Fallback to Explainer
                self.explainer = shap.Explainer(model.xgb_model.model, X_train)
        
        return self
    
    def explain(self, X: pd.DataFrame, prediction_proba: float) -> Dict:
        """
        Generate explanation for a prediction.
        
        Returns dict with:
        - risk_level: CRITICAL/HIGH/MEDIUM/LOW
        - probability: Attack probability
        - reasons: List of human-readable explanations
        - feature_contributions: SHAP values for each feature
        """
        # Risk level
        if prediction_proba >= 0.7:
            risk_level = 'CRITICAL'
        elif prediction_proba >= 0.5:
            risk_level = 'HIGH'
        elif prediction_proba >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Get SHAP values or feature importance
        if HAS_SHAP and self.explainer:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            contributions = pd.DataFrame({
                'feature': X.columns,
                'value': X.values[0] if len(X) == 1 else X.mean().values,
                'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values
            }).sort_values('shap_value', key=abs, ascending=False)
        else:
            # Fallback: Use feature importance
            importance = self.model.get_feature_importance() if self.model else pd.DataFrame()
            contributions = importance.rename(columns={'importance': 'shap_value'})
            contributions['value'] = [X[f].values[0] if f in X.columns else 0 for f in contributions['feature']]
        
        # Generate human-readable reasons
        reasons = self._generate_reasons(contributions, X)
        
        return {
            'risk_level': risk_level,
            'probability': prediction_proba,
            'reasons': reasons,
            'feature_contributions': contributions.to_dict('records')[:10]
        }
    
    def _generate_reasons(self, contributions: pd.DataFrame, X: pd.DataFrame) -> List[str]:
        """Convert feature contributions to human-readable reasons."""
        reasons = []
        
        # Get top contributing features
        top_features = contributions.head(5)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            value = row['value']
            shap_val = row.get('shap_value', 0)
            
            # Get description
            description = self.feature_descriptions.get(feature, feature)
            
            # Generate reason based on feature type and value
            reason = self._feature_to_reason(feature, value, shap_val)
            if reason:
                reasons.append(reason)
        
        return reasons[:5]  # Top 5 reasons
    
    def _feature_to_reason(self, feature: str, value: float, shap_val: float) -> Optional[str]:
        """Convert a single feature to a human-readable reason."""
        
        # Direction of contribution
        direction = 'increases' if shap_val > 0 else 'decreases'
        
        # Feature-specific interpretations
        if 'intel_velocity' in feature and value > 1.5:
            return f"Intelligence activity {value:.1f}x higher than normal"
        
        elif 'urgency_velocity' in feature and value > 1.5:
            return f"High-urgency reports {value:.1f}x above baseline"
        
        elif 'days_since_last_attack' in feature and value <= 7:
            return f"Only {int(value)} days since last attack (retaliation window)"
        
        elif 'high_threat_keyword' in feature and value >= 5:
            return f"{int(value)} high-threat keywords detected (IED, blast, explosive)"
        
        elif 'ied_mention' in feature and value >= 3:
            return f"{int(value)} direct IED/explosive mentions in intelligence"
        
        elif 'source_corroboration' in feature and value >= 0.5:
            return f"Multiple independent sources reporting same area"
        
        elif 'high_reliability_ratio' in feature and value >= 0.4:
            return f"{int(value*100)}% reports from high-reliability sources"
        
        elif 'min_distance_to_attack' in feature and value <= 20:
            return f"Activity within {value:.0f}km of previous attack site"
        
        elif 'max_grid_density' in feature and value >= 15:
            return f"Intelligence hotspot detected ({int(value)} reports in single grid)"
        
        elif 'silence_indicator' in feature and value > 0:
            return f"Unusual silence in communications (potential preparation)"
        
        elif 'total_high_urgency' in feature and value >= 10:
            return f"{int(value)} high-urgency reports today"
        
        elif 'deception_risk' in feature and value >= 0.3:
            return f"⚠️ {int(value*100)}% deception risk - verify sources"
        
        else:
            return None
    
    def generate_alert(self, 
                       date: datetime,
                       probability: float,
                       reasons: List[str],
                       location: Optional[str] = None) -> str:
        """
        Generate formatted alert text.
        
        Example:
            ═══════════════════════════════════════════════════════
            ⚠️  THREAT ALERT - 15 JAN 2025
            ═══════════════════════════════════════════════════════
            RISK LEVEL: CRITICAL (78%)
            
            KEY INDICATORS:
            ├─ Intelligence activity 2.3x higher than normal
            ├─ 3 direct IED mentions in HUMINT reports
            ├─ Multiple sources reporting Narayanpur area
            └─ Only 1 day since last attack (Jan 16)
            
            RECOMMENDED ACTIONS:
            ☐ Delay patrol on high-risk routes
            ☐ Deploy EOD sweep on Garpa-BSF corridor
            ☐ Increase UAV coverage
            ═══════════════════════════════════════════════════════
        """
        # Risk level
        if probability >= 0.7:
            risk_emoji = '🔴'
            risk_text = 'CRITICAL'
        elif probability >= 0.5:
            risk_emoji = '🟠'
            risk_text = 'HIGH'
        elif probability >= 0.3:
            risk_emoji = '🟡'
            risk_text = 'MEDIUM'
        else:
            risk_emoji = '🟢'
            risk_text = 'LOW'
        
        # Format date
        date_str = date.strftime('%d %b %Y').upper()
        
        # Build alert
        lines = [
            "═" * 60,
            f"⚠️  THREAT ALERT - {date_str}",
            "═" * 60,
            f"RISK LEVEL: {risk_text} ({probability*100:.0f}%)",
        ]
        
        if location:
            lines.append(f"LOCATION: {location}")
        
        lines.append("")
        lines.append("KEY INDICATORS:")
        
        for i, reason in enumerate(reasons):
            prefix = "├─" if i < len(reasons) - 1 else "└─"
            lines.append(f"{prefix} {reason}")
        
        if probability >= 0.5:
            lines.extend([
                "",
                "RECOMMENDED ACTIONS:",
                "☐ Delay patrol on high-risk routes",
                "☐ Deploy EOD sweep before movement",
                "☐ Increase UAV coverage",
                "☐ Cross-verify with additional sources",
            ])
        
        lines.append("═" * 60)
        
        return "\n".join(lines)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    from pathlib import Path
    import pickle
    
    project_root = Path(__file__).parent.parent.parent
    
    # Load model
    model_path = project_root / "results" / "ensemble_model.pkl"
    
    if not model_path.exists():
        print("Model not found. Run models.py first.")
        exit(1)
    
    from src.models.models import EnsembleModel
    model = EnsembleModel.load(str(model_path))
    
    # Load feature matrix
    feature_path = project_root / "data" / "feature_matrix.csv"
    features = pd.read_csv(feature_path)
    features['date'] = pd.to_datetime(features['date'])
    
    # Get features for Jan 15 (2 days before attack #4)
    jan15_mask = features['date'].dt.date == datetime(2025, 1, 15).date()
    jan15_features = features.loc[jan15_mask]
    
    feature_cols = [c for c in features.columns 
                   if c not in ['date', 'target_attack_imminent', 'target_attack_tomorrow']]
    X = jan15_features[feature_cols]
    
    # Get prediction
    proba = model.predict_proba(X)[0]
    
    # Create explainer
    explainer = SHAPExplainer(model)
    
    # Get training data for SHAP background
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
    
    print("\n" + alert)
