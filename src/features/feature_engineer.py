"""
Feature Engineering Module

Extracts location-agnostic features that enable transfer learning to new conflict zones.

Feature Categories:
1. SPATIAL: Grid-based density, distance metrics
2. TEMPORAL: Velocity, cycles, patterns
3. SEMANTIC: Keywords, threat language
4. NETWORK: Source reliability, corroboration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Extract location-agnostic features for attack prediction.
    
    Key innovation: Features like 'intel_density_per_grid' work in any
    conflict zone, enabling transfer learning.
    """
    
    def __init__(self, 
                 grid_size_km: float = 5.0,
                 lat_range: Tuple[float, float] = (17.5, 20.0),
                 lon_range: Tuple[float, float] = (80.0, 82.5)):
        """
        Initialize feature engineer.
        
        Args:
            grid_size_km: Size of spatial grid cells in km
            lat_range: (min_lat, max_lat) for the operational area
            lon_range: (min_lon, max_lon) for the operational area
        """
        self.grid_size_km = grid_size_km
        self.lat_range = lat_range
        self.lon_range = lon_range
        
        # Approximate km per degree at this latitude
        self.km_per_lat = 111.0  # ~111 km per degree latitude
        self.km_per_lon = 102.0  # ~102 km per degree longitude at 18°N
        
        # Grid dimensions
        self.grid_lat_step = grid_size_km / self.km_per_lat
        self.grid_lon_step = grid_size_km / self.km_per_lon
        
        # Known attacks for distance calculations (will be set during fit)
        self.known_attacks = []
        
        # Threat keywords for semantic features
        self.threat_keywords = {
            'high_threat': ['IED', 'blast', 'explosive', 'attack', 'ambush', 'bomb', 'mine', 'detonate'],
            'medium_threat': ['movement', 'suspicious', 'cadre', 'preparation', 'target', 'convoy'],
            'low_threat': ['patrol', 'routine', 'observation', 'unclear'],
            'deception_indicators': ['CRITICAL', 'URGENT', 'capture', 'weapons cache', 'top leader']
        }
    
    def _lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert latitude/longitude to grid cell indices."""
        if pd.isna(lat) or pd.isna(lon):
            return (-1, -1)
        
        grid_row = int((lat - self.lat_range[0]) / self.grid_lat_step)
        grid_col = int((lon - self.lon_range[0]) / self.grid_lon_step)
        
        return (grid_row, grid_col)
    
    def _grid_to_lat_lon(self, grid_row: int, grid_col: int) -> Tuple[float, float]:
        """Convert grid cell to center latitude/longitude."""
        lat = self.lat_range[0] + (grid_row + 0.5) * self.grid_lat_step
        lon = self.lon_range[0] + (grid_col + 0.5) * self.grid_lon_step
        return (lat, lon)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km."""
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return float('inf')
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def set_known_attacks(self, attacks: List[Dict]):
        """Set known attack locations for distance features."""
        self.known_attacks = attacks
    
    def extract_spatial_features(self, df: pd.DataFrame, target_date: datetime) -> Dict[Tuple[int, int], Dict]:
        """
        Extract spatial features for each grid cell.
        
        Returns dict: {(grid_row, grid_col): {feature_name: value}}
        """
        # Filter to target date
        date_df = df[pd.to_datetime(df['timestamp']).dt.date == target_date.date()].copy()
        
        # Assign grid cells
        date_df['grid_cell'] = date_df.apply(
            lambda row: self._lat_lon_to_grid(row.get('location_lat'), row.get('location_lon')), 
            axis=1
        )
        
        # Initialize grid features
        grid_features = defaultdict(lambda: {
            'intel_density': 0,
            'humint_count': 0,
            'sigint_count': 0,
            'patrol_count': 0,
            'geoint_count': 0,
            'high_reliability_count': 0,
            'high_urgency_count': 0,
            'threat_keyword_score': 0,
            'unique_sources': set(),
            'avg_confidence': 0,
            'distance_to_nearest_attack': float('inf'),
        })
        
        # Aggregate by grid cell
        for _, row in date_df.iterrows():
            grid = row['grid_cell']
            if grid == (-1, -1):
                continue
            
            grid_features[grid]['intel_density'] += 1
            
            # Type counts
            intel_type = row.get('type', '')
            if intel_type == 'HUMINT':
                grid_features[grid]['humint_count'] += 1
            elif intel_type == 'SIGINT':
                grid_features[grid]['sigint_count'] += 1
            elif intel_type == 'PATROL_REPORT':
                grid_features[grid]['patrol_count'] += 1
            elif intel_type == 'GEOINT':
                grid_features[grid]['geoint_count'] += 1
            
            # Quality metrics
            reliability = row.get('source_reliability', 0)
            if pd.notna(reliability) and reliability >= 7:
                grid_features[grid]['high_reliability_count'] += 1
            
            urgency = row.get('urgency', '')
            if urgency in ['HIGH', 'CRITICAL']:
                grid_features[grid]['high_urgency_count'] += 1
            
            # Unique sources
            source_id = row.get('source_id')
            if pd.notna(source_id):
                grid_features[grid]['unique_sources'].add(source_id)
            
            # Confidence
            conf = row.get('confidence_score', 0)
            if pd.notna(conf):
                # Running average
                n = grid_features[grid]['intel_density']
                old_avg = grid_features[grid]['avg_confidence']
                grid_features[grid]['avg_confidence'] = old_avg + (conf - old_avg) / n
            
            # Threat keywords
            keywords = row.get('keywords', [])
            if isinstance(keywords, str):
                keywords = keywords.strip('[]').replace("'", "").split(', ')
            if isinstance(keywords, list):
                for kw in keywords:
                    if kw in self.threat_keywords['high_threat']:
                        grid_features[grid]['threat_keyword_score'] += 3
                    elif kw in self.threat_keywords['medium_threat']:
                        grid_features[grid]['threat_keyword_score'] += 1
        
        # Convert unique_sources set to count
        for grid in grid_features:
            grid_features[grid]['source_diversity'] = len(grid_features[grid]['unique_sources'])
            del grid_features[grid]['unique_sources']
            
            # Calculate distance to nearest known attack
            grid_lat, grid_lon = self._grid_to_lat_lon(grid[0], grid[1])
            min_dist = float('inf')
            for attack in self.known_attacks:
                if attack['date'].date() < target_date.date():  # Only past attacks
                    dist = self._haversine_distance(
                        grid_lat, grid_lon,
                        attack['location']['lat'], attack['location']['lon']
                    )
                    min_dist = min(min_dist, dist)
            grid_features[grid]['distance_to_nearest_attack'] = min_dist if min_dist != float('inf') else 100
        
        return dict(grid_features)
    
    def extract_temporal_features(self, df: pd.DataFrame, target_date: datetime, lookback_days: int = 7) -> Dict:
        """
        Extract temporal features for the target date.
        
        Features:
        - Intel velocity (rate of change)
        - Days since last attack
        - Attack cycle phase
        - Silence before storm indicator
        """
        features = {}
        
        # Get data for lookback window
        start_date = target_date - timedelta(days=lookback_days)
        window_df = df[
            (pd.to_datetime(df['timestamp']).dt.date >= start_date.date()) &
            (pd.to_datetime(df['timestamp']).dt.date <= target_date.date())
        ]
        
        # Daily counts
        daily_counts = window_df.groupby(pd.to_datetime(window_df['timestamp']).dt.date).size()
        
        # Intel velocity (today vs 7-day average)
        today_count = daily_counts.get(target_date.date(), 0)
        avg_count = daily_counts.mean() if len(daily_counts) > 0 else 1
        features['intel_velocity'] = today_count / avg_count if avg_count > 0 else 1.0
        
        # High urgency velocity
        high_urgency_df = window_df[window_df['urgency'].isin(['HIGH', 'CRITICAL'])]
        daily_high_urgency = high_urgency_df.groupby(pd.to_datetime(high_urgency_df['timestamp']).dt.date).size()
        today_high = daily_high_urgency.get(target_date.date(), 0)
        avg_high = daily_high_urgency.mean() if len(daily_high_urgency) > 0 else 0.1
        features['urgency_velocity'] = today_high / avg_high if avg_high > 0 else 1.0
        
        # Days since last attack
        days_since_attack = float('inf')
        for attack in self.known_attacks:
            if attack['date'].date() < target_date.date():
                days = (target_date.date() - attack['date'].date()).days
                days_since_attack = min(days_since_attack, days)
        features['days_since_last_attack'] = days_since_attack if days_since_attack != float('inf') else 30
        
        # Attack cycle phase (0-1, where 1 = typical attack interval)
        typical_interval = 6  # Average days between attacks in this cluster
        features['attack_cycle_phase'] = min(1.0, features['days_since_last_attack'] / typical_interval)
        
        # Silence before storm: Drop in SIGINT activity
        sigint_df = window_df[window_df['type'] == 'SIGINT']
        daily_sigint = sigint_df.groupby(pd.to_datetime(sigint_df['timestamp']).dt.date).size()
        if len(daily_sigint) >= 2:
            recent_sigint = daily_sigint.iloc[-1] if len(daily_sigint) > 0 else 0
            avg_sigint = daily_sigint.iloc[:-1].mean() if len(daily_sigint) > 1 else recent_sigint
            features['silence_indicator'] = 1.0 if recent_sigint < 0.5 * avg_sigint else 0.0
        else:
            features['silence_indicator'] = 0.0
        
        # Day of week pattern (some days historically higher risk)
        features['day_of_week'] = target_date.weekday()
        features['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
        
        return features
    
    def extract_semantic_features(self, df: pd.DataFrame, target_date: datetime) -> Dict:
        """
        Extract semantic features from intelligence text.
        
        Features:
        - Threat keyword density
        - IED-specific mentions
        - Urgency sentiment
        - Cross-source corroboration
        """
        date_df = df[pd.to_datetime(df['timestamp']).dt.date == target_date.date()].copy()
        
        features = {
            'high_threat_keyword_count': 0,
            'medium_threat_keyword_count': 0,
            'ied_mention_count': 0,
            'deception_indicator_count': 0,
            'avg_report_urgency_score': 0,
        }
        
        urgency_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        urgency_sum = 0
        count = 0
        
        for _, row in date_df.iterrows():
            keywords = row.get('keywords', [])
            if isinstance(keywords, str):
                keywords = keywords.strip('[]').replace("'", "").split(', ')
            
            if isinstance(keywords, list):
                for kw in keywords:
                    kw_lower = kw.lower() if isinstance(kw, str) else ''
                    
                    if kw in self.threat_keywords['high_threat']:
                        features['high_threat_keyword_count'] += 1
                    elif kw in self.threat_keywords['medium_threat']:
                        features['medium_threat_keyword_count'] += 1
                    
                    if 'ied' in kw_lower or 'blast' in kw_lower or 'explosive' in kw_lower:
                        features['ied_mention_count'] += 1
                    
                    if kw in self.threat_keywords['deception_indicators']:
                        features['deception_indicator_count'] += 1
            
            urgency = row.get('urgency', 'LOW')
            if urgency in urgency_scores:
                urgency_sum += urgency_scores[urgency]
                count += 1
        
        if count > 0:
            features['avg_report_urgency_score'] = urgency_sum / count
        
        return features
    
    def extract_network_features(self, df: pd.DataFrame, target_date: datetime) -> Dict:
        """
        Extract source network features.
        
        Features:
        - Source reliability distribution
        - Cross-source corroboration
        - Deception risk score
        """
        date_df = df[pd.to_datetime(df['timestamp']).dt.date == target_date.date()].copy()
        
        # Filter HUMINT for source analysis
        humint_df = date_df[date_df['type'] == 'HUMINT']
        
        features = {
            'avg_source_reliability': 0,
            'high_reliability_ratio': 0,
            'source_corroboration_score': 0,
            'deception_risk_score': 0,
            'unique_source_count': 0,
        }
        
        if len(humint_df) == 0:
            return features
        
        # Source reliability
        reliabilities = humint_df['source_reliability'].dropna()
        if len(reliabilities) > 0:
            features['avg_source_reliability'] = reliabilities.mean()
            features['high_reliability_ratio'] = (reliabilities >= 7).sum() / len(reliabilities)
        
        # Unique sources
        unique_sources = humint_df['source_id'].nunique()
        features['unique_source_count'] = unique_sources
        
        # Corroboration: Multiple sources reporting similar locations
        if 'district' in humint_df.columns:
            district_counts = humint_df['district'].value_counts()
            max_corroboration = district_counts.max() if len(district_counts) > 0 else 0
            features['source_corroboration_score'] = min(1.0, max_corroboration / 5.0)  # Normalize
        
        # Deception risk: Low reliability + high urgency
        low_rel_high_urgency = humint_df[
            (humint_df['source_reliability'] <= 4) & 
            (humint_df['urgency'].isin(['HIGH', 'CRITICAL']))
        ]
        features['deception_risk_score'] = len(low_rel_high_urgency) / len(humint_df) if len(humint_df) > 0 else 0
        
        return features
    
    def extract_all_features(self, df: pd.DataFrame, target_date: datetime) -> Dict:
        """
        Extract all feature categories for a given date.
        
        Returns flattened feature dictionary.
        """
        # Temporal features (global for the day)
        temporal = self.extract_temporal_features(df, target_date)
        
        # Semantic features (global for the day)
        semantic = self.extract_semantic_features(df, target_date)
        
        # Network features (global for the day)
        network = self.extract_network_features(df, target_date)
        
        # Spatial features (per grid) - aggregate to global
        spatial_by_grid = self.extract_spatial_features(df, target_date)
        
        # Aggregate spatial features
        spatial_agg = {
            'total_intel_density': 0,
            'max_grid_density': 0,
            'high_density_grid_count': 0,
            'total_humint': 0,
            'total_sigint': 0,
            'total_patrol': 0,
            'total_geoint': 0,
            'total_high_reliability': 0,
            'total_high_urgency': 0,
            'max_threat_score': 0,
            'min_distance_to_attack': float('inf'),
            'grids_near_attack': 0,  # Grids within 20km of past attack
        }
        
        for grid, features in spatial_by_grid.items():
            spatial_agg['total_intel_density'] += features['intel_density']
            spatial_agg['max_grid_density'] = max(spatial_agg['max_grid_density'], features['intel_density'])
            spatial_agg['total_humint'] += features['humint_count']
            spatial_agg['total_sigint'] += features['sigint_count']
            spatial_agg['total_patrol'] += features['patrol_count']
            spatial_agg['total_geoint'] += features['geoint_count']
            spatial_agg['total_high_reliability'] += features['high_reliability_count']
            spatial_agg['total_high_urgency'] += features['high_urgency_count']
            spatial_agg['max_threat_score'] = max(spatial_agg['max_threat_score'], features['threat_keyword_score'])
            
            dist = features['distance_to_nearest_attack']
            if dist < spatial_agg['min_distance_to_attack']:
                spatial_agg['min_distance_to_attack'] = dist
            if dist < 20:  # Within 20km
                spatial_agg['grids_near_attack'] += 1
            
            if features['intel_density'] >= 10:  # High density threshold
                spatial_agg['high_density_grid_count'] += 1
        
        if spatial_agg['min_distance_to_attack'] == float('inf'):
            spatial_agg['min_distance_to_attack'] = 100
        
        # Combine all features
        all_features = {
            'date': target_date.date(),
            **{f'temporal_{k}': v for k, v in temporal.items()},
            **{f'semantic_{k}': v for k, v in semantic.items()},
            **{f'network_{k}': v for k, v in network.items()},
            **{f'spatial_{k}': v for k, v in spatial_agg.items()},
        }
        
        return all_features
    
    def create_feature_matrix(self, df: pd.DataFrame, 
                               start_date: datetime, 
                               end_date: datetime,
                               attacks: List[Dict] = None) -> pd.DataFrame:
        """
        Create feature matrix for date range.
        
        Args:
            df: Intelligence dataframe
            start_date: Start of feature extraction period
            end_date: End of feature extraction period
            attacks: List of known attacks for target variable
        
        Returns:
            DataFrame with one row per date, all features as columns
        """
        if attacks:
            self.set_known_attacks(attacks)
        
        all_rows = []
        current = start_date
        
        while current <= end_date:
            features = self.extract_all_features(df, current)
            
            # Add target variable: Attack in next 1-3 days?
            attack_imminent = 0
            attack_tomorrow = 0
            
            for attack in self.known_attacks:
                days_until = (attack['date'].date() - current.date()).days
                if 1 <= days_until <= 3:
                    attack_imminent = 1
                if days_until == 1:
                    attack_tomorrow = 1
            
            features['target_attack_imminent'] = attack_imminent
            features['target_attack_tomorrow'] = attack_tomorrow
            
            all_rows.append(features)
            current += timedelta(days=1)
        
        return pd.DataFrame(all_rows)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Load data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "bastar_intelligence_15k.csv"
    
    if not data_path.exists():
        print("Data file not found. Run data_generator.py first.")
        exit(1)
    
    print("Loading intelligence data...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} records")
    
    # Define attacks
    attacks = [
        {'id': 0, 'date': datetime(2025, 1, 6, 14, 20), 'location': {'lat': 18.50, 'lon': 81.00, 'district': 'Bijapur'}},
        {'id': 1, 'date': datetime(2025, 1, 12, 19, 0), 'location': {'lat': 18.15, 'lon': 81.25, 'district': 'Sukma'}},
        {'id': 2, 'date': datetime(2025, 1, 16, 10, 30), 'location': {'lat': 18.62, 'lon': 80.88, 'district': 'Bijapur'}},
        {'id': 3, 'date': datetime(2025, 1, 17, 7, 15), 'location': {'lat': 18.45, 'lon': 80.95, 'district': 'Narayanpur'}},
    ]
    
    # Create feature engineer
    fe = FeatureEngineer(grid_size_km=5.0)
    
    # Generate feature matrix
    print("\nExtracting features...")
    feature_matrix = fe.create_feature_matrix(
        df,
        start_date=datetime(2024, 12, 25),
        end_date=datetime(2025, 1, 18),
        attacks=attacks
    )
    
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Features: {list(feature_matrix.columns)}")
    
    # Save
    output_path = project_root / "data" / "feature_matrix.csv"
    feature_matrix.to_csv(output_path, index=False)
    print(f"\n✓ Feature matrix saved to: {output_path}")
    
    # Show sample for pre-attack day
    print("\n" + "=" * 60)
    print("FEATURES FOR JAN 15, 2025 (2 days before Attack #4)")
    print("=" * 60)
    jan15 = feature_matrix[feature_matrix['date'] == datetime(2025, 1, 15).date()]
    if len(jan15) > 0:
        for col in jan15.columns:
            print(f"  {col}: {jan15[col].values[0]}")
