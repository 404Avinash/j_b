"""
Real Data Loader for JATAYU
===========================
Loads and processes actual IED incident data from the Red Corridor.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
import os


class RealDataLoader:
    """
    Load and process real IED incident data for ML training.
    """
    
    # District to approximate coordinates mapping (centroids)
    DISTRICT_COORDS = {
        # Chhattisgarh - Bastar Division
        'Bijapur': (18.8387, 80.7718),
        'Sukma': (18.3861, 81.6610),
        'Dantewada': (18.8974, 81.3457),
        'Narayanpur': (19.7217, 81.1028),
        'Bastar': (19.1071, 81.9535),
        'Kanker': (20.2720, 81.4915),
        'Kondagaon': (19.5986, 81.6636),
        
        # Jharkhand
        'West Singhbhum': (22.3631, 85.8245),
        'Lohardaga': (23.4354, 84.6836),
        'Gumla': (23.0434, 84.5426),
        'Latehar': (23.7416, 84.5021),
        'Palamu': (24.0269, 84.0528),
        
        # Odisha
        'Malkangiri': (18.3501, 81.8873),
        'Koraput': (18.8136, 82.7123),
        'Kalahandi': (19.9137, 83.1649),
        'Kandhamal': (20.4683, 84.2291),
        'Boudh-Kandhamal': (20.4683, 84.2291),
        'Sundargarh': (22.1208, 84.0436),
        
        # Maharashtra
        'Gadchiroli': (20.1052, 80.0032),
        
        # Telangana
        'Bhadradri Kothagudem': (17.5547, 80.6199),
        
        # Andhra Pradesh - approximate
        'Not specified': (18.5, 81.5),  # Default to Bastar region center
    }
    
    # Attack type keywords
    ATTACK_TYPES = {
        'pressure_ied': ['pressure ied', 'pressure plate', 'pressure-plate'],
        'vehicle_ied': ['blew up vehicle', 'blew up bus', 'blew up suv', 'vehicle with ied'],
        'landmine': ['landmine', 'land mine'],
        'ambush_ied': ['ambush', 'encounter'],
        'infrastructure': ['bridge', 'highway', 'road', 'rail track'],
        'standard_ied': ['ied', 'blast', 'explosion']
    }
    
    # Target type keywords
    TARGET_TYPES = {
        'security_forces': ['crpf', 'itbp', 'bsf', 'cobra', 'drg', 'stf', 'caf', 'ssb', 'sog', 
                           'patrol', 'trooper', 'jawan', 'constable', 'personnel', 'commando',
                           'security', 'police', 'forces', 'battalion'],
        'civilian': ['civilian', 'villager', 'woman', 'child', 'boy', 'girl', 'laborer', 
                    'worker', 'tribal', 'youth', 'man'],
        'infrastructure': ['bridge', 'road', 'highway', 'rail', 'mine']
    }
    
    def __init__(self, data_path: str = None):
        """Initialize the real data loader."""
        if data_path is None:
            # Default path relative to project
            self.data_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'data', 'raw_incidents.csv'
            )
        else:
            self.data_path = data_path
        
        self.df = None
        self.processed_df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load raw incident data from CSV."""
        print(f"[DATA] Loading real incident data from: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"[DATA] Loaded {len(self.df)} incidents")
        
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and standardize the raw data."""
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Standardize state names
        df['State'] = df['State'].str.strip()
        
        # Handle border areas
        df.loc[df['State'].str.contains('border', case=False, na=False), 'State'] = 'Multi-State Border'
        
        # Standardize district names
        df['District'] = df['District'].fillna('Not specified').str.strip()
        
        # Convert casualties to integers
        df['Killed'] = pd.to_numeric(df['Killed'], errors='coerce').fillna(0).astype(int)
        df['Injured'] = pd.to_numeric(df['Injured'], errors='coerce').fillna(0).astype(int)
        df['Total_Casualties'] = df['Killed'] + df['Injured']
        
        # Extract attack type
        df['Attack_Type'] = df['Description'].apply(self._extract_attack_type)
        
        # Extract target type
        df['Target_Type'] = df.apply(self._extract_target_type, axis=1)
        
        # Add coordinates
        df['Latitude'] = df['District'].apply(lambda x: self._get_coords(x)[0])
        df['Longitude'] = df['District'].apply(lambda x: self._get_coords(x)[1])
        
        # Add jitter to coordinates for visualization
        np.random.seed(42)
        df['Latitude'] = df['Latitude'] + np.random.uniform(-0.15, 0.15, len(df))
        df['Longitude'] = df['Longitude'] + np.random.uniform(-0.15, 0.15, len(df))
        
        # Extract year and month for analysis
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Day_of_Month'] = df['Date'].dt.day
        
        # Is it a high-casualty attack?
        df['Is_Major_Attack'] = (df['Total_Casualties'] >= 3).astype(int)
        
        # Severity score
        df['Severity'] = df['Killed'] * 2 + df['Injured']
        
        self.processed_df = df
        
        print(f"[DATA] Cleaned data: {len(df)} incidents")
        print(f"[DATA] Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"[DATA] States covered: {df['State'].nunique()}")
        print(f"[DATA] Total killed: {df['Killed'].sum()}, Total injured: {df['Injured'].sum()}")
        
        return df
    
    def _extract_attack_type(self, description: str) -> str:
        """Extract attack type from description."""
        if pd.isna(description):
            return 'unknown'
        
        desc_lower = description.lower()
        
        for attack_type, keywords in self.ATTACK_TYPES.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return attack_type
        
        return 'standard_ied'
    
    def _extract_target_type(self, row) -> str:
        """Extract target type from description and casualty details."""
        text = f"{row.get('Description', '')} {row.get('Killed_Details', '')} {row.get('Injured_Details', '')}"
        text_lower = text.lower()
        
        # Check security forces first
        for keyword in self.TARGET_TYPES['security_forces']:
            if keyword in text_lower:
                return 'security_forces'
        
        # Then civilians
        for keyword in self.TARGET_TYPES['civilian']:
            if keyword in text_lower:
                return 'civilian'
        
        # Then infrastructure
        for keyword in self.TARGET_TYPES['infrastructure']:
            if keyword in text_lower:
                return 'infrastructure'
        
        return 'unknown'
    
    def _get_coords(self, district: str) -> Tuple[float, float]:
        """Get approximate coordinates for a district."""
        district = district.strip() if isinstance(district, str) else 'Not specified'
        
        # Try exact match
        if district in self.DISTRICT_COORDS:
            return self.DISTRICT_COORDS[district]
        
        # Try partial match
        for key in self.DISTRICT_COORDS:
            if key.lower() in district.lower() or district.lower() in key.lower():
                return self.DISTRICT_COORDS[key]
        
        # Default to Bastar region center
        return self.DISTRICT_COORDS['Not specified']
    
    def get_statistics(self) -> Dict:
        """Get summary statistics of the data."""
        if self.processed_df is None:
            self.clean_data()
        
        df = self.processed_df
        
        stats = {
            'total_incidents': len(df),
            'date_range': {
                'start': str(df['Date'].min().date()),
                'end': str(df['Date'].max().date())
            },
            'by_year': df.groupby('Year').size().to_dict(),
            'by_state': df.groupby('State').size().to_dict(),
            'by_district': df.groupby('District').size().to_dict(),
            'total_killed': int(df['Killed'].sum()),
            'total_injured': int(df['Injured'].sum()),
            'major_attacks': int(df['Is_Major_Attack'].sum()),
            'by_attack_type': df.groupby('Attack_Type').size().to_dict(),
            'by_target_type': df.groupby('Target_Type').size().to_dict(),
            'monthly_avg': float(len(df) / ((df['Date'].max() - df['Date'].min()).days / 30))
        }
        
        return stats
    
    def get_temporal_features(self) -> pd.DataFrame:
        """Create temporal features for ML training."""
        if self.processed_df is None:
            self.clean_data()
        
        df = self.processed_df.copy()
        
        # Days since last attack
        df = df.sort_values('Date')
        df['Days_Since_Last'] = df['Date'].diff().dt.days.fillna(0)
        
        # Rolling attack counts
        df['Attacks_Last_7_Days'] = df.set_index('Date').rolling('7D').size().values
        df['Attacks_Last_30_Days'] = df.set_index('Date').rolling('30D').size().values
        
        # Attack velocity (acceleration/deceleration)
        df['Attack_Velocity'] = df['Days_Since_Last'].diff().fillna(0)
        
        return df
    
    def split_for_prediction(self, 
                             train_end_date: str,
                             test_start_date: str = None,
                             test_end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data for temporal prediction.
        
        Args:
            train_end_date: Last date to include in training (format: 'YYYY-MM-DD')
            test_start_date: First date of test period (optional, defaults to day after train_end)
            test_end_date: Last date of test period (optional)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.processed_df is None:
            self.clean_data()
        
        df = self.processed_df.copy()
        
        train_end = pd.to_datetime(train_end_date)
        test_start = pd.to_datetime(test_start_date) if test_start_date else train_end + timedelta(days=1)
        
        train_df = df[df['Date'] <= train_end]
        
        if test_end_date:
            test_end = pd.to_datetime(test_end_date)
            test_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
        else:
            test_df = df[df['Date'] >= test_start]
        
        print(f"[DATA] Train set: {len(train_df)} incidents (up to {train_end_date})")
        print(f"[DATA] Test set: {len(test_df)} incidents")
        
        return train_df, test_df
    
    def get_attack_clusters(self, days_threshold: int = 7) -> List[Dict]:
        """
        Identify attack clusters (multiple attacks within days_threshold).
        """
        if self.processed_df is None:
            self.clean_data()
        
        df = self.processed_df.sort_values('Date').copy()
        
        clusters = []
        current_cluster = []
        
        for idx, row in df.iterrows():
            if not current_cluster:
                current_cluster.append(row)
            else:
                days_diff = (row['Date'] - current_cluster[-1]['Date']).days
                if days_diff <= days_threshold:
                    current_cluster.append(row)
                else:
                    if len(current_cluster) >= 2:
                        clusters.append({
                            'start_date': current_cluster[0]['Date'],
                            'end_date': current_cluster[-1]['Date'],
                            'num_attacks': len(current_cluster),
                            'total_killed': sum(r['Killed'] for r in current_cluster),
                            'total_injured': sum(r['Injured'] for r in current_cluster),
                            'locations': [r['District'] for r in current_cluster],
                            'attacks': current_cluster
                        })
                    current_cluster = [row]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            clusters.append({
                'start_date': current_cluster[0]['Date'],
                'end_date': current_cluster[-1]['Date'],
                'num_attacks': len(current_cluster),
                'total_killed': sum(r['Killed'] for r in current_cluster),
                'total_injured': sum(r['Injured'] for r in current_cluster),
                'locations': [r['District'] for r in current_cluster],
                'attacks': current_cluster
            })
        
        print(f"[DATA] Found {len(clusters)} attack clusters (threshold: {days_threshold} days)")
        
        return clusters
    
    def filter_by_region(self, states: List[str] = None, districts: List[str] = None) -> pd.DataFrame:
        """Filter data by state or district."""
        if self.processed_df is None:
            self.clean_data()
        
        df = self.processed_df.copy()
        
        if states:
            df = df[df['State'].isin(states)]
        
        if districts:
            df = df[df['District'].isin(districts)]
        
        return df
    
    def export_processed_data(self, output_path: str = None):
        """Export processed data to CSV."""
        if self.processed_df is None:
            self.clean_data()
        
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(self.data_path),
                'processed_incidents.csv'
            )
        
        self.processed_df.to_csv(output_path, index=False)
        print(f"[DATA] Exported processed data to: {output_path}")


def analyze_data():
    """Quick analysis of the real data."""
    loader = RealDataLoader()
    df = loader.clean_data()
    stats = loader.get_statistics()
    
    print("\n" + "="*60)
    print("JATAYU - Real Data Analysis")
    print("="*60)
    
    print(f"\n[OVERVIEW]")
    print(f"  Total Incidents: {stats['total_incidents']}")
    print(f"  Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  Monthly Average: {stats['monthly_avg']:.1f} incidents")
    
    print(f"\n[CASUALTIES]")
    print(f"  Total Killed: {stats['total_killed']}")
    print(f"  Total Injured: {stats['total_injured']}")
    print(f"  Major Attacks (3+ casualties): {stats['major_attacks']}")
    
    print(f"\n[BY YEAR]")
    for year, count in sorted(stats['by_year'].items()):
        print(f"  {year}: {count} incidents")
    
    print(f"\n[BY STATE]")
    for state, count in sorted(stats['by_state'].items(), key=lambda x: -x[1]):
        print(f"  {state}: {count} incidents")
    
    print(f"\n[TOP DISTRICTS]")
    top_districts = sorted(stats['by_district'].items(), key=lambda x: -x[1])[:10]
    for district, count in top_districts:
        print(f"  {district}: {count} incidents")
    
    print(f"\n[ATTACK TYPES]")
    for atype, count in sorted(stats['by_attack_type'].items(), key=lambda x: -x[1]):
        print(f"  {atype}: {count}")
    
    print(f"\n[TARGET TYPES]")
    for ttype, count in sorted(stats['by_target_type'].items(), key=lambda x: -x[1]):
        print(f"  {ttype}: {count}")
    
    # Find clusters
    print(f"\n[ATTACK CLUSTERS]")
    clusters = loader.get_attack_clusters(days_threshold=7)
    for i, cluster in enumerate(clusters[:5], 1):
        print(f"  Cluster {i}: {cluster['start_date'].strftime('%Y-%m-%d')} to {cluster['end_date'].strftime('%Y-%m-%d')}")
        print(f"    Attacks: {cluster['num_attacks']}, Killed: {cluster['total_killed']}, Injured: {cluster['total_injured']}")
    
    return loader, df, stats


if __name__ == "__main__":
    analyze_data()
