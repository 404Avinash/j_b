"""
JATAYU - Reverse Intel Generator
=================================
Generate realistic 500+ intelligence reports by reverse-engineering
patterns from actual IED incidents.

Concept:
1. Find attack clusters (closest incidents in same region)
2. Reverse-engineer what intel SHOULD have looked like before attacks
3. Generate HUMINT, SIGINT, PATROL, OSINT that builds up to attacks
4. Model the "signal before the noise" pattern
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
import hashlib
import os

from src.data.real_data_loader import RealDataLoader


class ReverseIntelGenerator:
    """
    Generate realistic intelligence reports by reverse-engineering
    patterns from actual IED attack data.
    
    Key Insight: Before every attack, there ARE signals - we just need
    to model what those signals would look like.
    """
    
    # Intelligence types and their characteristics
    INTEL_TYPES = {
        'HUMINT': {
            'weight': 0.35,  # 35% of intel
            'reliability_range': (0.3, 0.9),
            'noise_rate': 0.45,  # 45% are noise
            'deception_rate': 0.10,  # 10% are deception
        },
        'SIGINT': {
            'weight': 0.25,  # 25% of intel
            'reliability_range': (0.5, 0.95),
            'noise_rate': 0.35,
            'deception_rate': 0.05,
        },
        'PATROL': {
            'weight': 0.25,  # 25% of intel
            'reliability_range': (0.6, 0.9),
            'noise_rate': 0.40,
            'deception_rate': 0.03,
        },
        'OSINT': {
            'weight': 0.15,  # 15% of intel
            'reliability_range': (0.4, 0.8),
            'noise_rate': 0.50,
            'deception_rate': 0.08,
        }
    }
    
    # Pre-attack signal patterns (days before attack → signal intensity)
    PRE_ATTACK_PATTERN = {
        -10: 0.15,  # 10 days before: weak signals
        -7: 0.25,   # 7 days before: building up
        -5: 0.40,   # 5 days before: moderate signals
        -3: 0.60,   # 3 days before: strong signals
        -2: 0.75,   # 2 days before: very strong
        -1: 0.85,   # 1 day before: peak signals
        0: 0.95,    # Attack day: highest signal
    }
    
    # Templates for each intel type
    HUMINT_TEMPLATES = {
        'pre_attack': [
            "Informant reports unusual movement of {count} individuals near {location}",
            "Source indicates possible IED materials being transported to {location} area",
            "Asset reports meeting of {count} suspected cadres near {location}",
            "Reliable source warns of planned attack on {target_type} in {location}",
            "Informant observed reconnaissance activity near {location} route",
            "Source reports increased Maoist presence in {location} forest area",
            "Asset indicates possible IED placement planned for {location} road",
            "Contact reports unusual night movements in {location} village",
        ],
        'noise': [
            "Unconfirmed report of strangers seen near {location}",
            "Villager reports hearing sounds in forest near {location}",
            "Informant mentions routine activity in {location} area",
            "Source reports no unusual activity in {location}",
            "Asset unable to confirm any suspicious movement near {location}",
        ],
        'deception': [
            "Source claims attack planned for {wrong_location} (redirect attempt)",
            "Informant reports all clear in {location} - safe to proceed",
            "Asset indicates Maoists have left {location} area permanently",
            "Contact claims dismantled IED found in {wrong_location} (false)",
        ]
    }
    
    SIGINT_TEMPLATES = {
        'pre_attack': [
            "Intercepted communication mentions 'package' delivery to {location}",
            "Increased radio chatter detected in {location} sector",
            "Voice intercept references 'action' on {date_ref}",
            "Communication pattern shows coordination between {location} groups",
            "Intercepted message mentions security force patrol schedule",
            "Signal intelligence indicates command coordination for {location}",
            "Communication burst detected from {location} forest coordinates",
            "Encrypted traffic spike from known Maoist frequency in {location}",
        ],
        'noise': [
            "Routine communication intercepted from {location} area",
            "No significant signal activity in {location} sector",
            "Standard civilian traffic patterns in {location}",
        ],
        'deception': [
            "Deliberate disinformation broadcast about {wrong_location}",
            "Suspected false flag communication mentioning {wrong_location}",
        ]
    }
    
    PATROL_TEMPLATES = {
        'pre_attack': [
            "Patrol reports fresh tracks of {count} individuals near {location}",
            "Route reconnaissance found disturbed soil on {location} road",
            "Team observed suspicious digging activity near {location}",
            "Patrol noted absence of usual villagers on {location} route",
            "Report of wire/cable remnants found near {location}",
            "Unusual vehicle markings observed on {location} road",
            "Patrol found recently cut vegetation near {location} path",
            "Team reports hostile atmosphere from villagers near {location}",
        ],
        'noise': [
            "Routine patrol in {location} - no incidents",
            "Area domination in {location} sector completed without incident",
            "Road opening party cleared {location} route - normal",
        ],
        'deception': [
            "Patrol reports area is clear near {location} (compromised info)",
        ]
    }
    
    OSINT_TEMPLATES = {
        'pre_attack': [
            "Social media post warns travelers to avoid {location} area",
            "Local news reports increased bandh calls in {location} region",
            "Twitter mentions Maoist movement in {location} district",
            "WhatsApp forward warns of 'danger' on {location} road",
            "News reports villagers fleeing {location} area",
            "Social media shows photos of pamphleteering in {location}",
        ],
        'noise': [
            "General news about development activities in {location}",
            "Social media discusses local politics in {location}",
            "Routine weather updates for {location} region",
        ],
        'deception': [
            "Fake news spreading about security operations in {wrong_location}",
        ]
    }
    
    # Location-specific details
    LOCATION_DETAILS = {
        'Bijapur': {
            'villages': ['Kutru', 'Basaguda', 'Gangaloor', 'Mirtur', 'Usoor', 'Tarrem', 'Farsegarh', 'Pegdapalli'],
            'routes': ['Bijapur-Bhopalpatnam NH', 'Kutru-Farsegarh Road', 'Basaguda-Mirtur Road', 'Gangaloor Route'],
            'terrain': ['forest', 'hilly', 'riverine', 'agricultural'],
        },
        'Narayanpur': {
            'villages': ['Orchha', 'Sonpur', 'Kohkameta', 'Dhanora', 'Chhotedongar', 'Abujhmad'],
            'routes': ['Narayanpur-Kurusnar Road', 'Orchha-Dhanora Route', 'Kohkameta Road'],
            'terrain': ['dense forest', 'Abujhmad jungles', 'hilly'],
        },
        'Sukma': {
            'villages': ['Chintalnar', 'Kistaram', 'Burkapal', 'Minpa', 'Timmapuram'],
            'routes': ['Sukma-Dornapal Road', 'Chintalnar Route', 'Kistaram Road'],
            'terrain': ['dense forest', 'riverine', 'remote'],
        },
        'Dantewada': {
            'villages': ['Aranpur', 'Katekalyan', 'Barsur', 'Geedam', 'Bhanupratappur'],
            'routes': ['Aranpur Road', 'Dantewada-Bijapur Route', 'National Highway'],
            'terrain': ['forest', 'semi-urban', 'agricultural'],
        },
        'West Singhbhum': {
            'villages': ['Chaibasa', 'Goilkera', 'Saranda', 'Jareikela', 'Tonto'],
            'routes': ['Chaibasa-Saranda Route', 'Jareikela Forest Road', 'Kolhan Route'],
            'terrain': ['Saranda forest', 'hilly', 'mining area'],
        }
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the generator."""
        random.seed(seed)
        np.random.seed(seed)
        
        self.loader = RealDataLoader()
        self.incidents_df = None
        self.clusters = []
        
    def load_incidents(self) -> pd.DataFrame:
        """Load real incident data."""
        self.incidents_df = self.loader.clean_data()
        return self.incidents_df
    
    def find_attack_clusters(self, 
                             region: str = None,
                             max_gap_days: int = 7) -> List[Dict]:
        """
        Find clusters of attacks in the same region with shortest time gaps.
        
        Returns list of clusters, each containing consecutive attacks.
        """
        if self.incidents_df is None:
            self.load_incidents()
        
        df = self.incidents_df.copy()
        
        # Filter by region if specified
        if region:
            df = df[df['District'] == region]
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        clusters = []
        current_cluster = []
        
        for i, row in df.iterrows():
            if not current_cluster:
                current_cluster = [row.to_dict()]
            else:
                last = current_cluster[-1]
                days_gap = (row['Date'] - pd.to_datetime(last['Date'])).days
                same_region = row['District'] == last['District']
                
                if days_gap <= max_gap_days and same_region:
                    current_cluster.append(row.to_dict())
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(self._summarize_cluster(current_cluster))
                    current_cluster = [row.to_dict()]
        
        # Don't forget last cluster
        if len(current_cluster) >= 2:
            clusters.append(self._summarize_cluster(current_cluster))
        
        # Sort by severity (total casualties)
        clusters = sorted(clusters, key=lambda x: -x['total_casualties'])
        
        self.clusters = clusters
        print(f"[INTEL] Found {len(clusters)} attack clusters")
        
        return clusters
    
    def _summarize_cluster(self, attacks: List[Dict]) -> Dict:
        """Summarize an attack cluster."""
        return {
            'attacks': attacks,
            'district': attacks[0]['District'],
            'start_date': attacks[0]['Date'],
            'end_date': attacks[-1]['Date'],
            'num_attacks': len(attacks),
            'total_killed': sum(a['Killed'] for a in attacks),
            'total_injured': sum(a['Injured'] for a in attacks),
            'total_casualties': sum(a['Killed'] + a['Injured'] for a in attacks),
            'gap_days': [(attacks[i+1]['Date'] - attacks[i]['Date']).days 
                        for i in range(len(attacks)-1)]
        }
    
    def generate_intel_for_cluster(self,
                                    cluster: Dict,
                                    intel_per_day: int = 50,
                                    pre_days: int = 10,
                                    post_days: int = 3) -> pd.DataFrame:
        """
        Generate realistic intelligence reports for an attack cluster.
        
        Args:
            cluster: Attack cluster dict from find_attack_clusters()
            intel_per_day: Base number of intel reports per day (~500+ over period)
            pre_days: Days before first attack to generate intel
            post_days: Days after last attack
        
        Returns:
            DataFrame of generated intelligence reports
        """
        district = cluster['district']
        start_date = pd.to_datetime(cluster['start_date']) - timedelta(days=pre_days)
        end_date = pd.to_datetime(cluster['end_date']) + timedelta(days=post_days)
        
        attacks = cluster['attacks']
        attack_dates = [pd.to_datetime(a['Date']) for a in attacks]
        
        print(f"[INTEL] Generating intel for {district} cluster:")
        print(f"        Attacks: {cluster['num_attacks']} ({cluster['start_date'].strftime('%Y-%m-%d')} to {cluster['end_date'].strftime('%Y-%m-%d')})")
        print(f"        Intel period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        intel_records = []
        current_date = start_date
        
        while current_date <= end_date:
            # Calculate signal intensity based on proximity to attacks
            signal_intensity = self._calculate_signal_intensity(current_date, attack_dates)
            
            # Adjust intel volume based on signal
            day_intel_count = int(intel_per_day * (0.7 + 0.6 * signal_intensity))
            
            # Generate intel for this day
            day_records = self._generate_day_intel(
                current_date, 
                district, 
                signal_intensity,
                day_intel_count,
                attack_dates
            )
            
            intel_records.extend(day_records)
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(intel_records)
        
        print(f"[INTEL] Generated {len(df)} intelligence reports")
        print(f"        True signals: {(df['Label'] == 'TRUE_SIGNAL').sum()}")
        print(f"        Noise: {(df['Label'] == 'NOISE').sum()}")
        print(f"        Deception: {(df['Label'] == 'DECEPTION').sum()}")
        
        return df
    
    def _calculate_signal_intensity(self, 
                                     current_date: datetime,
                                     attack_dates: List[datetime]) -> float:
        """
        Calculate signal intensity based on proximity to attacks.
        
        Signal builds up before attacks and spikes on attack days.
        """
        min_distance = float('inf')
        
        for attack_date in attack_dates:
            days_diff = (attack_date - current_date).days
            
            # Only consider upcoming attacks (not past)
            if days_diff >= 0:
                min_distance = min(min_distance, days_diff)
        
        # Map distance to intensity using pre-attack pattern
        if min_distance == 0:
            return 0.95  # Attack day
        elif min_distance == 1:
            return 0.85
        elif min_distance == 2:
            return 0.75
        elif min_distance == 3:
            return 0.60
        elif min_distance <= 5:
            return 0.40
        elif min_distance <= 7:
            return 0.25
        elif min_distance <= 10:
            return 0.15
        else:
            return 0.05  # Far from any attack
    
    def _generate_day_intel(self,
                            date: datetime,
                            district: str,
                            signal_intensity: float,
                            count: int,
                            attack_dates: List[datetime]) -> List[Dict]:
        """Generate intelligence reports for a single day."""
        
        records = []
        location_info = self.LOCATION_DETAILS.get(district, self.LOCATION_DETAILS['Bijapur'])
        
        # Check if it's an attack day
        is_attack_day = any((date.date() == ad.date()) for ad in attack_dates)
        
        for i in range(count):
            # Select intel type based on weights
            intel_type = self._select_intel_type()
            type_config = self.INTEL_TYPES[intel_type]
            
            # Determine if this is signal, noise, or deception
            roll = random.random()
            
            # Higher signal intensity = more true signals
            adjusted_noise_rate = type_config['noise_rate'] * (1 - signal_intensity * 0.5)
            adjusted_deception_rate = type_config['deception_rate']
            
            if roll < (1 - adjusted_noise_rate - adjusted_deception_rate) * signal_intensity:
                label = 'TRUE_SIGNAL'
                template_type = 'pre_attack'
            elif roll < (1 - adjusted_deception_rate):
                label = 'NOISE'
                template_type = 'noise'
            else:
                label = 'DECEPTION'
                template_type = 'deception'
            
            # Generate content
            content = self._generate_content(intel_type, template_type, location_info, date, attack_dates)
            
            # Generate reliability score
            if label == 'TRUE_SIGNAL':
                reliability = random.uniform(0.6, 0.95)
            elif label == 'DECEPTION':
                reliability = random.uniform(0.4, 0.8)  # Deception often looks reliable
            else:
                reliability = random.uniform(0.2, 0.7)
            
            # Generate urgency
            if is_attack_day or signal_intensity > 0.7:
                urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.6, 0.3, 0.1])[0]
            elif signal_intensity > 0.4:
                urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.3, 0.5, 0.2])[0]
            else:
                urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.1, 0.3, 0.6])[0]
            
            # Generate source ID
            source_id = self._generate_source_id(intel_type, district)
            
            # Create timestamp
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            timestamp = date.replace(hour=hour, minute=minute)
            
            records.append({
                'Timestamp': timestamp,
                'Date': date.date(),
                'Intel_Type': intel_type,
                'Source_ID': source_id,
                'District': district,
                'Location': random.choice(location_info['villages']),
                'Content': content,
                'Reliability': round(reliability, 2),
                'Urgency': urgency,
                'Signal_Intensity': round(signal_intensity, 2),
                'Label': label,
                'Is_Attack_Day': is_attack_day,
            })
        
        return records
    
    def _select_intel_type(self) -> str:
        """Select intel type based on weights."""
        types = list(self.INTEL_TYPES.keys())
        weights = [self.INTEL_TYPES[t]['weight'] for t in types]
        return random.choices(types, weights=weights)[0]
    
    def _generate_content(self,
                          intel_type: str,
                          template_type: str,
                          location_info: Dict,
                          date: datetime,
                          attack_dates: List[datetime]) -> str:
        """Generate intel content from templates."""
        
        templates = {
            'HUMINT': self.HUMINT_TEMPLATES,
            'SIGINT': self.SIGINT_TEMPLATES,
            'PATROL': self.PATROL_TEMPLATES,
            'OSINT': self.OSINT_TEMPLATES,
        }
        
        template_list = templates[intel_type].get(template_type, templates[intel_type]['noise'])
        template = random.choice(template_list)
        
        # Fill in template variables
        content = template.format(
            location=random.choice(location_info['villages']),
            wrong_location=random.choice(['Bijapur', 'Sukma', 'Narayanpur', 'Dantewada']),
            count=random.randint(3, 15),
            target_type=random.choice(['patrol', 'convoy', 'camp', 'road opening party']),
            date_ref=random.choice(['tomorrow', 'soon', 'coming days', 'this week']),
        )
        
        return content
    
    def _generate_source_id(self, intel_type: str, district: str) -> str:
        """Generate a realistic source ID."""
        prefixes = {
            'HUMINT': ['HUM', 'SRC', 'INF', 'AST'],
            'SIGINT': ['SIG', 'INT', 'COM', 'ELT'],
            'PATROL': ['PTL', 'ROP', 'QRT', 'DOM'],
            'OSINT': ['OSI', 'SOC', 'MED', 'NEW'],
        }
        
        prefix = random.choice(prefixes[intel_type])
        district_code = district[:3].upper()
        number = random.randint(100, 999)
        
        return f"{prefix}-{district_code}-{number}"
    
    def generate_full_dataset(self,
                              target_records: int = 500,
                              region: str = 'Bijapur') -> pd.DataFrame:
        """
        Generate a full dataset of 500+ intel records for a specific region.
        
        Args:
            target_records: Minimum number of records to generate
            region: Target region/district
        
        Returns:
            DataFrame with intel records
        """
        # Find clusters in the region
        clusters = self.find_attack_clusters(region=region, max_gap_days=10)
        
        if not clusters:
            print(f"[WARN] No clusters found in {region}. Using all clusters.")
            clusters = self.find_attack_clusters(max_gap_days=10)
        
        # Select the most severe cluster
        if clusters:
            cluster = clusters[0]  # Most severe by casualties
            
            # Calculate intel per day to reach target
            total_days = (cluster['end_date'] - cluster['start_date']).days + 13  # +10 pre, +3 post
            intel_per_day = max(30, target_records // total_days)
            
            df = self.generate_intel_for_cluster(
                cluster,
                intel_per_day=intel_per_day,
                pre_days=10,
                post_days=3
            )
            
            return df
        else:
            print("[ERROR] No clusters found in data!")
            return pd.DataFrame()
    
    def save_intel(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save generated intel to CSV."""
        if filename is None:
            filename = os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'data', 'generated_intel.csv'
            )
        
        df.to_csv(filename, index=False)
        print(f"[INTEL] Saved {len(df)} records to {filename}")
        return filename


def demo_intel_generation():
    """Demonstrate intelligence generation."""
    
    print("="*70)
    print("JATAYU - Reverse Intel Generator Demo")
    print("="*70)
    
    generator = ReverseIntelGenerator()
    
    # Load incidents
    generator.load_incidents()
    
    # Find clusters
    print("\n[STEP 1] Finding attack clusters...")
    clusters = generator.find_attack_clusters(max_gap_days=7)
    
    print("\nTop 5 Most Severe Clusters:")
    for i, c in enumerate(clusters[:5], 1):
        print(f"\n  Cluster {i}: {c['district']}")
        print(f"    Period: {c['start_date'].strftime('%Y-%m-%d')} to {c['end_date'].strftime('%Y-%m-%d')}")
        print(f"    Attacks: {c['num_attacks']}, Killed: {c['total_killed']}, Injured: {c['total_injured']}")
        print(f"    Gap days between attacks: {c['gap_days']}")
    
    # Generate intel for the most severe cluster
    print("\n[STEP 2] Generating intel for most severe cluster...")
    best_cluster = clusters[0]
    
    intel_df = generator.generate_intel_for_cluster(
        best_cluster,
        intel_per_day=50,
        pre_days=10,
        post_days=3
    )
    
    # Save to file
    print("\n[STEP 3] Saving generated intel...")
    generator.save_intel(intel_df)
    
    # Show sample records
    print("\n[SAMPLE] First 10 records:")
    print(intel_df[['Date', 'Intel_Type', 'Urgency', 'Label', 'Content']].head(10).to_string())
    
    # Show pattern
    print("\n[PATTERN] Signal intensity by day:")
    daily_pattern = intel_df.groupby('Date').agg({
        'Signal_Intensity': 'mean',
        'Label': lambda x: (x == 'TRUE_SIGNAL').sum(),
        'Is_Attack_Day': 'any'
    }).reset_index()
    daily_pattern.columns = ['Date', 'Avg_Signal', 'True_Signals', 'Attack_Day']
    
    for _, row in daily_pattern.iterrows():
        marker = "** ATTACK **" if row['Attack_Day'] else ""
        bar = "#" * int(row['Avg_Signal'] * 20)
        print(f"  {row['Date']}: {bar} ({row['True_Signals']} signals) {marker}")
    
    return generator, intel_df


if __name__ == "__main__":
    generator, intel_df = demo_intel_generation()
