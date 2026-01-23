"""
JATAYU - Daily 500+ Intel Generator (Corrected)
================================================
Generate 500+ intelligence reports per day from 2020 to 2026.

Requirements:
- 500+ intel per day (every single day)
- 50% True Signals (actionable intel)
- 40% Noise (routine/false positives)
- 10% Deception (deliberate misinformation)
- Organized by: Region → Year → Month → Date
- From first 2020 blast to current 2026 data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
import os
from collections import defaultdict

from src.data.real_data_loader import RealDataLoader


class DailyIntelGenerator:
    """
    Generate 500+ intel reports per day with correct ratios:
    - 50% True Signals
    - 40% Noise
    - 10% Deception
    """
    
    # Exact ratios from problem statement
    SIGNAL_RATIO = 0.50   # 50% actionable intelligence
    NOISE_RATIO = 0.40    # 40% noise/false positives
    DECEPTION_RATIO = 0.10  # 10% deliberate deception
    
    # Intel type distribution: 35% HUMINT, 25% SIGINT, 25% PATROL, 15% OSINT
    INTEL_WEIGHTS = {
        'HUMINT': 0.35,
        'SIGINT': 0.25,
        'PATROL': 0.25,
        'OSINT': 0.15,
    }
    
    # Region data
    REGIONS = {
        'Bijapur': {
            'villages': ['Kutru', 'Basaguda', 'Gangaloor', 'Mirtur', 'Usoor', 'Tarrem', 
                        'Farsegarh', 'Pegdapalli', 'Awapalli', 'Modakpal', 'Bhopalpatnam',
                        'Madded', 'Jangla', 'Somanpalli', 'Bhejji', 'Pidia'],
            'routes': ['NH-30', 'Bijapur-Bhopalpatanam Road', 'Kutru-Farsegarh Road',
                      'Basaguda-Mirtur Road', 'Gangaloor-Usoor Route', 'Awapalli Route'],
            'lat': 18.84, 'lon': 80.77,
        },
        'Narayanpur': {
            'villages': ['Orchha', 'Sonpur', 'Kohkameta', 'Dhanora', 'Chhotedongar',
                        'Abujhmad', 'Dhaudai', 'Kanhargaon', 'Batum', 'Kurusnar', 'Becha'],
            'routes': ['Narayanpur-Kurusnar Road', 'Orchha-Dhanora Route', 
                      'Abujhmad Interior Road', 'Kohkameta Road'],
            'lat': 19.72, 'lon': 81.10,
        },
        'Sukma': {
            'villages': ['Chintalnar', 'Kistaram', 'Burkapal', 'Minpa', 'Timmapuram',
                        'Pallodi', 'Dornapal', 'Errabore', 'Chintagufa', 'Bhejji'],
            'routes': ['Sukma-Dornapal Road', 'Kistaram Route', 'Chintalnar Road'],
            'lat': 18.39, 'lon': 81.66,
        },
        'Dantewada': {
            'villages': ['Aranpur', 'Katekalyan', 'Barsur', 'Geedam', 'Bhanupratappur',
                        'Kuakonda', 'Sameli', 'Tokpal', 'Telam', 'Malewadhi'],
            'routes': ['Dantewada-Bijapur Road', 'Aranpur Route', 'Geedam Highway'],
            'lat': 18.90, 'lon': 81.35,
        },
        'West Singhbhum': {
            'villages': ['Chaibasa', 'Goilkera', 'Saranda', 'Jareikela', 'Tonto',
                        'Seraikela', 'Manoharpur', 'Jhinkpani', 'Hatiburu', 'Kolhan'],
            'routes': ['Chaibasa-Saranda Route', 'Kolhan Road', 'Jareikela Forest Road'],
            'lat': 22.36, 'lon': 85.82,
        },
        'Kanker': {
            'villages': ['Koylibeda', 'Antagarh', 'Pakhanjur', 'Bhanupratappur',
                        'Chilparas', 'Alparas', 'Rengagondi', 'Rengawahi'],
            'routes': ['Kanker-Antagarh Road', 'Koylibeda Route', 'Pakhanjur Road'],
            'lat': 20.27, 'lon': 81.49,
        },
        'Gadchiroli': {
            'villages': ['Bhamragad', 'Etapalli', 'Aheri', 'Sironcha', 'Bodmeta'],
            'routes': ['Gadchiroli-Aheri Road', 'Bhamragad Route'],
            'lat': 20.11, 'lon': 80.00,
        },
        'Lohardaga': {
            'villages': ['Serendag', 'Dundru', 'Pesra', 'Kuru', 'Bulbul'],
            'routes': ['Lohardaga-Gumla Road', 'Serendag Forest Route'],
            'lat': 23.44, 'lon': 84.68,
        },
        'Gumla': {
            'villages': ['Marwa', 'Kurumgarh', 'Bishunpur', 'Judwani', 'Ghagra'],
            'routes': ['Gumla Forest Road', 'Marwa Route', 'Bishunpur Road'],
            'lat': 23.04, 'lon': 84.54,
        },
        'Latehar': {
            'villages': ['Bulbul', 'Tarwadih', 'Peso', 'Kadapani'],
            'routes': ['Latehar Forest Road', 'Peso Route'],
            'lat': 23.74, 'lon': 84.50,
        },
    }
    
    # Intel templates
    TEMPLATES = {
        'HUMINT': {
            'signal': [
                "Informant reports {count} suspects near {village}",
                "Source warns of planned attack on {route}",
                "Asset observed IED materials at {village}",
                "Contact reports suspicious meeting at {village}",
                "Reliable source indicates activity on {route}",
                "Informant heard 'package' discussed for {village}",
                "Asset reports night movements near {village}",
                "Source confirms cadre presence at {village}",
            ],
            'noise': [
                "Villager mentions nothing unusual at {village}",
                "Source reports normal activity in {village}",
                "Contact unable to verify report about {village}",
                "Informant says quiet day at {village}",
                "Asset reports routine matters in {village}",
            ],
            'deception': [
                "Source claims attack planned for {wrong_loc} (FALSE)",
                "Informant says {village} is completely safe (TRAP)",
                "Contact reports cadres left area permanently (FALSE)",
            ],
        },
        'SIGINT': {
            'signal': [
                "Intercept mentions {village} coordinates",
                "Radio spike detected near {village}",
                "Voice intercept references {route}",
                "Encrypted burst from {village} area",
                "Communication about 'delivery' at {village}",
                "Frequency monitoring shows activity near {village}",
            ],
            'noise': [
                "Routine traffic in {village} sector",
                "No significant signals near {village}",
                "Standard civilian communication at {village}",
            ],
            'deception': [
                "False transmission about {wrong_loc} detected",
                "Decoy signal mentioning {wrong_loc}",
            ],
        },
        'PATROL': {
            'signal': [
                "Patrol found tracks of {count} persons near {village}",
                "ROP observed disturbed soil on {route}",
                "Team found wire near {village} culvert",
                "Patrol noted villagers avoiding {route}",
                "Fresh digging near {village} junction",
                "Team reports hostile reception at {village}",
            ],
            'noise': [
                "Routine patrol at {village} - clear",
                "Area domination in {village} - normal",
                "ROP cleared {route} - no issues",
            ],
            'deception': [
                "Patrol reports {route} safe (COMPROMISED)",
            ],
        },
        'OSINT': {
            'signal': [
                "WhatsApp warns about {route}",
                "News reports tension in {village}",
                "Social media shows pamphlets at {village}",
                "Twitter mentions activity near {village}",
            ],
            'noise': [
                "News about development at {village}",
                "Weather updates for {village}",
            ],
            'deception': [
                "Fake news about ops in {wrong_loc}",
            ],
        },
    }
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.loader = RealDataLoader()
        self.incidents_df = None
        self.incidents_by_region = {}
        self.incidents_by_date = {}
        
    def load_incidents(self) -> pd.DataFrame:
        """Load and organize incidents by region and date."""
        self.incidents_df = self.loader.clean_data()
        
        # Organize by region
        for district in self.incidents_df['District'].unique():
            self.incidents_by_region[district] = self.incidents_df[
                self.incidents_df['District'] == district
            ].sort_values('Date')
        
        # Organize by date
        for _, row in self.incidents_df.iterrows():
            date_key = row['Date'].date()
            if date_key not in self.incidents_by_date:
                self.incidents_by_date[date_key] = []
            self.incidents_by_date[date_key].append(row.to_dict())
        
        print(f"[DATA] Loaded {len(self.incidents_df)} incidents")
        print(f"[DATA] Regions: {len(self.incidents_by_region)}")
        print(f"[DATA] Date range: {self.incidents_df['Date'].min().date()} to {self.incidents_df['Date'].max().date()}")
        
        return self.incidents_df
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get the full date range from first to last incident."""
        start = self.incidents_df['Date'].min()
        end = self.incidents_df['Date'].max()
        return start, end
    
    def get_signal_intensity(self, 
                              date: datetime,
                              region: str) -> float:
        """
        Calculate signal intensity for a region on a given date.
        Higher intensity near attack dates.
        """
        region_incidents = self.incidents_by_region.get(region, pd.DataFrame())
        
        if len(region_incidents) == 0:
            return 0.1  # Base level for regions with no incidents
        
        # Find closest upcoming attack
        future_attacks = region_incidents[region_incidents['Date'] > date]
        past_attacks = region_incidents[region_incidents['Date'] <= date]
        
        min_days_ahead = float('inf')
        if len(future_attacks) > 0:
            next_attack = future_attacks.iloc[0]['Date']
            min_days_ahead = (next_attack - date).days
        
        # Signal intensity based on days to attack
        if min_days_ahead == 0:
            return 1.0
        elif min_days_ahead == 1:
            return 0.90
        elif min_days_ahead == 2:
            return 0.80
        elif min_days_ahead == 3:
            return 0.65
        elif min_days_ahead <= 5:
            return 0.50
        elif min_days_ahead <= 7:
            return 0.35
        elif min_days_ahead <= 14:
            return 0.20
        else:
            return 0.10  # Base level
    
    def generate_daily_intel(self,
                              date: datetime,
                              region: str,
                              daily_count: int = 500) -> List[Dict]:
        """
        Generate 500+ intel for a single day in a region.
        
        Ratios:
        - 50% True Signals
        - 40% Noise
        - 10% Deception
        """
        records = []
        region_data = self.REGIONS.get(region, list(self.REGIONS.values())[0])
        
        # Check if this is an attack day
        date_key = date.date() if hasattr(date, 'date') else date
        is_attack_day = date_key in self.incidents_by_date
        attack_in_region = False
        
        if is_attack_day:
            for attack in self.incidents_by_date[date_key]:
                if attack['District'] == region:
                    attack_in_region = True
                    break
        
        # Get signal intensity
        signal_intensity = self.get_signal_intensity(date, region)
        
        # Calculate counts based on exact ratios
        signal_count = int(daily_count * self.SIGNAL_RATIO)  # 50%
        noise_count = int(daily_count * self.NOISE_RATIO)    # 40%
        deception_count = int(daily_count * self.DECEPTION_RATIO)  # 10%
        
        # Adjust based on signal intensity (more signals closer to attacks)
        if signal_intensity > 0.5:
            # Near attack: boost signals, reduce noise
            signal_count = int(signal_count * (1 + signal_intensity * 0.3))
            noise_count = int(noise_count * (1 - signal_intensity * 0.2))
        
        # Generate each type
        for i in range(signal_count):
            records.append(self._create_intel_record(
                date, region, region_data, 'signal', signal_intensity, is_attack_day
            ))
        
        for i in range(noise_count):
            records.append(self._create_intel_record(
                date, region, region_data, 'noise', signal_intensity, is_attack_day
            ))
        
        for i in range(deception_count):
            records.append(self._create_intel_record(
                date, region, region_data, 'deception', signal_intensity, is_attack_day
            ))
        
        return records
    
    def _create_intel_record(self,
                              date: datetime,
                              region: str,
                              region_data: Dict,
                              label_type: str,
                              signal_intensity: float,
                              is_attack_day: bool) -> Dict:
        """Create a single intel record."""
        
        # Select intel type based on weights
        intel_type = random.choices(
            list(self.INTEL_WEIGHTS.keys()),
            weights=list(self.INTEL_WEIGHTS.values())
        )[0]
        
        # Get template
        templates = self.TEMPLATES[intel_type][label_type]
        template = random.choice(templates)
        
        # Fill template
        content = template.format(
            village=random.choice(region_data['villages']),
            route=random.choice(region_data['routes']),
            count=random.randint(3, 15),
            wrong_loc=random.choice(['Sukma', 'Dantewada', 'Kanker', 'Bastar', 'Jagdalpur']),
        )
        
        # Labels
        label_map = {'signal': 'TRUE_SIGNAL', 'noise': 'NOISE', 'deception': 'DECEPTION'}
        
        # Reliability
        if label_type == 'signal':
            reliability = random.uniform(0.55, 0.95)
        elif label_type == 'deception':
            reliability = random.uniform(0.40, 0.80)
        else:
            reliability = random.uniform(0.20, 0.60)
        
        # Urgency
        if signal_intensity > 0.7:
            urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.6, 0.3, 0.1])[0]
        elif signal_intensity > 0.4:
            urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.3, 0.5, 0.2])[0]
        else:
            urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.15, 0.35, 0.50])[0]
        
        # Timestamp with time
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        if hasattr(date, 'date'):
            timestamp = pd.Timestamp(date.year, date.month, date.day, hour, minute, second)
        else:
            timestamp = pd.Timestamp(date.year, date.month, date.day, hour, minute, second)
        
        # Source ID
        prefix_map = {'HUMINT': 'HUM', 'SIGINT': 'SIG', 'PATROL': 'PTL', 'OSINT': 'OSI'}
        source_id = f"{prefix_map[intel_type]}-{region[:3].upper()}-{random.randint(100,999)}"
        
        return {
            'Timestamp': timestamp,
            'Date': date.date() if hasattr(date, 'date') else date,
            'Year': timestamp.year,
            'Month': timestamp.month,
            'Day': timestamp.day,
            'Hour': hour,
            'Region': region,
            'Intel_Type': intel_type,
            'Source_ID': source_id,
            'Content': content,
            'Reliability': round(reliability, 2),
            'Urgency': urgency,
            'Signal_Intensity': round(signal_intensity, 2),
            'Label': label_map[label_type],
            'Is_Attack_Day': is_attack_day,
        }
    
    def generate_all_daily_intel(self,
                                  daily_per_region: int = 70,
                                  save: bool = True) -> Dict[int, pd.DataFrame]:
        """
        Generate 500+ intel per day for the entire date range.
        
        With ~7-8 active regions, 70 per region = ~500+ per day
        """
        if self.incidents_df is None:
            self.load_incidents()
        
        start_date, end_date = self.get_date_range()
        
        # Get active regions (those with incidents)
        active_regions = list(self.incidents_by_region.keys())
        
        print("\n" + "="*70)
        print("JATAYU - Daily 500+ Intel Generation")
        print("="*70)
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        print(f"Total Days: {(end_date - start_date).days + 1}")
        print(f"Active Regions: {len(active_regions)}")
        print(f"Intel per Region per Day: {daily_per_region}")
        print(f"Total per Day: ~{daily_per_region * len(active_regions)}")
        print(f"Ratios: 50% Signal, 40% Noise, 10% Deception")
        print("="*70)
        
        # Generate by year
        intel_by_year = defaultdict(list)
        
        current_date = start_date
        total_generated = 0
        last_year = None
        
        while current_date <= end_date:
            year = current_date.year
            
            # Progress indicator
            if year != last_year:
                if last_year is not None:
                    print(f"  Completed {last_year}: {sum(len(df) for df in intel_by_year[last_year])} records")
                print(f"\n[{year}] Generating...")
                last_year = year
            
            # Generate for each active region
            for region in active_regions:
                # Skip if region has no data near this date
                daily_records = self.generate_daily_intel(
                    current_date, region, daily_per_region
                )
                intel_by_year[year].extend(daily_records)
                total_generated += len(daily_records)
            
            current_date += timedelta(days=1)
        
        # Final year
        if last_year:
            print(f"  Completed {last_year}: {sum(len(intel_by_year[last_year]) for _ in [1])} records")
        
        # Convert to DataFrames
        result = {}
        for year, records in intel_by_year.items():
            result[year] = pd.DataFrame(records)
        
        print(f"\n[TOTAL] Generated {total_generated:,} intel records")
        
        # Save files
        if save:
            self._save_intel(result)
        
        return result
    
    def _save_intel(self, intel_by_year: Dict[int, pd.DataFrame]):
        """Save intel files organized by year."""
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'daily_intel'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[SAVING] Output: {output_dir}")
        
        # Save yearly files
        for year, df in sorted(intel_by_year.items()):
            filepath = os.path.join(output_dir, f'intel_{year}.csv')
            df.to_csv(filepath, index=False)
            
            # Stats
            signals = (df['Label'] == 'TRUE_SIGNAL').sum()
            noise = (df['Label'] == 'NOISE').sum()
            deception = (df['Label'] == 'DECEPTION').sum()
            total = len(df)
            
            print(f"  {year}: {total:,} records "
                  f"(Sig: {signals} [{100*signals/total:.0f}%], "
                  f"Noise: {noise} [{100*noise/total:.0f}%], "
                  f"Dec: {deception} [{100*deception/total:.0f}%])")
        
        # Combined file
        combined = pd.concat(intel_by_year.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, '..', 'complete_daily_intel.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined: complete_daily_intel.csv ({len(combined):,} records)")
        
        # Verify ratios
        print("\n[VERIFICATION] Overall Ratios:")
        for label in ['TRUE_SIGNAL', 'NOISE', 'DECEPTION']:
            count = (combined['Label'] == label).sum()
            pct = 100 * count / len(combined)
            expected = {'TRUE_SIGNAL': 50, 'NOISE': 40, 'DECEPTION': 10}[label]
            print(f"  {label}: {count:,} ({pct:.1f}%) - Expected: {expected}%")


def run_daily_generation():
    """Run daily 500+ intel generation."""
    
    generator = DailyIntelGenerator()
    generator.load_incidents()
    
    # Generate with ~70 intel per region per day
    # With ~7 active regions = ~500 per day
    intel_by_year = generator.generate_all_daily_intel(
        daily_per_region=70,
        save=True
    )
    
    return generator, intel_by_year


if __name__ == "__main__":
    generator, intel_by_year = run_daily_generation()
