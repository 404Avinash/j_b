"""
JATAYU - Comprehensive Intel Reverse Engineering
==================================================
Generate 500+ daily intel for EVERY gap between attacks.

Process:
1. Go through ALL incidents (2020-2026)
2. For each region, find consecutive attack pairs
3. Generate daily intel for each day in the gap
4. Show: What was received, what was missed, what caused delay

Output: Year-wise, Region-wise, Month-wise intel datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
import os
import json

from src.data.real_data_loader import RealDataLoader


class ComprehensiveIntelGenerator:
    """
    Generate comprehensive intel datasets by reverse-engineering
    all attack patterns from 2020-2026.
    """
    
    # Intel templates per type
    HUMINT_SIGNALS = [
        "Informant reports {count} unidentified persons near {location}",
        "Source indicates material movement towards {location} area",
        "Asset reports meeting of suspected cadres at {location}",
        "Reliable informant warns of planned activity on {route}",
        "Contact observed reconnaissance near {location} junction",
        "Source reports increased movement in {location} forest",
        "Informant heard discussion about 'package' at {location}",
        "Asset indicates possible IED materials at {location}",
        "Source reports night activity near {location} village",
        "Informant mentions route change of patrols discussed",
    ]
    
    HUMINT_NOISE = [
        "Villager reports nothing unusual near {location}",
        "Source confirms routine activity in {location}",
        "Contact unable to verify earlier report about {location}",
        "Asset reports area quiet near {location}",
        "Informant mentions normal market activity at {location}",
        "Source says cadres moved to different area",
        "Contact reports false alarm about {location}",
    ]
    
    HUMINT_DECEPTION = [
        "Source claims group left {location} permanently (FALSE)",
        "Informant says attack planned for {wrong_location} (MISDIRECT)",
        "Asset confirms {location} is safe for patrol (TRAP)",
        "Contact claims IED found and defused (FALSE)",
    ]
    
    SIGINT_SIGNALS = [
        "Intercepted communication mentions {location} coordinates",
        "Increased radio traffic detected in {location} sector",
        "Voice intercept references action on {date_ref}",
        "Communication spike from known frequency near {location}",
        "Encrypted message mentions 'delivery' at {location}",
        "Signal intelligence shows coordination between cells",
        "Intercept mentions patrol timing discussion",
        "Radio chatter about 'package ready' near {location}",
    ]
    
    SIGINT_NOISE = [
        "Routine communication in {location} sector",
        "No significant signal activity detected",
        "Standard civilian traffic patterns observed",
        "Normal frequency usage in {location}",
    ]
    
    SIGINT_DECEPTION = [
        "Deliberate false transmission about {wrong_location}",
        "Decoy signal detected mentioning {wrong_location}",
    ]
    
    PATROL_SIGNALS = [
        "Patrol found fresh tracks of {count} persons near {location}",
        "ROP observed disturbed soil on {route}",
        "Team found wire remnants near {location} culvert",
        "Patrol noted villagers avoiding {route}",
        "Fresh digging marks observed near {location}",
        "Team reports hostile atmosphere at {location}",
        "Patrol found recently cut vegetation on {route}",
        "Suspicious vehicle seen near {location}",
        "Dogs detected unusual scent near {location}",
    ]
    
    PATROL_NOISE = [
        "Routine patrol completed - {location} clear",
        "Area domination in {location} - no contact",
        "ROP cleared {route} without incident",
        "Night patrol in {location} - normal",
        "Team completed route reconnaissance - clear",
    ]
    
    PATROL_DECEPTION = [
        "Patrol confirms {location} route is safe (COMPROMISED)",
    ]
    
    OSINT_SIGNALS = [
        "WhatsApp forward warns about {location} road",
        "Local news reports tension in {location} area",
        "Social media shows pamphlets in {location}",
        "Villagers posting warnings about {route}",
        "News mentions increased activity in {location}",
        "Facebook post warns travelers about {location}",
    ]
    
    OSINT_NOISE = [
        "General news about {location} development",
        "Weather updates for {location} region",
        "Local politics discussed on social media",
        "Routine news from {location} district",
    ]
    
    OSINT_DECEPTION = [
        "Fake news spreading about {wrong_location}",
        "False rumor about security ops in {wrong_location}",
    ]
    
    # Region-specific locations
    REGIONS = {
        'Bijapur': {
            'villages': ['Kutru', 'Basaguda', 'Gangaloor', 'Mirtur', 'Usoor', 'Tarrem', 
                        'Farsegarh', 'Pegdapalli', 'Awapalli', 'Modakpal', 'Bhopalpatnam',
                        'Madded', 'Jangla', 'Bhejji', 'Somanpalli'],
            'routes': ['NH-30', 'Bijapur-Bhopalpatanam Road', 'Kutru-Farsegarh Road',
                      'Basaguda-Mirtur Road', 'Gangaloor-Usoor Route'],
        },
        'Narayanpur': {
            'villages': ['Orchha', 'Sonpur', 'Kohkameta', 'Dhanora', 'Chhotedongar',
                        'Abujhmad', 'Dhaudai', 'Kanhargaon', 'Batum', 'Kurusnar'],
            'routes': ['Narayanpur-Kurusnar Road', 'Orchha-Dhanora Route', 
                      'Abujhmad Interior Road'],
        },
        'Sukma': {
            'villages': ['Chintalnar', 'Kistaram', 'Burkapal', 'Minpa', 'Timmapuram',
                        'Pallodi', 'Dornapal', 'Errabore', 'Chintagufa'],
            'routes': ['Sukma-Dornapal Road', 'Kistaram Route', 'Chintalnar Road'],
        },
        'Dantewada': {
            'villages': ['Aranpur', 'Katekalyan', 'Barsur', 'Geedam', 'Bhanupratappur',
                        'Kuakonda', 'Sameli', 'Tokpal'],
            'routes': ['Dantewada-Bijapur Road', 'Aranpur Route', 'Geedam Highway'],
        },
        'West Singhbhum': {
            'villages': ['Chaibasa', 'Goilkera', 'Saranda', 'Jareikela', 'Tonto',
                        'Seraikela', 'Manoharpur', 'Jhinkpani'],
            'routes': ['Chaibasa-Saranda Route', 'Kolhan Road', 'Jareikela Forest Road'],
        },
        'Kanker': {
            'villages': ['Koylibeda', 'Antagarh', 'Pakhanjur', 'Bhanupratappur',
                        'Chilparas', 'Alparas'],
            'routes': ['Kanker-Antagarh Road', 'Koylibeda Route'],
        },
        'Lohardaga': {
            'villages': ['Serendag', 'Dundru', 'Pesra', 'Kuru'],
            'routes': ['Lohardaga-Gumla Road', 'Forest Route'],
        },
        'Gumla': {
            'villages': ['Marwa', 'Kurumgarh', 'Bishunpur', 'Judwani'],
            'routes': ['Gumla Forest Road', 'Marwa Route'],
        },
    }
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.loader = RealDataLoader()
        self.incidents_df = None
        self.all_intel = []
        
    def load_incidents(self) -> pd.DataFrame:
        """Load all real incidents."""
        self.incidents_df = self.loader.clean_data()
        print(f"[DATA] Loaded {len(self.incidents_df)} incidents")
        return self.incidents_df
    
    def get_attack_pairs_by_region(self) -> Dict[str, List[Dict]]:
        """
        Get all consecutive attack pairs organized by region.
        
        Returns: {region: [{attack1, attack2, gap_days}, ...]}
        """
        if self.incidents_df is None:
            self.load_incidents()
        
        df = self.incidents_df.sort_values('Date')
        
        pairs_by_region = {}
        
        for district in df['District'].unique():
            district_df = df[df['District'] == district].sort_values('Date')
            
            if len(district_df) < 2:
                continue
            
            pairs = []
            for i in range(len(district_df) - 1):
                attack1 = district_df.iloc[i].to_dict()
                attack2 = district_df.iloc[i + 1].to_dict()
                gap_days = (attack2['Date'] - attack1['Date']).days
                
                pairs.append({
                    'attack1': attack1,
                    'attack2': attack2,
                    'gap_days': gap_days,
                    'year': attack1['Date'].year,
                    'month': attack1['Date'].month,
                })
            
            pairs_by_region[district] = pairs
        
        return pairs_by_region
    
    def generate_daily_intel(self,
                              date: datetime,
                              district: str,
                              days_to_attack: int,
                              attack_severity: int,
                              intel_per_day: int = 500) -> List[Dict]:
        """
        Generate intel for a single day.
        
        Args:
            date: The day to generate intel for
            district: Region name
            days_to_attack: Days until next attack (0 = attack day)
            attack_severity: Casualties in the attack (affects signal strength)
            intel_per_day: Base number of intel reports
        """
        records = []
        
        # Get region-specific locations
        region_data = self.REGIONS.get(district, self.REGIONS.get('Bijapur'))
        villages = region_data['villages']
        routes = region_data['routes']
        
        # Calculate signal intensity based on proximity to attack
        if days_to_attack <= 0:
            signal_intensity = 0.95
        elif days_to_attack == 1:
            signal_intensity = 0.85
        elif days_to_attack == 2:
            signal_intensity = 0.70
        elif days_to_attack == 3:
            signal_intensity = 0.55
        elif days_to_attack <= 5:
            signal_intensity = 0.40
        elif days_to_attack <= 7:
            signal_intensity = 0.25
        else:
            signal_intensity = 0.10 + 0.05 * min(attack_severity, 5)  # Higher severity = more early signals
        
        # Ensure minimum 500+ intel per day as per problem statement
        # Base is intel_per_day, slight variation based on signal intensity
        day_count = max(500, int(intel_per_day * (0.9 + 0.2 * signal_intensity)))
        
        # Distribution: 35% HUMINT, 25% SIGINT, 25% PATROL, 15% OSINT
        type_counts = {
            'HUMINT': int(day_count * 0.35),
            'SIGINT': int(day_count * 0.25),
            'PATROL': int(day_count * 0.25),
            'OSINT': int(day_count * 0.15),
        }
        
        for intel_type, count in type_counts.items():
            for i in range(count):
                record = self._generate_single_intel(
                    date, district, intel_type, 
                    signal_intensity, villages, routes
                )
                records.append(record)
        
        return records
    
    def _generate_single_intel(self,
                                date: datetime,
                                district: str,
                                intel_type: str,
                                signal_intensity: float,
                                villages: List[str],
                                routes: List[str]) -> Dict:
        """Generate a single intel record."""
        
        # Determine if this is signal, noise, or deception
        # FIXED: Exact 50% True Signal, 40% Noise, 10% Deception as per problem statement
        roll = random.random()
        
        # Fixed thresholds: 50% TRUE, 40% NOISE, 10% DECEPTION
        # Signal intensity slightly affects TRUE signals (45-55% range) but maintains overall 50/40/10 average
        signal_threshold = 0.50  # Exactly 50% TRUE signals
        noise_threshold = 0.90   # 40% NOISE (0.50 to 0.90)
        # Remaining 10% is DECEPTION (0.90 to 1.00)
        
        if roll < signal_threshold:
            label = 'TRUE_SIGNAL'
            templates = getattr(self, f'{intel_type}_SIGNALS')
        elif roll < noise_threshold:
            label = 'NOISE'
            templates = getattr(self, f'{intel_type}_NOISE')
        else:
            label = 'DECEPTION'
            templates = getattr(self, f'{intel_type}_DECEPTION')
        
        # Select and fill template
        template = random.choice(templates)
        content = template.format(
            location=random.choice(villages),
            route=random.choice(routes),
            count=random.randint(3, 15),
            date_ref=random.choice(['tomorrow', 'soon', 'coming days', 'next week']),
            wrong_location=random.choice(['Sukma', 'Dantewada', 'Kanker', 'Bastar']),
        )
        
        # Reliability score
        if label == 'TRUE_SIGNAL':
            reliability = random.uniform(0.5, 0.95)
        elif label == 'DECEPTION':
            reliability = random.uniform(0.4, 0.85)  # Deception looks credible
        else:
            reliability = random.uniform(0.2, 0.6)
        
        # Urgency
        if signal_intensity > 0.7:
            urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.6, 0.3, 0.1])[0]
        elif signal_intensity > 0.4:
            urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.3, 0.5, 0.2])[0]
        else:
            urgency = random.choices(['HIGH', 'MEDIUM', 'LOW'], weights=[0.1, 0.3, 0.6])[0]
        
        # Timestamp
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        timestamp = pd.Timestamp(date.year, date.month, date.day, hour, minute)
        
        # Source ID
        type_prefix = {'HUMINT': 'HUM', 'SIGINT': 'SIG', 'PATROL': 'PTL', 'OSINT': 'OSI'}
        source_id = f"{type_prefix[intel_type]}-{district[:3].upper()}-{random.randint(100,999)}"
        
        return {
            'Timestamp': timestamp,
            'Date': date.date(),
            'Year': date.year,
            'Month': date.month,
            'District': district,
            'Intel_Type': intel_type,
            'Source_ID': source_id,
            'Content': content,
            'Reliability': round(reliability, 2),
            'Urgency': urgency,
            'Signal_Intensity': round(signal_intensity, 2),
            'Label': label,
            'Days_To_Attack': 0,  # Will be set later
        }
    
    def generate_intel_for_pair(self,
                                 attack1: Dict,
                                 attack2: Dict,
                                 intel_per_day: int = 45) -> pd.DataFrame:
        """
        Generate intel for all days between two attacks.
        
        This shows:
        - What intel was received leading to attack1
        - What intel came during the gap
        - What signals built up before attack2
        """
        date1 = attack1['Date']
        date2 = attack2['Date']
        district = attack1['District']
        severity1 = attack1['Killed'] + attack1['Injured']
        severity2 = attack2['Killed'] + attack2['Injured']
        
        all_records = []
        
        # Generate for 5 days before attack1 through attack2
        start_date = date1 - timedelta(days=5)
        end_date = date2
        
        current_date = start_date
        while current_date <= end_date:
            # Calculate days to next attack
            days_to_attack1 = (date1 - current_date).days
            days_to_attack2 = (date2 - current_date).days
            
            # Use whichever attack is closer
            if days_to_attack1 >= 0 and days_to_attack1 <= 3:
                days_to_attack = days_to_attack1
                attack_severity = severity1
            elif days_to_attack2 >= 0:
                days_to_attack = days_to_attack2
                attack_severity = severity2
            else:
                days_to_attack = 10  # Between attacks, lower intensity
                attack_severity = 0
            
            records = self.generate_daily_intel(
                current_date, district, days_to_attack, attack_severity, intel_per_day
            )
            
            # Update days_to_attack field
            for r in records:
                r['Days_To_Attack'] = min(days_to_attack1 if days_to_attack1 >= 0 else 999,
                                          days_to_attack2 if days_to_attack2 >= 0 else 999)
                r['Is_Attack_Day'] = (current_date.date() == date1.date() or 
                                      current_date.date() == date2.date())
            
            all_records.extend(records)
            current_date += timedelta(days=1)
        
        return pd.DataFrame(all_records)
    
    def generate_all_intel(self, 
                           intel_per_day: int = 500,
                           save_yearly: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate intel for ALL attack pairs, organized by year and region.
        
        Returns: Dictionary of DataFrames by year
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE INTEL REVERSE ENGINEERING")
        print("="*70)
        
        pairs_by_region = self.get_attack_pairs_by_region()
        
        # Organize by year
        intel_by_year = {}
        
        for district, pairs in pairs_by_region.items():
            print(f"\n[{district}] Processing {len(pairs)} attack pairs...")
            
            for pair in pairs:
                year = pair['year']
                
                # Generate intel for this pair
                intel_df = self.generate_intel_for_pair(
                    pair['attack1'], 
                    pair['attack2'],
                    intel_per_day
                )
                
                if year not in intel_by_year:
                    intel_by_year[year] = []
                
                intel_by_year[year].append(intel_df)
                
                print(f"  {pair['attack1']['Date'].strftime('%Y-%m-%d')} -> "
                      f"{pair['attack2']['Date'].strftime('%Y-%m-%d')} "
                      f"({pair['gap_days']} days): {len(intel_df)} records")
        
        # Combine by year
        combined_by_year = {}
        for year, dfs in intel_by_year.items():
            combined_by_year[year] = pd.concat(dfs, ignore_index=True)
            print(f"\n[{year}] Total: {len(combined_by_year[year])} intel records")
        
        # Save to files
        if save_yearly:
            self._save_yearly_intel(combined_by_year)
        
        return combined_by_year
    
    def _save_yearly_intel(self, intel_by_year: Dict[str, pd.DataFrame]):
        """Save intel files organized by year."""
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'intel_by_year'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[SAVE] Saving to {output_dir}")
        
        for year, df in intel_by_year.items():
            filepath = os.path.join(output_dir, f'intel_{year}.csv')
            df.to_csv(filepath, index=False)
            print(f"  Saved: intel_{year}.csv ({len(df)} records)")
        
        # Also save combined file
        combined = pd.concat(intel_by_year.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, '..', 'all_intel_2020_2026.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\n  Saved: all_intel_2020_2026.csv ({len(combined)} total records)")
    
    def generate_summary_report(self, intel_by_year: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_records': sum(len(df) for df in intel_by_year.values()),
            'by_year': {},
            'by_region': {},
            'by_type': {},
            'by_label': {},
        }
        
        combined = pd.concat(intel_by_year.values(), ignore_index=True)
        
        # By year
        for year, df in intel_by_year.items():
            summary['by_year'][year] = {
                'total': len(df),
                'true_signals': (df['Label'] == 'TRUE_SIGNAL').sum(),
                'noise': (df['Label'] == 'NOISE').sum(),
                'deception': (df['Label'] == 'DECEPTION').sum(),
            }
        
        # By region
        for district in combined['District'].unique():
            district_df = combined[combined['District'] == district]
            summary['by_region'][district] = len(district_df)
        
        # By type
        for intel_type in ['HUMINT', 'SIGINT', 'PATROL', 'OSINT']:
            summary['by_type'][intel_type] = (combined['Intel_Type'] == intel_type).sum()
        
        # By label
        for label in ['TRUE_SIGNAL', 'NOISE', 'DECEPTION']:
            summary['by_label'][label] = (combined['Label'] == label).sum()
        
        return summary


def run_comprehensive_generation():
    """Run comprehensive intel generation for all years."""
    
    print("="*70)
    print("JATAYU - Comprehensive Intel Reverse Engineering")
    print("Generating 500+ daily intel for ALL attack gaps (2020-2026)")
    print("="*70)
    
    generator = ComprehensiveIntelGenerator()
    generator.load_incidents()
    
    # Generate all intel
    intel_by_year = generator.generate_all_intel(intel_per_day=500, save_yearly=True)
    
    # Summary
    summary = generator.generate_summary_report(intel_by_year)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nTotal Intel Records: {summary['total_records']}")
    
    print("\n[BY YEAR]")
    for year, stats in sorted(summary['by_year'].items()):
        print(f"  {year}: {stats['total']:,} records "
              f"(Signals: {stats['true_signals']}, Noise: {stats['noise']}, "
              f"Deception: {stats['deception']})")
    
    print("\n[BY REGION]")
    for region, count in sorted(summary['by_region'].items(), key=lambda x: -x[1]):
        print(f"  {region}: {count:,} records")
    
    print("\n[BY TYPE]")
    for intel_type, count in summary['by_type'].items():
        pct = 100 * count / summary['total_records']
        print(f"  {intel_type}: {count:,} ({pct:.1f}%)")
    
    print("\n[BY LABEL]")
    for label, count in summary['by_label'].items():
        pct = 100 * count / summary['total_records']
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    return generator, intel_by_year, summary


if __name__ == "__main__":
    generator, intel_by_year, summary = run_comprehensive_generation()
