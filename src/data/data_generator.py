"""
Bastar Intelligence Data Generator

Generates 15,000 realistic intelligence records for the Jan 2025 Bastar IED cluster.
Multi-INT fusion: HUMINT, SIGINT, PATROL, GEOINT, OSINT

Signal-to-noise ratio:
- 2-3% TRUE SIGNALS (actionable intelligence)
- 8% FALSE POSITIVES (looks suspicious but isn't)
- 1% DECEPTION (planted false intel)
- 89% NOISE (routine, irrelevant)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


class BastarIntelligenceGenerator:
    """Generate realistic synthetic intelligence data for Bastar region."""
    
    def __init__(self, 
                 start_date: datetime = datetime(2024, 12, 20),
                 end_date: datetime = datetime(2025, 1, 20),
                 daily_volume: int = 500,
                 random_seed: int = 42):
        """
        Initialize the generator.
        
        Args:
            start_date: Start of intelligence collection period
            end_date: End of intelligence collection period
            daily_volume: Average number of intelligence inputs per day
            random_seed: For reproducibility
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.start_date = start_date
        self.end_date = end_date
        self.daily_volume = daily_volume
        self.days = (end_date - start_date).days
        
        # Ground truth: Known attacks
        self.attacks = [
            {
                'id': 0,
                'date': datetime(2025, 1, 6, 14, 20),
                'location': {'lat': 18.50, 'lon': 81.00, 'district': 'Bijapur', 'village': 'Ambeli'},
                'type': 'COMMAND_IED',
                'casualties': {'kia': 9, 'wia': 0},
                'target': 'DRG_vehicle',
                'weight_kg': 65,
                'triggering': 'command_wire'
            },
            {
                'id': 1,
                'date': datetime(2025, 1, 12, 19, 0),
                'location': {'lat': 18.15, 'lon': 81.25, 'district': 'Sukma', 'village': 'Timmapuram'},
                'type': 'PRESSURE_IED',
                'casualties': {'kia': 0, 'wia': 1},
                'target': 'civilian',  # Unintended
                'weight_kg': 15,
                'triggering': 'pressure_plate'
            },
            {
                'id': 2,
                'date': datetime(2025, 1, 16, 10, 30),
                'location': {'lat': 18.62, 'lon': 80.88, 'district': 'Bijapur', 'village': 'Putkel'},
                'type': 'PRESSURE_IED',
                'casualties': {'kia': 0, 'wia': 2},
                'target': 'CoBRA_patrol',
                'weight_kg': 20,
                'triggering': 'pressure_plate'
            },
            {
                'id': 3,
                'date': datetime(2025, 1, 17, 7, 15),
                'location': {'lat': 18.45, 'lon': 80.95, 'district': 'Narayanpur', 'village': 'Garpa'},
                'type': 'COMMAND_IED',
                'casualties': {'kia': 0, 'wia': 2},
                'target': 'BSF_ROP',
                'weight_kg': 25,
                'triggering': 'command_wire'
            }
        ]
        
        # Districts in Bastar region
        self.districts = ['Bijapur', 'Sukma', 'Dantewada', 'Narayanpur', 'Kanker', 'Kondagaon']
        
        # Villages (simplified set)
        self.villages = {
            'Bijapur': ['Ambeli', 'Karkeli', 'Kutru', 'Basaguda', 'Putkel', 'Tarrem', 'Gangaloor'],
            'Sukma': ['Timmapuram', 'Chintalnar', 'Konta', 'Chintagufa', 'Errabore', 'Dornapal'],
            'Dantewada': ['Kirandul', 'Kuakonda', 'Barsoor', 'Geedam', 'Sameli'],
            'Narayanpur': ['Garpa', 'Orchha', 'Chhote Dongar', 'Kohkameta', 'Benur'],
            'Kanker': ['Antagarh', 'Bhanupratappur', 'Charama', 'Pakhanjur'],
            'Kondagaon': ['Mardapal', 'Baderajpur', 'Keshkal', 'Vishrampuri']
        }
        
        # Security force units
        self.security_units = [
            'DRG_Bijapur', 'DRG_Sukma', 'CRPF_206', 'CRPF_210', 'CRPF_168',
            'BSF_128', 'BSF_114', 'CoBRA_206', 'CoBRA_208', 'STF_Alpha', 'STF_Bravo'
        ]
        
        # Patrol routes
        self.patrol_routes = [
            {'id': 'ALPHA', 'district': 'Bijapur', 'risk': 'HIGH'},
            {'id': 'BRAVO', 'district': 'Bijapur', 'risk': 'MEDIUM'},
            {'id': 'CHARLIE', 'district': 'Sukma', 'risk': 'HIGH'},
            {'id': 'DELTA', 'district': 'Dantewada', 'risk': 'MEDIUM'},
            {'id': 'ECHO', 'district': 'Narayanpur', 'risk': 'HIGH'},
            {'id': 'FOXTROT', 'district': 'Narayanpur', 'risk': 'MEDIUM'},
            {'id': 'GOLF', 'district': 'Kanker', 'risk': 'LOW'},
        ]
        
        # Initialize informant network
        self.informants = self._generate_informant_network(50)
        
        # Threat keywords for NLP features later
        self.threat_keywords = [
            'IED', 'blast', 'explosive', 'attack', 'ambush', 'mine', 'bomb',
            'movement', 'suspicious', 'cadre', 'dalam', 'preparation', 'target',
            'patrol', 'vehicle', 'convoy', 'road', 'morning', 'night', 'digging'
        ]
        
        self.routine_keywords = [
            'routine', 'normal', 'livestock', 'farming', 'villagers', 'market',
            'forest', 'collection', 'tendu', 'mahua', 'unclear', 'rumor'
        ]
    
    def _generate_informant_network(self, n: int = 50) -> List[Dict]:
        """Create realistic informant profiles with varying reliability."""
        informants = []
        
        for i in range(n):
            # Reliability follows normal distribution (mean=5, sd=2)
            reliability = max(1, min(10, int(np.random.normal(5, 2))))
            
            district = random.choice(self.districts[:4])  # Focus on active districts
            
            informants.append({
                'id': f'HUMINT_SOURCE_{i+1:03d}',
                'codename': f'{random.choice(["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA"])}-{random.randint(10,99)}',
                'reliability_score': reliability,
                'location_district': district,
                'active_since': self.start_date - timedelta(days=random.randint(100, 1000)),
                'reports_submitted': 0,
                'reports_verified': 0,
                'last_contact': None,
                'handler': f'Officer_{random.randint(1,10)}',
                'specialization': random.choice(['cadre_movement', 'logistics', 'ied_activity', 'general'])
            })
        
        return informants
    
    def _get_next_attack(self, current_date: datetime) -> Optional[Dict]:
        """Find the next attack after current date."""
        for attack in self.attacks:
            if attack['date'] > current_date:
                return attack
        return None
    
    def _is_pre_attack_window(self, current_date: datetime, window_days: int = 3) -> Tuple[bool, Optional[int], Optional[Dict]]:
        """Check if current date is within pre-attack window."""
        for attack in self.attacks:
            days_until = (attack['date'].date() - current_date.date()).days
            if 0 < days_until <= window_days:
                return True, days_until, attack
        return False, None, None
    
    def _generate_true_signal_text(self, attack: Dict, days_to_attack: int) -> str:
        """Generate realistic pre-attack intelligence text."""
        village = attack['location']['village']
        district = attack['location']['district']
        target_type = attack['target'].replace('_', ' ')
        
        if days_to_attack == 1:
            templates = [
                f"URGENT: Unusual activity near {village}. 15-20 cadres spotted with heavy loads. Possible IED preparation.",
                f"Source reports imminent action planned on {district} patrol routes. Digging observed near main road.",
                f"Warning: Local villagers told to stay indoors tomorrow. Maoists preparing 'big action' near {village}.",
                f"Fresh digging marks observed on Kutru-Bedre road. Consistent with IED placement. High confidence.",
            ]
        elif days_to_attack == 2:
            templates = [
                f"Movement of 10-15 suspected Maoists near {village} forest area. Carrying unidentified heavy materials.",
                f"SIGINT indicates increased chatter mentioning '{village}' and 'heavy vehicle' targets.",
                f"Informant reports preparation activity in {district}. Cadres avoiding usual routes.",
                f"Local market activity reduced near {village}. Villagers seem anxious. Possible operation imminent.",
            ]
        else:
            templates = [
                f"Maoist squad of 8-10 cadres spotted moving towards {district} district. Intent unclear.",
                f"Routine surveillance detects increased forest activity near {village}. May be logistics or preparation.",
                f"Source reports hearing discussions about targeting security forces in {district} area.",
                f"Unverified: Maoists collecting explosive materials in remote areas of {district}.",
            ]
        
        return random.choice(templates)
    
    def _generate_noise_text(self) -> str:
        """Generate routine/irrelevant intelligence text."""
        templates = [
            "Group of villagers seen collecting tendu leaves in forest. Initially reported as suspicious movement.",
            "Distant explosions heard. Likely security forces conducting training or road construction blasting.",
            "Unknown individuals spotted in forest. Lost visual contact. Unable to confirm identity or intentions.",
            "Routine Maoist logistics activity observed. Discussing food supplies and medical needs.",
            "Livestock missing from village. May be wildlife predation or theft. No security implications.",
            "Rumors of Maoist presence in area, but no concrete evidence. Third-hand information.",
            "Village meeting observed. Appears to be routine jan adalat (people's court). No security threat.",
            "Movement on forest trail. Likely villagers traveling between villages for weekly market.",
            "Smoke observed in forest. Likely cooking fire or agricultural burning.",
            "Old IED crater observed. Area already cleared. No new activity.",
        ]
        return random.choice(templates)
    
    def _generate_deception_text(self) -> str:
        """Generate deliberately misleading intelligence (planted by insurgents)."""
        templates = [
            "CRITICAL: Major IED attack planned on Kanker highway tomorrow. 50+ cadres involved. [DECEPTION]",
            "Source claims top Maoist leader hiding in Kondagaon. Ready for capture. [DECEPTION]",
            "Large weapons cache reported at specific coordinates. Recommends immediate action. [DECEPTION]",
            "Informant insists attack imminent on Route Golf. Very specific but unverified. [DECEPTION]",
        ]
        return random.choice(templates)
    
    def generate_humint(self, target_date: datetime) -> List[Dict]:
        """Generate HUMINT (Human Intelligence) reports for a specific day."""
        reports = []
        n_reports = int(self.daily_volume * 0.40)  # 40% of daily volume
        
        is_pre_attack, days_to_attack, next_attack = self._is_pre_attack_window(target_date)
        
        # Determine signal distribution
        if is_pre_attack and days_to_attack <= 2:
            n_true_signals = int(n_reports * 0.12)  # 12% true signals in critical window
            n_deception = int(n_reports * 0.02)
        elif is_pre_attack:
            n_true_signals = int(n_reports * 0.06)  # 6% in wider window
            n_deception = int(n_reports * 0.01)
        else:
            n_true_signals = int(n_reports * 0.02)  # 2% baseline
            n_deception = int(n_reports * 0.01)
        
        n_noise = n_reports - n_true_signals - n_deception
        
        # Generate TRUE SIGNALS
        for _ in range(n_true_signals):
            # Prefer high-reliability sources for true signals
            high_rel_sources = [s for s in self.informants if s['reliability_score'] >= 7]
            source = random.choice(high_rel_sources) if high_rel_sources else random.choice(self.informants)
            
            if next_attack:
                lat_jitter = np.random.uniform(-0.08, 0.08)
                lon_jitter = np.random.uniform(-0.08, 0.08)
                
                report = {
                    'record_id': None,  # Will be assigned later
                    'timestamp': target_date + timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59)),
                    'type': 'HUMINT',
                    'source_id': source['id'],
                    'source_codename': source['codename'],
                    'source_reliability': source['reliability_score'],
                    'district': next_attack['location']['district'],
                    'village': next_attack['location']['village'],
                    'location_lat': next_attack['location']['lat'] + lat_jitter,
                    'location_lon': next_attack['location']['lon'] + lon_jitter,
                    'report_text': self._generate_true_signal_text(next_attack, days_to_attack),
                    'keywords': random.sample(self.threat_keywords, k=random.randint(3, 6)),
                    'urgency': 'CRITICAL' if days_to_attack == 1 else 'HIGH' if days_to_attack == 2 else 'MEDIUM',
                    'verified': False,
                    'ground_truth_label': 'TRUE_SIGNAL',
                    'related_attack_id': next_attack['id'],
                    'confidence_score': np.random.uniform(0.6, 0.9)
                }
                reports.append(report)
        
        # Generate DECEPTION
        for _ in range(n_deception):
            source = random.choice([s for s in self.informants if s['reliability_score'] <= 4])
            wrong_district = random.choice([d for d in self.districts if d not in ['Bijapur', 'Sukma', 'Narayanpur']])
            
            report = {
                'record_id': None,
                'timestamp': target_date + timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59)),
                'type': 'HUMINT',
                'source_id': source['id'],
                'source_codename': source['codename'],
                'source_reliability': source['reliability_score'],
                'district': wrong_district,
                'village': random.choice(self.villages.get(wrong_district, ['Unknown'])),
                'location_lat': np.random.uniform(18.0, 19.5),
                'location_lon': np.random.uniform(80.5, 82.0),
                'report_text': self._generate_deception_text(),
                'keywords': random.sample(self.threat_keywords, k=random.randint(4, 7)),
                'urgency': random.choice(['CRITICAL', 'HIGH']),  # Deception often sounds urgent
                'verified': False,
                'ground_truth_label': 'DECEPTION',
                'related_attack_id': None,
                'confidence_score': np.random.uniform(0.7, 0.95)  # High confidence (trap)
            }
            reports.append(report)
        
        # Generate NOISE
        for _ in range(n_noise):
            source = random.choice(self.informants)
            district = random.choice(self.districts)
            
            report = {
                'record_id': None,
                'timestamp': target_date + timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59)),
                'type': 'HUMINT',
                'source_id': source['id'],
                'source_codename': source['codename'],
                'source_reliability': source['reliability_score'],
                'district': district,
                'village': random.choice(self.villages.get(district, ['Unknown'])),
                'location_lat': np.random.uniform(18.0, 19.5),
                'location_lon': np.random.uniform(80.5, 82.0),
                'report_text': self._generate_noise_text(),
                'keywords': random.sample(self.routine_keywords, k=random.randint(2, 4)),
                'urgency': random.choice(['LOW', 'LOW', 'LOW', 'MEDIUM']),
                'verified': False,
                'ground_truth_label': 'NOISE',
                'related_attack_id': None,
                'confidence_score': np.random.uniform(0.2, 0.5)
            }
            reports.append(report)
        
        return reports
    
    def generate_sigint(self, target_date: datetime) -> List[Dict]:
        """Generate SIGINT (Signals Intelligence) intercepts."""
        intercepts = []
        n_intercepts = int(self.daily_volume * 0.30)  # 30% of daily volume
        
        is_pre_attack, days_to_attack, next_attack = self._is_pre_attack_window(target_date)
        
        for i in range(n_intercepts):
            # Pre-attack signals
            is_signal = is_pre_attack and random.random() < (0.10 if days_to_attack <= 2 else 0.05)
            
            if is_signal and next_attack:
                transcript = random.choice([
                    f"...tomorrow...{next_attack['location']['village']}...wait for signal...",
                    "...convoy expected...detonate on command...",
                    "...devices ready...morning patrol...",
                    f"...heavy vehicle...{next_attack['location']['district']} road...",
                    "...final preparation...be ready at dawn...",
                ])
                lat = next_attack['location']['lat'] + np.random.uniform(-0.1, 0.1)
                lon = next_attack['location']['lon'] + np.random.uniform(-0.1, 0.1)
                confidence = 'HIGH'
                ground_truth = 'TRUE_SIGNAL'
            else:
                transcript = random.choice([
                    "...food supplies...medical needs...",
                    "...training schedule...next week...",
                    "...unclear...signal breaking up...",
                    "...routine communication...logistics...",
                    "...[static]...cannot decipher...",
                    "...village meeting...jan adalat...",
                ])
                lat = np.random.uniform(18.0, 19.5)
                lon = np.random.uniform(80.5, 82.0)
                confidence = random.choice(['LOW', 'MEDIUM', 'MEDIUM'])
                ground_truth = 'NOISE'
            
            intercept = {
                'record_id': None,
                'timestamp': target_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                'type': 'SIGINT',
                'frequency_mhz': round(np.random.uniform(150.0, 160.0), 2),
                'duration_sec': random.randint(15, 300),
                'district': random.choice(self.districts[:4]),
                'village': None,
                'location_lat': lat,
                'location_lon': lon,
                'location_accuracy_km': round(np.random.uniform(1, 8), 1),
                'signal_strength': random.choice(['WEAK', 'MODERATE', 'STRONG']),
                'transcript': transcript,
                'language': random.choice(['Hindi', 'Gondi', 'Halbi', 'Mixed']),
                'voice_matched': random.random() < 0.08,  # 8% match known cadre
                'keywords': self._extract_keywords(transcript),
                'urgency': 'HIGH' if is_signal else random.choice(['LOW', 'MEDIUM']),
                'confidence_score': np.random.uniform(0.6, 0.9) if is_signal else np.random.uniform(0.2, 0.5),
                'ground_truth_label': ground_truth,
                'related_attack_id': next_attack['id'] if is_signal and next_attack else None
            }
            intercepts.append(intercept)
        
        return intercepts
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        text_lower = text.lower()
        found = []
        for kw in self.threat_keywords + self.routine_keywords:
            if kw.lower() in text_lower:
                found.append(kw)
        return found if found else ['unclear']
    
    def generate_patrol_reports(self, target_date: datetime) -> List[Dict]:
        """Generate patrol activity reports from security forces."""
        reports = []
        n_patrols = int(self.daily_volume * 0.20)  # 20% of daily volume
        
        is_pre_attack, days_to_attack, next_attack = self._is_pre_attack_window(target_date)
        
        for _ in range(n_patrols):
            unit = random.choice(self.security_units)
            route = random.choice(self.patrol_routes)
            district = route['district']
            
            # Some patrols observe IED indicators
            observe_indicators = random.random() < 0.12  # 12% observe something
            
            if observe_indicators:
                observations = random.choice([
                    'Fresh digging observed on patrol route',
                    'Disturbed earth near culvert',
                    'Suspicious wires spotted, EOD called',
                    'Villagers warned us about specific route section',
                    'Unusual absence of civilians in normally busy area',
                ])
                risk_level = 'HIGH'
            else:
                observations = random.choice([
                    'No contact. Area dominated.',
                    'Routine patrol. No suspicious activity.',
                    'Patrol completed. Route clear.',
                    'Light civilian traffic observed. Normal.',
                    'Forest patrol. No contact with hostiles.',
                ])
                risk_level = route['risk']
            
            # Check if this patrol is near actual attack site
            near_attack_site = False
            if next_attack and district == next_attack['location']['district']:
                near_attack_site = True
            
            report = {
                'record_id': None,
                'timestamp': target_date + timedelta(hours=random.randint(6, 18), minutes=random.randint(0, 59)),
                'type': 'PATROL_REPORT',
                'unit': unit,
                'route_id': route['id'],
                'district': district,
                'village': random.choice(self.villages.get(district, ['Unknown'])),
                'location_lat': np.random.uniform(18.0, 19.5),
                'location_lon': np.random.uniform(80.5, 82.0),
                'start_time': f"{random.randint(5,8):02d}:00",
                'end_time': f"{random.randint(15,18):02d}:00",
                'observations': observations,
                'keywords': self._extract_keywords(observations),
                'ied_indicators': observe_indicators,
                'eod_sweep_conducted': random.random() < 0.25,
                'risk_level': risk_level,
                'urgency': 'HIGH' if observe_indicators else 'LOW',
                'confidence_score': 0.8 if observe_indicators else 0.3,
                'ground_truth_label': 'TRUE_SIGNAL' if observe_indicators and near_attack_site else 'NOISE',
                'related_attack_id': next_attack['id'] if observe_indicators and near_attack_site else None
            }
            reports.append(report)
        
        return reports
    
    def generate_geoint(self, target_date: datetime) -> List[Dict]:
        """Generate GEOINT (Geospatial Intelligence) from satellites/drones."""
        observations = []
        n_obs = int(self.daily_volume * 0.06)  # 6% of daily volume
        
        is_pre_attack, days_to_attack, next_attack = self._is_pre_attack_window(target_date)
        
        for _ in range(n_obs):
            # Some observations detect anomalies
            detect_anomaly = random.random() < 0.15  # 15% detect something
            
            if detect_anomaly and is_pre_attack and next_attack:
                analysis = random.choice([
                    'Vehicle tracks inconsistent with civilian patterns',
                    'Heat signatures detected in forest at night',
                    'Fresh excavation marks along patrol route',
                    'Group of 10-15 individuals observed moving through forest',
                ])
                lat = next_attack['location']['lat'] + np.random.uniform(-0.15, 0.15)
                lon = next_attack['location']['lon'] + np.random.uniform(-0.15, 0.15)
                ground_truth = 'TRUE_SIGNAL'
                recommendation = 'GROUND_VERIFICATION_REQUIRED'
            elif detect_anomaly:
                analysis = random.choice([
                    'Vegetation disturbance detected',
                    'Unknown vehicle observed on forest track',
                    'Possible camp structure in remote area',
                ])
                lat = np.random.uniform(18.0, 19.5)
                lon = np.random.uniform(80.5, 82.0)
                ground_truth = 'FALSE_POSITIVE'
                recommendation = 'ENHANCED_SURVEILLANCE'
            else:
                analysis = random.choice([
                    'No significant change from baseline',
                    'Normal civilian activity observed',
                    'Cloud cover obscured target area',
                    'Area clear. No anomalies.',
                ])
                lat = np.random.uniform(18.0, 19.5)
                lon = np.random.uniform(80.5, 82.0)
                ground_truth = 'NOISE'
                recommendation = 'NO_ACTION'
            
            obs = {
                'record_id': None,
                'timestamp': target_date + timedelta(hours=random.randint(10, 16), minutes=random.randint(0, 59)),
                'type': 'GEOINT',
                'sensor': random.choice(['CARTOSAT-3', 'Heron_UAV', 'Rustom-2_UAV', 'Commercial_Satellite']),
                'resolution_m': random.choice([0.3, 0.5, 1.0, 2.5]),
                'district': random.choice(self.districts[:4]),
                'village': None,
                'location_lat': lat,
                'location_lon': lon,
                'coverage_km2': random.randint(10, 100),
                'anomaly_detected': detect_anomaly,
                'analysis': analysis,
                'keywords': self._extract_keywords(analysis),
                'urgency': 'HIGH' if ground_truth == 'TRUE_SIGNAL' else 'MEDIUM' if detect_anomaly else 'LOW',
                'recommendation': recommendation,
                'confidence_score': np.random.uniform(0.5, 0.8) if detect_anomaly else np.random.uniform(0.2, 0.4),
                'ground_truth_label': ground_truth,
                'related_attack_id': next_attack['id'] if ground_truth == 'TRUE_SIGNAL' and next_attack else None
            }
            observations.append(obs)
        
        return observations
    
    def generate_osint(self, target_date: datetime) -> List[Dict]:
        """Generate OSINT (Open Source Intelligence) from news and social media."""
        reports = []
        n_reports = int(self.daily_volume * 0.04)  # 4% of daily volume
        
        is_post_attack = any(
            attack['date'].date() <= target_date.date() <= attack['date'].date() + timedelta(days=2)
            for attack in self.attacks
        )
        
        for _ in range(n_reports):
            if is_post_attack:
                source_type = random.choice(['news_article', 'twitter', 'local_report'])
                content = random.choice([
                    'Breaking: IED blast reported in Bastar region. Casualties feared.',
                    'Security forces convoy attacked in Chhattisgarh forest area.',
                    'Naxal activity intensifies in Red Corridor districts.',
                    'Local residents report increased Maoist presence.',
                ])
                relevance = 'HIGH'
            else:
                source_type = random.choice(['news_article', 'twitter', 'local_report', 'government_statement'])
                content = random.choice([
                    'Security forces conduct area domination in Bastar.',
                    'Development work continues in Naxal-affected areas.',
                    'CM reviews law and order situation in Red Corridor.',
                    'Surrender of Maoist cadres reported in Dantewada.',
                    'Road construction progresses despite Naxal threats.',
                ])
                relevance = 'LOW'
            
            report = {
                'record_id': None,
                'timestamp': target_date + timedelta(hours=random.randint(8, 22), minutes=random.randint(0, 59)),
                'type': 'OSINT',
                'source_type': source_type,
                'source_name': random.choice(['NDTV', 'Times of India', 'Local Stringer', 'PTI', 'ANI']),
                'district': random.choice(self.districts),
                'village': None,
                'location_lat': None,
                'location_lon': None,
                'content': content,
                'keywords': self._extract_keywords(content),
                'relevance': relevance,
                'urgency': 'MEDIUM' if is_post_attack else 'LOW',
                'confidence_score': 0.9,  # OSINT is verifiable
                'ground_truth_label': 'POST_ATTACK' if is_post_attack else 'NOISE',
                'related_attack_id': None
            }
            reports.append(report)
        
        return reports
    
    def generate_full_dataset(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Generate complete intelligence dataset."""
        all_records = []
        record_id = 0
        
        print(f"Generating {self.days} days of intelligence data...")
        print(f"Target: ~{self.daily_volume} records/day = ~{self.days * self.daily_volume} total records")
        print("-" * 60)
        
        for day in tqdm(range(self.days), desc="Generating data"):
            current_date = self.start_date + timedelta(days=day)
            
            # Generate all intelligence types for this day
            humint = self.generate_humint(current_date)
            sigint = self.generate_sigint(current_date)
            patrol = self.generate_patrol_reports(current_date)
            geoint = self.generate_geoint(current_date)
            osint = self.generate_osint(current_date)
            
            daily_records = humint + sigint + patrol + geoint + osint
            
            # Assign record IDs
            for record in daily_records:
                record['record_id'] = f"INTEL_{record_id:06d}"
                record_id += 1
            
            all_records.extend(daily_records)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived columns
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        print("\n" + "=" * 60)
        print("DATASET GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nBreakdown by type:")
        print(df['type'].value_counts().to_string())
        print(f"\nBreakdown by ground truth:")
        print(df['ground_truth_label'].value_counts().to_string())
        print(f"\nSignal-to-noise ratio: {len(df[df['ground_truth_label'] == 'TRUE_SIGNAL']) / len(df) * 100:.2f}%")
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"\n✓ Dataset saved to: {save_path}")
        
        return df
    
    def get_attacks_df(self) -> pd.DataFrame:
        """Return attacks as DataFrame for reference."""
        attacks_flat = []
        for attack in self.attacks:
            attacks_flat.append({
                'attack_id': attack['id'],
                'timestamp': attack['date'],
                'district': attack['location']['district'],
                'village': attack['location']['village'],
                'lat': attack['location']['lat'],
                'lon': attack['location']['lon'],
                'ied_type': attack['type'],
                'target': attack['target'],
                'kia': attack['casualties']['kia'],
                'wia': attack['casualties']['wia'],
                'weight_kg': attack['weight_kg']
            })
        return pd.DataFrame(attacks_flat)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Create data directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    generator = BastarIntelligenceGenerator(
        start_date=datetime(2024, 12, 20),
        end_date=datetime(2025, 1, 20),
        daily_volume=500,
        random_seed=42
    )
    
    # Generate and save
    df = generator.generate_full_dataset(save_path=str(data_dir / "bastar_intelligence_15k.csv"))
    
    # Save attacks reference
    attacks_df = generator.get_attacks_df()
    attacks_df.to_csv(data_dir / "attacks_ground_truth.csv", index=False)
    print(f"✓ Attacks reference saved to: {data_dir / 'attacks_ground_truth.csv'}")
    
    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS (TRUE SIGNALS)")
    print("=" * 60)
    true_signals = df[df['ground_truth_label'] == 'TRUE_SIGNAL'].head(3)
    for _, row in true_signals.iterrows():
        print(f"\n[{row['record_id']}] {row['type']} - {row['timestamp']}")
        print(f"  District: {row['district']}, Urgency: {row['urgency']}")
        if 'report_text' in row and pd.notna(row.get('report_text')):
            print(f"  Text: {row['report_text'][:100]}...")
        elif 'transcript' in row and pd.notna(row.get('transcript')):
            print(f"  Transcript: {row['transcript']}")
