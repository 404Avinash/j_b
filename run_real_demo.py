"""
JATAYU - Complete Real Data Pipeline Demo
==========================================
End-to-end demonstration of real IED prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.real_data_loader import RealDataLoader
from src.models.real_trainer import RealMLTrainer


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """Print a section header."""
    print(f"\n[{text}]")
    print("-" * 50)


def run_complete_demo():
    """Run the complete JATAYU demonstration with real data."""
    
    print_header("JATAYU - Predictive Intelligence Fusion System")
    print("Operation: Silent Watch")
    print("Dataset: Real IED Incidents from Red Corridor (2020-2026)")
    print(f"Demo Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # PHASE 1: DATA LOADING & ANALYSIS
    # =========================================================================
    print_header("PHASE 1: Intelligence Data Loading")
    
    loader = RealDataLoader()
    df = loader.clean_data()
    stats = loader.get_statistics()
    
    print_section("Dataset Overview")
    print(f"  Total IED Incidents: {stats['total_incidents']}")
    print(f"  Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  States Covered: {len(stats['by_state'])}")
    
    print_section("Casualty Summary")
    print(f"  Total Killed: {stats['total_killed']}")
    print(f"  Total Injured: {stats['total_injured']}")
    print(f"  Major Attacks (3+ casualties): {stats['major_attacks']}")
    print(f"  Monthly Average: {stats['monthly_avg']:.1f} incidents")
    
    print_section("Geographic Distribution")
    for state, count in sorted(stats['by_state'].items(), key=lambda x: -x[1])[:5]:
        pct = 100 * count / stats['total_incidents']
        print(f"  {state}: {count} incidents ({pct:.1f}%)")
    
    print_section("Hotspot Districts")
    top_districts = sorted(stats['by_district'].items(), key=lambda x: -x[1])[:5]
    for district, count in top_districts:
        killed = df[df['District'] == district]['Killed'].sum()
        print(f"  {district}: {count} incidents, {killed} killed")
    
    # =========================================================================
    # PHASE 2: ATTACK PATTERN ANALYSIS
    # =========================================================================
    print_header("PHASE 2: Attack Pattern Analysis")
    
    print_section("Year-over-Year Trend")
    for year, count in sorted(stats['by_year'].items()):
        killed = df[df['Year'] == year]['Killed'].sum()
        bar = "#" * (count // 2)
        print(f"  {year}: {count:2d} incidents, {killed:2d} killed {bar}")
    
    print_section("Attack Types")
    for atype, count in sorted(stats['by_attack_type'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_incidents']
        print(f"  {atype}: {count} ({pct:.1f}%)")
    
    print_section("Attack Clusters (Multiple attacks within 7 days)")
    clusters = loader.get_attack_clusters(days_threshold=7)
    print(f"  Found {len(clusters)} distinct clusters")
    
    # Show the most severe clusters
    severe_clusters = sorted(clusters, key=lambda x: -x['total_killed'])[:3]
    for i, c in enumerate(severe_clusters, 1):
        print(f"\n  Cluster {i}: {c['start_date'].strftime('%Y-%m-%d')} to {c['end_date'].strftime('%Y-%m-%d')}")
        print(f"    Attacks: {c['num_attacks']}")
        print(f"    Casualties: {c['total_killed']} killed, {c['total_injured']} injured")
        print(f"    Districts: {', '.join(set(c['locations']))}")
    
    # =========================================================================
    # PHASE 3: ML MODEL TRAINING
    # =========================================================================
    print_header("PHASE 3: ML Model Training")
    
    trainer = RealMLTrainer()
    trainer.prepare_data()
    
    print_section("Temporal Split Strategy")
    print("  Training: 2020-01-01 to 2024-12-31 (5 years)")
    print("  Testing:  2025-01-01 onwards")
    print("  Task: Predict if attack occurs within next 7 days")
    
    results = trainer.train_model(
        train_end_date='2024-12-31',
        prediction_window_days=7
    )
    
    # =========================================================================
    # PHASE 4: VALIDATION ON JANUARY 2025 CLUSTER
    # =========================================================================
    print_header("PHASE 4: Validation on Known Attack Cluster")
    
    print_section("Target Cluster: January 2025 (Bastar Region)")
    
    # Show the actual attacks in January 2025
    jan_2025 = df[(df['Date'] >= '2025-01-01') & (df['Date'] <= '2025-01-31')]
    print(f"  Actual attacks in January 2025: {len(jan_2025)}")
    for _, row in jan_2025.iterrows():
        print(f"    {row['Date'].strftime('%Y-%m-%d')}: {row['District']} - {row['Killed']} killed, {row['Injured']} injured")
    
    print_section("Pre-Attack Prediction (as of 2025-01-05)")
    print("  Model prediction BEFORE the Jan 6 attack:")
    
    prediction = trainer.predict_next_attack(as_of_date='2025-01-05')
    
    # =========================================================================
    # PHASE 5: CURRENT RISK ASSESSMENT
    # =========================================================================
    print_header("PHASE 5: Current Threat Assessment")
    
    # Get the latest data point
    latest_date = df['Date'].max()
    print(f"  Intelligence cutoff: {latest_date.strftime('%Y-%m-%d')}")
    
    current_prediction = trainer.predict_next_attack(as_of_date=latest_date)
    
    print_section("Recommended Actions")
    if current_prediction['probability'] > 0.5:
        print("  [!] HIGH ALERT: Increase patrols in high-risk districts")
        print("  [!] Deploy additional EOD teams")
        print("  [!] Activate road-opening party protocols")
    else:
        print("  [*] Standard patrol protocols")
        print("  [*] Continue monitoring HUMINT sources")
        print("  [*] Maintain readiness in high-risk districts")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("JATAYU SYSTEM SUMMARY")
    
    print("""
    MISSION: Predict IED attacks before they occur
    
    KEY FINDINGS:
    1. Bijapur district is the highest-risk area (63 incidents, 27 killed)
    2. Attack tempo increases during certain periods (clustering detected)
    3. Model identifies key predictive signals:
       - Recent casualty momentum
       - District historical risk
       - Days since last attack
    
    SYSTEM CAPABILITIES:
    - 23 engineered features from incident data
    - Temporal train-test split (no data leakage)
    - Real-time risk scoring by district
    - Attack cluster detection
    
    NEXT STEPS:
    1. Integrate OSINT feeds for real-time monitoring
    2. Add SIGINT/HUMINT correlation
    3. Deploy as operational dashboard
    """)
    
    print("\n" + "=" * 70)
    print("  Demo Complete. Stay Vigilant.")
    print("=" * 70 + "\n")
    
    return {
        'loader': loader,
        'trainer': trainer,
        'stats': stats,
        'results': results,
        'prediction': current_prediction
    }


if __name__ == "__main__":
    demo_results = run_complete_demo()
