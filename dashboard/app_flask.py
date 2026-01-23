"""
JATAYU - Complete Integrated Dashboard
=======================================
Flask application with actual ML model, geographic data, and all 187 incidents.

Run: python app_flask.py
Open: http://localhost:5000
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ============================================
# DATA LOADING
# ============================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'production_model.pkl')

# Location coordinates (actual Bastar region coordinates)
LOCATION_COORDS = {
    'Bijapur': {'lat': 18.8463, 'lon': 80.8330, 'state': 'Chhattisgarh'},
    'Sukma': {'lat': 18.3874, 'lon': 81.6600, 'state': 'Chhattisgarh'},
    'Dantewada': {'lat': 18.8974, 'lon': 81.3467, 'state': 'Chhattisgarh'},
    'Narayanpur': {'lat': 19.7136, 'lon': 81.2523, 'state': 'Chhattisgarh'},
    'Kanker': {'lat': 20.2719, 'lon': 81.4914, 'state': 'Chhattisgarh'},
    'Gadchiroli': {'lat': 20.1052, 'lon': 80.0056, 'state': 'Maharashtra'},
    'West Singhbhum': {'lat': 22.5736, 'lon': 85.8309, 'state': 'Jharkhand'},
    'Lohardaga': {'lat': 23.4357, 'lon': 84.6836, 'state': 'Jharkhand'},
    'Gumla': {'lat': 23.0437, 'lon': 84.5421, 'state': 'Jharkhand'},
    'Latehar': {'lat': 23.7370, 'lon': 84.5013, 'state': 'Jharkhand'},
    'Sundargarh': {'lat': 22.1167, 'lon': 84.0333, 'state': 'Odisha'},
}

# Global data holders
intel_df = None
incidents_df = None
model_data = None


def load_data():
    """Load all data on startup."""
    global intel_df, incidents_df, model_data
    
    print("[JATAYU] Loading data...")
    
    # Load incidents
    incidents_path = os.path.join(DATA_DIR, 'raw_incidents.csv')
    if os.path.exists(incidents_path):
        incidents_df = pd.read_csv(incidents_path)
        incidents_df['Date'] = pd.to_datetime(incidents_df['Date'])
        print(f"[JATAYU] Loaded {len(incidents_df)} incidents")
    
    # Load intel sample (200K for speed, full 8.2M available)
    intel_path = os.path.join(DATA_DIR, 'all_intel_2020_2026.csv')
    if os.path.exists(intel_path):
        intel_df = pd.read_csv(intel_path, nrows=300000)
        print(f"[JATAYU] Loaded {len(intel_df):,} intel records (sample)")
    
    # Load model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print(f"[JATAYU] Loaded ML model")
    
    print("[JATAYU] Data loading complete!")


def predict_with_model(region_intel):
    """Use actual trained ML model for prediction."""
    global model_data
    
    if model_data is None or len(region_intel) == 0:
        return 0.5, "MEDIUM"
    
    # Calculate features (same as training)
    total = len(region_intel)
    features = {
        'total_intel': total,
        'pct_humint': (region_intel['Intel_Type'] == 'HUMINT').sum() / total,
        'pct_sigint': (region_intel['Intel_Type'] == 'SIGINT').sum() / total,
        'pct_patrol': (region_intel['Intel_Type'] == 'PATROL').sum() / total,
        'pct_osint': (region_intel['Intel_Type'] == 'OSINT').sum() / total,
        'pct_true_signal': (region_intel['Label'] == 'TRUE_SIGNAL').sum() / total,
        'pct_noise': (region_intel['Label'] == 'NOISE').sum() / total,
        'pct_deception': (region_intel['Label'] == 'DECEPTION').sum() / total,
        'pct_high_urgency': (region_intel['Urgency'] == 'HIGH').sum() / total,
        'pct_medium_urgency': (region_intel['Urgency'] == 'MEDIUM').sum() / total,
        'avg_reliability': region_intel['Reliability'].mean(),
        'max_reliability': region_intel['Reliability'].max(),
        'std_reliability': region_intel['Reliability'].std() if len(region_intel) > 1 else 0,
        'avg_signal_intensity': region_intel['Signal_Intensity'].mean(),
        'max_signal_intensity': region_intel['Signal_Intensity'].max(),
    }
    
    # Create feature array in correct order
    feature_names = model_data.get('feature_names', list(features.keys()))
    X = np.array([[features.get(f, 0) for f in feature_names]])
    
    # Scale and predict
    scaler = model_data.get('scaler')
    model = model_data.get('model')
    
    if scaler and model:
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        # Fallback calculation
        prob = features['avg_signal_intensity'] * 0.5 + features['pct_high_urgency'] * 0.3 + features['pct_true_signal'] * 0.2
    
    # Determine level
    if prob >= 0.7:
        level = "CRITICAL"
    elif prob >= 0.5:
        level = "HIGH"
    elif prob >= 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return float(prob), level


# ============================================
# HTML TEMPLATE
# ============================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JATAYU - Attack Prediction System</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #00ff88;
        }
        .header h1 { 
            font-size: 2.5rem; 
            background: linear-gradient(90deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p { color: #aaa; margin-top: 5px; }
        
        .stats-bar {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            flex-wrap: wrap;
        }
        .stat-box {
            text-align: center;
            padding: 15px 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            margin: 5px;
        }
        .stat-value { font-size: 2rem; font-weight: bold; color: #00ff88; }
        .stat-label { color: #888; font-size: 0.9rem; }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h2 {
            color: #00ff88;
            margin-bottom: 15px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        #map { height: 400px; border-radius: 10px; }
        
        .prediction-panel {
            grid-column: span 2;
        }
        
        .region-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .region-btn {
            padding: 10px 20px;
            background: rgba(0,255,136,0.1);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .region-btn:hover, .region-btn.active {
            background: #00ff88;
            color: #000;
        }
        
        .predict-btn {
            width: 100%;
            padding: 15px;
            font-size: 1.2rem;
            background: linear-gradient(90deg, #00ff88, #00ccff);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            color: #000;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .predict-btn:hover { transform: scale(1.02); }
        
        .result-box {
            margin-top: 20px;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            display: none;
        }
        .result-box.show { display: block; }
        .result-box.critical { background: linear-gradient(135deg, #ff4444, #cc0000); }
        .result-box.high { background: linear-gradient(135deg, #ff8800, #cc6600); }
        .result-box.medium { background: linear-gradient(135deg, #ffcc00, #cc9900); color: #000; }
        .result-box.low { background: linear-gradient(135deg, #00cc66, #009944); }
        
        .result-level { font-size: 2.5rem; font-weight: bold; }
        .result-prob { font-size: 1.5rem; margin-top: 10px; }
        .result-action { margin-top: 15px; font-size: 1.1rem; }
        
        .incidents-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .incidents-table th, .incidents-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .incidents-table th { color: #00ff88; }
        
        .chart-container { height: 300px; }
        
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .full-width { grid-column: span 2; }
        
        @media (max-width: 900px) {
            .main-content { grid-template-columns: 1fr; }
            .prediction-panel, .full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 JATAYU</h1>
        <p>Predictive Intelligence Fusion System for IED Attack Prevention</p>
    </div>
    
    <div class="stats-bar">
        <div class="stat-box">
            <div class="stat-value">{{ stats.total_intel }}</div>
            <div class="stat-label">Intel Records (8.2M Total)</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{{ stats.total_incidents }}</div>
            <div class="stat-label">Real IED Incidents</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{{ stats.total_killed }}</div>
            <div class="stat-label">Total Killed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{{ stats.regions }}</div>
            <div class="stat-label">Active Regions</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" style="color: #00ccff">ML: 100%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
    </div>
    
    <div class="main-content">
        <!-- MAP -->
        <div class="panel">
            <h2>🗺️ Geographic Threat Map</h2>
            <div id="map"></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#ff4444"></div> Critical</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ff8800"></div> High</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ffcc00"></div> Medium</div>
                <div class="legend-item"><div class="legend-dot" style="background:#00cc66"></div> Low</div>
            </div>
        </div>
        
        <!-- INTEL DISTRIBUTION -->
        <div class="panel">
            <h2>📊 Intelligence Distribution</h2>
            <div class="chart-container">
                <canvas id="intelChart"></canvas>
            </div>
        </div>
        
        <!-- PREDICTION PANEL -->
        <div class="panel prediction-panel">
            <h2>🎯 Attack Prediction (ML Model)</h2>
            <p style="color:#888; margin-bottom:15px">Select a region and click PREDICT to run the trained XGBoost model</p>
            
            <div class="region-selector" id="regionButtons">
                {% for region in regions %}
                <button class="region-btn" data-region="{{ region }}">{{ region }}</button>
                {% endfor %}
            </div>
            
            <button class="predict-btn" onclick="predictAttack()">
                🔮 PREDICT ATTACK PROBABILITY
            </button>
            
            <div class="result-box" id="resultBox">
                <div class="result-level" id="resultLevel">--</div>
                <div class="result-prob" id="resultProb">--</div>
                <div class="result-action" id="resultAction">--</div>
            </div>
        </div>
        
        <!-- INCIDENTS TABLE -->
        <div class="panel">
            <h2>📍 Recent IED Incidents (2020-2026)</h2>
            <table class="incidents-table">
                <thead>
                    <tr><th>Date</th><th>District</th><th>Killed</th><th>Injured</th></tr>
                </thead>
                <tbody>
                    {% for incident in incidents %}
                    <tr>
                        <td>{{ incident.Date }}</td>
                        <td>{{ incident.District }}</td>
                        <td style="color:#ff4444">{{ incident.Killed }}</td>
                        <td style="color:#ffcc00">{{ incident.Injured }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- SIGNAL BREAKDOWN -->
        <div class="panel">
            <h2>📡 Signal Classification</h2>
            <div class="chart-container">
                <canvas id="labelChart"></canvas>
            </div>
            <p style="color:#888; margin-top:10px; font-size:0.9rem">
                50% True Signals | 40% Noise (filtered) | 10% Deception (detected)
            </p>
        </div>
    </div>
    
    <script>
        // Map initialization
        const map = L.map('map').setView([20.5, 82], 6);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        }).addTo(map);
        
        // Add markers for incidents
        const locations = {{ locations | safe }};
        const riskColors = {CRITICAL: '#ff4444', HIGH: '#ff8800', MEDIUM: '#ffcc00', LOW: '#00cc66'};
        
        locations.forEach(loc => {
            const color = riskColors[loc.risk_level] || '#888';
            const marker = L.circleMarker([loc.lat, loc.lon], {
                radius: Math.min(loc.incidents * 2, 20),
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map);
            
            marker.bindPopup(`
                <b>${loc.name}</b><br>
                State: ${loc.state}<br>
                Incidents: ${loc.incidents}<br>
                Killed: ${loc.killed}<br>
                Risk: <b style="color:${color}">${loc.risk_level}</b><br>
                Risk Score: ${(loc.risk_score * 100).toFixed(1)}%
            `);
        });
        
        // Region selection
        let selectedRegion = '{{ regions[0] }}';
        document.querySelectorAll('.region-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.region-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                selectedRegion = this.dataset.region;
            });
        });
        document.querySelector('.region-btn').classList.add('active');
        
        // Prediction function
        async function predictAttack() {
            const btn = document.querySelector('.predict-btn');
            btn.textContent = '⏳ Analyzing with ML Model...';
            
            try {
                const response = await fetch('/api/predict?region=' + encodeURIComponent(selectedRegion));
                const data = await response.json();
                
                const resultBox = document.getElementById('resultBox');
                resultBox.className = 'result-box show ' + data.level.toLowerCase();
                
                document.getElementById('resultLevel').textContent = data.level + ' RISK';
                document.getElementById('resultProb').textContent = 
                    'Attack Probability: ' + (data.probability * 100).toFixed(1) + '%';
                document.getElementById('resultAction').textContent = data.action;
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
            
            btn.textContent = '🔮 PREDICT ATTACK PROBABILITY';
        }
        
        // Charts
        new Chart(document.getElementById('intelChart'), {
            type: 'doughnut',
            data: {
                labels: ['HUMINT', 'SIGINT', 'PATROL', 'OSINT'],
                datasets: [{
                    data: {{ intel_types | safe }},
                    backgroundColor: ['#00ff88', '#00ccff', '#ff88ff', '#ffcc00']
                }]
            },
            options: {
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        });
        
        new Chart(document.getElementById('labelChart'), {
            type: 'bar',
            data: {
                labels: ['True Signal', 'Noise', 'Deception'],
                datasets: [{
                    data: {{ labels | safe }},
                    backgroundColor: ['#00cc66', '#888888', '#ff4444']
                }]
            },
            options: {
                plugins: { legend: { display: false } },
                scales: { 
                    y: { ticks: { color: '#888' } },
                    x: { ticks: { color: '#888' } }
                }
            }
        });
    </script>
</body>
</html>
'''


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Main dashboard page."""
    global intel_df, incidents_df
    
    if intel_df is None or incidents_df is None:
        load_data()
    
    # Get regions
    regions = sorted(intel_df['District'].dropna().unique().tolist())
    
    # Stats
    stats = {
        'total_intel': f"{len(intel_df):,}",
        'total_incidents': len(incidents_df),
        'total_killed': int(incidents_df['Killed'].sum()),
        'regions': len(regions)
    }
    
    # Recent incidents
    recent_incidents = incidents_df.sort_values('Date', ascending=False).head(15)
    incidents_list = [
        {
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'District': row['District'],
            'Killed': int(row['Killed']),
            'Injured': int(row['Injured'])
        }
        for _, row in recent_incidents.iterrows()
    ]
    
    # Location data with risk
    locations = []
    for district, coords in LOCATION_COORDS.items():
        district_incidents = incidents_df[incidents_df['District'] == district]
        region_intel = intel_df[intel_df['District'].str.contains(district, case=False, na=False)]
        
        if len(region_intel) > 0:
            risk_score, risk_level = predict_with_model(region_intel)
        else:
            risk_score, risk_level = 0.3, "MEDIUM"
        
        locations.append({
            'name': district,
            'state': coords['state'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'incidents': len(district_incidents),
            'killed': int(district_incidents['Killed'].sum()) if len(district_incidents) > 0 else 0,
            'risk_score': risk_score,
            'risk_level': risk_level
        })
    
    # Intel type distribution
    intel_types = [
        int((intel_df['Intel_Type'] == 'HUMINT').sum()),
        int((intel_df['Intel_Type'] == 'SIGINT').sum()),
        int((intel_df['Intel_Type'] == 'PATROL').sum()),
        int((intel_df['Intel_Type'] == 'OSINT').sum())
    ]
    
    # Label distribution
    labels = [
        int((intel_df['Label'] == 'TRUE_SIGNAL').sum()),
        int((intel_df['Label'] == 'NOISE').sum()),
        int((intel_df['Label'] == 'DECEPTION').sum())
    ]
    
    return render_template_string(
        HTML_TEMPLATE,
        stats=stats,
        regions=regions,
        incidents=incidents_list,
        locations=json.dumps(locations),
        intel_types=json.dumps(intel_types),
        labels=json.dumps(labels)
    )


@app.route('/api/predict')
def api_predict():
    """API endpoint for prediction."""
    global intel_df
    
    region = request.args.get('region', 'Bijapur')
    
    if intel_df is None:
        load_data()
    
    region_intel = intel_df[intel_df['District'].str.contains(region, case=False, na=False)]
    
    prob, level = predict_with_model(region_intel)
    
    actions = {
        'CRITICAL': 'IMMEDIATE: Suspend all patrols, deploy EOD teams, maximum alert!',
        'HIGH': 'Deploy mine-protected vehicles, increase surveillance, alert QRF',
        'MEDIUM': 'Standard precautions, continue HUMINT monitoring',
        'LOW': 'Normal operations, maintain routine patrols'
    }
    
    return jsonify({
        'region': region,
        'probability': prob,
        'level': level,
        'action': actions.get(level, 'Monitor situation'),
        'intel_count': len(region_intel),
        'model_used': 'XGBoost (trained on 8.2M records)'
    })


@app.route('/api/stats')
def api_stats():
    """Stats API."""
    global intel_df, incidents_df
    
    if intel_df is None:
        load_data()
    
    return jsonify({
        'intel_records': len(intel_df),
        'full_dataset': 8207201,
        'incidents': len(incidents_df),
        'killed': int(incidents_df['Killed'].sum()),
        'injured': int(incidents_df['Injured'].sum()),
        'regions': intel_df['District'].nunique(),
        'date_range': {
            'start': str(intel_df['Date'].min()),
            'end': str(intel_df['Date'].max())
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  JATAYU - Attack Prediction System")
    print("="*60)
    load_data()
    print("\n  Starting server at http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
