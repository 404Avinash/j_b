"""
JATAYU - Full Integration Server
=================================
Serves the HTML frontend + real 8.2M intel data via API.

Run: python jatayu_server.py
Open: http://localhost:5000

This integrates the ACTUAL 8.2M intel dataset!
"""

import os
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*60)
print("  JATAYU - Full Data Integration Server")
print("="*60)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'all_intel_2020_2026.csv')
INCIDENTS_PATH = os.path.join(BASE_DIR, 'data', 'raw_incidents.csv')

# Global data holders
intel_df = None
incidents_df = None
intel_by_date_region = {}

def load_data():
    """Load the full 8.2M dataset."""
    global intel_df, incidents_df, intel_by_date_region
    
    print("\n[1/3] Loading incidents...")
    incidents_df = pd.read_csv(INCIDENTS_PATH)
    incidents_df['Date'] = pd.to_datetime(incidents_df['Date']).dt.strftime('%Y-%m-%d')
    print(f"      Loaded {len(incidents_df)} incidents")
    
    print("\n[2/3] Loading 8.2M intel records (this takes ~2 minutes)...")
    print("      Please wait...")
    
    # Load in chunks for progress
    chunks = []
    chunk_size = 500000
    total_rows = 0
    
    # Check if main file exists, else look for parts
    if os.path.exists(DATA_PATH):
        print(f"      Loading from {DATA_PATH}...")
        for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size)):
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"      Loaded {total_rows:,} records...", end='\r')
    else:
        # Load parts
        print(f"      Main file not found. Checking for split parts...")
        import glob
        base_name = os.path.basename(DATA_PATH).replace('.csv', '_part*.csv')
        part_pattern = os.path.join(os.path.dirname(DATA_PATH), base_name)
        parts = sorted(glob.glob(part_pattern))
        
        if not parts:
            print("[ERROR] No data found! Please run data generation.")
            return

        print(f"      Found {len(parts)} split files. Loading...")
        for part_file in parts:
            print(f"      Loading {os.path.basename(part_file)}...")
            chunk = pd.read_csv(part_file)
            chunks.append(chunk)
            total_rows += len(chunk)
            
    if chunks:
        intel_df = pd.concat(chunks, ignore_index=True)
    else:
        intel_df = pd.DataFrame()
        
    print(f"\n      Total: {len(intel_df):,} intel records")
    
    print("\n[3/3] Preparing data index...")
    # Just keep the dataframe - we'll filter on demand
    # Create a simple district index for faster filtering
    intel_df['DateStr'] = intel_df['Date']
    print(f"      Index ready!")
    print(f"\n[READY] Starting server on http://localhost:5000")
    print(f"        Open this URL in your browser!")


class JatayuHandler(SimpleHTTPRequestHandler):
    """Custom handler for JATAYU server."""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # API endpoints
        if parsed.path == '/api/intel':
            self.handle_intel_request(parsed.query)
        elif parsed.path == '/api/stats':
            self.handle_stats_request()
        elif parsed.path == '/api/regions':
            self.handle_regions_request()
        elif parsed.path == '/api/dates':
            self.handle_dates_request(parsed.query)
        elif parsed.path == '/' or parsed.path == '':
            # Serve the HTML
            self.path = '/jatayu_full.html'
            super().do_GET()
        else:
            super().do_GET()
    
    def send_json(self, data):
        """Send JSON response."""
        response = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode())
    
    def handle_intel_request(self, query):
        """Get intel for specific date and region."""
        global intel_df
        params = parse_qs(query)
        date = params.get('date', [''])[0]
        region = params.get('region', [''])[0]
        
        if not date or not region:
            self.send_json({'error': 'date and region required', 'intel': []})
            return
        
        # Filter from dataframe
        mask = (intel_df['Date'].astype(str).str.contains(date)) & \
               (intel_df['District'].str.contains(region, case=False, na=False))
        filtered = intel_df[mask]
        
        # Convert to list of dicts (first 50)
        intel_list = filtered.head(50).to_dict('records')
        
        self.send_json({
            'date': date,
            'region': region,
            'total': len(filtered),
            'intel': intel_list
        })
    
    def handle_stats_request(self):
        """Get overall stats."""
        global intel_df
        
        total = len(intel_df)
        true_count = int((intel_df['Label'] == 'TRUE_SIGNAL').sum())
        noise_count = int((intel_df['Label'] == 'NOISE').sum())
        deception_count = int((intel_df['Label'] == 'DECEPTION').sum())
        
        stats = {
            'total_intel': total,
            'true_signals': true_count,
            'true_pct': round(true_count/total*100, 1),
            'noise': noise_count,
            'noise_pct': round(noise_count/total*100, 1),
            'deception': deception_count,
            'deception_pct': round(deception_count/total*100, 1),
            'date_range': {
                'start': intel_df['Date'].min(),
                'end': intel_df['Date'].max()
            },
            'regions': intel_df['District'].nunique(),
            'incidents': len(incidents_df)
        }
        self.send_json(stats)
    
    def handle_regions_request(self):
        """Get list of regions."""
        regions = sorted(intel_df['District'].dropna().unique().tolist())
        self.send_json({'regions': regions})
    
    def handle_dates_request(self, query):
        """Get dates with intel for a region."""
        params = parse_qs(query)
        region = params.get('region', [''])[0]
        
        if region:
            region_intel = intel_df[intel_df['District'].str.contains(region, case=False, na=False)]
            dates = sorted(region_intel['Date'].unique().tolist())
        else:
            dates = sorted(intel_df['Date'].unique().tolist())
        
        self.send_json({'dates': dates[-100:]})  # Last 100 dates


def main():
    load_data()
    
    server = HTTPServer(('0.0.0.0', 5000), JatayuHandler)
    print(f"\n{'='*60}")
    print(f"  Server running at: http://localhost:5000")
    print(f"  8.2M intel records loaded and ready!")
    print(f"{'='*60}\n")
    
    # Auto-open browser
    webbrowser.open('http://localhost:5000')
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
