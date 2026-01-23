"""
Bastar Intelligence Fusion Dashboard

Streamlit-based analyst interface for threat monitoring.

Features:
- Interactive threat map with attack locations
- Daily risk timeline
- Alert generation with explanations
- Feature importance visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="JATAYU - Bastar Intelligence Fusion",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLES
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-critical {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #fd7e14 0%, #e55a14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-low {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    .alert-box {
        background: #2d2d2d;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    """Load intelligence data and feature matrix."""
    project_root = Path(__file__).parent.parent.parent
    
    # Load intelligence data
    intel_path = project_root / "data" / "bastar_intelligence_15k.csv"
    if intel_path.exists():
        intel_df = pd.read_csv(intel_path)
        intel_df['timestamp'] = pd.to_datetime(intel_df['timestamp'])
        intel_df['date'] = intel_df['timestamp'].dt.date
    else:
        intel_df = None
    
    # Load feature matrix
    feature_path = project_root / "data" / "feature_matrix.csv"
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        feature_df['date'] = pd.to_datetime(feature_df['date'])
    else:
        feature_df = None
    
    # Load model
    model_path = project_root / "results" / "ensemble_model.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None
    
    return intel_df, feature_df, model

@st.cache_data
def get_attacks():
    """Return known attacks."""
    return [
        {'id': 1, 'date': datetime(2025, 1, 6), 'district': 'Bijapur', 'village': 'Ambeli', 
         'lat': 18.50, 'lon': 81.00, 'kia': 9, 'wia': 0, 'target': 'DRG Vehicle'},
        {'id': 2, 'date': datetime(2025, 1, 12), 'district': 'Sukma', 'village': 'Timmapuram',
         'lat': 18.15, 'lon': 81.25, 'kia': 0, 'wia': 1, 'target': 'Civilian'},
        {'id': 3, 'date': datetime(2025, 1, 16), 'district': 'Bijapur', 'village': 'Putkel',
         'lat': 18.62, 'lon': 80.88, 'kia': 0, 'wia': 2, 'target': 'CoBRA Patrol'},
        {'id': 4, 'date': datetime(2025, 1, 17), 'district': 'Narayanpur', 'village': 'Garpa',
         'lat': 18.45, 'lon': 80.95, 'kia': 0, 'wia': 2, 'target': 'BSF ROP'},
    ]

# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 JATAYU</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Bastar Intelligence Fusion System | Operation Silent Watch</p>', unsafe_allow_html=True)
    
    # Load data
    intel_df, feature_df, model = load_data()
    attacks = get_attacks()
    
    # Sidebar
    st.sidebar.header("⚙️ Controls")
    
    if feature_df is not None:
        min_date = feature_df['date'].min().date()
        max_date = feature_df['date'].max().date()
        selected_date = st.sidebar.date_input(
            "Analysis Date",
            value=datetime(2025, 1, 15).date(),
            min_value=min_date,
            max_value=max_date
        )
    else:
        selected_date = datetime(2025, 1, 15).date()
    
    threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    if intel_df is not None:
        st.sidebar.metric("Total Records", f"{len(intel_df):,}")
        st.sidebar.metric("Date Range", f"{intel_df['timestamp'].min().date()} to {intel_df['timestamp'].max().date()}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Risk Overview", "🗺️ Threat Map", "🔍 Intelligence Analysis", "📋 Alerts"])
    
    # ============================================
    # TAB 1: RISK OVERVIEW
    # ============================================
    with tab1:
        st.header("Daily Risk Assessment")
        
        if feature_df is not None:
            # Calculate risk for all dates
            feature_cols = [c for c in feature_df.columns 
                          if c not in ['date', 'target_attack_imminent', 'target_attack_tomorrow']]
            
            # Simple risk score based on features
            feature_df['risk_score'] = (
                feature_df['spatial_total_high_urgency'] * 0.2 +
                feature_df['semantic_ied_mention_count'] * 0.3 +
                feature_df['network_source_corroboration_score'] * 0.2 +
                feature_df['temporal_urgency_velocity'] * 0.15 +
                (100 - feature_df['spatial_min_distance_to_attack']) / 100 * 0.15
            )
            feature_df['risk_score'] = feature_df['risk_score'].clip(0, 1)
            
            # Current day metrics
            col1, col2, col3, col4 = st.columns(4)
            
            selected_row = feature_df[feature_df['date'].dt.date == selected_date]
            
            if len(selected_row) > 0:
                risk_score = selected_row['risk_score'].values[0]
                
                # Determine risk level
                if risk_score >= 0.7:
                    risk_level = "CRITICAL"
                    risk_class = "risk-critical"
                elif risk_score >= 0.5:
                    risk_level = "HIGH"
                    risk_class = "risk-high"
                elif risk_score >= 0.3:
                    risk_level = "MEDIUM"
                    risk_class = "risk-medium"
                else:
                    risk_level = "LOW"
                    risk_class = "risk-low"
                
                with col1:
                    st.markdown(f'<div class="{risk_class}"><h2>{risk_level}</h2><p>Threat Level</p></div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Risk Score", f"{risk_score*100:.0f}%")
                
                with col3:
                    st.metric("High-Urgency Reports", int(selected_row['spatial_total_high_urgency'].values[0]))
                
                with col4:
                    st.metric("IED Mentions", int(selected_row['semantic_ied_mention_count'].values[0]))
            
            # Risk timeline
            st.subheader("📊 Risk Timeline")
            
            fig = go.Figure()
            
            # Add risk line
            fig.add_trace(go.Scatter(
                x=feature_df['date'],
                y=feature_df['risk_score'] * 100,
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#dc3545', width=2),
                marker=dict(size=6)
            ))
            
            # Add threshold line
            fig.add_hline(y=threshold*100, line_dash="dash", line_color="orange",
                         annotation_text=f"Threshold ({threshold*100:.0f}%)")
            
            # Mark attacks
            for attack in attacks:
                fig.add_vline(x=attack['date'], line_dash="dot", line_color="red", opacity=0.7)
                fig.add_annotation(
                    x=attack['date'],
                    y=100,
                    text=f"Attack #{attack['id']}",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=10)
                )
            
            # Mark selected date
            fig.add_vline(x=datetime.combine(selected_date, datetime.min.time()), 
                         line_color="blue", line_width=2)
            
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Risk Score (%)",
                yaxis_range=[0, 105],
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Attack summary
            st.subheader("⚔️ Attack Summary")
            attack_df = pd.DataFrame(attacks)
            attack_df['date'] = pd.to_datetime(attack_df['date']).dt.strftime('%d %b %Y')
            attack_df['casualties'] = attack_df['kia'].astype(str) + ' KIA, ' + attack_df['wia'].astype(str) + ' WIA'
            st.dataframe(
                attack_df[['id', 'date', 'district', 'village', 'target', 'casualties']],
                use_container_width=True,
                hide_index=True
            )
    
    # ============================================
    # TAB 2: THREAT MAP
    # ============================================
    with tab2:
        st.header("🗺️ Geographic Threat Analysis")
        
        if intel_df is not None and HAS_FOLIUM:
            # Create map
            m = folium.Map(location=[18.5, 81.0], zoom_start=8, tiles='cartodbdark_matter')
            
            # Add attack markers
            for attack in attacks:
                color = 'red' if attack['kia'] > 0 else 'orange'
                folium.CircleMarker(
                    [attack['lat'], attack['lon']],
                    radius=10 + attack['kia'] * 2,
                    color=color,
                    fill=True,
                    popup=f"<b>Attack #{attack['id']}</b><br>{attack['date'].strftime('%d %b %Y')}<br>{attack['village']}, {attack['district']}<br>{attack['kia']} KIA, {attack['wia']} WIA"
                ).add_to(m)
            
            # Add heatmap of intelligence activity
            selected_intel = intel_df[intel_df['date'] == selected_date]
            heat_data = selected_intel[
                selected_intel['location_lat'].notna() & selected_intel['location_lon'].notna()
            ][['location_lat', 'location_lon']].values.tolist()
            
            if heat_data:
                HeatMap(heat_data, radius=15).add_to(m)
            
            # Show map
            from streamlit_folium import st_folium
            st_folium(m, width=None, height=500)
        else:
            st.info("Map visualization requires folium and streamlit-folium packages.")
            
            # Fallback: Plotly scatter
            if intel_df is not None:
                selected_intel = intel_df[intel_df['date'] == selected_date]
                
                fig = px.scatter(
                    selected_intel[selected_intel['location_lat'].notna()],
                    x='location_lon',
                    y='location_lat',
                    color='type',
                    size_max=10,
                    title=f"Intelligence Activity on {selected_date}"
                )
                
                # Add attack points
                for attack in attacks:
                    fig.add_trace(go.Scatter(
                        x=[attack['lon']],
                        y=[attack['lat']],
                        mode='markers',
                        marker=dict(size=20, color='red', symbol='star'),
                        name=f"Attack #{attack['id']}"
                    ))
                
                fig.update_layout(height=500, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 3: INTELLIGENCE ANALYSIS
    # ============================================
    with tab3:
        st.header("🔍 Intelligence Breakdown")
        
        if intel_df is not None:
            selected_intel = intel_df[intel_df['date'] == selected_date]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Intelligence by Type")
                type_counts = selected_intel['type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index, 
                            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Urgency Distribution")
                urgency_counts = selected_intel['urgency'].value_counts()
                colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
                fig = px.bar(x=urgency_counts.index, y=urgency_counts.values,
                            color=urgency_counts.index, color_discrete_map=colors)
                fig.update_layout(height=300, showlegend=False, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            # High-urgency reports
            st.subheader("⚠️ High-Urgency Intelligence")
            high_urgency = selected_intel[selected_intel['urgency'].isin(['HIGH', 'CRITICAL'])]
            
            if len(high_urgency) > 0:
                display_cols = ['timestamp', 'type', 'district', 'urgency']
                text_col = None
                for col in ['report_text', 'transcript', 'analysis', 'observations']:
                    if col in high_urgency.columns:
                        text_col = col
                        display_cols.append(col)
                        break
                
                st.dataframe(high_urgency[display_cols].head(10), use_container_width=True, hide_index=True)
            else:
                st.info("No high-urgency reports for this date.")
            
            # Source reliability
            st.subheader("👤 Source Analysis")
            humint = selected_intel[selected_intel['type'] == 'HUMINT']
            if 'source_reliability' in humint.columns and len(humint) > 0:
                fig = px.histogram(humint, x='source_reliability', nbins=10,
                                  title='Source Reliability Distribution')
                fig.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 4: ALERTS
    # ============================================
    with tab4:
        st.header("📋 Alert Generation")
        
        if feature_df is not None:
            selected_row = feature_df[feature_df['date'].dt.date == selected_date]
            
            if len(selected_row) > 0:
                risk_score = selected_row['risk_score'].values[0]
                
                # Generate alert text
                if risk_score >= 0.7:
                    risk_level = "CRITICAL"
                elif risk_score >= 0.5:
                    risk_level = "HIGH"
                elif risk_score >= 0.3:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                # Build reasons
                reasons = []
                
                if selected_row['semantic_ied_mention_count'].values[0] >= 3:
                    reasons.append(f"{int(selected_row['semantic_ied_mention_count'].values[0])} direct IED mentions in intelligence")
                
                if selected_row['spatial_total_high_urgency'].values[0] >= 5:
                    reasons.append(f"{int(selected_row['spatial_total_high_urgency'].values[0])} high-urgency reports")
                
                if selected_row['temporal_urgency_velocity'].values[0] >= 1.5:
                    reasons.append(f"Urgency velocity {selected_row['temporal_urgency_velocity'].values[0]:.1f}x above normal")
                
                if selected_row['network_source_corroboration_score'].values[0] >= 0.5:
                    reasons.append("Multiple sources corroborating same area")
                
                if selected_row['spatial_min_distance_to_attack'].values[0] <= 20:
                    reasons.append(f"Activity within {selected_row['spatial_min_distance_to_attack'].values[0]:.0f}km of past attack")
                
                if not reasons:
                    reasons.append("Routine intelligence activity levels")
                
                # Format alert
                alert_text = f"""
═══════════════════════════════════════════════════════════
⚠️  THREAT ALERT - {selected_date.strftime('%d %b %Y').upper()}
═══════════════════════════════════════════════════════════
RISK LEVEL: {risk_level} ({risk_score*100:.0f}%)

KEY INDICATORS:
"""
                for i, reason in enumerate(reasons):
                    prefix = "├─" if i < len(reasons) - 1 else "└─"
                    alert_text += f"{prefix} {reason}\n"
                
                if risk_score >= 0.5:
                    alert_text += """
RECOMMENDED ACTIONS:
☐ Delay patrol on high-risk routes
☐ Deploy EOD sweep before movement  
☐ Increase UAV coverage
☐ Cross-verify with additional sources
"""
                
                alert_text += "═══════════════════════════════════════════════════════════"
                
                st.markdown(f'<div class="alert-box">{alert_text}</div>', unsafe_allow_html=True)
                
                # Feature breakdown
                st.subheader("📊 Feature Values")
                
                feature_cols = [c for c in selected_row.columns 
                              if c.startswith(('temporal_', 'semantic_', 'network_', 'spatial_')) 
                              and selected_row[c].values[0] != 0]
                
                feature_data = []
                for col in feature_cols:
                    feature_data.append({
                        'Feature': col.replace('_', ' ').title(),
                        'Value': f"{selected_row[col].values[0]:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>JATAYU v1.0 | Bastar Intelligence Fusion System | Operation Silent Watch</p>
            <p>⚠️ CLASSIFIED - FOR AUTHORIZED PERSONNEL ONLY</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_dashboard():
    """Entry point for dashboard."""
    main()


if __name__ == "__main__":
    main()
