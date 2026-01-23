"""
JATAYU Real Data Dashboard
===========================
Interactive Streamlit dashboard using actual IED incident data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from src.data.real_data_loader import RealDataLoader
from src.models.real_trainer import RealMLTrainer


@st.cache_data
def load_data():
    """Load and cache the incident data."""
    loader = RealDataLoader()
    df = loader.clean_data()
    stats = loader.get_statistics()
    clusters = loader.get_attack_clusters(days_threshold=7)
    return df, stats, clusters


@st.cache_resource
def train_model():
    """Train and cache the ML model."""
    trainer = RealMLTrainer()
    trainer.prepare_data()
    results = trainer.train_model(train_end_date='2024-12-31', prediction_window_days=7)
    return trainer, results


def main():
    """Main dashboard function."""
    
    st.set_page_config(
        page_title="JATAYU - Predictive Intelligence System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .risk-critical { background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; }
    .risk-high { background-color: #fd7e14; color: white; padding: 10px; border-radius: 5px; }
    .risk-medium { background-color: #ffc107; color: black; padding: 10px; border-radius: 5px; }
    .risk-low { background-color: #28a745; color: white; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🛡️ JATAYU</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive Intelligence Fusion System | Operation Silent Watch</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real IED Incident Data | Red Corridor (2020-2026)</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Load data
    with st.spinner("Loading intelligence data..."):
        df, stats, clusters = load_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
        st.title("Control Panel")
        
        st.subheader("📅 Date Filter")
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
        
        st.subheader("📍 Region Filter")
        selected_states = st.multiselect(
            "States",
            options=df['State'].unique().tolist(),
            default=df['State'].unique().tolist()
        )
        
        st.subheader("🎯 Attack Type Filter")
        selected_types = st.multiselect(
            "Attack Types",
            options=df['Attack_Type'].unique().tolist(),
            default=df['Attack_Type'].unique().tolist()
        )
        
        st.divider()
        
        st.subheader("🤖 ML Model")
        if st.button("🔄 Train/Refresh Model"):
            st.cache_resource.clear()
            st.rerun()
    
    # Apply filters
    filtered_df = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['State'].isin(selected_states)) &
        (df['Attack_Type'].isin(selected_types))
    ]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🗺️ Threat Map", "📈 Trends", "🔮 Predictions", "📋 Data"
    ])
    
    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tab1:
        st.header("Intelligence Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Incidents", len(filtered_df), 
                     delta=f"{len(filtered_df) - len(df)//2}" if len(filtered_df) > len(df)//2 else None)
        
        with col2:
            st.metric("Total Killed", filtered_df['Killed'].sum(),
                     delta_color="inverse")
        
        with col3:
            st.metric("Total Injured", filtered_df['Injured'].sum(),
                     delta_color="inverse")
        
        with col4:
            major = (filtered_df['Total_Casualties'] >= 3).sum()
            st.metric("Major Attacks", major)
        
        with col5:
            monthly_avg = len(filtered_df) / max(1, (date_range[1] - date_range[0]).days / 30)
            st.metric("Monthly Rate", f"{monthly_avg:.1f}")
        
        st.divider()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Incidents by State")
            state_counts = filtered_df.groupby('State').size().reset_index(name='Count')
            fig = px.pie(state_counts, values='Count', names='State', hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Attack Types Distribution")
            type_counts = filtered_df.groupby('Attack_Type').size().reset_index(name='Count')
            fig = px.bar(type_counts, x='Attack_Type', y='Count',
                        color='Count', color_continuous_scale='Reds')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top hotspots
        st.subheader("🔥 Hotspot Districts")
        district_stats = filtered_df.groupby('District').agg({
            'Killed': 'sum',
            'Injured': 'sum',
            'Is_Major_Attack': 'sum'
        }).reset_index()
        district_stats['Total_Incidents'] = filtered_df.groupby('District').size().values
        district_stats['Risk_Score'] = (
            district_stats['Total_Incidents'] * 0.4 +
            district_stats['Killed'] * 0.3 +
            district_stats['Is_Major_Attack'] * 0.3
        )
        district_stats = district_stats.sort_values('Risk_Score', ascending=False).head(10)
        
        fig = px.bar(district_stats, y='District', x='Total_Incidents', orientation='h',
                    color='Killed', color_continuous_scale='YlOrRd',
                    hover_data=['Killed', 'Injured', 'Is_Major_Attack'])
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 2: THREAT MAP
    # =========================================================================
    with tab2:
        st.header("Geographic Threat Map")
        
        if HAS_FOLIUM:
            # Create base map centered on Bastar
            m = folium.Map(location=[19.5, 81.5], zoom_start=7, tiles='cartodbpositron')
            
            # Add incident markers
            for _, row in filtered_df.iterrows():
                color = 'red' if row['Killed'] > 0 else 'orange' if row['Injured'] > 0 else 'yellow'
                radius = 5 + row['Total_Casualties'] * 2
                
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    popup=f"""
                    <b>{row['Date'].strftime('%Y-%m-%d')}</b><br>
                    {row['District']}, {row['State']}<br>
                    Killed: {row['Killed']}, Injured: {row['Injured']}<br>
                    Type: {row['Attack_Type']}
                    """
                ).add_to(m)
            
            # Add heatmap layer
            from folium.plugins import HeatMap
            heat_data = filtered_df[['Latitude', 'Longitude', 'Total_Casualties']].values.tolist()
            HeatMap(heat_data, radius=20, blur=15).add_to(m)
            
            st_folium(m, width=None, height=600)
            
            # Legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🔴 **Fatal Incident** (Killed > 0)")
            with col2:
                st.markdown("🟠 **Injury Incident** (Injured only)")
            with col3:
                st.markdown("🟡 **No Casualty Incident**")
        else:
            st.warning("Folium not installed. Install with: pip install folium streamlit-folium")
            
            # Fallback scatter plot
            fig = px.scatter_geo(
                filtered_df,
                lat='Latitude',
                lon='Longitude',
                color='District',
                size='Total_Casualties',
                hover_name='District',
                hover_data=['Date', 'Killed', 'Injured'],
                title="IED Incidents Map"
            )
            fig.update_geos(
                center=dict(lat=19.5, lon=81.5),
                projection_scale=15,
                showland=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 3: TRENDS
    # =========================================================================
    with tab3:
        st.header("Temporal Trends Analysis")
        
        # Timeline
        st.subheader("📅 Attack Timeline")
        timeline_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).agg({
            'Killed': 'sum',
            'Injured': 'sum'
        }).reset_index()
        timeline_df['Date'] = timeline_df['Date'].astype(str)
        timeline_df['Total_Incidents'] = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).size().values
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline_df['Date'], y=timeline_df['Total_Incidents'],
                                mode='lines+markers', name='Incidents', line=dict(color='blue')))
        fig.add_trace(go.Bar(x=timeline_df['Date'], y=timeline_df['Killed'],
                            name='Killed', marker_color='red', opacity=0.6))
        fig.add_trace(go.Bar(x=timeline_df['Date'], y=timeline_df['Injured'],
                            name='Injured', marker_color='orange', opacity=0.6))
        fig.update_layout(height=400, barmode='group', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Year-over-Year")
            yearly = filtered_df.groupby('Year').agg({
                'Killed': 'sum',
                'Injured': 'sum'
            }).reset_index()
            yearly['Incidents'] = filtered_df.groupby('Year').size().values
            
            fig = px.bar(yearly, x='Year', y='Incidents', text='Incidents',
                        color='Killed', color_continuous_scale='YlOrRd')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📆 Monthly Pattern")
            monthly = filtered_df.groupby('Month').size().reset_index(name='Incidents')
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly['Month_Name'] = monthly['Month'].apply(lambda x: months[x-1])
            
            fig = px.bar(monthly, x='Month_Name', y='Incidents',
                        color='Incidents', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Attack clusters
        st.subheader("⚠️ Attack Clusters (Multiple attacks within 7 days)")
        
        cluster_data = []
        for c in clusters:
            cluster_data.append({
                'Start': c['start_date'].strftime('%Y-%m-%d'),
                'End': c['end_date'].strftime('%Y-%m-%d'),
                'Attacks': c['num_attacks'],
                'Killed': c['total_killed'],
                'Injured': c['total_injured'],
                'Districts': ', '.join(set(c['locations']))
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        cluster_df = cluster_df.sort_values('Killed', ascending=False)
        st.dataframe(cluster_df, use_container_width=True, height=300)
    
    # =========================================================================
    # TAB 4: PREDICTIONS
    # =========================================================================
    with tab4:
        st.header("🔮 ML Predictions")
        
        with st.spinner("Loading ML model..."):
            trainer, results = train_model()
        
        # Model performance
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Model Performance")
            metrics = results.get('metrics', {})
            
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
            st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
            st.metric("F1 Score", f"{metrics.get('f1', 0)*100:.1f}%")
            st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        
        with col2:
            st.subheader("Feature Importance")
            importance = results.get('feature_importance', pd.DataFrame())
            if len(importance) > 0:
                fig = px.bar(importance.head(10), y='feature', x='importance',
                            orientation='h', color='importance',
                            color_continuous_scale='Viridis')
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Current prediction
        st.subheader("⚡ Current Risk Assessment")
        
        prediction = trainer.predict_next_attack(as_of_date=df['Date'].max())
        
        prob = prediction['probability']
        risk_level = prediction['risk_level']
        
        # Risk meter
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Attack Probability (Next 7 Days)<br><b>Risk: {risk_level}</b>"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 50], 'color': "yellow"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk districts
        st.subheader("🎯 High-Risk Districts")
        risk_districts = prediction['high_risk_districts']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        for i, (_, row) in enumerate(risk_districts.head(5).iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5365c, #f56036); 
                            padding: 15px; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0;">{row['District']}</h3>
                    <p style="margin: 5px 0; font-size: 24px;"><b>{int(row['total_incidents'])}</b></p>
                    <p style="margin: 0;">incidents</p>
                    <p style="margin: 5px 0; font-size: 18px;">💀 {int(row['Killed'])} killed</p>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 5: DATA
    # =========================================================================
    with tab5:
        st.header("📋 Raw Incident Data")
        
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            filtered_df.to_csv(index=False),
            "jatayu_incidents.csv",
            "text/csv"
        )
        
        st.dataframe(
            filtered_df[[
                'Date', 'State', 'District', 'Location', 'Attack_Type',
                'Killed', 'Injured', 'Description'
            ]].sort_values('Date', ascending=False),
            use_container_width=True,
            height=600
        )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><b>JATAYU - Predictive Intelligence Fusion System</b></p>
        <p>Operation Silent Watch | Developed for Hackathon 2025</p>
        <p>⚠️ CLASSIFIED - FOR DEMONSTRATION PURPOSES ONLY</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
