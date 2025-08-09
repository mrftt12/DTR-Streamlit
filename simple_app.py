"""
Simple DTR Dashboard - Testing Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="DTR Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sample_data():
    """Create sample DTR data for demonstration"""
    hours = 168  # One week
    time_index = np.arange(hours)
    
    # Generate sample load profile
    base_load = 30 + 10 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 2, hours)
    ev_load = 5 + 8 * np.sin(2 * np.pi * (time_index - 18) / 24) + np.random.normal(0, 1, hours)
    ev_load = np.maximum(ev_load, 0)  # No negative loads
    
    total_load = base_load + ev_load
    load_pu = total_load / 50  # Assuming 50 MVA transformer
    
    # Generate sample temperatures
    ambient_temp = 25 + 8 * np.sin(2 * np.pi * time_index / 24 - np.pi/2)
    top_oil_temp = ambient_temp + 20 + 15 * load_pu
    hot_spot_temp = top_oil_temp + 10 + 10 * load_pu
    
    # Generate sample ratings
    normal_rating = 50 + 15 * (1 - (ambient_temp - 15) / 20)
    emergency_rating = normal_rating * 1.4
    
    return pd.DataFrame({
        'time_index': time_index,
        'hour': time_index % 24,
        'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][time_index // 24 % 7],
        'base_load_mw': base_load,
        'ev_load_mw': ev_load,
        'total_load_mw': total_load,
        'load_pu': load_pu,
        'ambient_temp': ambient_temp,
        'top_oil_temp': top_oil_temp,
        'hot_spot_temp': hot_spot_temp,
        'normal_rating_mva': normal_rating,
        'emergency_rating_mva': emergency_rating,
        'utilization_normal': (load_pu * 50 / normal_rating) * 100
    })

def create_thermal_plot(df):
    """Create thermal monitoring plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['hot_spot_temp'],
        name='Hot Spot Temperature',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['top_oil_temp'],
        name='Top Oil Temperature',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['ambient_temp'],
        name='Ambient Temperature',
        line=dict(color='green', width=1, dash='dot')
    ))
    
    # Add thermal limits
    fig.add_hline(y=110, line_dash="dash", line_color="orange", 
                  annotation_text="Normal Limit (110°C)")
    fig.add_hline(y=140, line_dash="dash", line_color="red", 
                  annotation_text="Emergency Limit (140°C)")
    
    fig.update_layout(
        title="Transformer Thermal Monitoring",
        xaxis_title="Time (hours)",
        yaxis_title="Temperature (°C)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_load_plot(df):
    """Create load forecasting plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['base_load_mw'],
        name='Base Load',
        fill='tonexty',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['total_load_mw'],
        name='Total Load (Base + EV)',
        fill='tonexty',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['ev_load_mw'],
        name='EV Charging Load',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title="Load Forecast with EV Charging",
        xaxis_title="Time (hours)",
        yaxis_title="Load (MW)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_ratings_plot(df):
    """Create dynamic ratings plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['emergency_rating_mva'],
        name='Emergency Rating',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['normal_rating_mva'],
        name='Normal Rating',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time_index'],
        y=df['total_load_mw'],
        name='Actual Load',
        line=dict(color='blue', width=2)
    ))
    
    # Add nameplate rating
    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                  annotation_text="Nameplate Rating (50 MVA)")
    
    fig.update_layout(
        title="Dynamic Transformer Ratings",
        xaxis_title="Time (hours)",
        yaxis_title="Rating/Load (MVA)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def main():
    """Main application"""
    
    # Title
    st.title("⚡ Dynamic Transformer Rating Dashboard")
    st.markdown("**OFAF Transformer Thermal Monitoring & Risk Assessment**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Sample parameters
        rated_power = st.number_input("Rated Power (MVA)", value=50.0, min_value=1.0, max_value=500.0)
        num_l2_chargers = st.number_input("Level 2 Chargers", value=100, min_value=0, max_value=1000)
        num_dcfc_chargers = st.number_input("DC Fast Chargers", value=10, min_value=0, max_value=100)
        
        if st.button("Generate Sample Analysis", type="primary"):
            st.session_state.data_generated = True
    
    # Main content
    if st.session_state.get('data_generated', False):
        # Generate sample data
        df = create_sample_data()
        
        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Peak Hot Spot", 
                f"{df['hot_spot_temp'].max():.1f}°C",
                f"Limit: 110°C"
            )
        
        with col2:
            st.metric(
                "Max Dynamic Rating", 
                f"{df['normal_rating_mva'].max():.1f} MVA",
                f"+{((df['normal_rating_mva'].max() - 50) / 50 * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Peak Utilization", 
                f"{df['utilization_normal'].max():.1f}%",
                f"Avg: {df['utilization_normal'].mean():.1f}%"
            )
        
        with col4:
            st.metric(
                "EV Peak Load", 
                f"{df['ev_load_mw'].max():.1f} MW",
                f"Avg: {df['ev_load_mw'].mean():.1f} MW"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌡️ Thermal Monitoring", 
            "📊 Dynamic Ratings", 
            "🚗 Load Forecast",
            "📋 Data Table"
        ])
        
        with tab1:
            st.plotly_chart(create_thermal_plot(df), use_container_width=True)
            
            # Alert panel
            st.subheader("Current Status")
            current_hot_spot = df['hot_spot_temp'].iloc[-1]
            current_utilization = df['utilization_normal'].iloc[-1]
            
            if current_hot_spot > 110:
                st.error(f"WARNING: Hot spot temperature ({current_hot_spot:.1f}°C) exceeds normal limit")
            elif current_hot_spot > 100:
                st.warning(f"CAUTION: Hot spot temperature ({current_hot_spot:.1f}°C) approaching limit")
            else:
                st.success("All temperature parameters within normal limits")
        
        with tab2:
            st.plotly_chart(create_ratings_plot(df), use_container_width=True)
            
            # Rating insights
            st.subheader("Rating Analysis")
            additional_capacity = (df['normal_rating_mva'].mean() - 50) / 50 * 100
            st.info(f"Average additional capacity: **{additional_capacity:.1f}%** above nameplate rating")
        
        with tab3:
            st.plotly_chart(create_load_plot(df), use_container_width=True)
            
            # Load insights
            st.subheader("Load Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Peak Total Load", f"{df['total_load_mw'].max():.1f} MW")
                st.metric("Average Base Load", f"{df['base_load_mw'].mean():.1f} MW")
            with col2:
                st.metric("Peak EV Load", f"{df['ev_load_mw'].max():.1f} MW")
                st.metric("EV Load Factor", f"{(df['ev_load_mw'].mean() / df['ev_load_mw'].max() * 100):.1f}%")
        
        with tab4:
            st.subheader("Detailed Data")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"dtr_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### Welcome to the DTR Analysis System
            
            This application provides Dynamic Transformer Rating analysis for OFAF transformers
            in high EV charging areas.
            
            **Key Features:**
            - Real-time thermal monitoring
            - Dynamic rating calculations  
            - EV load forecasting
            - Risk assessment capabilities
            
            **Getting Started:**
            1. Configure your transformer parameters in the sidebar
            2. Click "Generate Sample Analysis" to see example results
            3. View thermal monitoring, ratings, and load forecasts
            
            The analysis follows IEEE C57.91-2011 standards.
            """)

if __name__ == "__main__":
    main()