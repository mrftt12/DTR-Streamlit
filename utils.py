"""
Utility functions for the DTR Streamlit application
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
import io
import base64


def create_thermal_monitoring_plot(thermal_data: Dict) -> go.Figure:
    """Create thermal monitoring plot with temperature curves"""
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Hot Spot Temperature', 'Top Oil Temperature', 'Load Profile'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    time = thermal_data['time']
    
    # Hot spot temperature
    fig.add_trace(
        go.Scatter(
            x=time, 
            y=thermal_data['hot_spot'],
            name='Hot Spot Temp',
            line=dict(color='red', width=2),
            hovertemplate='Hour: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add thermal limits
    fig.add_hline(y=110, line_dash="dash", line_color="orange", 
                  annotation_text="Normal Limit (110°C)", row=1, col=1)
    fig.add_hline(y=140, line_dash="dash", line_color="red", 
                  annotation_text="Emergency Limit (140°C)", row=1, col=1)
    
    # Top oil temperature
    fig.add_trace(
        go.Scatter(
            x=time,
            y=thermal_data['top_oil'],
            name='Top Oil Temp',
            line=dict(color='blue', width=2),
            hovertemplate='Hour: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Ambient temperature
    fig.add_trace(
        go.Scatter(
            x=time,
            y=thermal_data['ambient'],
            name='Ambient Temp',
            line=dict(color='green', width=1, dash='dot'),
            hovertemplate='Hour: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Load profile
    fig.add_trace(
        go.Scatter(
            x=time,
            y=thermal_data['load_pu'],
            name='Loading (p.u.)',
            line=dict(color='purple', width=2),
            fill='tonexty',
            hovertemplate='Hour: %{x}<br>Load: %{y:.2f} p.u.<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add rated load line
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                  annotation_text="Rated Load (1.0 p.u.)", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Transformer Thermal Monitoring",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Loading (p.u.)", row=3, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    
    return fig


def create_dynamic_ratings_plot(ratings_df: pd.DataFrame) -> go.Figure:
    """Create dynamic ratings visualization"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Dynamic Ratings vs Load', 'Utilization Percentage'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    time = ratings_df['time_index']
    
    # Ratings comparison
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ratings_df['emergency_rating_mva'],
            name='Emergency Rating',
            line=dict(color='red', width=2),
            hovertemplate='Hour: %{x}<br>Rating: %{y:.1f} MVA<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ratings_df['normal_rating_mva'],
            name='Normal Rating',
            line=dict(color='orange', width=2),
            hovertemplate='Hour: %{x}<br>Rating: %{y:.1f} MVA<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ratings_df['load_pu'] * 50,  # Convert to MVA
            name='Actual Load',
            line=dict(color='blue', width=2),
            hovertemplate='Hour: %{x}<br>Load: %{y:.1f} MVA<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add nameplate rating
    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                  annotation_text="Nameplate Rating (50 MVA)", row=1, col=1)
    
    # Utilization percentage
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ratings_df['utilization_normal'],
            name='Normal Utilization',
            line=dict(color='green', width=2),
            fill='tonexty',
            hovertemplate='Hour: %{x}<br>Utilization: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add utilization limits
    fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                  annotation_text="100% Utilization", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="yellow", 
                  annotation_text="80% Utilization", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="Dynamic Transformer Ratings",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Rating (MVA)", row=1, col=1)
    fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    
    return fig


def create_ev_load_forecast_plot(forecast_df: pd.DataFrame) -> go.Figure:
    """Create EV load forecasting visualization"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Load Components', 'Daily Load Patterns'),
        vertical_spacing=0.12
    )
    
    time = forecast_df['time_index']
    
    # Stacked area chart for load components
    fig.add_trace(
        go.Scatter(
            x=time,
            y=forecast_df['base_load_mw'],
            name='Base Load',
            line=dict(color='blue'),
            fill='tonexty',
            hovertemplate='Hour: %{x}<br>Base Load: %{y:.1f} MW<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=forecast_df['base_load_mw'] + forecast_df['ev_load_mw'],
            name='Total Load (Base + EV)',
            line=dict(color='red'),
            fill='tonexty',
            hovertemplate='Hour: %{x}<br>Total Load: %{y:.1f} MW<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=forecast_df['ev_load_mw'],
            name='EV Charging Load',
            line=dict(color='green'),
            hovertemplate='Hour: %{x}<br>EV Load: %{y:.1f} MW<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Daily patterns (show first 48 hours)
    if len(forecast_df) >= 48:
        first_48h = forecast_df.head(48)
        
        # Group by hour of day
        hourly_avg = first_48h.groupby('hour').agg({
            'base_load_mw': 'mean',
            'ev_load_mw': 'mean',
            'total_load_mw': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=hourly_avg['hour'],
                y=hourly_avg['base_load_mw'],
                name='Avg Base Load',
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=hourly_avg['hour'],
                y=hourly_avg['ev_load_mw'],
                name='Avg EV Load',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="EV Load Forecasting Analysis",
        title_x=0.5,
        showlegend=True,
        barmode='stack'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Load (MW)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    
    return fig


def create_risk_assessment_plot(risk_results: Dict, health_data: Dict) -> go.Figure:
    """Create risk assessment visualization"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss of Life Distribution', 'Hot Spot Temperature Distribution', 
                       'Health Index Components', 'Risk Metrics'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Loss of life percentiles (create mock distribution for visualization)
    lol_percentiles = risk_results['lol_percentiles']
    lol_values = np.random.normal(risk_results['mean_lol'], risk_results['std_lol'], 1000)
    
    fig.add_trace(
        go.Histogram(
            x=lol_values,
            name='Loss of Life',
            nbinsx=30,
            marker_color='orange',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Hot spot temperature percentiles
    hot_spot_percentiles = risk_results['hot_spot_max_percentiles']
    hot_spot_values = np.random.normal(hot_spot_percentiles[2], 5, 1000)  # Use median
    
    fig.add_trace(
        go.Histogram(
            x=hot_spot_values,
            name='Max Hot Spot Temp',
            nbinsx=30,
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Health index components
    components = ['Thermal', 'Loading', 'Temperature', 'Operational']
    values = [
        health_data['thermal_component'],
        health_data['loading_component'],
        health_data['temperature_component'],
        health_data['operational_component']
    ]
    
    fig.add_trace(
        go.Bar(
            x=components,
            y=values,
            name='Health Components',
            marker_color=['red', 'blue', 'orange', 'green']
        ),
        row=2, col=1
    )
    
    # Risk indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_data['health_index'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Health Index"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Risk Assessment Dashboard",
        title_x=0.5,
        showlegend=False
    )
    
    return fig


def create_cooling_optimization_plot(cooling_data: Dict, ratings_df: pd.DataFrame) -> go.Figure:
    """Create cooling system optimization visualization"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cooling Schedule Optimization', 'Cost Savings Analysis'),
        vertical_spacing=0.12
    )
    
    time = ratings_df['time_index']
    cooling_schedule = cooling_data['cooling_schedule']
    
    # Cooling schedule
    cooling_names = ['ONAN', 'Stage 1 Fans', 'Full Fans']
    cooling_colors = ['blue', 'orange', 'red']
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=cooling_schedule,
            mode='lines+markers',
            name='Cooling Stage',
            line=dict(color='purple', width=2),
            hovertemplate='Hour: %{x}<br>Cooling: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add load for context
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ratings_df['load_pu'] * 2,  # Scale for visibility
            name='Load (scaled)',
            line=dict(color='gray', width=1, dash='dot'),
            yaxis='y2'
        ),
        row=1, col=1
    )
    
    # Cost savings
    hourly_savings = cooling_data['hourly_savings']
    
    fig.add_trace(
        go.Bar(
            x=time,
            y=hourly_savings,
            name='Hourly Savings',
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="Cooling System Optimization",
        title_x=0.5,
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Cooling Stage", row=1, col=1)
    fig.update_yaxes(title_text="Cost Savings ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    
    return fig


def create_summary_metrics_cards(results: Dict) -> List[Dict]:
    """Create summary metrics for dashboard cards"""
    
    thermal_data = results['thermal']
    ratings_data = results['ratings']
    health_data = results['health']
    economic_data = results['economic']
    cooling_data = results['cooling']
    
    metrics = [
        {
            'title': 'Peak Hot Spot Temperature',
            'value': f"{np.max(thermal_data['hot_spot']):.1f}°C",
            'delta': f"Limit: 110°C",
            'delta_color': 'normal' if np.max(thermal_data['hot_spot']) < 110 else 'alert'
        },
        {
            'title': 'Maximum Dynamic Rating',
            'value': f"{ratings_data['normal_rating_mva'].max():.1f} MVA",
            'delta': f"+{((ratings_data['normal_rating_mva'].max() - 50) / 50 * 100):.1f}% vs nameplate",
            'delta_color': 'positive'
        },
        {
            'title': 'Health Index',
            'value': f"{health_data['health_index']:.1f}",
            'delta': health_data['category'],
            'delta_color': 'positive' if health_data['health_index'] > 60 else 'alert'
        },
        {
            'title': 'Annual Energy Savings',
            'value': f"${cooling_data['annual_savings_estimate']:.0f}",
            'delta': f"{cooling_data['energy_efficiency']*100:.1f}% efficiency",
            'delta_color': 'positive'
        },
        {
            'title': 'Simple Payback',
            'value': f"{economic_data['simple_payback_years']:.1f} years",
            'delta': f"ROI: {economic_data['roi_percent']:.1f}%",
            'delta_color': 'positive' if economic_data['simple_payback_years'] < 5 else 'normal'
        },
        {
            'title': 'Peak Utilization',
            'value': f"{ratings_data['utilization_normal'].max():.1f}%",
            'delta': f"Avg: {ratings_data['utilization_normal'].mean():.1f}%",
            'delta_color': 'alert' if ratings_data['utilization_normal'].max() > 100 else 'normal'
        }
    ]
    
    return metrics


def export_results_to_csv(results: Dict) -> str:
    """Export analysis results to CSV format"""
    
    # Combine key data into a comprehensive DataFrame
    forecast_df = results['forecast']
    ratings_df = results['ratings']
    thermal_data = results['thermal']
    
    # Create comprehensive export DataFrame
    export_df = pd.DataFrame({
        'time_index': forecast_df['time_index'],
        'day': forecast_df['day'],
        'hour': forecast_df['hour'],
        'base_load_mw': forecast_df['base_load_mw'],
        'ev_load_mw': forecast_df['ev_load_mw'],
        'total_load_mw': forecast_df['total_load_mw'],
        'load_pu': forecast_df['load_pu'],
        'ambient_temp': thermal_data['ambient'],
        'top_oil_temp': thermal_data['top_oil'],
        'hot_spot_temp': thermal_data['hot_spot'],
        'normal_rating_mva': ratings_df['normal_rating_mva'],
        'emergency_rating_mva': ratings_df['emergency_rating_mva'],
        'utilization_normal': ratings_df['utilization_normal'],
        'margin_normal': ratings_df['margin_normal']
    })
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def create_download_link(data: str, filename: str, link_text: str) -> str:
    """Create a download link for data"""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def format_alert_message(thermal_data: Dict, ratings_data: pd.DataFrame) -> List[str]:
    """Format alert messages based on current conditions"""
    
    alerts = []
    
    # Current values (last hour)
    current_hot_spot = thermal_data['hot_spot'][-1]
    current_top_oil = thermal_data['top_oil'][-1]
    current_load = thermal_data['load_pu'][-1]
    current_utilization = ratings_data['utilization_normal'].iloc[-1]
    
    # Temperature alerts
    if current_hot_spot > 140:
        alerts.append("🚨 CRITICAL: Hot spot temperature exceeds emergency limit!")
    elif current_hot_spot > 120:
        alerts.append("⚠️ WARNING: Hot spot temperature approaching emergency limit")
    elif current_hot_spot > 110:
        alerts.append("⚠️ CAUTION: Hot spot temperature exceeds normal limit")
    
    if current_top_oil > 110:
        alerts.append("🚨 CRITICAL: Top oil temperature exceeds emergency limit!")
    elif current_top_oil > 95:
        alerts.append("⚠️ WARNING: Top oil temperature approaching limit")
    
    # Loading alerts
    if current_utilization > 100:
        alerts.append("⚠️ WARNING: Transformer utilization exceeds normal rating")
    elif current_utilization > 90:
        alerts.append("ℹ️ INFO: High transformer utilization - monitor closely")
    
    # Trending alerts
    if len(thermal_data['hot_spot']) >= 3:
        recent_trend = np.mean(thermal_data['hot_spot'][-3:]) - np.mean(thermal_data['hot_spot'][-6:-3])
        if recent_trend > 5:
            alerts.append("📈 TREND: Hot spot temperature rising rapidly")
    
    if not alerts:
        alerts.append("✅ All parameters within normal operating limits")
    
    return alerts


def validate_uploaded_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate uploaded CSV data format"""
    
    errors = []
    required_columns = [
        'time_index', 'load_pu', 'ambient_temp', 'hot_spot_temp', 'top_oil_temp'
    ]
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and ranges
    if 'load_pu' in df.columns:
        if df['load_pu'].min() < 0 or df['load_pu'].max() > 5:
            errors.append("Load values should be between 0 and 5 p.u.")
    
    if 'hot_spot_temp' in df.columns:
        if df['hot_spot_temp'].min() < -50 or df['hot_spot_temp'].max() > 200:
            errors.append("Hot spot temperature values seem unrealistic")
    
    if 'ambient_temp' in df.columns:
        if df['ambient_temp'].min() < -40 or df['ambient_temp'].max() > 60:
            errors.append("Ambient temperature values seem unrealistic")
    
    # Check for missing data
    if df.isnull().any().any():
        errors.append("Data contains missing values")
    
    return len(errors) == 0, errors
