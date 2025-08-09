"""
Dynamic Transformer Rating (DTR) Streamlit Application
For visualizing OFAF transformer thermal monitoring, load forecasting, and risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Import DTR system components
from dtr_system import (
    TransformerParameters, EVChargingProfile, ThermalModel, 
    EVLoadForecaster, DynamicRatingCalculator, RiskAssessment,
    DTRSystemIntegration, run_comprehensive_analysis
)

# Import utility functions
from utils import (
    create_thermal_monitoring_plot, create_dynamic_ratings_plot,
    create_ev_load_forecast_plot, create_risk_assessment_plot,
    create_cooling_optimization_plot, create_summary_metrics_cards,
    export_results_to_csv, create_download_link, format_alert_message,
    validate_uploaded_data
)

# Configure page
st.set_page_config(
    page_title="DTR Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #ff6b35;
    }
    .alert-critical {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #fecaca;
    }
    .alert-warning {
        background-color: #fef3cd;
        color: #d97706;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #fde047;
    }
    .alert-info {
        background-color: #dbeafe;
        color: #2563eb;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #93c5fd;
    }
    .alert-success {
        background-color: #dcfce7;
        color: #16a34a;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #86efac;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Title and header
    st.title("⚡ Dynamic Transformer Rating Dashboard")
    st.markdown("**OFAF Transformer Thermal Monitoring & Risk Assessment**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Run New Analysis", "Upload CSV Data"],
            help="Choose to run fresh calculations or upload existing data"
        )
        
        if data_source == "Run New Analysis":
            st.subheader("Transformer Parameters")
            
            # Transformer configuration
            rated_power = st.number_input("Rated Power (MVA)", value=50.0, min_value=1.0, max_value=500.0)
            rated_voltage_hv = st.number_input("HV Voltage (kV)", value=138.0, min_value=1.0, max_value=500.0)
            rated_voltage_lv = st.number_input("LV Voltage (kV)", value=13.8, min_value=1.0, max_value=50.0)
            
            st.subheader("EV Charging Configuration")
            
            # EV charging parameters
            num_l2_chargers = st.number_input("Level 2 Chargers", value=100, min_value=0, max_value=1000)
            num_dcfc_chargers = st.number_input("DC Fast Chargers", value=10, min_value=0, max_value=100)
            
            # Analysis period
            st.subheader("Analysis Period")
            analysis_days = st.selectbox("Analysis Duration", [7, 14, 30], index=0)
            
            # Run analysis button
            run_analysis = st.button("🚀 Run DTR Analysis", type="primary")
            
        else:
            st.subheader("Upload Data")
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload CSV file with DTR analysis results"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    is_valid, errors = validate_uploaded_data(uploaded_df)
                    
                    if is_valid:
                        st.success("✅ Data uploaded successfully!")
                        st.write(f"Loaded {len(uploaded_df)} records")
                    else:
                        st.error("❌ Data validation failed:")
                        for error in errors:
                            st.write(f"• {error}")
                            
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    uploaded_df = None
    
    # Main content area
    if data_source == "Run New Analysis":
        if 'run_analysis' in locals() and run_analysis:
            run_new_analysis(
                rated_power, rated_voltage_hv, rated_voltage_lv,
                num_l2_chargers, num_dcfc_chargers, analysis_days
            )
        elif 'results' not in st.session_state:
            show_welcome_screen()
        else:
            display_analysis_results(st.session_state.results)
    
    else:  # Upload CSV Data
        if 'uploaded_df' in locals() and uploaded_df is not None and is_valid:
            display_uploaded_data_analysis(uploaded_df)
        else:
            show_upload_instructions()


def show_welcome_screen():
    """Display welcome screen with system overview"""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 🏭 Welcome to the DTR Analysis System
        
        This application provides comprehensive Dynamic Transformer Rating analysis for OFAF transformers
        in high EV charging penetration areas.
        
        **Key Features:**
        - 🌡️ Real-time thermal monitoring
        - 📊 Dynamic rating calculations
        - 🚗 EV load forecasting
        - ⚠️ Risk assessment and health indices
        - 💰 Economic analysis
        - 🎯 Cooling system optimization
        
        **Getting Started:**
        1. Configure your transformer parameters in the sidebar
        2. Set EV charging configuration
        3. Click "Run DTR Analysis" to begin
        
        The analysis follows IEEE C57.91-2011 and IEC 60076-7 standards.
        """)
    
    # Display sample metrics
    st.markdown("---")
    st.subheader("📈 Sample Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Additional Capacity", "25-45%", "vs nameplate rating")
    
    with col2:
        st.metric("Energy Savings", "55-75%", "cooling optimization")
    
    with col3:
        st.metric("Simple Payback", "2-3 years", "typical ROI")
    
    with col4:
        st.metric("EV Growth", "3-5 years", "accommodation period")


def show_upload_instructions():
    """Display instructions for data upload"""
    
    st.markdown("---")
    st.subheader("📁 Data Upload Instructions")
    
    st.markdown("""
    Upload a CSV file containing DTR analysis results with the following required columns:
    
    **Required Columns:**
    - `time_index`: Hour index (0, 1, 2, ...)
    - `load_pu`: Load in per-unit (0.0 to 3.0)
    - `ambient_temp`: Ambient temperature in °C
    - `hot_spot_temp`: Hot spot temperature in °C
    - `top_oil_temp`: Top oil temperature in °C
    
    **Optional Columns:**
    - `normal_rating_mva`: Normal rating in MVA
    - `emergency_rating_mva`: Emergency rating in MVA
    - `utilization_normal`: Utilization percentage
    - `ev_load_mw`: EV charging load in MW
    """)
    
    # Sample data format
    st.subheader("📝 Sample Data Format")
    
    sample_data = pd.DataFrame({
        'time_index': [0, 1, 2, 3, 4],
        'load_pu': [0.8, 0.9, 1.1, 1.0, 0.7],
        'ambient_temp': [25, 26, 28, 30, 29],
        'hot_spot_temp': [85, 90, 105, 95, 80],
        'top_oil_temp': [65, 68, 75, 70, 62]
    })
    
    st.dataframe(sample_data)
    
    # Download sample CSV
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv,
        file_name="sample_dtr_data.csv",
        mime="text/csv"
    )


def run_new_analysis(rated_power, rated_voltage_hv, rated_voltage_lv, 
                    num_l2_chargers, num_dcfc_chargers, analysis_days):
    """Run new DTR analysis with user parameters"""
    
    with st.spinner("🔄 Running DTR Analysis... This may take a moment."):
        try:
            # Create custom parameters
            transformer_params = TransformerParameters(
                rated_power=rated_power,
                rated_voltage_hv=rated_voltage_hv,
                rated_voltage_lv=rated_voltage_lv
            )
            
            ev_profile = EVChargingProfile(
                num_chargers_l2=num_l2_chargers,
                num_chargers_dcfc=num_dcfc_chargers
            )
            
            # Initialize system components
            thermal_model = ThermalModel(transformer_params)
            ev_forecaster = EVLoadForecaster(ev_profile)
            rating_calculator = DynamicRatingCalculator(thermal_model)
            risk_assessor = RiskAssessment(thermal_model)
            
            # Generate extended forecast if needed
            if analysis_days > 7:
                # Generate extended forecast
                weekly_forecast = ev_forecaster.forecast_weekly_load(base_load_mw=rated_power * 0.6)
                
                # Replicate for longer periods
                extended_forecast = []
                for week in range(analysis_days // 7 + 1):
                    week_data = weekly_forecast.copy()
                    week_data['time_index'] += week * 168  # 168 hours per week
                    extended_forecast.append(week_data)
                
                forecast_df = pd.concat(extended_forecast).head(analysis_days * 24)
                forecast_df = forecast_df.reset_index(drop=True)
            else:
                forecast_df = ev_forecaster.forecast_weekly_load(base_load_mw=rated_power * 0.6)
            
            # Generate ambient temperature profile
            hours = len(forecast_df)
            ambient_base = 25  # °C
            daily_variation = 8  # °C peak-to-peak
            ambient_temps = ambient_base + daily_variation * np.sin(
                2 * np.pi * np.arange(hours) / 24 - np.pi/2
            ) / 2
            
            # Calculate dynamic ratings
            ratings_df = rating_calculator.calculate_hourly_ratings(forecast_df, ambient_temps)
            
            # Run thermal simulation
            thermal_response = thermal_model.simulate_thermal_response(
                forecast_df['load_pu'].values, 
                ambient_temps
            )
            
            # Cooling optimization
            cooling_optimization = rating_calculator.optimize_cooling_schedule(ratings_df)
            
            # Risk assessment
            risk_results = risk_assessor.monte_carlo_analysis(
                forecast_df, ambient_temps, num_simulations=500
            )
            
            # Health assessment
            health_assessment = risk_assessor.calculate_health_index(
                thermal_response, 
                {'operational_score': 85}
            )
            
            # Economic analysis
            dtr_benefits = {
                'additional_capacity_mva': (ratings_df['normal_rating_mva'].mean() - rated_power) / rated_power * 100,
                'cooling_savings': cooling_optimization['total_daily_savings'],
                'deferred_investment': 2000000
            }
            
            economic_results = risk_assessor.economic_analysis(dtr_benefits)
            
            # Store results in session state
            st.session_state.results = {
                'forecast': forecast_df,
                'ratings': ratings_df,
                'thermal': thermal_response,
                'cooling': cooling_optimization,
                'risk': risk_results,
                'health': health_assessment,
                'economic': economic_results
            }
            
            st.success("✅ Analysis completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)


def display_analysis_results(results):
    """Display comprehensive analysis results"""
    
    # Alert panel
    display_alert_panel(results)
    
    # Summary metrics
    display_summary_metrics(results)
    
    # Main visualizations tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌡️ Thermal Monitoring", "📊 Dynamic Ratings", "🚗 EV Load Forecast",
        "⚠️ Risk Assessment", "❄️ Cooling Optimization", "📋 Detailed Results"
    ])
    
    with tab1:
        display_thermal_monitoring(results)
    
    with tab2:
        display_dynamic_ratings(results)
    
    with tab3:
        display_ev_load_forecast(results)
    
    with tab4:
        display_risk_assessment(results)
    
    with tab5:
        display_cooling_optimization(results)
    
    with tab6:
        display_detailed_results(results)


def display_uploaded_data_analysis(uploaded_df):
    """Display analysis for uploaded data"""
    
    st.markdown("---")
    st.subheader("📊 Uploaded Data Analysis")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(uploaded_df))
    
    with col2:
        if 'hot_spot_temp' in uploaded_df.columns:
            st.metric("Peak Hot Spot", f"{uploaded_df['hot_spot_temp'].max():.1f}°C")
    
    with col3:
        if 'load_pu' in uploaded_df.columns:
            st.metric("Peak Load", f"{uploaded_df['load_pu'].max():.2f} p.u.")
    
    with col4:
        if 'ambient_temp' in uploaded_df.columns:
            st.metric("Avg Ambient", f"{uploaded_df['ambient_temp'].mean():.1f}°C")
    
    # Simple visualizations
    st.subheader("Temperature Trends")
    
    if all(col in uploaded_df.columns for col in ['hot_spot_temp', 'top_oil_temp', 'ambient_temp']):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=uploaded_df['time_index'],
            y=uploaded_df['hot_spot_temp'],
            name='Hot Spot',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=uploaded_df['time_index'],
            y=uploaded_df['top_oil_temp'],
            name='Top Oil',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=uploaded_df['time_index'],
            y=uploaded_df['ambient_temp'],
            name='Ambient',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="Temperature Profile",
            xaxis_title="Time (hours)",
            yaxis_title="Temperature (°C)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Data Preview")
    st.dataframe(uploaded_df.head(50), use_container_width=True)
    
    # Download processed data
    csv_data = uploaded_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Processed Data",
        data=csv_data,
        file_name=f"processed_dtr_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )


def display_alert_panel(results):
    """Display alert panel with current status"""
    
    st.markdown("---")
    st.subheader("🚨 Current Status & Alerts")
    
    alerts = format_alert_message(results['thermal'], results['ratings'])
    
    for alert in alerts:
        if "CRITICAL" in alert:
            st.markdown(f'<div class="alert-critical">{alert}</div>', unsafe_allow_html=True)
        elif "WARNING" in alert:
            st.markdown(f'<div class="alert-warning">{alert}</div>', unsafe_allow_html=True)
        elif "INFO" in alert or "TREND" in alert:
            st.markdown(f'<div class="alert-info">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success">{alert}</div>', unsafe_allow_html=True)


def display_summary_metrics(results):
    """Display summary metrics cards"""
    
    st.markdown("---")
    st.subheader("📈 Key Performance Indicators")
    
    metrics = create_summary_metrics_cards(results)
    
    cols = st.columns(3)
    
    for i, metric in enumerate(metrics):
        with cols[i % 3]:
            delta_color = "normal"
            if metric['delta_color'] == 'positive':
                delta_color = "normal"
            elif metric['delta_color'] == 'alert':
                delta_color = "inverse"
            
            st.metric(
                label=metric['title'],
                value=metric['value'],
                delta=metric['delta']
            )


def display_thermal_monitoring(results):
    """Display thermal monitoring tab"""
    
    st.subheader("🌡️ Thermal Monitoring Dashboard")
    
    # Create and display thermal plot
    thermal_fig = create_thermal_monitoring_plot(results['thermal'])
    st.plotly_chart(thermal_fig, use_container_width=True)
    
    # Thermal statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Statistics")
        
        thermal_stats = pd.DataFrame({
            'Parameter': ['Hot Spot', 'Top Oil', 'Ambient'],
            'Current (°C)': [
                results['thermal']['hot_spot'][-1],
                results['thermal']['top_oil'][-1],
                results['thermal']['ambient'][-1]
            ],
            'Peak (°C)': [
                np.max(results['thermal']['hot_spot']),
                np.max(results['thermal']['top_oil']),
                np.max(results['thermal']['ambient'])
            ],
            'Average (°C)': [
                np.mean(results['thermal']['hot_spot']),
                np.mean(results['thermal']['top_oil']),
                np.mean(results['thermal']['ambient'])
            ]
        })
        
        st.dataframe(thermal_stats, hide_index=True)
    
    with col2:
        st.subheader("Thermal Limits")
        
        limits_data = pd.DataFrame({
            'Limit Type': ['Normal Hot Spot', 'Emergency Hot Spot', 'Top Oil Emergency'],
            'Temperature (°C)': [110, 140, 110],
            'Current Status': [
                '✅ OK' if results['thermal']['hot_spot'][-1] < 110 else '⚠️ EXCEEDED',
                '✅ OK' if results['thermal']['hot_spot'][-1] < 140 else '🚨 EXCEEDED',
                '✅ OK' if results['thermal']['top_oil'][-1] < 110 else '🚨 EXCEEDED'
            ]
        })
        
        st.dataframe(limits_data, hide_index=True)


def display_dynamic_ratings(results):
    """Display dynamic ratings tab"""
    
    st.subheader("📊 Dynamic Rating Analysis")
    
    # Create and display ratings plot
    ratings_fig = create_dynamic_ratings_plot(results['ratings'])
    st.plotly_chart(ratings_fig, use_container_width=True)
    
    # Rating statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Summary")
        
        rating_stats = pd.DataFrame({
            'Rating Type': ['Nameplate', 'Normal (Min)', 'Normal (Max)', 'Emergency (Max)'],
            'Value (MVA)': [
                50.0,
                results['ratings']['normal_rating_mva'].min(),
                results['ratings']['normal_rating_mva'].max(),
                results['ratings']['emergency_rating_mva'].max()
            ],
            'Increase (%)': [
                0,
                (results['ratings']['normal_rating_mva'].min() - 50) / 50 * 100,
                (results['ratings']['normal_rating_mva'].max() - 50) / 50 * 100,
                (results['ratings']['emergency_rating_mva'].max() - 50) / 50 * 100
            ]
        })
        
        st.dataframe(rating_stats, hide_index=True)
    
    with col2:
        st.subheader("Utilization Analysis")
        
        util_stats = pd.DataFrame({
            'Metric': ['Current', 'Peak', 'Average', 'Hours > 100%'],
            'Utilization (%)': [
                results['ratings']['utilization_normal'].iloc[-1],
                results['ratings']['utilization_normal'].max(),
                results['ratings']['utilization_normal'].mean(),
                len(results['ratings'][results['ratings']['utilization_normal'] > 100])
            ]
        })
        
        st.dataframe(util_stats, hide_index=True)


def display_ev_load_forecast(results):
    """Display EV load forecast tab"""
    
    st.subheader("🚗 EV Load Forecasting")
    
    # Create and display EV load plot
    ev_fig = create_ev_load_forecast_plot(results['forecast'])
    st.plotly_chart(ev_fig, use_container_width=True)
    
    # Load composition analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Load Composition")
        
        total_base = results['forecast']['base_load_mw'].sum()
        total_ev = results['forecast']['ev_load_mw'].sum()
        total_load = total_base + total_ev
        
        composition_data = pd.DataFrame({
            'Load Type': ['Base Load', 'EV Charging', 'Total'],
            'Energy (MWh)': [total_base, total_ev, total_load],
            'Percentage (%)': [
                total_base / total_load * 100,
                total_ev / total_load * 100,
                100
            ]
        })
        
        st.dataframe(composition_data, hide_index=True)
    
    with col2:
        st.subheader("Peak Demand Analysis")
        
        peak_stats = pd.DataFrame({
            'Period': ['Peak Hour', 'Off-Peak', 'Average'],
            'Base Load (MW)': [
                results['forecast']['base_load_mw'].max(),
                results['forecast']['base_load_mw'].min(),
                results['forecast']['base_load_mw'].mean()
            ],
            'EV Load (MW)': [
                results['forecast']['ev_load_mw'].max(),
                results['forecast']['ev_load_mw'].min(),
                results['forecast']['ev_load_mw'].mean()
            ]
        })
        
        st.dataframe(peak_stats, hide_index=True)


def display_risk_assessment(results):
    """Display risk assessment tab"""
    
    st.subheader("⚠️ Risk Assessment & Health Index")
    
    # Create and display risk plot
    risk_fig = create_risk_assessment_plot(results['risk'], results['health'])
    st.plotly_chart(risk_fig, use_container_width=True)
    
    # Risk metrics and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")
        
        risk_data = pd.DataFrame({
            'Metric': [
                'Overload Probability',
                'Mean Loss of Life',
                'Health Index',
                'Health Category'
            ],
            'Value': [
                f"{results['risk']['overload_probability']:.1f}%",
                f"{results['risk']['mean_lol']:.4f}%",
                f"{results['health']['health_index']:.1f}",
                results['health']['category']
            ]
        })
        
        st.dataframe(risk_data, hide_index=True)
        
        # Economic metrics
        st.subheader("Economic Analysis")
        
        economic_data = pd.DataFrame({
            'Metric': [
                'Implementation Cost',
                'Annual Benefits',
                'Simple Payback',
                'NPV (20 years)',
                'ROI'
            ],
            'Value': [
                f"${results['economic']['implementation_cost']:,.0f}",
                f"${results['economic']['total_annual_benefits']:,.0f}",
                f"{results['economic']['simple_payback_years']:.1f} years",
                f"${results['economic']['npv_20_year']:,.0f}",
                f"{results['economic']['roi_percent']:.1f}%"
            ]
        })
        
        st.dataframe(economic_data, hide_index=True)
    
    with col2:
        st.subheader("Health Assessment Components")
        
        health_components = pd.DataFrame({
            'Component': ['Thermal', 'Loading', 'Temperature', 'Operational'],
            'Score': [
                results['health']['thermal_component'],
                results['health']['loading_component'],
                results['health']['temperature_component'],
                results['health']['operational_component']
            ],
            'Weight (%)': [40, 30, 20, 10]
        })
        
        st.dataframe(health_components, hide_index=True)
        
        st.subheader("Recommendations")
        
        for i, rec in enumerate(results['health']['recommendations'], 1):
            st.write(f"{i}. {rec}")


def display_cooling_optimization(results):
    """Display cooling optimization tab"""
    
    st.subheader("❄️ Cooling System Optimization")
    
    # Create and display cooling plot
    cooling_fig = create_cooling_optimization_plot(results['cooling'], results['ratings'])
    st.plotly_chart(cooling_fig, use_container_width=True)
    
    # Cooling analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimization Results")
        
        cooling_stats = pd.DataFrame({
            'Metric': [
                'Total Daily Savings',
                'Annual Savings Estimate',
                'Energy Efficiency',
                'Average Cooling Stage'
            ],
            'Value': [
                f"${results['cooling']['total_daily_savings']:.2f}",
                f"${results['cooling']['annual_savings_estimate']:.0f}",
                f"{results['cooling']['energy_efficiency']*100:.1f}%",
                f"{np.mean(results['cooling']['cooling_schedule']):.1f}"
            ]
        })
        
        st.dataframe(cooling_stats, hide_index=True)
    
    with col2:
        st.subheader("Cooling Stage Distribution")
        
        cooling_schedule = results['cooling']['cooling_schedule']
        stage_counts = pd.Series(cooling_schedule).value_counts().sort_index()
        
        stage_data = pd.DataFrame({
            'Cooling Stage': ['ONAN (0)', 'Stage 1 (1)', 'Full Fans (2)'],
            'Hours': [
                stage_counts.get(0, 0),
                stage_counts.get(1, 0),
                stage_counts.get(2, 0)
            ],
            'Percentage': [
                stage_counts.get(0, 0) / len(cooling_schedule) * 100,
                stage_counts.get(1, 0) / len(cooling_schedule) * 100,
                stage_counts.get(2, 0) / len(cooling_schedule) * 100
            ]
        })
        
        st.dataframe(stage_data, hide_index=True)


def display_detailed_results(results):
    """Display detailed results tab"""
    
    st.subheader("📋 Detailed Analysis Results")
    
    # Data export section
    st.subheader("📥 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export full results
        csv_data = export_results_to_csv(results)
        st.download_button(
            label="📊 Download Full Results (CSV)",
            data=csv_data,
            file_name=f"dtr_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export summary report
        summary_data = create_summary_report(results)
        st.download_button(
            label="📄 Download Summary Report (CSV)",
            data=summary_data,
            file_name=f"dtr_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # Detailed data tables
    st.subheader("📊 Detailed Data Tables")
    
    data_view = st.selectbox(
        "Select Data View",
        ["Load Forecast", "Dynamic Ratings", "Thermal Response", "Risk Analysis"]
    )
    
    if data_view == "Load Forecast":
        st.write("**Load Forecasting Results**")
        st.dataframe(results['forecast'], use_container_width=True)
    
    elif data_view == "Dynamic Ratings":
        st.write("**Dynamic Rating Calculations**")
        st.dataframe(results['ratings'], use_container_width=True)
    
    elif data_view == "Thermal Response":
        st.write("**Thermal Simulation Results**")
        thermal_df = pd.DataFrame({
            'Time': results['thermal']['time'],
            'Load (p.u.)': results['thermal']['load_pu'],
            'Ambient (°C)': results['thermal']['ambient'],
            'Top Oil (°C)': results['thermal']['top_oil'],
            'Hot Spot (°C)': results['thermal']['hot_spot']
        })
        st.dataframe(thermal_df, use_container_width=True)
    
    elif data_view == "Risk Analysis":
        st.write("**Risk Assessment Summary**")
        risk_summary = pd.DataFrame({
            'Metric': [
                'Overload Probability (%)',
                'Mean Loss of Life (%)',
                'Std Loss of Life (%)',
                'Peak Hot Spot 95th Percentile (°C)',
                'Health Index',
                'Health Category'
            ],
            'Value': [
                results['risk']['overload_probability'],
                results['risk']['mean_lol'],
                results['risk']['std_lol'],
                results['risk']['hot_spot_max_percentiles'][4],
                results['health']['health_index'],
                results['health']['category']
            ]
        })
        st.dataframe(risk_summary, hide_index=True)


def create_summary_report(results):
    """Create summary report for export"""
    
    summary_data = pd.DataFrame({
        'Metric': [
            'Analysis Date',
            'Peak Hot Spot Temperature (°C)',
            'Peak Top Oil Temperature (°C)',
            'Maximum Normal Rating (MVA)',
            'Maximum Emergency Rating (MVA)',
            'Peak Utilization (%)',
            'Health Index',
            'Health Category',
            'Overload Probability (%)',
            'Annual Energy Savings ($)',
            'Simple Payback (years)',
            'NPV 20-year ($)'
        ],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            f"{np.max(results['thermal']['hot_spot']):.1f}",
            f"{np.max(results['thermal']['top_oil']):.1f}",
            f"{results['ratings']['normal_rating_mva'].max():.1f}",
            f"{results['ratings']['emergency_rating_mva'].max():.1f}",
            f"{results['ratings']['utilization_normal'].max():.1f}",
            f"{results['health']['health_index']:.1f}",
            results['health']['category'],
            f"{results['risk']['overload_probability']:.1f}",
            f"{results['cooling']['annual_savings_estimate']:.0f}",
            f"{results['economic']['simple_payback_years']:.1f}",
            f"{results['economic']['npv_20_year']:,.0f}"
        ]
    })
    
    return summary_data.to_csv(index=False)


if __name__ == "__main__":
    main()
