# DTR Analysis Dashboard

## Overview

This is a Dynamic Transformer Rating (DTR) Streamlit application designed for monitoring and analyzing OFAF (Oil Forced Air Forced) transformers in high EV charging areas. The system provides comprehensive thermal monitoring, load forecasting, dynamic rating calculations, and risk assessment capabilities. It implements IEEE C57.91-2011 thermal equations and includes specialized features for electric vehicle charging infrastructure planning and transformer health management.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with interactive dashboard
- **Visualization**: Plotly for dynamic charts and graphs including thermal monitoring plots, load forecasts, and risk assessments
- **Layout**: Wide layout with expandable sidebar for configuration and multi-column metric displays
- **Styling**: Custom CSS for professional appearance with metric cards and alert styling

### Backend Architecture
- **Core Engine**: Modular DTR system (`dtr_system.py`) containing specialized classes for different analysis components
- **Thermal Modeling**: IEEE C57.91-2011 compliant thermal calculations using differential equations solved with scipy.odeint
- **Load Forecasting**: EV charging pattern analysis with Level 2 and DC fast charging considerations
- **Rating Calculations**: Dynamic normal and emergency ratings with multi-stage cooling optimization
- **Risk Assessment**: Monte Carlo simulation for uncertainty quantification and reliability metrics

### Key Components
- **TransformerParameters**: OFAF-specific thermal parameters (80-90 min oil time constant, 6-7 min winding time constant)
- **ThermalModel**: Hot spot and top oil temperature calculations with loss of life assessment
- **EVLoadForecaster**: Realistic EV charging patterns with coincidence factors (25% L2, 60% DCFC)
- **DynamicRatingCalculator**: Hourly rating calculations with cooling system optimization
- **RiskAssessment**: Health index calculation and economic risk evaluation
- **DTRSystemIntegration**: SCADA export and IEC 61850 report generation

### Data Processing
- **Input Validation**: Comprehensive data validation for uploaded transformer data
- **Export Functionality**: CSV export capabilities for analysis results
- **Real-time Analysis**: Live thermal monitoring and dynamic rating updates

### Algorithms
- **Optimization**: Differential evolution for cooling system optimization
- **Search**: Binary search algorithms for rating calculations
- **Simulation**: Monte Carlo methods for risk assessment
- **Numerical**: Differential equation solving for thermal dynamics

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for dashboard interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **plotly**: Interactive visualization and charting (graph_objects, express, subplots)
- **scipy**: Scientific computing for differential equation solving and optimization
- **datetime**: Date and time handling for temporal analysis

### Standards Compliance
- **IEEE C57.91-2011**: Thermal modeling standards for transformer analysis
- **IEC 61850**: Communication protocol for power system automation and report generation

### Data Formats
- **CSV**: Export format for analysis results and data interchange
- **JSON**: Configuration and parameter storage
- **Base64**: File encoding for download functionality

### Potential Future Integrations
- **SCADA Systems**: Real-time data acquisition and control system integration
- **Weather APIs**: Ambient temperature and environmental data for enhanced thermal modeling
- **Grid Management Systems**: Integration with utility management platforms
- **Database Systems**: Persistent storage for historical data and analysis results