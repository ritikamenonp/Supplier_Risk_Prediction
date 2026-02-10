# Supplier Risk Prediction and Demand Forecasting

Supply chain analytics project for predictive supplier risk detection and demand trend analysis using business intelligence dashboards.

## Project Overview

This project provides:
- **Demand Forecasting**: 3-month ahead predictions using Prophet time series model
- **Supplier Risk Detection**: ML-based classification using Logistic Regression
- **Interactive Dashboard**: Streamlit web application for visualization and analysis

## Technologies Used

- **Python 3.12**
- **Prophet**: Time series forecasting (MAPE: 4.72%)
- **Scikit-learn**: Logistic Regression for risk classification (Accuracy: 97.2%)
- **Streamlit**: Interactive web dashboard
- **Plotly**: Dynamic visualizations
- **Pandas**: Data manipulation and feature engineering

## Project Structure

```
Supplier_Risk_Prediction/
├── app.py                          # Streamlit dashboard
├── run_pipeline.py                 # Master pipeline script
├── 01_generate_synthetic_data.py   # Data generation
├── 02_feature_engineering.py       # Feature engineering
├── 03a_demand_forecasting.py       # Prophet model
├── 03b_supplier_risk_model.py      # Risk classification
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## How to Run

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Data and Train Models

```bash
python run_pipeline.py
```

This will:
- Generate synthetic supplier and demand data
- Engineer features (rolling KPIs)
- Train Prophet forecasting model
- Train Logistic Regression risk model
- Output results to CSV files

### 3. Launch Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open at `http://localhost:8501` or `http://localhost:8502`

## Features

### Overview Page
- Key metrics (total suppliers, risk distribution)
- High-risk supplier alerts
- 3-month demand forecast summary
- Supplier status table

### Demand Forecast Page
- Historical trends (Jan 2023 - Dec 2025)
- 3-month forecast (Jan - Mar 2026)
- Confidence intervals (95%)
- Category-wise analysis

### Supplier Risk Page
- Individual supplier performance
- Risk score trends
- On-Time Delivery (SOTD) trends
- Defects Per Million (DPPM) trends
- Current period KPI summary

## Model Performance

- **Demand Forecast**: MAPE 4.72%
- **Risk Detection**: 97.2% Accuracy

## Data Sources (Production)

For real-world deployment:
- ERP System (SAP, Oracle) via SQL connector
- SharePoint via Microsoft Graph API
- Supplier Portal APIs

## Future Enhancements

- Database integration (PostgreSQL)
- Automated pipeline scheduling (Airflow / Task Scheduler)
- Real-time data refresh
- Power BI integration
- Model monitoring and alerts

