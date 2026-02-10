"""
Master Pipeline: Supplier Risk Prediction Project
==================================================
This script runs the complete end-to-end pipeline:
1. Generate synthetic datasets
2. Feature engineering and preprocessing
3. Demand forecasting with Prophet
4. Supplier risk detection with Logistic Regression
5. Export all outputs for Power BI

Usage:
    python run_pipeline.py

Author: Data Engineering Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "."
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
N_SUPPLIERS = 8
PART_CATEGORIES = ["Electronics", "Mechanical", "Hydraulics"]
REGIONS = ["APAC", "EMEA", "NA", "LATAM"]
COMMODITIES = ["Sensors", "Fasteners", "Pumps", "Cables"]
RANDOM_SEED = 42

# Feature engineering parameters
ROLLING_WINDOW = 3
SOTD_THRESHOLD = 90
DPPM_THRESHOLD = 500

# Model parameters
FORECAST_HORIZON = 3
TEST_SIZE = 0.25

# Feature columns for risk model
FEATURE_COLUMNS = [
    'rolling_sotd_3m',
    'rolling_dppm_3m',
    'lead_time_variance',
    'late_delivery_rate'
]


# ============================================================================
# STEP 1: SYNTHETIC DATA GENERATION
# ============================================================================
def generate_supplier_master(n_suppliers: int) -> pd.DataFrame:
    """Generate master supplier list."""
    supplier_names = [
        "TechParts Inc.", "GlobalSupply Co.", "PrecisionMfg Ltd.",
        "QualityFirst Industries", "FastTrack Components", "ReliableSourcing LLC",
        "PrimeParts Group", "ValueChain Solutions"
    ]
    
    suppliers = []
    for i in range(1, n_suppliers + 1):
        suppliers.append({
            "supplier_id": f"SUP{i:03d}",
            "supplier_name": supplier_names[i - 1],
            "region": REGIONS[(i - 1) % len(REGIONS)],
            "commodity": COMMODITIES[(i - 1) % len(COMMODITIES)]
        })
    
    df = pd.DataFrame(suppliers)
    df.to_csv(f"{DATA_PATH}/supplier_master.csv", index=False)
    return df


def generate_demand_data(start_date: str, end_date: str, categories: list) -> pd.DataFrame:
    """Generate synthetic demand data with trend and seasonality."""
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    data = []
    
    category_base_demand = {"Electronics": 1500, "Mechanical": 1000, "Hydraulics": 600}
    
    for category in categories:
        base_demand = category_base_demand.get(category, 1000)
        
        for i, date in enumerate(dates):
            trend_multiplier = 1.0 + (i * 0.015)
            month_of_year = date.month
            seasonal_multiplier = 1.0 + 0.3 * np.sin((month_of_year - 4) * np.pi / 6)
            noise_multiplier = np.random.uniform(0.95, 1.05)
            demand = int(base_demand * trend_multiplier * seasonal_multiplier * noise_multiplier)
            
            data.append({
                "date": date.strftime('%Y-%m-%d'),
                "part_category": category,
                "demand_qty": demand
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['part_category', 'date']).reset_index(drop=True)
    df.to_csv(f"{DATA_PATH}/demand_data.csv", index=False)
    return df


def generate_supplier_kpi_data(supplier_ids: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate synthetic supplier KPI data with degradation patterns."""
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    data = []

    np.random.seed(RANDOM_SEED + 1)
    degrading_suppliers = np.random.choice(supplier_ids, size=3, replace=False)

    supplier_baselines = {}
    for idx, supplier_id in enumerate(supplier_ids):
        np.random.seed(RANDOM_SEED + idx + 10)
        supplier_baselines[supplier_id] = {
            "sotd_base": np.random.uniform(95, 99),
            "dppm_base": np.random.randint(80, 250),
            "lead_time_base": np.random.randint(18, 35),
            "deliveries_base": np.random.randint(60, 120)
        }

    np.random.seed(RANDOM_SEED)

    for supplier_id in supplier_ids:
        baseline = supplier_baselines[supplier_id]
        is_degrading = supplier_id in degrading_suppliers
        
        for date in dates:
            degradation_factor = 0
            if is_degrading and date > pd.Timestamp("2024-06-01"):
                months_since_degradation = (date.year - 2024) * 12 + (date.month - 6)
                degradation_factor = months_since_degradation * 0.8
            
            sotd_pct = baseline["sotd_base"] - degradation_factor * np.random.uniform(0.4, 0.6)
            sotd_pct = round(np.clip(sotd_pct * np.random.uniform(0.99, 1.01), 82, 99.5), 2)
            
            dppm = baseline["dppm_base"] + degradation_factor * np.random.uniform(15, 25)
            dppm = int(np.clip(dppm * np.random.uniform(0.95, 1.05), 50, 750))
            
            lead_time_days = baseline["lead_time_base"] + int(degradation_factor * 0.3)
            lead_time_days = int(np.clip(lead_time_days * np.random.uniform(0.92, 1.08), 12, 50))
            
            total_deliveries = int(baseline["deliveries_base"] * np.random.uniform(0.85, 1.15))
            on_time_rate = sotd_pct / 100.0
            late_deliveries = int(total_deliveries * (1 - on_time_rate))
            
            data.append({
                "supplier_id": supplier_id,
                "date": date.strftime('%Y-%m-%d'),
                "sotd_pct": sotd_pct,
                "dppm": dppm,
                "lead_time_days": lead_time_days,
                "total_deliveries": total_deliveries,
                "late_deliveries": late_deliveries
            })

    df = pd.DataFrame(data)
    df.to_csv(f"{DATA_PATH}/supplier_kpi_data.csv", index=False)
    return df


def step1_generate_data():
    """Execute Step 1: Generate all synthetic datasets."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Synthetic Datasets")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    
    supplier_master = generate_supplier_master(N_SUPPLIERS)
    print(f"  ✓ supplier_master.csv ({len(supplier_master)} suppliers)")
    
    demand_data = generate_demand_data(START_DATE, END_DATE, PART_CATEGORIES)
    print(f"  ✓ demand_data.csv ({len(demand_data)} records)")
    
    kpi_data = generate_supplier_kpi_data(
        supplier_master['supplier_id'].tolist(), START_DATE, END_DATE
    )
    print(f"  ✓ supplier_kpi_data.csv ({len(kpi_data)} records)")
    
    return supplier_master, demand_data, kpi_data


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
def step2_feature_engineering():
    """Execute Step 2: Feature engineering and preprocessing."""
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    
    # Load data
    supplier_master = pd.read_csv(f"{DATA_PATH}/supplier_master.csv")
    demand_data = pd.read_csv(f"{DATA_PATH}/demand_data.csv")
    supplier_kpi = pd.read_csv(f"{DATA_PATH}/supplier_kpi_data.csv")
    
    demand_data['date'] = pd.to_datetime(demand_data['date'], format='%Y-%m-%d')
    supplier_kpi['date'] = pd.to_datetime(supplier_kpi['date'], format='%Y-%m-%d')
    
    # Engineer features
    df = supplier_kpi.sort_values(['supplier_id', 'date']).reset_index(drop=True)
    
    df['rolling_sotd_3m'] = df.groupby('supplier_id')['sotd_pct'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    
    df['rolling_dppm_3m'] = df.groupby('supplier_id')['dppm'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    
    df['lead_time_variance'] = df.groupby('supplier_id')['lead_time_days'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).var()
    ).fillna(0)
    
    df['late_delivery_rate'] = np.where(
        df['total_deliveries'] > 0,
        df['late_deliveries'] / df['total_deliveries'],
        0
    )
    
    # Round features
    df['rolling_sotd_3m'] = df['rolling_sotd_3m'].round(2)
    df['rolling_dppm_3m'] = df['rolling_dppm_3m'].round(2)
    df['lead_time_variance'] = df['lead_time_variance'].round(2)
    df['late_delivery_rate'] = df['late_delivery_rate'].round(4)
    
    # Create risk labels
    df['risk_label'] = np.where(
        (df['rolling_sotd_3m'] < SOTD_THRESHOLD) | (df['rolling_dppm_3m'] > DPPM_THRESHOLD),
        1, 0
    )
    
    # Merge with supplier master
    df = df.merge(
        supplier_master[['supplier_id', 'supplier_name', 'region', 'commodity']],
        on='supplier_id', how='left'
    )
    
    # Save processed data
    df.to_csv(f"{DATA_PATH}/supplier_features.csv", index=False)
    demand_data.to_csv(f"{DATA_PATH}/demand_cleaned.csv", index=False)
    
    high_risk_count = df['risk_label'].sum()
    print(f"  ✓ Rolling features created (3-month window)")
    print(f"  ✓ Risk labels: {high_risk_count} high-risk records ({high_risk_count/len(df)*100:.1f}%)")
    print(f"  ✓ supplier_features.csv saved")
    print(f"  ✓ demand_cleaned.csv saved")
    
    return df, demand_data


# ============================================================================
# STEP 3A: DEMAND FORECASTING
# ============================================================================
def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def step3a_demand_forecasting():
    """Execute Step 3A: Demand forecasting with Prophet."""
    from prophet import Prophet
    
    print("\n" + "=" * 60)
    print("STEP 3A: Demand Forecasting (Prophet)")
    print("=" * 60)
    
    demand_df = pd.read_csv(f"{DATA_PATH}/demand_cleaned.csv")
    demand_df['date'] = pd.to_datetime(demand_df['date'], format='%Y-%m-%d')
    
    categories = demand_df['part_category'].unique()
    all_forecasts = []
    mape_scores = {}
    
    for category in categories:
        # Prepare Prophet data
        cat_df = demand_df[demand_df['part_category'] == category].copy()
        prophet_df = cat_df[['date', 'demand_qty']].rename(columns={'date': 'ds', 'demand_qty': 'y'})
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Split for validation
        train_df = prophet_df.iloc[:-FORECAST_HORIZON]
        valid_df = prophet_df.iloc[-FORECAST_HORIZON:]
        
        # Train validation model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        model.fit(train_df)
        
        # Calculate MAPE
        valid_forecast = model.predict(valid_df[['ds']])
        mape = calculate_mape(valid_df['y'].values, valid_forecast['yhat'].values)
        mape_scores[category] = mape
        
        # Train final model on all data
        full_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        full_model.fit(prophet_df)
        
        # Generate forecast
        future_dates = full_model.make_future_dataframe(periods=FORECAST_HORIZON, freq='MS')
        forecast = full_model.predict(future_dates)
        
        # Extract future predictions
        last_date = prophet_df['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_date].copy()
        future_forecast['part_category'] = category
        all_forecasts.append(future_forecast)
        
        print(f"  ✓ {category}: MAPE = {mape:.2f}%")
    
    # Combine and save
    combined = pd.concat(all_forecasts, ignore_index=True)
    output_df = combined[['ds', 'part_category', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    output_df = output_df.rename(columns={
        'ds': 'forecast_date',
        'yhat': 'forecast_qty',
        'yhat_lower': 'forecast_lower',
        'yhat_upper': 'forecast_upper'
    })
    
    output_df['forecast_qty'] = output_df['forecast_qty'].round(0).astype(int)
    output_df['forecast_lower'] = output_df['forecast_lower'].round(0).astype(int)
    output_df['forecast_upper'] = output_df['forecast_upper'].round(0).astype(int)
    output_df['forecast_date'] = output_df['forecast_date'].dt.strftime('%Y-%m-%d')
    
    output_df.to_csv(f"{DATA_PATH}/demand_forecast.csv", index=False)
    
    avg_mape = np.mean(list(mape_scores.values()))
    print(f"  ✓ Average MAPE: {avg_mape:.2f}%")
    print(f"  ✓ demand_forecast.csv saved ({len(output_df)} records)")
    
    return output_df, mape_scores


# ============================================================================
# STEP 3B: SUPPLIER RISK MODEL
# ============================================================================
def step3b_supplier_risk_model():
    """Execute Step 3B: Supplier risk detection with Logistic Regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    print("\n" + "=" * 60)
    print("STEP 3B: Supplier Risk Model (Logistic Regression)")
    print("=" * 60)
    
    # Load data
    supplier_df = pd.read_csv(f"{DATA_PATH}/supplier_features.csv")
    supplier_df['date'] = pd.to_datetime(supplier_df['date'], format='%Y-%m-%d')
    
    # Prepare features
    X = supplier_df[FEATURE_COLUMNS].values
    y = supplier_df['risk_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        random_state=RANDOM_SEED,
        class_weight='balanced',
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  ✓ Model trained (Accuracy: {accuracy*100:.1f}%)")
    
    # Feature importance
    print("  ✓ Feature Coefficients:")
    for feature, coef in zip(FEATURE_COLUMNS, model.coef_[0]):
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"      {feature}: {coef:+.3f} ({direction})")
    
    # Generate risk scores for all records
    X_all_scaled = scaler.transform(supplier_df[FEATURE_COLUMNS].values)
    risk_scores = model.predict_proba(X_all_scaled)[:, 1]
    
    # Create output DataFrame
    output_df = supplier_df[['supplier_id', 'date', 'supplier_name', 'region', 'commodity']].copy()
    output_df['sotd_pct'] = supplier_df['sotd_pct']
    output_df['dppm'] = supplier_df['dppm']
    output_df['rolling_sotd_3m'] = supplier_df['rolling_sotd_3m']
    output_df['rolling_dppm_3m'] = supplier_df['rolling_dppm_3m']
    output_df['late_delivery_rate'] = supplier_df['late_delivery_rate']
    output_df['risk_score'] = np.round(risk_scores, 4)
    output_df['risk_class'] = np.where(risk_scores >= 0.5, 'High', 'Low')
    output_df['risk_label_actual'] = supplier_df['risk_label'].map({0: 'Low', 1: 'High'})
    output_df['date'] = output_df['date'].dt.strftime('%Y-%m-%d')
    
    # Save full history
    output_df.to_csv(f"{DATA_PATH}/supplier_risk_scores.csv", index=False)
    
    # Create latest snapshot
    output_df['date_dt'] = pd.to_datetime(output_df['date'])
    latest = output_df.loc[output_df.groupby('supplier_id')['date_dt'].idxmax()]
    latest = latest.drop(columns=['date_dt']).reset_index(drop=True)
    latest.to_csv(f"{DATA_PATH}/supplier_risk_latest.csv", index=False)
    
    print(f"  ✓ supplier_risk_scores.csv saved ({len(output_df)} records)")
    print(f"  ✓ supplier_risk_latest.csv saved ({len(latest)} records)")
    
    # Identify high-risk suppliers
    high_risk = latest[latest['risk_class'] == 'High']
    if len(high_risk) > 0:
        print(f"\n  ⚠ HIGH RISK SUPPLIERS ({len(high_risk)}):")
        for _, row in high_risk.iterrows():
            print(f"      {row['supplier_id']} ({row['supplier_name']}): {row['risk_score']:.2f}")
    
    return output_df, model, accuracy


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Run the complete pipeline."""
    start_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("  SUPPLIER RISK PREDICTION PIPELINE")
    print("  " + "=" * 56)
    print(f"  Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Generate Data
    supplier_master, demand_data, kpi_data = step1_generate_data()
    
    # Step 2: Feature Engineering
    supplier_features, demand_cleaned = step2_feature_engineering()
    
    # Step 3A: Demand Forecasting
    demand_forecast, mape_scores = step3a_demand_forecasting()
    
    # Step 3B: Risk Model
    risk_scores, model, accuracy = step3b_supplier_risk_model()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Duration: {duration:.1f} seconds")
    print(f"  Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n  OUTPUT FILES FOR POWER BI:")
    print("  " + "-" * 40)
    
    output_files = [
        ("demand_forecast.csv", "3-month demand predictions"),
        ("supplier_risk_scores.csv", "Full risk history"),
        ("supplier_risk_latest.csv", "Current supplier risk snapshot"),
        ("supplier_master.csv", "Supplier reference data"),
        ("demand_data.csv", "Historical demand data")
    ]
    
    for filename, description in output_files:
        filepath = f"{DATA_PATH}/{filename}"
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"    ✓ {filename} ({size:,} bytes)")
            print(f"      → {description}")
    
    print("\n  MODEL PERFORMANCE:")
    print("  " + "-" * 40)
    print(f"    Demand Forecast MAPE: {np.mean(list(mape_scores.values())):.2f}%")
    print(f"    Risk Model Accuracy: {accuracy*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("  Ready for Power BI import!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
