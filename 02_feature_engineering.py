"""
Step 2: Data Preprocessing and Feature Engineering
===================================================
This script:
1. Loads the raw synthetic datasets
2. Engineers rolling KPI features for supplier risk analysis
3. Creates risk labels based on business rules
4. Prepares data for model training

Features Created:
- rolling_sotd_3m: 3-month rolling average of SOTD %
- rolling_dppm_3m: 3-month rolling average of DPPM
- lead_time_variance: 3-month rolling variance of lead time
- late_delivery_rate: late_deliveries / total_deliveries

Risk Label Logic:
- High Risk (1): rolling_sotd_3m < 90 OR rolling_dppm_3m > 500
- Low Risk (0): Otherwise

Author: Data Engineering Team
Date: February 2026
"""

import pandas as pd
import numpy as np

# --- Configuration ---
DATA_PATH = "."
ROLLING_WINDOW = 3  # 3-month rolling window for KPI calculations

# Risk thresholds (business-defined)
SOTD_THRESHOLD = 90       # Below 90% on-time delivery = risk
DPPM_THRESHOLD = 500      # Above 500 defects per million = risk


def load_data() -> tuple:
    """
    Load all raw CSV files.
    
    Returns:
        Tuple of (supplier_master_df, demand_df, kpi_df)
    """
    print("Loading raw data files...")
    
    supplier_master = pd.read_csv(f"{DATA_PATH}/supplier_master.csv")
    demand_data = pd.read_csv(f"{DATA_PATH}/demand_data.csv")
    supplier_kpi = pd.read_csv(f"{DATA_PATH}/supplier_kpi_data.csv")
    
    # Convert date columns to datetime (explicit format for cleaner parsing)
    demand_data['date'] = pd.to_datetime(demand_data['date'], format='%Y-%m-%d')
    supplier_kpi['date'] = pd.to_datetime(supplier_kpi['date'], format='%Y-%m-%d')
    
    print(f"  ✓ supplier_master: {len(supplier_master)} records")
    print(f"  ✓ demand_data: {len(demand_data)} records")
    print(f"  ✓ supplier_kpi_data: {len(supplier_kpi)} records")
    
    return supplier_master, demand_data, supplier_kpi


def engineer_supplier_features(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling KPI features for supplier risk analysis.
    
    Why rolling averages:
    - Single-month KPIs can be noisy/volatile
    - 3-month rolling captures sustained performance trends
    - Aligns with typical quarterly review cycles
    
    Args:
        kpi_df: Raw supplier KPI dataframe
        
    Returns:
        DataFrame with engineered features
    """
    print("\nEngineering supplier features...")
    
    # Sort by supplier and date to ensure correct rolling calculations
    df = kpi_df.copy()
    df = df.sort_values(['supplier_id', 'date']).reset_index(drop=True)
    
    # Calculate features for each supplier separately
    # Using groupby + transform to maintain original DataFrame structure
    
    # 1. Rolling SOTD (3-month average)
    # Why: Smooths out monthly fluctuations, shows sustained delivery performance
    df['rolling_sotd_3m'] = df.groupby('supplier_id')['sotd_pct'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    
    # 2. Rolling DPPM (3-month average)
    # Why: Quality issues often emerge gradually; rolling average catches trends
    df['rolling_dppm_3m'] = df.groupby('supplier_id')['dppm'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    
    # 3. Lead Time Variance (3-month rolling)
    # Why: High variance indicates unreliable delivery times, a supply chain risk
    df['lead_time_variance'] = df.groupby('supplier_id')['lead_time_days'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).var()
    )
    # Fill NaN variance (occurs when window has only 1 value) with 0
    df['lead_time_variance'] = df['lead_time_variance'].fillna(0)
    
    # 4. Late Delivery Rate
    # Why: Direct measure of delivery reliability as a ratio
    # Guard against division by zero
    df['late_delivery_rate'] = np.where(
        df['total_deliveries'] > 0,
        df['late_deliveries'] / df['total_deliveries'],
        0
    )
    
    # Round numeric columns for cleaner output
    df['rolling_sotd_3m'] = df['rolling_sotd_3m'].round(2)
    df['rolling_dppm_3m'] = df['rolling_dppm_3m'].round(2)
    df['lead_time_variance'] = df['lead_time_variance'].round(2)
    df['late_delivery_rate'] = df['late_delivery_rate'].round(4)
    
    print(f"  ✓ rolling_sotd_3m: 3-month rolling average of SOTD %")
    print(f"  ✓ rolling_dppm_3m: 3-month rolling average of DPPM")
    print(f"  ✓ lead_time_variance: 3-month rolling variance of lead time")
    print(f"  ✓ late_delivery_rate: late_deliveries / total_deliveries")
    
    return df


def create_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business rules to create risk labels.
    
    Risk Logic (explicitly defined by business):
    - High Risk (1): rolling_sotd_3m < 90 OR rolling_dppm_3m > 500
    - Low Risk (0): Otherwise
    
    Why these thresholds:
    - SOTD < 90%: Industry standard for acceptable on-time delivery
    - DPPM > 500: High defect rate indicating quality control issues
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        DataFrame with risk labels
    """
    print("\nCreating risk labels...")
    
    # Apply risk logic using explicit business rules
    # Using np.where for clarity and performance
    df['risk_label'] = np.where(
        (df['rolling_sotd_3m'] < SOTD_THRESHOLD) | (df['rolling_dppm_3m'] > DPPM_THRESHOLD),
        1,  # High Risk
        0   # Low Risk
    )
    
    # Count risk distribution
    risk_counts = df['risk_label'].value_counts()
    high_risk_count = risk_counts.get(1, 0)
    low_risk_count = risk_counts.get(0, 0)
    total_records = len(df)
    
    print(f"  ✓ Risk labels created using thresholds:")
    print(f"      - SOTD threshold: < {SOTD_THRESHOLD}%")
    print(f"      - DPPM threshold: > {DPPM_THRESHOLD}")
    print(f"  ✓ Risk distribution:")
    print(f"      - High Risk (1): {high_risk_count} records ({high_risk_count/total_records*100:.1f}%)")
    print(f"      - Low Risk (0): {low_risk_count} records ({low_risk_count/total_records*100:.1f}%)")
    
    return df


def prepare_demand_data(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare demand data for Prophet forecasting.
    
    Prophet requires specific column names:
    - 'ds': datestamp column
    - 'y': target variable to forecast
    
    We'll create separate datasets per category for individual forecasts.
    
    Args:
        demand_df: Raw demand dataframe
        
    Returns:
        Cleaned demand dataframe
    """
    print("\nPreparing demand data for forecasting...")
    
    df = demand_df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values(['part_category', 'date']).reset_index(drop=True)
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ⚠ Warning: {missing_count} missing values found")
    else:
        print(f"  ✓ No missing values in demand data")
    
    # Summary statistics
    print(f"  ✓ Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"  ✓ Categories: {df['part_category'].unique().tolist()}")
    
    return df


def save_processed_data(supplier_features_df: pd.DataFrame, demand_df: pd.DataFrame):
    """
    Save processed datasets for model training.
    
    Args:
        supplier_features_df: Supplier data with features and labels
        demand_df: Cleaned demand data
    """
    print("\nSaving processed data...")
    
    # Save supplier features (for risk model training)
    supplier_features_df.to_csv(f"{DATA_PATH}/supplier_features.csv", index=False)
    print(f"  ✓ supplier_features.csv saved ({len(supplier_features_df)} records)")
    
    # Save cleaned demand data (for Prophet forecasting)
    demand_df.to_csv(f"{DATA_PATH}/demand_cleaned.csv", index=False)
    print(f"  ✓ demand_cleaned.csv saved ({len(demand_df)} records)")


def display_feature_summary(df: pd.DataFrame):
    """
    Display summary statistics for engineered features.
    """
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY STATISTICS")
    print("=" * 60)
    
    feature_cols = ['rolling_sotd_3m', 'rolling_dppm_3m', 'lead_time_variance', 'late_delivery_rate']
    
    print("\nEngineered Features Statistics:")
    print(df[feature_cols].describe().round(2).to_string())
    
    # Show sample of high-risk records
    print("\n--- Sample High-Risk Records ---")
    high_risk = df[df['risk_label'] == 1][['supplier_id', 'date', 'rolling_sotd_3m', 'rolling_dppm_3m', 'risk_label']]
    if len(high_risk) > 0:
        print(high_risk.head(5).to_string(index=False))
    else:
        print("No high-risk records found.")
    
    # Show risk breakdown by supplier
    print("\n--- Risk Distribution by Supplier ---")
    risk_by_supplier = df.groupby('supplier_id')['risk_label'].agg(['sum', 'count'])
    risk_by_supplier.columns = ['high_risk_months', 'total_months']
    risk_by_supplier['risk_rate'] = (risk_by_supplier['high_risk_months'] / risk_by_supplier['total_months'] * 100).round(1)
    print(risk_by_supplier.to_string())


def main():
    """Main execution function."""
    print("=" * 60)
    print("STEP 2: Data Preprocessing & Feature Engineering")
    print("=" * 60)
    
    # Load raw data
    supplier_master, demand_data, supplier_kpi = load_data()
    
    # Engineer supplier features
    supplier_features = engineer_supplier_features(supplier_kpi)
    
    # Create risk labels
    supplier_features = create_risk_labels(supplier_features)
    
    # Merge with supplier master for complete context
    supplier_features = supplier_features.merge(
        supplier_master[['supplier_id', 'supplier_name', 'region', 'commodity']],
        on='supplier_id',
        how='left'
    )
    
    # Prepare demand data
    demand_cleaned = prepare_demand_data(demand_data)
    
    # Save processed data
    save_processed_data(supplier_features, demand_cleaned)
    
    # Display summary
    display_feature_summary(supplier_features)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print("\nProcessed files ready for model training:")
    print("  - supplier_features.csv (for Logistic Regression)")
    print("  - demand_cleaned.csv (for Prophet forecasting)")


if __name__ == "__main__":
    main()
