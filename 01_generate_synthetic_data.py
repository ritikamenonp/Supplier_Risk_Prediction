"""
Step 1: Generate Synthetic Datasets for Supplier Risk Prediction Project
=========================================================================
This script creates three CSV files with realistic synthetic data:
1. supplier_master.csv - Static supplier information
2. demand_data.csv - Monthly demand with trend and seasonality
3. supplier_kpi_data.csv - Monthly supplier performance metrics

Author: Data Engineering Team
Date: February 2026
"""

import pandas as pd
import numpy as np

# --- Configuration ---
# All configurable parameters in one place for easy adjustment
DATA_PATH = "."
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"  # 3 years of historical data
N_SUPPLIERS = 8
PART_CATEGORIES = ["Electronics", "Mechanical", "Hydraulics"]
REGIONS = ["APAC", "EMEA", "NA", "LATAM"]
COMMODITIES = ["Sensors", "Fasteners", "Pumps", "Cables"]

# Seed for reproducibility - critical for consistent results across runs
RANDOM_SEED = 42


def generate_supplier_master(n_suppliers: int) -> pd.DataFrame:
    """
    Generates a master list of suppliers with static attributes.
    
    Why this structure:
    - supplier_id: Unique identifier for joins across tables
    - supplier_name: Human-readable name for reporting
    - region: Geographic segmentation for risk analysis
    - commodity: Product category for spend analysis
    """
    # Predefined realistic supplier names for consistency
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
            "region": REGIONS[(i - 1) % len(REGIONS)],  # Distribute across regions
            "commodity": COMMODITIES[(i - 1) % len(COMMODITIES)]
        })
    
    df = pd.DataFrame(suppliers)
    df.to_csv(f"{DATA_PATH}/supplier_master.csv", index=False)
    print(f"✓ supplier_master.csv generated ({len(df)} suppliers)")
    return df


def generate_demand_data(start_date: str, end_date: str, categories: list) -> pd.DataFrame:
    """
    Generates synthetic monthly demand data with realistic patterns.
    
    Patterns included:
    - Upward trend: Simulates business growth over time
    - Seasonality: Higher demand mid-year (Q2-Q3), lower in Q1/Q4
    - Random noise: ±5% variation for realism
    
    Why monthly granularity:
    - Aligns with typical supply chain planning cycles
    - Sufficient data points for Prophet forecasting (36 months)
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    data = []
    
    # Different base demand for each category to simulate product mix
    category_base_demand = {
        "Electronics": 1500,
        "Mechanical": 1000,
        "Hydraulics": 600
    }
    
    for category in categories:
        base_demand = category_base_demand.get(category, 1000)
        n_months = len(dates)
        
        for i, date in enumerate(dates):
            # Trend component: 1.5% monthly growth compounded
            trend_multiplier = 1.0 + (i * 0.015)
            
            # Seasonal component: sinusoidal pattern peaking in summer
            # Month 7 (July) = peak, Month 1 (Jan) = trough
            month_of_year = date.month
            seasonal_multiplier = 1.0 + 0.3 * np.sin((month_of_year - 4) * np.pi / 6)
            
            # Random noise: ±5% for realism
            noise_multiplier = np.random.uniform(0.95, 1.05)
            
            demand = int(base_demand * trend_multiplier * seasonal_multiplier * noise_multiplier)
            
            data.append({
                "date": date.strftime('%Y-%m-%d'),
                "part_category": category,
                "demand_qty": demand
            })
    
    df = pd.DataFrame(data)
    # Sort by category and date for clarity
    df = df.sort_values(['part_category', 'date']).reset_index(drop=True)
    df.to_csv(f"{DATA_PATH}/demand_data.csv", index=False)
    print(f"✓ demand_data.csv generated ({len(df)} records, {len(dates)} months per category)")
    return df


def generate_supplier_kpi_data(supplier_ids: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates synthetic monthly supplier KPI data with realistic patterns.
    
    Business rules implemented:
    1. Some suppliers (3 out of 8) show gradual performance degradation after mid-2024
    2. Stable suppliers maintain consistent performance with minor fluctuations
    3. late_deliveries is directly derived from sotd_pct for logical consistency
    
    KPI definitions:
    - sotd_pct: Supplier On-Time Delivery percentage (target: >95%)
    - dppm: Defective Parts Per Million (target: <300)
    - lead_time_days: Average days from order to delivery
    - total_deliveries: Monthly delivery count
    - late_deliveries: Count of deliveries past due date
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    data = []

    # Identify 3 suppliers who will show performance degradation
    # This simulates real-world scenario where some suppliers start failing
    np.random.seed(RANDOM_SEED + 1)  # Separate seed for supplier selection
    degrading_suppliers = np.random.choice(supplier_ids, size=3, replace=False)
    print(f"  → Degrading suppliers: {list(degrading_suppliers)}")

    # Define baseline performance for each supplier
    supplier_baselines = {}
    for idx, supplier_id in enumerate(supplier_ids):
        np.random.seed(RANDOM_SEED + idx + 10)
        supplier_baselines[supplier_id] = {
            "sotd_base": np.random.uniform(95, 99),
            "dppm_base": np.random.randint(80, 250),
            "lead_time_base": np.random.randint(18, 35),
            "deliveries_base": np.random.randint(60, 120)
        }

    # Reset seed for consistent data generation
    np.random.seed(RANDOM_SEED)

    for supplier_id in supplier_ids:
        baseline = supplier_baselines[supplier_id]
        is_degrading = supplier_id in degrading_suppliers
        
        for date in dates:
            # Calculate degradation factor for at-risk suppliers
            # Degradation starts after June 2024 and accelerates over time
            degradation_factor = 0
            if is_degrading and date > pd.Timestamp("2024-06-01"):
                months_since_degradation = (date.year - 2024) * 12 + (date.month - 6)
                degradation_factor = months_since_degradation * 0.8  # Gradual increase
            
            # SOTD: Decreases for degrading suppliers
            sotd_pct = baseline["sotd_base"] - degradation_factor * np.random.uniform(0.4, 0.6)
            sotd_pct = round(np.clip(sotd_pct * np.random.uniform(0.99, 1.01), 82, 99.5), 2)
            
            # DPPM: Increases for degrading suppliers
            dppm = baseline["dppm_base"] + degradation_factor * np.random.uniform(15, 25)
            dppm = int(np.clip(dppm * np.random.uniform(0.95, 1.05), 50, 750))
            
            # Lead time: Slightly increases for degrading suppliers
            lead_time_days = baseline["lead_time_base"] + int(degradation_factor * 0.3)
            lead_time_days = int(np.clip(lead_time_days * np.random.uniform(0.92, 1.08), 12, 50))
            
            # Total deliveries: Relatively stable with minor fluctuation
            total_deliveries = int(baseline["deliveries_base"] * np.random.uniform(0.85, 1.15))
            
            # Late deliveries: Derived from SOTD for logical consistency
            # If SOTD is 95%, then 5% of deliveries are late
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
    print(f"✓ supplier_kpi_data.csv generated ({len(df)} records)")
    return df


def main():
    """Main execution function."""
    print("=" * 60)
    print("STEP 1: Generating Synthetic Datasets")
    print("=" * 60)
    
    # Set global random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Generate all three datasets
    print("\n1. Generating supplier master data...")
    supplier_master_df = generate_supplier_master(N_SUPPLIERS)
    
    print("\n2. Generating demand data...")
    demand_df = generate_demand_data(START_DATE, END_DATE, PART_CATEGORIES)
    
    print("\n3. Generating supplier KPI data...")
    kpi_df = generate_supplier_kpi_data(
        supplier_master_df['supplier_id'].tolist(),
        START_DATE,
        END_DATE
    )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles created in: {DATA_PATH}")
    print(f"  - supplier_master.csv: {len(supplier_master_df)} suppliers")
    print(f"  - demand_data.csv: {len(demand_df)} records")
    print(f"  - supplier_kpi_data.csv: {len(kpi_df)} records")
    
    # Quick data quality check
    print("\n--- Quick Data Preview ---")
    print("\nSupplier Master (first 3 rows):")
    print(supplier_master_df.head(3).to_string(index=False))
    
    print("\nDemand Data (first 3 rows):")
    print(demand_df.head(3).to_string(index=False))
    
    print("\nSupplier KPI Data (first 3 rows):")
    print(kpi_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
