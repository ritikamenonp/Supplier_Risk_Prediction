"""
Step 3A: Demand Forecasting with Prophet
========================================
This script:
1. Loads cleaned demand data
2. Trains Prophet models for each product category
3. Forecasts demand for the next 3 months
4. Evaluates model performance using MAPE
5. Exports forecasts to CSV for Power BI

Model: Facebook Prophet
- Handles seasonality automatically
- Works well with monthly data
- Provides uncertainty intervals (yhat_lower, yhat_upper)

Author: Data Engineering Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings

# Suppress Prophet's verbose logging
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = "."
FORECAST_HORIZON = 3  # Forecast 3 months ahead


def load_demand_data() -> pd.DataFrame:
    """Load cleaned demand data."""
    print("Loading demand data...")
    df = pd.read_csv(f"{DATA_PATH}/demand_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    print(f"  ✓ Loaded {len(df)} records")
    return df


def prepare_prophet_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prepare data in Prophet's required format.
    
    Prophet requires exactly two columns:
    - 'ds': datetime column (datestamp)
    - 'y': numeric column to forecast
    
    Args:
        df: Full demand dataframe
        category: Product category to filter
        
    Returns:
        DataFrame with 'ds' and 'y' columns
    """
    category_df = df[df['part_category'] == category].copy()
    prophet_df = category_df[['date', 'demand_qty']].rename(
        columns={'date': 'ds', 'demand_qty': 'y'}
    )
    return prophet_df.sort_values('ds').reset_index(drop=True)


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = (1/n) * Σ|actual - predicted| / |actual| * 100
    
    Why MAPE:
    - Easy to interpret (percentage error)
    - Scale-independent
    - Common metric for demand forecasting
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE as a percentage
    """
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def train_and_forecast(df: pd.DataFrame, category: str) -> tuple:
    """
    Train Prophet model and generate forecasts.
    
    Training approach:
    - Use all but last 3 months for training
    - Use last 3 months for validation (calculate MAPE)
    - Retrain on full data for final forecast
    
    Args:
        df: Prophet-formatted dataframe
        category: Category name for logging
        
    Returns:
        Tuple of (forecast_df, mape_score)
    """
    print(f"\n  Training model for: {category}")
    
    # Split data: hold out last 3 months for validation
    train_df = df.iloc[:-FORECAST_HORIZON]
    valid_df = df.iloc[-FORECAST_HORIZON:]
    
    # Initialize Prophet model
    # yearly_seasonality=True: Capture annual patterns
    # weekly_seasonality=False: Monthly data, no weekly patterns
    # daily_seasonality=False: Monthly data, no daily patterns
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for data with growing trend
        interval_width=0.95  # 95% confidence interval
    )
    
    # Train on training data
    model.fit(train_df)
    
    # Validate on held-out data
    valid_forecast = model.predict(valid_df[['ds']])
    mape = calculate_mape(
        valid_df['y'].values,
        valid_forecast['yhat'].values
    )
    print(f"    → Validation MAPE: {mape:.2f}%")
    
    # Retrain on full data for production forecast
    full_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    full_model.fit(df)
    
    # Generate future dates for forecasting
    future_dates = full_model.make_future_dataframe(periods=FORECAST_HORIZON, freq='MS')
    
    # Generate forecast
    forecast = full_model.predict(future_dates)
    
    # Extract only future predictions (not historical fitted values)
    last_historical_date = df['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_historical_date].copy()
    
    # Add category identifier
    future_forecast['part_category'] = category
    
    print(f"    → Generated {len(future_forecast)} month forecast")
    
    return future_forecast, mape


def run_demand_forecasting():
    """Main demand forecasting pipeline."""
    print("=" * 60)
    print("STEP 3A: Demand Forecasting with Prophet")
    print("=" * 60)
    
    # Load data
    demand_df = load_demand_data()
    
    # Get unique categories
    categories = demand_df['part_category'].unique()
    print(f"\nCategories to forecast: {list(categories)}")
    
    # Store results
    all_forecasts = []
    mape_scores = {}
    
    # Train model for each category
    print("\n" + "-" * 40)
    print("Training Prophet Models")
    print("-" * 40)
    
    for category in categories:
        # Prepare data
        prophet_df = prepare_prophet_data(demand_df, category)
        
        # Train and forecast
        forecast, mape = train_and_forecast(prophet_df, category)
        
        # Store results
        all_forecasts.append(forecast)
        mape_scores[category] = mape
    
    # Combine all forecasts
    combined_forecast = pd.concat(all_forecasts, ignore_index=True)
    
    # Select and rename columns for Power BI
    output_df = combined_forecast[[
        'ds', 'part_category', 'yhat', 'yhat_lower', 'yhat_upper'
    ]].copy()
    
    output_df = output_df.rename(columns={
        'ds': 'forecast_date',
        'yhat': 'forecast_qty',
        'yhat_lower': 'forecast_lower',
        'yhat_upper': 'forecast_upper'
    })
    
    # Round forecasts to integers (demand quantities)
    output_df['forecast_qty'] = output_df['forecast_qty'].round(0).astype(int)
    output_df['forecast_lower'] = output_df['forecast_lower'].round(0).astype(int)
    output_df['forecast_upper'] = output_df['forecast_upper'].round(0).astype(int)
    
    # Format date for Power BI
    output_df['forecast_date'] = output_df['forecast_date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    output_df.to_csv(f"{DATA_PATH}/demand_forecast.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEMAND FORECASTING COMPLETE")
    print("=" * 60)
    
    print("\nModel Performance (MAPE):")
    for category, mape in mape_scores.items():
        status = "✓ Good" if mape < 15 else "⚠ Review"
        print(f"  {category}: {mape:.2f}% {status}")
    
    avg_mape = np.mean(list(mape_scores.values()))
    print(f"\n  Average MAPE: {avg_mape:.2f}%")
    
    print("\nForecast Output:")
    print(output_df.to_string(index=False))
    
    print(f"\n✓ demand_forecast.csv saved ({len(output_df)} records)")
    
    return output_df, mape_scores


if __name__ == "__main__":
    run_demand_forecasting()
