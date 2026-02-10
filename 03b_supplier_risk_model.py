"""
Step 3B: Supplier Risk Detection with Logistic Regression
=========================================================
This script:
1. Loads supplier features with risk labels
2. Trains a Logistic Regression model
3. Predicts risk scores for all suppliers
4. Evaluates model with accuracy and classification report
5. Exports risk scores to CSV for Power BI

Model: Logistic Regression
- Interpretable coefficients (business can understand feature importance)
- Outputs probability scores (0-1)
- No hyperparameter tuning needed for baseline

Author: Data Engineering Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
DATA_PATH = "."
RANDOM_STATE = 42  # For reproducibility
TEST_SIZE = 0.25   # 25% holdout for testing

# Features to use for risk prediction (as specified in requirements)
FEATURE_COLUMNS = [
    'rolling_sotd_3m',
    'rolling_dppm_3m', 
    'lead_time_variance',
    'late_delivery_rate'
]


def load_supplier_features() -> pd.DataFrame:
    """Load supplier features data."""
    print("Loading supplier features...")
    df = pd.read_csv(f"{DATA_PATH}/supplier_features.csv")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    print(f"  ✓ Loaded {len(df)} records")
    print(f"  ✓ Features: {FEATURE_COLUMNS}")
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vector for training.
    
    Why we filter for records with enough history:
    - Rolling features need at least 3 months of data
    - First 2 months per supplier have partial rolling windows
    
    Args:
        df: Full supplier features dataframe
        
    Returns:
        Tuple of (X, y, filtered_df)
    """
    print("\nPreparing training data...")
    
    # Use all records (rolling features handle min_periods=1)
    # But filter out any rows with NaN in features
    clean_df = df.dropna(subset=FEATURE_COLUMNS + ['risk_label'])
    
    X = clean_df[FEATURE_COLUMNS].values
    y = clean_df['risk_label'].values
    
    print(f"  ✓ Training samples: {len(X)}")
    print(f"  ✓ Class distribution:")
    print(f"      - Low Risk (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"      - High Risk (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    
    return X, y, clean_df


def train_risk_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train Logistic Regression model for supplier risk.
    
    Model choice rationale:
    - Logistic Regression is interpretable (stakeholders can understand)
    - Outputs calibrated probabilities
    - Works well with small datasets
    - No overfitting concerns with 4 features
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (trained_model, scaler, metrics_dict)
    """
    print("\n" + "-" * 40)
    print("Training Logistic Regression Model")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Standardize features
    # Why: Logistic Regression converges faster with scaled features
    # Also makes coefficients comparable across features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    # class_weight='balanced': Handle imbalanced classes (few high-risk samples)
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight='balanced',  # Important: handles class imbalance
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])
    
    print(f"\n  Model Performance:")
    print(f"  → Accuracy: {accuracy*100:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Low    High")
    print(f"    Actual Low    {conf_matrix[0,0]:4d}   {conf_matrix[0,1]:4d}")
    print(f"    Actual High   {conf_matrix[1,0]:4d}   {conf_matrix[1,1]:4d}")
    print(f"\n  Classification Report:")
    print(class_report)
    
    # Feature importance (coefficients)
    print("  Feature Importance (Coefficients):")
    for feature, coef in zip(FEATURE_COLUMNS, model.coef_[0]):
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"    {feature}: {coef:+.3f} ({direction})")
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    return model, scaler, metrics


def generate_risk_scores(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Generate risk scores for all supplier records.
    
    Output columns:
    - risk_score: Probability of high risk (0-1)
    - risk_class: 'High' if risk_score >= 0.5, else 'Low'
    
    Args:
        df: Full supplier features dataframe
        model: Trained Logistic Regression model
        scaler: Fitted StandardScaler
        
    Returns:
        DataFrame with risk scores
    """
    print("\nGenerating risk scores for all records...")
    
    # Get features
    X = df[FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X)
    
    # Predict probabilities
    risk_scores = model.predict_proba(X_scaled)[:, 1]
    
    # Create output dataframe
    output_df = df[['supplier_id', 'date', 'supplier_name', 'region', 'commodity']].copy()
    output_df['sotd_pct'] = df['sotd_pct']
    output_df['dppm'] = df['dppm']
    output_df['rolling_sotd_3m'] = df['rolling_sotd_3m']
    output_df['rolling_dppm_3m'] = df['rolling_dppm_3m']
    output_df['late_delivery_rate'] = df['late_delivery_rate']
    output_df['risk_score'] = np.round(risk_scores, 4)
    output_df['risk_class'] = np.where(risk_scores >= 0.5, 'High', 'Low')
    output_df['risk_label_actual'] = df['risk_label'].map({0: 'Low', 1: 'High'})
    
    # Format date for Power BI
    output_df['date'] = pd.to_datetime(output_df['date']).dt.strftime('%Y-%m-%d')
    
    return output_df


def create_latest_supplier_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a snapshot of the latest risk score per supplier.
    
    Why: Power BI dashboards often need current state, not full history.
    
    Args:
        df: Full risk scores dataframe
        
    Returns:
        DataFrame with one row per supplier (latest date)
    """
    # Get the latest record for each supplier
    df_sorted = df.copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    latest = df_sorted.loc[df_sorted.groupby('supplier_id')['date'].idxmax()]
    latest['date'] = latest['date'].dt.strftime('%Y-%m-%d')
    
    return latest.reset_index(drop=True)


def run_risk_modeling():
    """Main supplier risk modeling pipeline."""
    print("=" * 60)
    print("STEP 3B: Supplier Risk Detection with Logistic Regression")
    print("=" * 60)
    
    # Load data
    supplier_df = load_supplier_features()
    
    # Prepare training data
    X, y, clean_df = prepare_training_data(supplier_df)
    
    # Train model
    model, scaler, metrics = train_risk_model(X, y)
    
    # Generate risk scores for all records
    risk_scores_df = generate_risk_scores(clean_df, model, scaler)
    
    # Save full history
    risk_scores_df.to_csv(f"{DATA_PATH}/supplier_risk_scores.csv", index=False)
    print(f"\n✓ supplier_risk_scores.csv saved ({len(risk_scores_df)} records)")
    
    # Create and save latest snapshot
    latest_snapshot = create_latest_supplier_snapshot(risk_scores_df)
    latest_snapshot.to_csv(f"{DATA_PATH}/supplier_risk_latest.csv", index=False)
    print(f"✓ supplier_risk_latest.csv saved ({len(latest_snapshot)} records)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUPPLIER RISK MODELING COMPLETE")
    print("=" * 60)
    
    print("\nLatest Supplier Risk Snapshot:")
    print(latest_snapshot[['supplier_id', 'supplier_name', 'risk_score', 'risk_class']].to_string(index=False))
    
    # Identify high-risk suppliers
    high_risk = latest_snapshot[latest_snapshot['risk_class'] == 'High']
    if len(high_risk) > 0:
        print(f"\n⚠ HIGH RISK SUPPLIERS ({len(high_risk)}):")
        for _, row in high_risk.iterrows():
            print(f"  - {row['supplier_id']} ({row['supplier_name']}): Score = {row['risk_score']:.2f}")
    else:
        print("\n✓ No suppliers currently in High Risk category")
    
    return risk_scores_df, model, scaler


if __name__ == "__main__":
    run_risk_modeling()
