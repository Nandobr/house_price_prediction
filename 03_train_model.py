import pandas as pd
import numpy as np
import os
import time
import hashlib
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Configuration
INPUT_FILE = 'engineered_features.csv'
EXPERIMENTS_FILE = 'experiments.csv'

# Default Experiment Settings
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2019
DEFAULT_FEATURES = [
    'SFLA', 'RMBED', 'YRBLT', 'NBHD', 'LUC', 'Month', 
    'HouseAge_Squared', 'Bed_Bath_Ratio',
    'NBHD_Median_Size', 'Size_vs_NBHD', 'SFLA_Squared'
]
DEFAULT_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_jobs': -1,
    'random_state': 42
}

def log_experiment(start_year, end_year, features, params, metrics):
    """Logs experiment results to CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feature_hash = hashlib.md5(str(sorted(features)).encode()).hexdigest()[:8]
    
    row = {
        'Timestamp': timestamp,
        'Period': f"{start_year}-{end_year}",
        'Features': str(features),
        'Feature_Hash': feature_hash,
        'Params': str(params),
        'R2_Mean': metrics['r2_mean'],
        'R2_Std': metrics['r2_std'],
        'RMSE_Mean': metrics['rmse_mean']
    }
    
    df_row = pd.DataFrame([row])
    
    if not os.path.exists(EXPERIMENTS_FILE):
        df_row.to_csv(EXPERIMENTS_FILE, index=False)
    else:
        df_row.to_csv(EXPERIMENTS_FILE, mode='a', header=False, index=False)
    
    print(f"Experiment logged to {EXPERIMENTS_FILE}")

def run_experiment(start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR, features=DEFAULT_FEATURES, params=DEFAULT_PARAMS):
    print(f"\n--- Running Experiment: {start_year}-{end_year} ---")
    print(f"Features ({len(features)}): {features}")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # 2. Filter Time Period
    if 'SaleYear' in df.columns:
        df = df[(df['SaleYear'] >= start_year) & (df['SaleYear'] <= end_year)]
    elif 'TAXYR' in df.columns: # Fallback if SaleYear missing
         df = df[(df['TAXYR'] >= start_year) & (df['TAXYR'] <= end_year)]
         
    print(f"Data for {start_year}-{end_year}: {len(df)} records")
    
    if len(df) == 0:
        print("No data found for this period.")
        return

    # 3. Prepare X and y
    target_col = 'PRICE'
    # Ensure all features exist
    valid_features = [f for f in features if f in df.columns]
    
    # Important: NBHD is needed for encoding, even if not in 'features' list directly 
    # (if we want to use it for encoding derived feature). 
    # But usually 'NBHD' IS in features list to be encoded.
    if 'NBHD' not in valid_features and 'NBHD' in df.columns:
        valid_features.append('NBHD')
        
    X = df[valid_features].copy()
    y = df[target_col].copy()
    
    # 4. 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"  Fold {fold}/5...", end='\r')
        
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # --- Preprocessing INSIDE Fold (Avoid Leakage) ---
        
        # A. Target Encoding for NBHD
        if 'NBHD' in X_train.columns:
            # Calculate means on TRAIN
            nbhd_means = y_train.groupby(X_train['NBHD']).mean()
            global_mean = y_train.mean()
            
            # Map to TRAIN and VAL
            X_train['NBHD_Encoded'] = X_train['NBHD'].map(nbhd_means).fillna(global_mean)
            X_val['NBHD_Encoded'] = X_val['NBHD'].map(nbhd_means).fillna(global_mean)
            
            # Drop original NBHD (categorical) unless handled by model
            X_train = X_train.drop(columns=['NBHD'])
            X_val = X_val.drop(columns=['NBHD'])
            
        # --- Stage 1: Binning (Paper Strategy) ---
        # Create bins based on TRAIN y
        try:
            # 100 bins
            y_train_bins, bin_edges = pd.qcut(y_train, q=100, labels=False, retbins=True, duplicates='drop')
            
            # Train Stage 1 Classifier (Using Regressor on bin index as per paper implication or simplifiction)
            # Paper likely used Classifier or Regressor on bin ID. Let's use Regressor for speed/simplicity
            # to predict "Bin Index"
            stage1_model = xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            stage1_model.fit(X_train, y_train_bins)
            
            # Predict Bins
            bin_pred_train = stage1_model.predict(X_train)
            bin_pred_val = stage1_model.predict(X_val)
            
            # Add as Feature
            X_train['Predicted_PriceBin'] = bin_pred_train
            X_val['Predicted_PriceBin'] = bin_pred_val
            
        except Exception as e:
            print(f"Stage 1 failed: {e}. Skipping to Stage 2.")
            
        # --- Stage 2: Final Regressor ---
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_val)
        
        r2 = r2_score(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        
        fold += 1
        
    print(f"\nExperiment Complete.")
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    
    print(f"Results: R2 = {mean_r2:.4f} (+/- {std_r2:.4f}), RMSE = {mean_rmse:,.0f}")
    
    # 5. Log
    metrics = {
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'rmse_mean': mean_rmse
    }
    log_experiment(start_year, end_year, valid_features, params, metrics)

if __name__ == "__main__":
    # Example Run
    run_experiment()
