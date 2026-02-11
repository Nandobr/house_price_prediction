import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# Define paths
INPUT_FILE = 'engineered_features.csv'
# Target the new project folder we created
OUTPUT_DIR = '../volusia_property_app'
ARTIFACT_PATH = os.path.join(OUTPUT_DIR, 'model_artifacts.pkl')

# Feature config (Must match what the app expects to be able to generate)
FEATURES = [
    'SFLA', 'RMBED', 'YRBLT', 'NBHD', 'LUC', 'Month', 
    'HouseAge_Squared', 'Bed_Bath_Ratio',
    'NBHD_Median_Size', 'Size_vs_NBHD', 'SFLA_Squared'
]

def train_and_export():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # 0. Basic Filtering
    # Remove rows with missing critical features if any
    # Assuming engineered_features.csv is mostly clean, but let's be safe
    df = df.dropna(subset=['PRICE', 'SFLA', 'NBHD'])
    df = df[df['PRICE'] > 1000] # Remove placeholders or corrupt low values
    
    target_col = 'PRICE'
    
    # 1. NBHD Target Encoding for Price
    print("Creating Neighborhood Encodings...")
    # Map: NBHD -> Mean Price
    nbhd_price_map = df.groupby('NBHD')[target_col].mean().to_dict()
    global_mean_price = df[target_col].mean()
    
    # Apply encoding to data so we can train
    df['NBHD_Encoded'] = df['NBHD'].map(nbhd_price_map).fillna(global_mean_price)
    
    # 2. NBHD Median Size Map (for Feature Engineering in App)
    # We need to recreate logic: df['Size_vs_NBHD'] = df['SFLA'] - df['NBHD_Median_Size']
    # So we need to assist the app in getting 'NBHD_Median_Size' for a new input.
    if 'NBHD_Median_Size' in df.columns:
         # Since it's already calculated, we can just grab unique values
         nbhd_size_map = df.groupby('NBHD')['NBHD_Median_Size'].first().to_dict()
    else:
         # Calculate it if missing
         nbhd_size_map = df.groupby('NBHD')['SFLA'].median().to_dict()

    # NBHD Name Map (Code -> Name) for UI
    if 'NBHD_DESC' in df.columns:
        # Create a map, assuming one description per code
        # If multiple, take first or most frequent.
        nbhd_name_map = df.groupby('NBHD')['NBHD_DESC'].first().to_dict()
    else:
        print("Warning: NBHD_DESC not found, using codes as names.")
        nbhd_name_map = {nbhd: str(nbhd) for nbhd in df['NBHD'].unique()}

    # LUC Map (NBHD -> Most Frequent LUC)
    if 'LUC' in df.columns:
        # Get the most common LUC for each neighborhood
        nbhd_luc_map = df.groupby('NBHD')['LUC'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).to_dict()
        # Top 5 LUCs for the dropdown
        top_lucs = df['LUC'].value_counts().head(5).index.tolist()
    else:
        nbhd_luc_map = {}
        top_lucs = [99]

    global_median_size = df['SFLA'].median()

    # 3. Prepare Training Data
    # 'NBHD' is in FEATURES list, but effectively replaced by 'NBHD_Encoded' for the model
    # We remove 'NBHD' (and 'NBHD_DESC' if present) from the training columns list
    train_cols = [f for f in FEATURES if f != 'NBHD'] + ['NBHD_Encoded']
    
    # Validation check
    missing_cols = [c for c in train_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    X = df[train_cols]
    y = df[target_col]
    
    print(f"Training XGBoost Model on {len(X)} records...")
    # Using robust settings
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)
    
    # 4. Gather Metadata/Stats for UI
    # Min/Max for sliders
    stats = {
        'SFLA': {'min': int(df['SFLA'].min()), 'max': int(df['SFLA'].quantile(0.99))}, # 99% to avoid outliers in UI
        'YRBLT': {'min': int(df['YRBLT'].min()), 'max': int(df['YRBLT'].max())},
        'RMBED': {'min': int(df['RMBED'].min()), 'max': int(df['RMBED'].max())},
        'nbhds': sorted(list(df['NBHD'].unique())), # Keep as native type for mapping keys
        'data_count': len(df),
        'date_range': '2015-2019' # Simplified, could extract from data
    }

    # 5. Save Everything
    artifacts = {
        'model': model,
        'features': train_cols, # The exact columns the model expects (order matters)
        'nbhd_price_map': nbhd_price_map,
        'global_mean_price': global_mean_price,
        'nbhd_size_map': nbhd_size_map,
        'nbhd_name_map': nbhd_name_map,
        'nbhd_luc_map': nbhd_luc_map, # New
        'top_lucs': top_lucs, # New
        'ui_stats': stats
    }
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
        
    print(f"Saving artifacts to {ARTIFACT_PATH}...")
    with open(ARTIFACT_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print("Export Complete.")

if __name__ == "__main__":
    train_and_export()
