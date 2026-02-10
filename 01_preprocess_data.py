import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
LEAKAGE_COLUMNS = [
    'APRTOT', 'APRLAND', 'APRBLDG', 
    'STXBL', 'NSTXBL', 'COTXBL', 'CITXBL',
    'SASD', 'NSASD', 'MSASD', 'MSTXBL', 'OITXBL'
]
OUTPUT_FILE = 'processed_data.csv'
STATS_FILE = 'data_stats.md'

# Data Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_and_merge_data(data_dir=DATA_DIR):
    """
    Loads Sales, Residential Building, and Parcel data, and merges them.
    Returns a consolidated DataFrame.
    """
    print("Loading and merging data sets...")
    
    # 1. Load Sales Data
    sales_path = os.path.join(data_dir, 'VCPA_CAMA_SALES.csv')
    if not os.path.exists(sales_path):
        print(f"Sales file not found: {sales_path}")
        return None
    
    print(f"Loading Sales from {sales_path}...")
    df_sales = pd.read_csv(sales_path, low_memory=False)
    
    # Identify Price column
    price_col = next((c for c in df_sales.columns if 'PRICE' in c.upper()), None)
    if not price_col:
        print("Could not find Price column in Sales.")
        return None
    
    df_sales = df_sales.dropna(subset=['PARID', price_col])
    # Filter out zero/low prices
    df_sales = df_sales[df_sales[price_col] > 1000] 
            
    print(f"Sales records loaded: {len(df_sales)}")

    # 2. Load Residential Building Data (Characteristics)
    bldg_path = os.path.join(data_dir, 'VCPA_CAMA_RES_BLDG.csv')
    if os.path.exists(bldg_path):
        print(f"Loading Building Data from {bldg_path}...")
        use_cols = ['PARID', 'YRBLT', 'RMBED', 'FIXBATH', 'SFLA', 'TOTAL_AREA', 'STORIES', 'EXTWALL_DESC', 'ROOF_COVER_DESC']
        try:
            df_bldg = pd.read_csv(bldg_path, usecols=lambda c: c in use_cols or c == 'PARID', low_memory=False)
            
            # Deduplicate Building data: One parcel might have multiple buildings. 
            if 'SFLA' in df_bldg.columns:
                df_bldg = df_bldg.sort_values('SFLA', ascending=False).drop_duplicates('PARID')
            else:
                df_bldg = df_bldg.drop_duplicates('PARID')
                
            print(f"Building records loaded: {len(df_bldg)}")
            
            # Merge
            df_sales = pd.merge(df_sales, df_bldg, on='PARID', how='inner')
            print(f"Merged Sales + Bldg: {len(df_sales)}")
            
        except ValueError as e:
            print(f"Error loading Building data columns: {e}")
    else:
        print("Warning: VCPA_CAMA_RES_BLDG.csv not found. Skipping building features.")

    # 3. Load Parcel Data (Location/Nbhd + Tax Values)
    # We need to load them to merge, even if we drop leakage later.
    parcel_path = os.path.join(data_dir, 'VCPA_CAMA_PARCEL.csv')
    if os.path.exists(parcel_path):
        print(f"Loading Parcel Data from {parcel_path}...")
        use_cols = ['PARID', 'NBHD', 'LUC', 'LUC_DESC', 
                    'APRLAND', 'APRBLDG', 'APRTOT', 
                    'SASD', 'NSASD', 'STXBL', 'NSTXBL', 'COTXBL', 'CITXBL']
        try:
            df_parcel = pd.read_csv(parcel_path, usecols=lambda c: c in use_cols or c == 'PARID', low_memory=False)
            df_parcel = df_parcel.drop_duplicates('PARID')
            
            print(f"Parcel records loaded: {len(df_parcel)}")
            
            # Merge
            df_sales = pd.merge(df_sales, df_parcel, on='PARID', how='inner')
            print(f"Merged Sales + Parcel: {len(df_sales)}")
            
        except ValueError as e:
            print(f"Error loading Parcel data columns: {e}")
    else:
        print("Warning: VCPA_CAMA_PARCEL.csv not found. Skipping parcel features.")

    return df_sales

def log_stats(text, mode='a'):
    """Appends stats to the markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATS_FILE, mode) as f:
        f.write(f"\n### Preprocessing Run: {timestamp}\n")
        f.write(text + "\n")

def get_basic_stats(df, label="Dataset"):
    stats = f"**{label}**\n"
    stats += f"- Shape: {df.shape}\n"
    if 'PRICE' in df.columns:
        stats += f"- Price Mean: ${df['PRICE'].mean():,.2f}\n"
        stats += f"- Price Median: ${df['PRICE'].median():,.2f}\n"
    return stats

def main():
    print("--- 01_PREPROCESS_DATA ---")
    
    # 1. Load Data
    df = load_and_merge_data()
    
    if df is None or df.empty:
        print("Error: No data loaded.")
        return

    # Initialize stats file if not exists
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'w') as f:
            f.write("# Data Statistics Log\n")

    raw_stats = get_basic_stats(df, "Raw Merged Data")

    # 2. Drop Leakage Columns
    print("Dropping Leakage Columns (2026 Tax Values)...")
    cols_to_drop = []
    for col in df.columns:
        upper_col = col.upper()
        # Check if any leakage keyword is in the column name
        if any(leak in upper_col for leak in LEAKAGE_COLUMNS):
            cols_to_drop.append(col)
            
    print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # 3. Basic Cleaning
    # Remove Price < 1000 (Already done in loader usually, but safety check)
    if 'PRICE' in df.columns:
         df = df[df['PRICE'] > 1000]

    # Fill NaNs for numericals (simple strategy for now)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 4. Save
    print(f"Saving processed data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # 5. Log Stats
    processed_stats = get_basic_stats(df, "Processed Data (Leakage Removed)")
    log_stats(raw_stats + "\n" + processed_stats)
    
    print("Done.")
    print(raw_stats)
    print(processed_stats)

if __name__ == "__main__":
    main()
