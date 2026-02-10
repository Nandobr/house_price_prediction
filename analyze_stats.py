from data_loader import load_and_merge_data
from model import HomePriceModel
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def analyze():
    print("--- Loading Data ---")
    df_raw = load_and_merge_data()
    if df_raw is None:
        return

    print(f"\nRaw Data Shape: {df_raw.shape}")
    
    print("\n--- Preprocessing ---")
    model = HomePriceModel()
    df_processed = model.preprocess(df_raw, is_training=True)
    
    if df_processed is None:
        print("Preprocessing returned None")
        return

    print("\n" + "="*50)
    print("DATASET STATISTICS (Before Train/Test Split)")
    print("="*50)
    
    print(f"\nFinal Dataset Shape: {df_processed.shape}")
    
    print("\n--- Numerical Feature Statistics ---")
    # Filter for numeric columns
    numeric_stats = df_processed.describe()
    print(numeric_stats)
    
    print("\n--- Target Variable (PRICE) Distribution ---")
    if 'PRICE' in df_processed.columns:
        print(df_processed['PRICE'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    
    print("\n--- Categorical Features (One-Hot Encoded Count) ---")
    # Just counting how many columns per original category prefix roughly
    # This is an estimation since we already one-hot encoded
    print(f"Total Columns: {len(df_processed.columns)}")
    print("Top 10 Columns by Standard Deviation (Likely most influential numeric or high variance features):")
    print(df_processed.std().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    analyze()
