import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('engineered_features.csv', low_memory=False)

print("--- Data Analysis ---")
print(f"Total Rows: {len(df)}")

# 1. SFLA Stats
print("\n--- SFLA Stats ---")
print(df['SFLA'].describe())

# 2. SFLA vs Price Correlation
corr = df['SFLA'].corr(df['PRICE'])
print(f"\nSFLA vs PRICE Correlation: {corr:.4f}")

# 3. Small vs Medium Homes
small_homes = df[(df['SFLA'] >= 400) & (df['SFLA'] <= 600)]
medium_homes = df[(df['SFLA'] >= 1400) & (df['SFLA'] <= 1600)]

print(f"\nSmall Homes (400-600 sqft): {len(small_homes)}")
print(f"Mean Price: ${small_homes['PRICE'].mean():,.0f}")
print(f"Median Price: ${small_homes['PRICE'].median():,.0f}")
print(f"Mean Efficiency Ratio: {small_homes['Efficiency_Ratio'].mean():.4f}")

print(f"\nMedium Homes (1400-1600 sqft): {len(medium_homes)}")
print(f"Mean Price: ${medium_homes['PRICE'].mean():,.0f}")
print(f"Median Price: ${medium_homes['PRICE'].median():,.0f}")
print(f"Mean Efficiency Ratio: {medium_homes['Efficiency_Ratio'].mean():.4f}")

# 4. Efficiency Ratio impact
print(f"\nOverall Efficiency Ratio Mean: {df['Efficiency_Ratio'].mean():.4f}")
print(f"Overall Efficiency Ratio Median: {df['Efficiency_Ratio'].median():.4f}")
