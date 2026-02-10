import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

files_to_check = ['VCPA_CAMA_RES_BLDG.csv', 'VCPA_CAMA_PARCEL.csv']

for fname in files_to_check:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, nrows=5)
            print(f"--- {fname} ---")
            print(df.columns.tolist())
            print(df.head(2))
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    else:
        print(f"File not found: {fname}")
