from data_loader import load_and_merge_data
import os

def create_sample():
    start_date = "2015-01-01"
    end_date = "2019-11-13"
    
    print(f"--- Creating Sample Dataset ({start_date} to {end_date}) ---")
    
    # Load with filter
    df = load_and_merge_data(start_date=start_date, end_date=end_date)
    
    if df is not None and not df.empty:
        # Save to CSV
        output_file = f"volusia_sales_{start_date}_{end_date}.csv"
        print(f"Saving {len(df)} records to {output_file}...")
        df.to_csv(output_file, index=False)
        
        # Show Statistics
        print("\n" + "="*50)
        print("FILTERED DATASET STATISTICS")
        print("="*50)
        
        print(f"Time Period: {start_date} to {end_date}")
        print(f"Total Records: {len(df)}")
        
        if 'PRICE' in df.columns:
            print("\nPrice Statistics:")
            print(df['PRICE'].describe().apply(lambda x: format(x, 'f')))
            
        print("\nOther Numeric Statistics:")
        # Select key numeric columns
        cols = ['SFLA', 'YRBLT', 'RMBED', 'FIXBATH'] 
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            print(df[existing_cols].describe())
            
    else:
        print("No data found for the specified date range.")

if __name__ == "__main__":
    create_sample()
