"""
preprocess.py
-------------
Loads raw retail data, handles missing values, type conversions,
feature extraction from dates, and exports clean data.
"""

import pandas as pd
import numpy as np
import os

def load_and_clean(filepath='data/retail_sales_data.csv'):
    """Load raw CSV and perform all cleaning steps."""
    print("=" * 50)
    print("STEP 1: Loading Data")
    print("=" * 50)
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ── 1. Parse dates ──────────────────────────────
    print("\nSTEP 2: Parsing Dates")
    df['date'] = pd.to_datetime(df['date'])
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    # ── 2. Check missing values ──────────────────────
    print("\nSTEP 3: Checking Missing Values")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ✓")

    # ── 3. Remove duplicates ─────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\nSTEP 4: Duplicates removed: {before - len(df)}")

    # ── 4. Ensure correct dtypes ─────────────────────
    df['units_sold']    = df['units_sold'].astype(int)
    df['opening_stock'] = df['opening_stock'].astype(int)
    df['closing_stock'] = df['closing_stock'].astype(int)
    df['promotion']     = df['promotion'].astype(int)
    df['stockout_flag'] = df['stockout_flag'].astype(int)
    print("STEP 5: Data types corrected ✓")

    # ── 5. Extract date features ─────────────────────
    print("\nSTEP 6: Extracting Date Features")
    df['year']        = df['date'].dt.year
    df['month']       = df['date'].dt.month
    df['quarter']     = df['date'].dt.quarter
    df['week']        = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek       # 0=Mon, 6=Sun
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['month_name']  = df['date'].dt.strftime('%B')
    print("  year, month, quarter, week, day_of_week, is_weekend, month_name added ✓")

    # ── 6. Add festival flag ─────────────────────────
    # Diwali season: Oct-Nov, Christmas: Dec
    df['festival_season'] = df['month'].isin([10, 11, 12]).astype(int)

    # ── 7. Summary stats ─────────────────────────────
    print("\nSTEP 7: Basic Summary")
    print(df[['units_sold', 'revenue', 'opening_stock', 'closing_stock']].describe().round(2))

    # ── 8. Save clean data ───────────────────────────
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/cleaned_data.csv', index=False)
    print("\nClean data saved → data/cleaned_data.csv")
    return df


if __name__ == '__main__':
    df = load_and_clean()
    print(f"\nFinal dataset shape: {df.shape}")
