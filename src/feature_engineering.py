"""
feature_engineering.py
-----------------------
Builds model-ready features:
- Lag features (sales 7, 14, 30 days ago)
- Rolling averages (7-day, 30-day)
- Exponentially weighted mean
- Category and store encodings
- All date-based features
Saves feature-engineered dataset for model training.
"""

import pandas as pd
import numpy as np
import os


def engineer_features(filepath='data/cleaned_data.csv'):
    print("=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values(['store', 'product', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Group key for computing per-product-per-store time series
    group_cols = ['store', 'product']

    # ── 1. Lag Features ──────────────────────────────────────────
    print("Adding lag features...")
    for lag in [7, 14, 30]:
        df[f'lag_{lag}d'] = (
            df.groupby(group_cols)['units_sold']
            .shift(lag)
        )

    # ── 2. Rolling Mean Features ─────────────────────────────────
    print("Adding rolling average features...")
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}d'] = (
            df.groupby(group_cols)['units_sold']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # ── 3. Rolling Std (demand variability) ──────────────────────
    df['rolling_std_7d'] = (
        df.groupby(group_cols)['units_sold']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
    )

    # ── 4. Exponentially Weighted Mean ───────────────────────────
    print("Adding EWM feature...")
    df['ewm_7d'] = (
        df.groupby(group_cols)['units_sold']
        .transform(lambda x: x.shift(1).ewm(span=7, min_periods=1).mean())
    )

    # ── 5. Month-over-Month growth ────────────────────────────────
    df['month_revenue'] = df.groupby(
        group_cols + ['year', 'month'])['revenue'].transform('sum')

    # ── 6. Categorical Encoding ──────────────────────────────────
    print("Encoding categorical features...")
    df['store_code']    = df['store'].astype('category').cat.codes
    df['category_code'] = df['category'].astype('category').cat.codes
    df['product_code']  = df['product'].astype('category').cat.codes

    # ── 7. Drop rows with NaN lags (first N days per group) ──────
    df.dropna(subset=['lag_30d'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 8. Select final feature columns ──────────────────────────
    feature_cols = [
        'date', 'store', 'category', 'product',
        'store_code', 'category_code', 'product_code',
        'year', 'month', 'quarter', 'week',
        'day_of_week', 'is_weekend', 'festival_season', 'promotion',
        'lag_7d', 'lag_14d', 'lag_30d',
        'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d',
        'rolling_std_7d', 'ewm_7d',
        'opening_stock', 'units_sold',   # target
        'revenue', 'unit_price',
    ]
    df_feat = df[feature_cols].copy()

    os.makedirs('data', exist_ok=True)
    df_feat.to_csv('data/features.csv', index=False)
    print(f"\n✅ Feature dataset saved → data/features.csv")
    print(f"   Shape: {df_feat.shape}")
    print(f"   Features: {[c for c in df_feat.columns if c not in ['date','store','category','product','units_sold']]}")
    return df_feat


if __name__ == '__main__':
    df = engineer_features()
