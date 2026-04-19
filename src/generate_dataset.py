

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
np.random.seed(42)

STORES      = ['Store_Mumbai', 'Store_Delhi', 'Store_Bangalore', 'Store_Chennai', 'Store_Pune']
CATEGORIES  = ['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Sports']
PRODUCTS = {
    'Electronics':    ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch'],
    'Clothing':       ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sneakers'],
    'Groceries':      ['Rice', 'Wheat Flour', 'Cooking Oil', 'Sugar', 'Milk'],
    'Home & Kitchen': ['Mixer', 'Pressure Cooker', 'Non-stick Pan', 'Water Bottle', 'Air Fryer'],
    'Sports':         ['Cricket Bat', 'Football', 'Yoga Mat', 'Dumbbell', 'Running Shoes'],
}

BASE_PRICES = {
    'Smartphone': 15000, 'Laptop': 55000, 'Headphones': 3000, 'Tablet': 20000, 'Smartwatch': 8000,
    'T-Shirt': 500,  'Jeans': 1200, 'Dress': 1500, 'Jacket': 2500, 'Sneakers': 3500,
    'Rice': 60,   'Wheat Flour': 40, 'Cooking Oil': 130, 'Sugar': 45, 'Milk': 55,
    'Mixer': 3500, 'Pressure Cooker': 2500, 'Non-stick Pan': 1200, 'Water Bottle': 400, 'Air Fryer': 4500,
    'Cricket Bat': 1500, 'Football': 800, 'Yoga Mat': 700, 'Dumbbell': 2000, 'Running Shoes': 4000,
}

START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)


def add_seasonality(date, category):
    """Return a seasonal multiplier based on month and category."""
    month = date.month
    # Festival boost: Oct-Dec
    festival_boost = 1.0
    if month in [10, 11, 12]:
        festival_boost = 1.4 if category in ['Electronics', 'Clothing'] else 1.2
    # Summer boost for certain categories
    summer_boost = 1.0
    if month in [5, 6] and category == 'Sports':
        summer_boost = 1.3
    # Rainy season for groceries
    rainy_boost = 1.0
    if month in [7, 8] and category == 'Groceries':
        rainy_boost = 1.15
    return festival_boost * summer_boost * rainy_boost


def add_trend(date, category):
    """Slight upward trend over years for Electronics and Sports."""
    year_offset = (date.year - 2022)
    if category == 'Electronics':
        return 1 + 0.08 * year_offset
    elif category == 'Sports':
        return 1 + 0.05 * year_offset
    return 1.0


def generate_dataset():
    """Generate synthetic retail sales data."""
    records = []
    date_range = pd.date_range(START_DATE, END_DATE, freq='D')

    print("Generating synthetic retail dataset...")
    for store in STORES:
        for category, products in PRODUCTS.items():
            for product in products:
                base_price  = BASE_PRICES[product]
                base_demand = np.random.randint(5, 50)   # avg daily units sold

                for date in date_range:
                    seasonal   = add_seasonality(date, category)
                    trend_mult = add_trend(date, category)
                    noise      = np.random.normal(1.0, 0.15)      # ±15% random noise
                    promotion  = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% promo days

                    promo_mult = 1.35 if promotion else 1.0

                    # Final units sold
                    units_sold = max(0, int(
                        base_demand * seasonal * trend_mult * noise * promo_mult
                    ))

                    # Opening stock (simulate inventory level)
                    opening_stock  = np.random.randint(50, 300)
                    closing_stock  = max(0, opening_stock - units_sold)
                    stockout_flag  = 1 if closing_stock == 0 else 0

                    # Revenue
                    price     = base_price * np.random.uniform(0.95, 1.05)
                    revenue   = round(units_sold * price, 2)

                    records.append({
                        'date':          date.strftime('%Y-%m-%d'),
                        'store':         store,
                        'category':      category,
                        'product':       product,
                        'units_sold':    units_sold,
                        'unit_price':    round(price, 2),
                        'revenue':       revenue,
                        'opening_stock': opening_stock,
                        'closing_stock': closing_stock,
                        'stockout_flag': stockout_flag,
                        'promotion':     promotion,
                    })

    df = pd.DataFrame(records)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/retail_sales_data.csv', index=False)
    print(f"Dataset saved → data/retail_sales_data.csv")
    print(f"Total records: {len(df):,}")
    print(df.head())
    return df


if __name__ == '__main__':
    df = generate_dataset()
