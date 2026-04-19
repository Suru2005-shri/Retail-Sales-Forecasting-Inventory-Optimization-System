"""
eda.py
------
Exploratory Data Analysis:
- Overall sales trends
- Category-wise analysis
- Store performance
- Seasonality patterns
- Stockout analysis
All charts saved to outputs/graphs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

OUTPUT_DIR = 'outputs/graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})


def plot_monthly_sales_trend(df):
    """Monthly total revenue across the full period."""
    monthly = (
        df.groupby(['year', 'month'])['revenue']
        .sum()
        .reset_index()
    )
    monthly['period'] = pd.to_datetime(
        monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
    )
    monthly.sort_values('period', inplace=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly['period'], monthly['revenue'] / 1e6,
            color='#2563EB', linewidth=2.2, marker='o', markersize=4)
    ax.fill_between(monthly['period'], monthly['revenue'] / 1e6, alpha=0.12, color='#2563EB')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.set_title('Monthly Total Revenue Trend (2022–2024)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (₹ Millions)')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/01_monthly_revenue_trend.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_category_sales(df):
    """Bar chart of total revenue by category."""
    cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False) / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    bars = ax.bar(cat_rev.index, cat_rev.values, color=colors, edgecolor='white', linewidth=0.8)
    ax.bar_label(bars, labels=[f'₹{v:.1f}M' for v in cat_rev.values], padding=5, fontsize=10)
    ax.set_title('Total Revenue by Category (2022–2024)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Revenue (₹ Millions)')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/02_category_revenue.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_store_performance(df):
    """Horizontal bar chart of store-level revenue."""
    store_rev = df.groupby('store')['revenue'].sum().sort_values() / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(store_rev.index, store_rev.values, color='#10B981', edgecolor='white')
    ax.bar_label(ax.containers[0], labels=[f'₹{v:.1f}M' for v in store_rev.values], padding=4)
    ax.set_title('Store-wise Total Revenue (2022–2024)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Revenue (₹ Millions)')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/03_store_performance.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_seasonality_heatmap(df):
    """Heatmap of average units sold by month and category."""
    pivot = df.pivot_table(values='units_sold', index='category', columns='month', aggfunc='mean')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title('Avg Daily Units Sold — Seasonality Heatmap (Category × Month)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Category')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/04_seasonality_heatmap.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_stockout_analysis(df):
    """Stockout frequency by category."""
    stockout = df.groupby('category')['stockout_flag'].mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#EF4444' if v > 5 else '#F59E0B' for v in stockout.values]
    bars = ax.bar(stockout.index, stockout.values, color=colors, edgecolor='white')
    ax.bar_label(bars, labels=[f'{v:.1f}%' for v in stockout.values], padding=4)
    ax.axhline(5, color='red', linestyle='--', linewidth=1.2, label='5% Threshold')
    ax.set_title('Stockout Rate by Category (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Stockout Rate (%)')
    ax.legend()
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/05_stockout_analysis.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_promotion_impact(df):
    """Boxplot comparing units sold on promo vs non-promo days."""
    fig, ax = plt.subplots(figsize=(9, 5))
    promo_map = {0: 'No Promotion', 1: 'Promotion'}
    df_plot = df.copy()
    df_plot['Promotion'] = df_plot['promotion'].map(promo_map)

    sns.boxplot(data=df_plot, x='Promotion', y='units_sold',
                palette=['#94A3B8', '#2563EB'], ax=ax, width=0.5)
    ax.set_title('Sales Distribution: Promotion vs No Promotion', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Units Sold per Day')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/06_promotion_impact.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_top_products(df):
    """Top 10 products by total revenue."""
    top10 = (
        df.groupby('product')['revenue']
        .sum()
        .sort_values(ascending=True)
        .tail(10)
    ) / 1e6

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10.index, top10.values, color='#8B5CF6', edgecolor='white')
    ax.bar_label(ax.containers[0], labels=[f'₹{v:.1f}M' for v in top10.values], padding=4)
    ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
    ax.set_xlabel('Revenue (₹ Millions)')
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/07_top_products.png'
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def run_eda(filepath='data/cleaned_data.csv'):
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"Loaded {len(df):,} records\n")

    plot_monthly_sales_trend(df)
    plot_category_sales(df)
    plot_store_performance(df)
    plot_seasonality_heatmap(df)
    plot_stockout_analysis(df)
    plot_promotion_impact(df)
    plot_top_products(df)

    print("\n✅ All EDA charts saved to outputs/graphs/")
    return df


if __name__ == '__main__':
    run_eda()
