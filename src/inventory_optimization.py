"""
inventory_optimization.py
--------------------------
Applies inventory management formulas to generate actionable decisions:
  - Reorder Point (ROP)
  - Safety Stock
  - Economic Order Quantity (EOQ)
  - Reorder recommendations (Yes/No + quantity)
  - Overstock and stockout alerts
  - Saves inventory report CSV and charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/graphs', exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LEAD_TIME_DAYS   = 3     # days between placing and receiving an order
SERVICE_LEVEL_Z  = 1.65  # Z-score for 95% service level
HOLDING_COST_PCT = 0.20  # 20% of unit cost per year
ORDERING_COST    = 500   # ₹500 per order placed


def compute_reorder_point(avg_daily_demand, lead_time, safety_stock):
    """ROP = (avg demand × lead time) + safety stock"""
    return round(avg_daily_demand * lead_time + safety_stock)


def compute_safety_stock(demand_std_dev, lead_time, z=SERVICE_LEVEL_Z):
    """Safety Stock = Z × σ_demand × √lead_time"""
    return round(z * demand_std_dev * np.sqrt(lead_time))


def compute_eoq(avg_daily_demand, unit_cost, ordering_cost=ORDERING_COST,
                holding_cost_pct=HOLDING_COST_PCT):
    """
    EOQ = √(2 × D × S / H)
    D = annual demand
    S = ordering cost per order
    H = holding cost per unit per year
    """
    annual_demand = avg_daily_demand * 365
    holding_cost  = unit_cost * holding_cost_pct
    if holding_cost == 0 or annual_demand == 0:
        return 0
    return round(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost))


def generate_inventory_report(
    features_path='data/features.csv',
    cleaned_path='data/cleaned_data.csv',
    predictions_path='outputs/tables/predictions.csv'
):
    print("=" * 50)
    print("INVENTORY OPTIMIZATION")
    print("=" * 50)

    df_feat = pd.read_csv(features_path, parse_dates=['date'])
    df_clean = pd.read_csv(cleaned_path, parse_dates=['date'])
    pred    = pd.read_csv(predictions_path, parse_dates=['date'])

    # Merge closing_stock and stockout_flag from cleaned data into features
    merge_cols = ['date', 'store', 'product', 'closing_stock', 'stockout_flag']
    df = df_feat.merge(df_clean[merge_cols], on=['date', 'store', 'product'], how='left')

    # ── Aggregate per product per store ──────────────────────────
    summary = (
        df.groupby(['store', 'category', 'product'])
        .agg(
            avg_daily_demand   = ('units_sold', 'mean'),
            demand_std_dev     = ('units_sold', 'std'),
            avg_unit_price     = ('unit_price', 'mean'),
            avg_opening_stock  = ('opening_stock', 'mean'),
            avg_closing_stock  = ('closing_stock', 'mean'),
            total_units_sold   = ('units_sold', 'sum'),
            stockout_days      = ('stockout_flag', 'sum'),
            total_days         = ('units_sold', 'count'),
        )
        .reset_index()
    )

    # Fill NaN std (single-record products)
    summary['demand_std_dev'].fillna(0, inplace=True)

    # ── Apply inventory formulas ──────────────────────────────────
    summary['lead_time_days'] = LEAD_TIME_DAYS

    summary['safety_stock'] = summary.apply(
        lambda r: compute_safety_stock(r['demand_std_dev'], LEAD_TIME_DAYS), axis=1
    )

    summary['reorder_point'] = summary.apply(
        lambda r: compute_reorder_point(r['avg_daily_demand'], LEAD_TIME_DAYS, r['safety_stock']),
        axis=1
    )

    summary['eoq'] = summary.apply(
        lambda r: compute_eoq(r['avg_daily_demand'], r['avg_unit_price']), axis=1
    )

    # ── Reorder recommendation ────────────────────────────────────
    summary['needs_reorder'] = summary['avg_closing_stock'] <= summary['reorder_point']
    summary['reorder_qty']   = np.where(
        summary['needs_reorder'],
        summary['eoq'],
        0
    )
    summary['reorder_value'] = (summary['reorder_qty'] * summary['avg_unit_price']).round(2)

    # ── Stockout rate ─────────────────────────────────────────────
    summary['stockout_rate_pct'] = (
        summary['stockout_days'] / summary['total_days'] * 100
    ).round(2)

    # ── Risk flag ─────────────────────────────────────────────────
    summary['risk_flag'] = 'Normal'
    summary.loc[summary['needs_reorder'], 'risk_flag'] = '⚠️ Reorder Now'
    summary.loc[summary['stockout_rate_pct'] > 10, 'risk_flag'] = '🔴 High Stockout Risk'
    summary.loc[
        (summary['avg_closing_stock'] > summary['avg_daily_demand'] * 60),
        'risk_flag'
    ] = 'Overstock'

    # ── Overstock flag ────────────────────────────────────────────
    summary['overstock_flag'] = summary['avg_closing_stock'] > (summary['avg_daily_demand'] * 60)

    # ── Round and tidy ────────────────────────────────────────────
    for col in ['avg_daily_demand', 'demand_std_dev', 'avg_unit_price',
                'avg_opening_stock', 'avg_closing_stock']:
        summary[col] = summary[col].round(2)

    # ── Save report ───────────────────────────────────────────────
    summary.to_csv('outputs/tables/inventory_report.csv', index=False)
    print(f"\nInventory report saved → outputs/tables/inventory_report.csv")
    print(f"   Total SKUs (product×store): {len(summary)}")

    reorder_count   = summary['needs_reorder'].sum()
    overstock_count = summary['overstock_flag'].sum()
    print(f"   SKUs needing reorder : {reorder_count}")
    print(f"   Overstock SKUs       : {overstock_count}")
    print(f"   Total reorder value  : ₹{summary['reorder_value'].sum():,.0f}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_reorder_alerts(summary)
    plot_risk_distribution(summary)
    plot_eoq_by_category(summary)
    plot_safety_stock_analysis(summary)

    return summary


def plot_reorder_alerts(df):
    """Top products needing immediate reorder."""
    reorder_df = df[df['needs_reorder']].sort_values('reorder_qty', ascending=False).head(15)

    if reorder_df.empty:
        print("No reorder alerts to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#EF4444' if r > 10 else '#F59E0B'
              for r in reorder_df['stockout_rate_pct']]
    bars = ax.barh(
        reorder_df['product'] + ' | ' + reorder_df['store'].str.replace('Store_', ''),
        reorder_df['reorder_qty'],
        color=colors, edgecolor='white'
    )
    ax.bar_label(bars, labels=[f'{int(v)} units' for v in reorder_df['reorder_qty']], padding=4)
    ax.set_title('Top Reorder Alerts — Recommended Order Quantities', fontsize=13, fontweight='bold')
    ax.set_xlabel('Recommended Order Quantity (EOQ)')
    red_patch    = mpatches.Patch(color='#EF4444', label='High Stockout Risk (>10%)')
    yellow_patch = mpatches.Patch(color='#F59E0B', label='Normal Reorder')
    ax.legend(handles=[red_patch, yellow_patch])
    plt.tight_layout()
    path = 'outputs/graphs/11_reorder_alerts.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_risk_distribution(df):
    """Pie chart of SKU risk distribution."""
    risk_counts = df['risk_flag'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#10B981', '#F59E0B', '#EF4444', '#3B82F6']
    wedges, texts, autotexts = ax.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(risk_counts)],
        startangle=140,
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title('SKU Risk Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = 'outputs/graphs/12_risk_distribution.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_eoq_by_category(df):
    """Average EOQ by category."""
    eoq_cat = df.groupby('category')['eoq'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(eoq_cat.index, eoq_cat.values, color='#2563EB', edgecolor='white')
    ax.bar_label(bars, labels=[f'{v:.0f}' for v in eoq_cat.values], padding=4)
    ax.set_title('Average EOQ by Category', fontsize=13, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Economic Order Quantity (units)')
    plt.tight_layout()
    path = 'outputs/graphs/13_eoq_by_category.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_safety_stock_analysis(df):
    """Safety stock vs average daily demand scatter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = df['category'].unique()
    colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

    for cat, color in zip(categories, colors):
        mask = df['category'] == cat
        ax.scatter(
            df[mask]['avg_daily_demand'],
            df[mask]['safety_stock'],
            label=cat, color=color, alpha=0.6, s=40
        )

    ax.set_title('Safety Stock vs Avg Daily Demand', fontsize=13, fontweight='bold')
    ax.set_xlabel('Avg Daily Demand (units/day)')
    ax.set_ylabel('Safety Stock (units)')
    ax.legend()
    plt.tight_layout()
    path = 'outputs/graphs/14_safety_stock_analysis.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    report = generate_inventory_report()
    print("\nTop 10 Reorder Alerts:")
    cols = ['store', 'product', 'avg_daily_demand', 'reorder_point',
            'safety_stock', 'eoq', 'reorder_qty', 'risk_flag']
    print(
        report[report['needs_reorder']][cols]
        .sort_values('reorder_qty', ascending=False)
        .head(10)
        .to_string(index=False)
    )
