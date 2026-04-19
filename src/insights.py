"""
insights.py
-----------
Generates a final business insights summary PDF-style report
and a consolidated insights visualization dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/graphs', exist_ok=True)


def generate_insights_dashboard(
    cleaned_path='data/cleaned_data.csv',
    inventory_path='outputs/tables/inventory_report.csv',
    predictions_path='outputs/tables/predictions.csv',
):
    print("=" * 50)
    print("GENERATING INSIGHTS DASHBOARD")
    print("=" * 50)

    df_clean = pd.read_csv(cleaned_path, parse_dates=['date'])
    df_inv   = pd.read_csv(inventory_path)
    df_pred  = pd.read_csv(predictions_path, parse_dates=['date'])

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#F8FAFC')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── KPI Banner ────────────────────────────────────────────────
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.set_facecolor('#1E3A5F')
    kpi_ax.axis('off')

    total_revenue  = df_clean['revenue'].sum()
    total_units    = df_clean['units_sold'].sum()
    avg_stockout   = df_clean['stockout_flag'].mean() * 100
    reorder_count  = df_inv['needs_reorder'].sum()
    overstock_pct  = (df_inv['overstock_flag'].sum() / len(df_inv) * 100)

    kpis = [
        ('Total Revenue',   f'₹{total_revenue/1e7:.1f} Cr'),
        ('Units Sold',      f'{total_units:,.0f}'),
        ('Stockout Rate',   f'{avg_stockout:.1f}%'),
        ('Reorder Alerts',  str(reorder_count)),
        ('Overstock SKUs',  f'{overstock_pct:.1f}%'),
    ]

    x_positions = [0.08, 0.26, 0.44, 0.62, 0.80]
    for (label, value), x in zip(kpis, x_positions):
        kpi_ax.text(x, 0.65, value, transform=kpi_ax.transAxes,
                    ha='center', va='center', fontsize=22, fontweight='bold',
                    color='#F0F9FF')
        kpi_ax.text(x, 0.25, label, transform=kpi_ax.transAxes,
                    ha='center', va='center', fontsize=10,
                    color='#94C7ED')

    kpi_ax.set_title('Retail Sales Forecasting & Inventory Optimization — Business Dashboard',
                     fontsize=15, fontweight='bold', color='#1E3A5F', pad=12)

    # ── Chart 1: Monthly revenue trend ───────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.set_facecolor('#FFFFFF')
    monthly = df_clean.groupby(['year', 'month'])['revenue'].sum().reset_index()
    monthly['period'] = pd.to_datetime(
        monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
    )
    monthly.sort_values('period', inplace=True)
    ax1.plot(monthly['period'], monthly['revenue'] / 1e6,
             color='#2563EB', linewidth=2, marker='o', markersize=3)
    ax1.fill_between(monthly['period'], monthly['revenue'] / 1e6, alpha=0.1, color='#2563EB')
    ax1.set_title('Monthly Revenue Trend', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Revenue (₹ M)')
    ax1.tick_params(axis='x', rotation=30, labelsize=8)

    # ── Chart 2: Category share donut ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_facecolor('#FFFFFF')
    cat_rev = df_clean.groupby('category')['revenue'].sum()
    colors  = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    wedges, texts, pcts = ax2.pie(
        cat_rev.values, labels=cat_rev.index,
        autopct='%1.0f%%', colors=colors,
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.55)
    )
    for t in texts:
        t.set_fontsize(7.5)
    ax2.set_title('Revenue Share by Category', fontsize=11, fontweight='bold')

    # ── Chart 3: Actual vs Predicted ─────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.set_facecolor('#FFFFFF')
    sample = df_pred.sample(200, random_state=1).sort_values('date')
    ax3.plot(range(200), sample['actual_units'].values,
             label='Actual', color='#1E40AF', linewidth=1.5, alpha=0.9)
    ax3.plot(range(200), sample['predicted_units'].values,
             label='Predicted', color='#EF4444', linewidth=1.5, linestyle='--', alpha=0.85)
    ax3.set_title('Actual vs Predicted Units Sold', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Units Sold')
    ax3.legend(fontsize=8)

    # ── Chart 4: Risk distribution bar ───────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.set_facecolor('#FFFFFF')
    risk_counts = df_inv['risk_flag'].value_counts()
    risk_colors = {'Normal': '#10B981', '⚠️ Reorder Now': '#F59E0B',
                   '🔴 High Stockout Risk': '#EF4444', '📦 Overstock': '#3B82F6'}
    bar_colors = [risk_colors.get(r, '#94A3B8') for r in risk_counts.index]
    bars = ax4.bar(range(len(risk_counts)), risk_counts.values,
                   color=bar_colors, edgecolor='white')
    ax4.bar_label(bars, padding=3, fontsize=9)
    ax4.set_xticks(range(len(risk_counts)))
    ax4.set_xticklabels(
        [r.replace('⚠️ ', '').replace('🔴 ', '').replace('📦 ', '')
         for r in risk_counts.index],
        rotation=15, fontsize=8
    )
    ax4.set_title('SKU Risk Distribution', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of SKUs')

    plt.suptitle('', y=0.98)
    path = 'outputs/graphs/15_insights_dashboard.png'
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Insights dashboard saved → {path}")


def generate_text_report(
    cleaned_path='data/cleaned_data.csv',
    inventory_path='outputs/tables/inventory_report.csv',
):
    """Write a plain-text executive summary report."""
    df_clean = pd.read_csv(cleaned_path, parse_dates=['date'])
    df_inv   = pd.read_csv(inventory_path)

    lines = []
    lines.append("=" * 65)
    lines.append("  RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION")
    lines.append("  EXECUTIVE SUMMARY REPORT")
    lines.append(f"  Period: {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")
    lines.append("=" * 65)

    lines.append("\n📊 BUSINESS KPIs")
    lines.append(f"  Total Revenue     : ₹{df_clean['revenue'].sum()/1e7:.2f} Crore")
    lines.append(f"  Total Units Sold  : {df_clean['units_sold'].sum():,}")
    lines.append(f"  Avg Daily Revenue : ₹{df_clean['revenue'].mean():,.0f}")
    lines.append(f"  Stockout Rate     : {df_clean['stockout_flag'].mean()*100:.1f}%")
    lines.append(f"  Promo Days        : {df_clean['promotion'].mean()*100:.1f}% of all days")

    lines.append("\n🏪 TOP PERFORMING STORE")
    top_store = df_clean.groupby('store')['revenue'].sum().idxmax()
    top_rev   = df_clean.groupby('store')['revenue'].sum().max()
    lines.append(f"  {top_store}  →  ₹{top_rev/1e6:.1f}M")

    lines.append("\n📦 TOP SELLING CATEGORY")
    top_cat = df_clean.groupby('category')['units_sold'].sum().idxmax()
    top_units = df_clean.groupby('category')['units_sold'].sum().max()
    lines.append(f"  {top_cat}  →  {top_units:,} units")

    lines.append("\n🔔 INVENTORY ALERTS")
    reorder_skus = df_inv[df_inv['needs_reorder']].sort_values('reorder_qty', ascending=False)
    lines.append(f"  SKUs needing reorder  : {len(reorder_skus)}")
    lines.append(f"  Total reorder value   : ₹{df_inv['reorder_value'].sum():,.0f}")
    lines.append(f"  Overstock SKUs        : {df_inv['overstock_flag'].sum()}")

    lines.append("\n  Top 5 Reorder Items:")
    for _, row in reorder_skus.head(5).iterrows():
        lines.append(
            f"    - {row['product']} ({row['store']})  →  "
            f"Order {int(row['reorder_qty'])} units  (₹{row['reorder_value']:,.0f})"
        )

    lines.append("\n💡 BUSINESS RECOMMENDATIONS")
    lines.append("  1. Increase safety stock for Electronics in Oct–Dec (festival season).")
    lines.append("  2. Run targeted promotions for Grocery items during Jul–Aug.")
    lines.append("  3. Reduce overstock of slow-moving Home & Kitchen items.")
    lines.append("  4. Review supplier lead times to reduce stockout risk below 3%.")
    lines.append("  5. Implement automated reorder triggers at the computed ROP levels.")

    lines.append("\n" + "=" * 65)
    report_text = "\n".join(lines)

    path = 'outputs/reports/executive_summary.txt'
    with open(path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n✅ Executive summary saved → {path}")


if __name__ == '__main__':
    generate_insights_dashboard()
    generate_text_report()
