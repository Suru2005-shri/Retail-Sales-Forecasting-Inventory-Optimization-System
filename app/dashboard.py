"""
app/dashboard.py
----------------
Interactive Streamlit dashboard for the Retail Sales Forecasting
& Inventory Optimization System.

Run with:
  streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #2563EB;
    }
    .stMetric { background: white; border-radius: 8px; padding: 12px; }
    h1 { color: #1E3A5F; }
    h2 { color: #1E40AF; }
    .alert-red { color: #EF4444; font-weight: bold; }
    .alert-yellow { color: #F59E0B; font-weight: bold; }
    .alert-green { color: #10B981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_cleaned():
    return pd.read_csv('data/cleaned_data.csv', parse_dates=['date'])

@st.cache_data
def load_inventory():
    return pd.read_csv('outputs/tables/inventory_report.csv')

@st.cache_data
def load_predictions():
    return pd.read_csv('outputs/tables/predictions.csv', parse_dates=['date'])


def check_data_exists():
    required = [
        'data/cleaned_data.csv',
        'outputs/tables/inventory_report.csv',
        'outputs/tables/predictions.csv',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    return missing


# ── App ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🛒 Retail Sales Forecasting & Inventory Optimization")
    st.caption("AI-powered demand forecasting and inventory intelligence system")

    missing = check_data_exists()
    if missing:
        st.error("⚠️ Data files not found. Please run `python main.py` first.")
        st.code("python main.py", language="bash")
        st.stop()

    df       = load_cleaned()
    df_inv   = load_inventory()
    df_pred  = load_predictions()

    # ── Sidebar filters ────────────────────────────────────────────
    st.sidebar.header("🔧 Filters")
    stores = ['All'] + sorted(df['store'].unique().tolist())
    cats   = ['All'] + sorted(df['category'].unique().tolist())
    years  = ['All'] + sorted(df['year'].unique().tolist())

    sel_store = st.sidebar.selectbox("Store",    stores)
    sel_cat   = st.sidebar.selectbox("Category", cats)
    sel_year  = st.sidebar.selectbox("Year",     years)

    df_f = df.copy()
    if sel_store != 'All': df_f = df_f[df_f['store']    == sel_store]
    if sel_cat   != 'All': df_f = df_f[df_f['category'] == sel_cat]
    if sel_year  != 'All': df_f = df_f[df_f['year']     == int(sel_year)]

    # ── KPI row ────────────────────────────────────────────────────
    st.markdown("### 📊 Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Revenue",    f"₹{df_f['revenue'].sum()/1e6:.1f}M")
    k2.metric("Units Sold",       f"{df_f['units_sold'].sum():,}")
    k3.metric("Avg Daily Sales",  f"{df_f['units_sold'].mean():.0f} units")
    k4.metric("Stockout Rate",    f"{df_f['stockout_flag'].mean()*100:.1f}%")
    k5.metric("Promo Impact",     f"+{(df_f[df_f['promotion']==1]['units_sold'].mean() / df_f[df_f['promotion']==0]['units_sold'].mean() - 1)*100:.0f}%")

    st.markdown("---")

    # ── Row 1: Monthly trend + Category donut ─────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📈 Monthly Revenue Trend")
        monthly = df_f.groupby(['year', 'month'])['revenue'].sum().reset_index()
        monthly['period'] = pd.to_datetime(
            monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
        )
        monthly.sort_values('period', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(monthly['period'], monthly['revenue']/1e6,
                color='#2563EB', linewidth=2, marker='o', markersize=4)
        ax.fill_between(monthly['period'], monthly['revenue']/1e6, alpha=0.1, color='#2563EB')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=30, fontsize=8)
        ax.set_ylabel("Revenue (₹M)")
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### 🥧 Category Share")
        cat_rev = df_f.groupby('category')['revenue'].sum()
        colors  = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        ax2.pie(cat_rev.values, labels=cat_rev.index, autopct='%1.0f%%',
                colors=colors[:len(cat_rev)], startangle=90,
                wedgeprops=dict(width=0.55), pctdistance=0.78)
        ax2.set_facecolor('#FAFAFA')
        fig2.patch.set_alpha(0)
        st.pyplot(fig2)
        plt.close()

    # ── Row 2: Forecast + Seasonality ─────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🔮 Actual vs Predicted Units Sold")
        sample = df_pred.sample(min(250, len(df_pred)), random_state=1).sort_values('date')
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        ax3.plot(sample['actual_units'].values,    label='Actual',    color='#1E40AF', linewidth=1.5)
        ax3.plot(sample['predicted_units'].values, label='Predicted', color='#EF4444',
                 linewidth=1.5, linestyle='--', alpha=0.85)
        ax3.legend(fontsize=8)
        ax3.set_xlabel("Sample")
        ax3.set_ylabel("Units Sold")
        ax3.set_facecolor('#FAFAFA')
        fig3.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col4:
        st.markdown("#### 🌡️ Seasonality Heatmap")
        pivot = df_f.pivot_table(values='units_sold', index='category', columns='month', aggfunc='mean')
        pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec']
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                    linewidths=0.4, linecolor='white', ax=ax4, cbar=False)
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        fig4.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # ── Inventory Alerts Table ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔔 Inventory Alerts & Reorder Recommendations")

    inv_filter = df_inv.copy()
    if sel_store != 'All': inv_filter = inv_filter[inv_filter['store'] == sel_store]
    if sel_cat   != 'All': inv_filter = inv_filter[inv_filter['category'] == sel_cat]

    reorder_df = inv_filter[inv_filter['needs_reorder']].sort_values('reorder_qty', ascending=False)

    a1, a2, a3 = st.columns(3)
    a1.metric("SKUs Needing Reorder",  len(reorder_df))
    a2.metric("Total Reorder Value",   f"₹{reorder_df['reorder_value'].sum():,.0f}")
    a3.metric("Overstock SKUs",        inv_filter['overstock_flag'].sum())

    display_cols = ['store', 'category', 'product', 'avg_daily_demand',
                    'safety_stock', 'reorder_point', 'eoq',
                    'reorder_qty', 'reorder_value', 'stockout_rate_pct', 'risk_flag']
    st.dataframe(
        reorder_df[display_cols].rename(columns={
            'avg_daily_demand':   'Avg Daily Demand',
            'safety_stock':       'Safety Stock',
            'reorder_point':      'Reorder Point',
            'eoq':                'EOQ',
            'reorder_qty':        'Order Qty',
            'reorder_value':      'Order Value (₹)',
            'stockout_rate_pct':  'Stockout %',
            'risk_flag':          'Risk',
        }),
        use_container_width=True,
        height=380
    )

    # ── Full inventory table (expandable) ─────────────────────────
    with st.expander("📋 View Full Inventory Report"):
        st.dataframe(inv_filter[display_cols], use_container_width=True)

    st.markdown("---")
    st.caption("Built by a Data Science student as portfolio proof · GitHub: [your-username]/Retail-Sales-Forecasting")


if __name__ == '__main__':
    main()
