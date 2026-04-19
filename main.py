"""
main.py
-------
Master pipeline runner for the Retail Sales Forecasting &
Inventory Optimization System.

Run this single file to execute the COMPLETE pipeline:
  Step 1 → Generate synthetic dataset
  Step 2 → Preprocess & clean data
  Step 3 → Exploratory Data Analysis
  Step 4 → Feature engineering
  Step 5 → Train forecasting models
  Step 6 → Inventory optimization
  Step 7 → Business insights & dashboard

Usage:
  python main.py
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_dataset      import generate_dataset
from preprocess            import load_and_clean
from eda                   import run_eda
from feature_engineering   import engineer_features
from forecasting_model     import train_models
from inventory_optimization import generate_inventory_report
from insights              import generate_insights_dashboard, generate_text_report


def section(title):
    print("\n")
    print("╔" + "═" * 56 + "╗")
    print(f"║  {title:<54}║")
    print("╚" + "═" * 56 + "╝")


def main():
    start = time.time()
    print("\n🚀 RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION")
    print("   Complete Pipeline Starting...\n")

    # ── Step 1: Generate Dataset ──────────────────────────────────
    section("STEP 1: Generating Synthetic Retail Dataset")
    generate_dataset()

    # ── Step 2: Preprocess ────────────────────────────────────────
    section("STEP 2: Data Preprocessing & Cleaning")
    load_and_clean('data/retail_sales_data.csv')

    # ── Step 3: EDA ───────────────────────────────────────────────
    section("STEP 3: Exploratory Data Analysis")
    run_eda('data/cleaned_data.csv')

    # ── Step 4: Feature Engineering ──────────────────────────────
    section("STEP 4: Feature Engineering")
    engineer_features('data/cleaned_data.csv')

    # ── Step 5: Forecasting Model ─────────────────────────────────
    section("STEP 5: Sales Forecasting Model Training & Evaluation")
    train_models('data/features.csv')

    # ── Step 6: Inventory Optimization ───────────────────────────
    section("STEP 6: Inventory Optimization & Reorder Planning")
    generate_inventory_report()

    # ── Step 7: Insights ──────────────────────────────────────────
    section("STEP 7: Business Insights Dashboard & Report")
    generate_insights_dashboard()
    generate_text_report()

    elapsed = time.time() - start
    print("\n")
    print("╔" + "═" * 56 + "╗")
    print("║  ✅ PIPELINE COMPLETE                                  ║")
    print(f"║  Total time: {elapsed:.1f}s                                    ║")
    print("╚" + "═" * 56 + "╝")
    print("\n📁 Output files:")
    print("   data/retail_sales_data.csv       ← Raw dataset")
    print("   data/cleaned_data.csv            ← Preprocessed data")
    print("   data/features.csv                ← Feature-engineered data")
    print("   models/best_model.pkl            ← Trained model")
    print("   outputs/tables/predictions.csv   ← Forecast predictions")
    print("   outputs/tables/inventory_report.csv ← Inventory decisions")
    print("   outputs/graphs/*.png             ← All visualization charts")
    print("   outputs/reports/executive_summary.txt ← Business report")
    print("\n🎯 Open outputs/graphs/ to view all charts!")


if __name__ == '__main__':
    main()
