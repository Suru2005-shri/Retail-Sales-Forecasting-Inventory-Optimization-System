"""
forecasting_model.py
--------------------
Trains multiple models to forecast daily units sold:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. XGBoost Regressor  ← best model
Evaluates with MAE, RMSE, R²
Saves best model, generates forecast plots and prediction CSVs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing    import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(" XGBoost not installed — using Random Forest as best model")

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/graphs', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)


# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = [
    'store_code', 'category_code', 'product_code',
    'year', 'month', 'quarter', 'week',
    'day_of_week', 'is_weekend', 'festival_season', 'promotion',
    'lag_7d', 'lag_14d', 'lag_30d',
    'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d',
    'rolling_std_7d', 'ewm_7d',
]
TARGET = 'units_sold'


def load_data(filepath='data/features.csv'):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.dropna(subset=FEATURE_COLS + [TARGET], inplace=True)
    return df


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.2f} units")
    print(f"    RMSE : {rmse:.2f} units")
    print(f"    R²   : {r2:.4f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def plot_actual_vs_predicted(y_test, y_pred_best, model_name, sample=300):
    """Plot actual vs predicted for a random sample."""
    idx = np.random.choice(len(y_test), sample, replace=False)
    idx_sorted = np.sort(idx)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── Line plot ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(y_test.values[idx_sorted], label='Actual', color='#1E40AF', linewidth=1.5, alpha=0.9)
    ax.plot(y_pred_best[idx_sorted],   label='Predicted', color='#EF4444', linewidth=1.5,
            linestyle='--', alpha=0.85)
    ax.set_title(f'Actual vs Predicted Units Sold — {model_name} (Sample of {sample})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Units Sold')
    ax.legend()

    # ── Scatter plot ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.scatter(y_test.values[idx_sorted], y_pred_best[idx_sorted],
                alpha=0.4, s=20, color='#2563EB')
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    ax2.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
    ax2.set_title('Predicted vs Actual Scatter Plot', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Actual Units Sold')
    ax2.set_ylabel('Predicted Units Sold')
    ax2.legend()

    plt.tight_layout()
    path = 'outputs/graphs/08_actual_vs_predicted.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nSaved: {path}")


def plot_feature_importance(model, model_name):
    """Bar chart of feature importances (for tree-based models)."""
    if not hasattr(model, 'feature_importances_'):
        return
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances.sort_values(ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot(kind='barh', ax=ax, color='#10B981', edgecolor='white')
    ax.set_title(f'Feature Importance — {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    path = 'outputs/graphs/09_feature_importance.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(results):
    """Compare all models on MAE, RMSE, R²."""
    df_res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ['MAE', 'RMSE', 'R2']
    colors  = ['#EF4444', '#F59E0B', '#10B981']

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(df_res['model'], df_res[metric], color=color, edgecolor='white')
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in df_res[metric]], padding=4, fontsize=9)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = 'outputs/graphs/10_model_comparison.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def train_models(filepath='data/features.csv'):
    print("=" * 50)
    print("FORECASTING MODEL TRAINING")
    print("=" * 50)

    df = load_data(filepath)
    X  = df[FEATURE_COLS]
    y  = df[TARGET]

    # ── Train/Test split (time-based) ────────────────────────────
    # Use last 20% of data as test (simulates future forecasting)
    split = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"\nTrain size: {len(X_train):,}   Test size: {len(X_test):,}")

    results    = []
    best_model = None
    best_rmse  = float('inf')
    best_name  = ''

    # ── Model 1: Linear Regression ───────────────────────────────
    print("\nTraining Linear Regression...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = np.maximum(0, lr.predict(X_test_sc))
    res = evaluate('Linear Regression', y_test, y_pred_lr)
    results.append(res)
    if res['RMSE'] < best_rmse:
        best_rmse  = res['RMSE']
        best_model = lr
        best_name  = 'Linear Regression'

    # ── Model 2: Random Forest ───────────────────────────────────
    print("\nTraining Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = np.maximum(0, rf.predict(X_test))
    res = evaluate('Random Forest', y_test, y_pred_rf)
    results.append(res)
    if res['RMSE'] < best_rmse:
        best_rmse  = res['RMSE']
        best_model = rf
        best_name  = 'Random Forest'

    # ── Model 3: XGBoost ─────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost Regressor...")
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbosity=0, n_jobs=-1)
        xgb.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False)
        y_pred_xgb = np.maximum(0, xgb.predict(X_test))
        res = evaluate('XGBoost', y_test, y_pred_xgb)
        results.append(res)
        if res['RMSE'] < best_rmse:
            best_rmse  = res['RMSE']
            best_model = xgb
            best_name  = 'XGBoost'

    # ── Best model predictions ───────────────────────────────────
    best_preds = np.maximum(0, best_model.predict(
        scaler.transform(X_test) if best_name == 'Linear Regression' else X_test
    ))

    print(f"\nBest Model: {best_name}  (RMSE: {best_rmse:.2f})")

    # ── Save predictions CSV ─────────────────────────────────────
    pred_df = df.iloc[split:][['date','store','product','category']].copy()
    pred_df['actual_units']    = y_test.values
    pred_df['predicted_units'] = best_preds.round().astype(int)
    pred_df['error']           = (pred_df['predicted_units'] - pred_df['actual_units']).abs()
    pred_df.to_csv('outputs/tables/predictions.csv', index=False)
    print("Saved: outputs/tables/predictions.csv")

    # ── Plots ────────────────────────────────────────────────────
    plot_actual_vs_predicted(y_test, best_preds, best_name)
    plot_feature_importance(best_model, best_name)
    plot_model_comparison(results)

    # ── Save model ───────────────────────────────────────────────
    joblib.dump(best_model, 'models/best_model.pkl')
    if best_name == 'Linear Regression':
        joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Model saved → models/best_model.pkl")

    # ── Results table ────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    res_df.to_csv('outputs/tables/model_comparison.csv', index=False)
    print("Saved: outputs/tables/model_comparison.csv")

    return best_model, pred_df


if __name__ == '__main__':
    model, preds = train_models()
    print("\nSample predictions:")
    print(preds.head(10).to_string(index=False))
