import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.x13 import x13_arima_select_order, x13_arima_analysis
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
import logging
import warnings
import os # Imported to handle X13 path if needed

# Suppress warnings for cleaner output but log important events.
warnings.filterwarnings("ignore")

# Set up logging for informed debugging.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- 1. Master Configuration for Individual Series Processing ---
PIPELINE_CONFIG = {
    # --- Quarterly Variables ---
    'GVA: Agriculture, Forestry and Fishing': {
        'freq': 'Q', 'type': 'flow', 'log_transform': False,
        'seasonal_method': 'x13', 'seasonal_periods': [4]
    },
    'Unemployment Rate: Quarterly Status: Urban': {
        'freq': 'Q', 'type': 'stock', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': [4]
    },
    # --- Monthly Variables ---
    'Industrial Production Index (IPI)': {
        'freq': 'M', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': None
    },
    'Manufacturing PMI: Headline: sa: India': {
        'freq': 'M', 'type': 'stock', 'log_transform': True,
        'seasonal_method': None, 'seasonal_periods': None
    },
    'Petroleum: Consumption': {
        'freq': 'M', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': None
    },
    'Imports: USD': {
        'freq': 'M', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': None
    },
    'GST: Gross Revenue': {
        'freq': 'M', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': None
    },
    'Exports: USD': {
        'freq': 'M', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': None
    },
    'Foreign Exchange Reserve: USD: Foreign Exchange': {
        'freq': 'M', 'type': 'stock', 'log_transform': True,
        'seasonal_method': 'x13', 'seasonal_periods': None
    },
    # --- Weekly Variables ---
    'Rainfall: All India: Actual': {
        'freq': 'W', 'type': 'flow', 'log_transform': False,
        'seasonal_method': 'mstl', 'seasonal_periods': [52]
    },
    'Bombay Stock Exchange: Index: SENSEX': {
        'freq': 'W', 'type': 'stock', 'log_transform': True,
        'seasonal_method': 'mstl', 'seasonal_periods': [52]
    },
    # --- Daily Variables ---
    'Payment Systems: Value: Payment Transactions: NPCI Operated: Unified Payments Interface (UPI)': {
        'freq': 'D', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'mstl', 'seasonal_periods': [7, 30]
    },
    'Consumer_Durables_Daily': {
        'freq': 'D', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'mstl', 'seasonal_periods': [7, 30]
    },
    'Financial_Stress_Daily': {
        'freq': 'D', 'type': 'flow', 'log_transform': True,
        'seasonal_method': 'mstl', 'seasonal_periods': [7, 30]
    },
}

# --- 2. Pipeline Functions (No Changes Here, plus new stationarity enforcement) ---

def adjust_seasonality(series: pd.Series, config: dict) -> pd.Series:
    """Performs seasonal adjustment based on the configuration."""
    method = config.get('seasonal_method')
    if method is None:
        logging.info(f" -> Step 1: Skipping seasonal adjustment for '{series.name}' as configured.")
        return series
    
    # Check for X-13-ARIMA-SEATS executable if that method is chosen
    if method == 'x13' and not x13_arima_select_order(series.dropna(), prefer_x13=True):
         logging.warning(f" -> Step 1: X-13 executable not found or failed for '{series.name}'. Set the X13PATH environment variable or check installation. Skipping seasonal adjustment.")
         return series
         
    series_to_adjust = series.copy()
    if config.get('log_transform'):
        series_to_adjust = np.log(series_to_adjust.replace(0, np.nan)) # Replace 0 with NaN before log
    series_to_adjust = series_to_adjust.dropna()
    sa_series = None
    
    if method == 'x13':
        try:
            result = x13_arima_analysis(series_to_adjust, prefer_x13=True, log=False)
            sa_series = result.seasadj
        except Exception as e:
            logging.error(f"X-13 failed for '{series.name}': {e}. Reverting.")
            return series
            
    elif method == 'mstl':
        try:
            all_periods = config.get('seasonal_periods', [])
            # Ensure data is long enough for all seasonal periods
            periods = tuple(int(round(p)) for p in all_periods if len(series_to_adjust) >= 2 * int(round(p)))
            if not periods:
                logging.warning(f"MSTL skipped for '{series.name}': data too short for specified periods.")
                return series
            res = MSTL(series_to_adjust, periods=periods).fit()
            sa_series = res.trend + res.resid
        except Exception as e:
            logging.error(f"MSTL failed for '{series.name}': {e}. Reverting.")
            return series
            
    if config.get('log_transform'):
        sa_series = np.exp(sa_series)
        
    logging.info(f" -> Step 1: Seasonal adjustment completed for '{series.name}'.")
    return sa_series

def make_stationary(series: pd.Series, series_type: str) -> pd.Series:
    """Transforms the series to become stationary using the initial differencing step."""
    series = series.dropna()
    if series_type == 'flow':
        return np.log(series.replace(0, np.nan)).diff().dropna()
    elif series_type == 'stock':
        return series.diff().dropna()
    return series

def enforce_stationarity(series: pd.Series, config: dict, max_diff: int = 3) -> pd.Series:
    """
    Checks for stationarity using the Augmented Dickey-Fuller (ADF) test.
    If non-stationary, applies additional differencing up to max_diff times.
    """
    current_series = series.copy()
    diff_count = 0
    
    while diff_count < max_diff:
        try:
            # Check if series is empty or has NaNs before ADF test
            if current_series.isna().any() or current_series.empty:
                logging.warning(f"Series '{series.name}' contains NaNs or is empty before ADF test. Halting differencing.")
                break
            p_value = adfuller(current_series)[1]
        except Exception as e:
            logging.error(f"ADF test failed for '{series.name}': {e}")
            break

        if p_value < 0.05:
            if diff_count > 0:
                logging.info(f" -> Step 3: Stationarity enforced after {diff_count} additional differencing step(s) for '{series.name}'. (p-value: {p_value:.4f})")
            else:
                logging.info(f" -> Step 3: Series '{series.name}' is stationary on the first pass. (p-value: {p_value:.4f})")
            return current_series
        else:
            diff_count += 1
            # Additional differencing step.
            current_series = current_series.diff(1).dropna()
            logging.info(f" -> Step 3: Applying additional differencing step {diff_count} for '{series.name}' (previous ADF p-value: {p_value:.4f}).")

    # If loop finished without becoming stationary
    if diff_count == max_diff:
        final_p_value = adfuller(current_series.dropna())[1]
        logging.warning(f"Series '{series.name}' may still be non-stationary after {max_diff} differencing steps (final p-value: {final_p_value:.4f}).")
        
    return current_series

def standardize(series: pd.Series) -> pd.Series:
    """Applies Z-score standardization."""
    return (series - series.mean()) / series.std()

# --- 3. NEW: Configuration for Economically-Linked Panels ---
CLUSTER_CONFIG = {
    "Industrial_Activity_Panel": [
        'Industrial Production Index (IPI)',
        'Manufacturing PMI: Headline: sa: India',
        'Petroleum: Consumption',
        'GST: Gross Revenue',
        'Exports: USD',
        'Imports: USD'
    ],
    "Financial_Sentiment_Panel": [
        'Bombay Stock Exchange: Index: SENSEX',
        'Foreign Exchange Reserve: USD: Foreign Exchange',
        'Financial_Stress_Daily'
    ],
    "Consumer_Demand_Panel": [
        'Payment Systems: Value: Payment Transactions: NPCI Operated: Unified Payments Interface (UPI)',
        'Consumer_Durables_Daily',
        'Unemployment Rate: Quarterly Status: Urban' # Reflects consumer economic health
    ],
    "Agricultural_Panel": [
        'GVA: Agriculture, Forestry and Fishing',
        'Rainfall: All India: Actual'
    ]
}

# --- 4. Main Execution Pipeline ---

def main():
    """Main function to run the entire preprocessing pipeline."""
    logging.info("==============================================")
    logging.info("=== Starting Data Preprocessing Pipeline ===")
    logging.info("==============================================")

    try:
        df_raw = pd.read_csv("[Sample] Data.csv", index_col='Date', parse_dates=True)
        logging.info(f"Raw data loaded: {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")
    except FileNotFoundError:
        logging.error("FATAL ERROR: '[Sample] Data.csv' not found. Please ensure the data file is in the correct directory. Halting execution.")
        return

    # --- Part A: Process Each Series Individually ---
    processed_series = {}
    for col, config in PIPELINE_CONFIG.items():
        if col not in df_raw.columns:
            logging.warning(f"Column '{col}' from config not found in data. Skipping.")
            continue

        logging.info(f"\n--- Processing Variable: '{col}' ---")
        series = df_raw[col].dropna()
        if series.empty:
            logging.warning(f"Series '{col}' is empty after dropping NaNs. Skipping.")
            continue

        # 1. Seasonal Adjustment
        sa_series = adjust_seasonality(series, config)
        if sa_series is None or sa_series.empty:
            logging.warning(f"Seasonal adjustment produced an empty series for '{col}'. Skipping further processing.")
            continue

        # 2. Initial Stationarity Transformation (differencing)
        logging.info(f" -> Step 2: Applying initial stationarity transform (type: {config['type']}) for '{col}'.")
        stat_series = make_stationary(sa_series, config['type'])
        if stat_series.empty:
            logging.warning(f"Initial stationarity transformation resulted in an empty series for '{col}'. Skipping.")
            continue

        # 3. Enforce further stationarity if needed
        stat_series.name = col
        stat_series = enforce_stationarity(stat_series, config)

        # 4. Final standardization
        logging.info(f" -> Step 4: Standardizing '{col}' with Z-score.")
        try:
            final_series = standardize(stat_series)
            final_series.name = f"{col}_processed"
            logging.info(f"--- Successfully processed '{col}' ---")
        except Exception as e:
            logging.error(f"Standardization failed for '{col}': {e}")
            continue

        processed_series[final_series.name] = final_series

    if not processed_series:
        logging.error("No series were successfully processed. Check config, data, and error logs.")
        return

    df_processed = pd.concat(processed_series.values(), axis=1)
    logging.info(f"\nIndividual series processing complete. Combined shape: {df_processed.shape}")

    # --- Part B: Create Panels Based on Economic Clusters ---
    logging.info("\n=================================================")
    logging.info("=== Creating Final Panels from Processed Data ===")
    logging.info("=================================================")

    for panel_name, var_list in CLUSTER_CONFIG.items():
        panel_cols = [f"{col}_processed" for col in var_list if f"{col}_processed" in df_processed.columns]

        if not panel_cols:
            logging.warning(f"No processed variables found for panel '{panel_name}'. Skipping.")
            continue

        panel_df = df_processed[panel_cols].dropna(how='all')

        filename = f"{panel_name}.csv"
        try:
            panel_df.to_csv(filename)
            logging.info(f"Successfully saved '{panel_name}' with {len(panel_cols)} variables to '{filename}'.")
        except Exception as e:
            logging.error(f"Failed to save panel file '{filename}': {e}")

        # Save correlation heatmap
        try:
            corr_matrix = panel_df.corr()
            plt.figure(figsize=(20, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
            plt.title(f"Correlation Heatmap: {panel_name}", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            heatmap_filename = f"{panel_name}_correlation_heatmap.png"
            plt.tight_layout()
            plt.savefig(heatmap_filename, dpi=300)
            plt.close()
            logging.info(f"Saved correlation heatmap for '{panel_name}' to '{heatmap_filename}'.")
        except Exception as e:
            logging.error(f"Failed to generate/save correlation heatmap for '{panel_name}': {e}")

    logging.info("\n=== Pipeline execution completed successfully. ===")


if __name__ == "__main__":
    main()
