# ======================================================================
#                       Imports & Configuration
# ======================================================================
import os
import ast
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.x13 import (
    x13_arima_select_order,
    x13_arima_analysis
)
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller

# Suppress warnings and set up logging
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


# ======================================================================
#                 1. METADATA LOADING & CONFIGURATION
# ======================================================================
def load_and_configure(metadata_path: str) -> dict | None:
    """
    1. Reads metadata CSV
    2. Strips whitespace from series names
    3. Constructs a dict of pipeline configs per series
    """
    try:
        df = pd.read_csv(metadata_path)
        logging.info("Loaded metadata from '%s'.", metadata_path)
    except FileNotFoundError:
        logging.error("Metadata file not found at '%s'.", metadata_path)
        return None

    # Clean up series names
    df['Series_Name'] = df['Series_Name'].str.strip()

    pipeline_config: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = row['Series_Name']
        try:
            # parse seasonal_periods if it's a list-string
            periods = row.get('seasonal_periods', None)
            if isinstance(periods, str) and periods.strip().startswith('['):
                seasonal_periods = ast.literal_eval(periods)
            else:
                seasonal_periods = None

            # parse rescale factor
            rescale_val = row['Rescaling by multiplication']
            rescale_factor = float(rescale_val) if pd.notna(rescale_val) else None

            # build config dict
            pipeline_config[name] = {
                'freq': row['Frequency'],
                'type': row['Type'],
                'seasonal_method': (
                    str(row.get('seasonal_method', '')).strip().lower()
                    or None
                ),
                'seasonal_periods': seasonal_periods,
                'rescale_factor': rescale_factor
            }
        except Exception as e:
            logging.error("Skipping '%s': %s", name, e)

    logging.info("Generated pipeline configuration for %d series.", len(pipeline_config))
    return pipeline_config


# ======================================================================
#            2. TRANSFORMATION: RESCALING & SEASONALITY
# ======================================================================
def rescale_series(series: pd.Series, cfg: dict) -> pd.Series:
    factor = cfg.get('rescale_factor')
    if factor is not None:
        logging.info("Rescaling '%s' by factor %s", series.name, factor)
        return series * factor
    return series


def adjust_seasonality(series: pd.Series, cfg: dict) -> pd.Series:
    method = cfg.get('seasonal_method')
    if not method:
        return series

    data = series.dropna().copy()

    if data.empty:
        return series

    try:
        if method == 'x13':
            if 'X13PATH' not in os.environ:
                logging.warning("X13PATH not set; X-13 may not run.")
            data.index.freq = pd.infer_freq(data.index)
            res = x13_arima_analysis(data, prefer_x13=True, log=False)
            adjusted = res.seasadj

        elif method == 'mstl':
            periods = cfg.get('seasonal_periods') or []
            valid = [int(p) for p in periods if len(data) >= 2 * p]
            if not valid:
                return series

            m = MSTL(data, periods=tuple(valid)).fit()
            adjusted = m.trend + m.resid

        else:
            return series

        logging.info("Seasonal adjustment done for '%s' (%s)", series.name, method)
        return adjusted

    except Exception as e:
        logging.error("Failed seasonal adjustment for '%s': %s", series.name, e)
        return series


# ======================================================================
#      3. STATIONARITY & STANDARDIZATION UTILITIES
# ======================================================================
def make_stationary(series: pd.Series, series_type: str) -> pd.Series:
    """
    - 'flow': period-on-period percentage change
    - 'stock': first difference
    """
    s = series.dropna()
    if s.empty:
        return s

    if series_type.lower() == 'flow':
        # Calculate period-on-period percentage change
        return ((s / s.shift(1)) - 1) * 100
    elif series_type.lower() == 'stock':
        # Calculate first difference
        return s.diff()
    return s


def enforce_stationarity(series: pd.Series, max_diff: int = 3) -> pd.Series:
    """
    Differ until ADF test p-value < 0.05 or max_diff reached.
    """
    s = series.copy().dropna()
    if len(s) < 10:  # Need minimum observations for ADF test
        logging.warning("Too few observations for ADF test, returning series as-is")
        return s
        
    for d in range(max_diff + 1):
        try:
            pval = adfuller(s)[1]
            if pval < 0.05:
                logging.info("Stationary after %d diffs (p=%.4f)", d, pval)
                return s
            s = s.diff().dropna()
            if len(s) < 10:  # Stop if too few observations left
                break
            logging.info("Applied diff %d, p=%.4f", d + 1, pval)
        except Exception as e:
            logging.warning("ADF test failed: %s, returning series", e)
            return s
    return s


def standardize(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty or s.std() == 0:
        return series
    return (s - s.mean()) / s.std()


def winsorize(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series
        
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:  # Handle case where all values are the same
        return series
        
    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR
    
    # Apply winsorization only to non-NaN values
    series_clean = series.copy()
    mask = series_clean.notna()
    series_clean.loc[mask] = series_clean.loc[mask].apply(
        lambda x: LB if x < LB else (UB if x > UB else x)
    )
    return series_clean


def process_single_series(series: pd.Series, cfg: dict, original_index: pd.Index) -> pd.Series:
    """
    Process a single series independently and map back to original index with NaN preservation.
    This is the key function that handles ragged-edge data properly.
    """
    series_name = series.name
    logging.info(f"Processing series '{series_name}'...")
    
    # Step 1: Work only with non-NaN values (preserving their original dates)
    clean_series = series.dropna()
    
    if clean_series.empty:
        logging.warning(f"Series '{series_name}' has no valid data points. Skipping.")
        return pd.Series(np.nan, index=original_index, name=f"{series_name}_processed")
    
    logging.info(f"  Original series: {len(series)} points, Valid data: {len(clean_series)} points")
    
    try:
        # Step 2: Apply all transformations to the clean series
        processed = clean_series.copy()
        
        # 2a) Rescale
        processed = rescale_series(processed, cfg)
        
        # 2b) Seasonal adjustment  
        processed = adjust_seasonality(processed, cfg)
        
        # 2c) Make stationary
        processed = make_stationary(processed, cfg['type'])
        
        # 2d) Enforce stationarity with ADF
        processed = enforce_stationarity(processed)
        
        # 2e) Winsorize
        processed = winsorize(processed)
        
        # 2f) Standardize
        processed = standardize(processed)
        
        # Remove any remaining NaN values after transformations
        processed = processed.dropna()
        
        logging.info(f"  After processing: {len(processed)} valid points")
        
        if processed.empty:
            logging.warning(f"Series '{series_name}' became empty after processing.")
            return pd.Series(np.nan, index=original_index, name=f"{series_name}_processed")
        
        # Step 3: Create result series with original index, filled with NaN
        result = pd.Series(np.nan, index=original_index, name=f"{series_name}_processed")
        
        # Step 4: Map processed values back to their original dates
        # Only fill dates that exist in both the processed series and original index
        common_dates = result.index.intersection(processed.index)
        result.loc[common_dates] = processed.loc[common_dates]
        
        filled_points = result.notna().sum()
        logging.info(f"  Final result: {filled_points} filled points out of {len(original_index)} total dates")
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to process series '{series_name}': {e}")
        return pd.Series(np.nan, index=original_index, name=f"{series_name}_processed")


# ======================================================================
#                     4. MAIN PIPELINE EXECUTION
# ======================================================================
def main():
    logging.info("=== Starting Mixed-Frequency Nowcasting Preprocessing Pipeline ===")

    META_FILE = 'Input Files/(Main) Metadata.csv'
    DATA_FILE = 'Input Files/Sample Clipped Data.csv'

    config = load_and_configure(META_FILE)
    if config is None:
        return

    try:
        df_raw = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
        logging.info("Loaded raw data: %s", df_raw.shape)
        
        # Clean up any duplicate dates (keeping last occurrence)
        if df_raw.index.duplicated().any():
            logging.warning("Found duplicate dates. Removing duplicates by keeping last occurrence.")
            df_raw = df_raw[~df_raw.index.duplicated(keep='last')]
            logging.info("After removing duplicates: %s", df_raw.shape)
        
        # Sort index to ensure proper time series order
        df_raw = df_raw.sort_index()
        
        # Store the original index for consistent mapping
        original_index = df_raw.index.copy()
        
        logging.info(f"Original time range: {original_index.min()} to {original_index.max()}")
        logging.info(f"Available series: {list(df_raw.columns)}")
        
    except FileNotFoundError:
        logging.error("Data file '%s' not found. Aborting.", DATA_FILE)
        return

    # Initialize the final processed DataFrame with the original index
    # This preserves the exact same date structure as the input
    processed_data = pd.DataFrame(index=original_index)

    # Process each series independently
    for series_name, cfg in config.items():
        if series_name not in df_raw.columns:
            logging.warning(f"Series '{series_name}' not found in data. Skipping.")
            continue
        
        # Extract the series (includes NaN values)
        original_series = df_raw[series_name].copy()
        
        # Process the series independently and map back to original index
        processed_series = process_single_series(original_series, cfg, original_index)
        
        # Add to the final dataset
        processed_data[processed_series.name] = processed_series

    if processed_data.empty or processed_data.shape[1] == 0:
        logging.error("No series were successfully processed. Exiting.")
        return

    # Summary statistics
    logging.info("=== Processing Summary ===")
    for col in processed_data.columns:
        valid_count = processed_data[col].notna().sum()
        total_count = len(processed_data)
        coverage = (valid_count / total_count) * 100
        logging.info(f"  {col}: {valid_count}/{total_count} points ({coverage:.1f}% coverage)")

    # Create output folder
    output_folder = 'Processed Data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the processed panel data (preserving ragged-edge structure)
    output_file = os.path.join(output_folder, 'All_Processed_data.csv')
    processed_data.to_csv(output_file)
    logging.info(f"Saved processed panel data to '{output_file}'")
    
    # Display info about the final dataset
    logging.info(f"Final dataset shape: {processed_data.shape}")
    logging.info(f"Date range preserved: {processed_data.index.min()} to {processed_data.index.max()}")

    logging.info("=== Nowcasting Preprocessing Pipeline Completed Successfully ===")
    logging.info("Dataset ready for Dynamic Factor Model (DFM) with EM-Kalman estimation.")


if __name__ == '__main__':
    main()