import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys

# Import preprocessing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Stage4_Factor_selection import FactorSelectionEngine
from Stage5_Latent_factor_extraction import RobustDynamicFactorExtractor
from Stage6_MIDAS_regression import RegularizedMIDASRegressor
from Stage1_GDP_preprocessing_pipeline import GDPDataProcessor
from Stage2_Series_Preprocessing_pipeline import load_and_configure, process_single_series

MIN_FACTORS = 1
MAX_FACTORS = 5

# --- Configuration Constants (Aligned with Stage 5) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDICATOR_PATH = os.path.join(BASE_DIR, "Input Files", "Sample Clipped Data.csv")
METADATA_PATH = os.path.join(BASE_DIR, "Input Files/(Main) Metadata.csv")
GDP_META_PATH = os.path.join(BASE_DIR, "Input Files", "Real GDP YoY Quarterly India-2.csv")

# Stage 5 aligned constants
MAX_ITERATIONS = 2000        # Matches Stage 5
EM_ITERATIONS = 30           # Matches Stage 5
CONVERGENCE_TOL = 1e-3       # Matches Stage 5

# Backtesting specific
HORIZON_DAYS = 90
TRAINING_QUARTERS = 24

class NowcastBacktester:
    """
    Backtesting system for GDP nowcasting pipeline.
    Evaluates performance of RobustDynamicFactorExtractor + MIDAS regression
    on historical data using expanding window approach.
    """

    def __init__(self):
        self.results = []
        self.logger = self._configure_logging()
        self.raw_gdp = None
        self.raw_indicators = None
        self.evaluation_quarters = []
        self.metadata_config = None
        self.temp_dir = tempfile.mkdtemp(prefix="backtest_temp_")
        self.logger.info(f"Created temporary directory: {self.temp_dir}")

    def _configure_logging(self) -> logging.Logger:
        """Set up logging for backtesting process"""
        logger = logging.getLogger('NowcastBacktester')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        os.makedirs("Backtest_Results", exist_ok=True)
        fh = logging.FileHandler(os.path.join("Backtest_Results", "backtest.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def load_data(self) -> None:
        """Load and prepare all necessary datasets"""
        try:
            # Log file paths for debugging
            self.logger.info(f"Base directory: {BASE_DIR}")
            self.logger.info(f"Metadata path: {METADATA_PATH}")
            self.logger.info(f"Indicator path: {INDICATOR_PATH}")
            self.logger.info(f"GDP path: {GDP_META_PATH}")
            
            # Check if files exist
            if not os.path.exists(METADATA_PATH):
                raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
            if not os.path.exists(INDICATOR_PATH):
                raise FileNotFoundError(f"Indicator file not found: {INDICATOR_PATH}")
            if not os.path.exists(GDP_META_PATH):
                raise FileNotFoundError(f"GDP file not found: {GDP_META_PATH}")
            
            # Load metadata configuration for Stage2 preprocessing
            self.metadata_config = load_and_configure(METADATA_PATH)
            if self.metadata_config is None:
                raise RuntimeError("Failed to load metadata configuration")
            
            # Load raw GDP data
            self.raw_gdp = pd.read_csv(
                GDP_META_PATH,
                parse_dates=['Date'],
                index_col='Date'
            ).squeeze()
            self.logger.info(f"Loaded GDP data: {len(self.raw_gdp)} points")
            
            # Load raw indicator data
            self.raw_indicators = pd.read_csv(
                INDICATOR_PATH,
                index_col='Date',
                parse_dates=True
            ).sort_index()
            self.logger.info(f"Loaded indicator data: {self.raw_indicators.shape}")

            # Determine evaluation quarters
            self._set_evaluation_period()

        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise

    def _set_evaluation_period(self) -> None:
        """Determine quarters for backtesting evaluation"""
        # Get GDP quarters sorted chronologically
        gdp_quarters = self.raw_gdp.dropna().index.sort_values()
        self.logger.info(f"GDP quarters: {gdp_quarters}")

        # Ensure we have enough training data
        if len(gdp_quarters) < TRAINING_QUARTERS + 1:
            raise ValueError(
                f"Insufficient GDP quarters for backtesting. Have {len(gdp_quarters)} "
                f"but need at least {TRAINING_QUARTERS + 1}"
            )

        # Use actual GDP quarter dates for evaluation
        self.evaluation_quarters = gdp_quarters[TRAINING_QUARTERS:]
        self.logger.info(f"Backtesting {len(self.evaluation_quarters)} quarters")
        self.logger.info(f"Training on first {TRAINING_QUARTERS} quarters")
        self.logger.info(f"First evaluation quarter: {self.evaluation_quarters[0]}")
        self.logger.info(f"Last evaluation quarter: {self.evaluation_quarters[-1]}")

    def _preprocess_gdp(self, end_date: datetime) -> Tuple[pd.Series, float, float]:
        """
        Preprocess GDP data up to but excluding the end_date
        Returns:
            processed_gdp: Standardized GDP series
            orig_mean: Mean of original GDP data before preprocessing
            orig_std: Standard deviation of original GDP data before preprocessing
        """
        try:
            # Filter GDP data to dates before end_date
            gdp_subset = self.raw_gdp[self.raw_gdp.index < end_date].copy()
            self.logger.info(f"GDP subset for {end_date}: {len(gdp_subset)} points")
            
            if len(gdp_subset) < 10:
                raise ValueError(f"Insufficient GDP data points: {len(gdp_subset)}")
            
            # Calculate original stats BEFORE preprocessing
            orig_mean = gdp_subset.mean()
            orig_std = gdp_subset.std()
            self.logger.info(f"Original GDP stats - Mean: {orig_mean:.2f}, Std: {orig_std:.2f}")
            
            # Create processor
            processor = GDPDataProcessor()
            
            # Create temporary files
            temp_input = os.path.join(self.temp_dir, f"gdp_input_{end_date.strftime('%Y%m%d')}.csv")
            temp_output = os.path.join(self.temp_dir, f"gdp_output_{end_date.strftime('%Y%m%d')}.csv")
            
            # Save subset to temp file
            gdp_subset.reset_index().to_csv(temp_input, index=False)
            self.logger.info(f"Saved GDP subset to temporary file: {temp_input}")
            
            # Process GDP data
            success = processor.process_gdp_data(
                input_file=temp_input,
                date_column='Date',
                value_column=self.raw_gdp.name,
                output_file=temp_output
            )
            
            if not success:
                raise RuntimeError("GDP preprocessing failed")
            
            # Load processed GDP
            processed_gdp = pd.read_csv(temp_output, index_col='Date', parse_dates=True).squeeze()
            self.logger.info(f"Processed GDP: {len(processed_gdp)} points")
            
            return processed_gdp, orig_mean, orig_std
        
        except Exception as e:
            self.logger.error(f"GDP preprocessing failed for {end_date}: {e}")
            raise

    def _preprocess_indicators(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess indicators using Stage2 pipeline"""
        try:
            # Create empty DataFrame with original index
            processed_data = pd.DataFrame(index=raw_df.index)
            self.logger.info(f"Starting indicator preprocessing for {len(raw_df)} periods")
            
            # Process each series independently
            for series_name, cfg in self.metadata_config.items():
                if series_name not in raw_df.columns:
                    self.logger.warning(f"Series '{series_name}' not found in data. Skipping.")
                    continue
                
                self.logger.info(f"Processing series: {series_name}")
                original_series = raw_df[series_name].copy()
                
                try:
                    processed_series = process_single_series(
                        original_series, 
                        cfg, 
                        raw_df.index
                    )
                    processed_data[processed_series.name] = processed_series
                except Exception as e:
                    self.logger.error(f"Failed to process series '{series_name}': {e}")
            
            if processed_data.empty:
                raise RuntimeError("No indicators were successfully processed")
                
            valid_count = processed_data.count().sum()
            self.logger.info(f"Indicator preprocessing complete. Valid points: {valid_count}")
            return processed_data
        
        except Exception as e:
            self.logger.error(f"Indicator preprocessing failed: {e}")
            raise

    def run_backtest(self) -> None:
        """Execute backtesting on historical data"""
        if len(self.evaluation_quarters) == 0:
            self.logger.error("No evaluation quarters defined")
            return

        self.logger.info("Starting backtesting...")

        for current_quarter in self.evaluation_quarters:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Nowcasting for quarter ending {current_quarter.strftime('%Y-%m-%d')}")
                self.logger.info(f"{'='*50}")

                # Prepare data available at current quarter
                indicator_subset, gdp_train, orig_gdp_mean, orig_gdp_std = self._prepare_data(current_quarter)

                # Run full pipeline
                nowcast_yoy, ci_lower, ci_upper, diagnostics = self._run_full_pipeline(
                    indicator_subset, gdp_train, current_quarter, orig_gdp_mean, orig_gdp_std
                )

                # Get actual GDP value (raw YoY)
                actual_yoy = self.raw_gdp.loc[current_quarter]

                # Store results
                self._store_results(
                    current_quarter,
                    nowcast_yoy,
                    ci_lower,
                    ci_upper,
                    actual_yoy,
                    orig_gdp_mean,
                    orig_gdp_std,
                    diagnostics
                )

                # Log nowcast summary
                self.logger.info(f"\nNowcast for {current_quarter.strftime('%Y-%m-%d')}:")
                self.logger.info(f"Original GDP Mean (training): {orig_gdp_mean:.2f}%")
                self.logger.info(f"Original GDP Std (training): {orig_gdp_std:.2f}%")
                self.logger.info(f"Predicted YoY Growth: {nowcast_yoy:.2f}%")
                self.logger.info(f"95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
                self.logger.info(f"Actual YoY Growth: {actual_yoy:.2f}%")

            except Exception as e:
                self.logger.error(f"Error processing {current_quarter}: {e}", exc_info=True)
                self._store_error_results(current_quarter, e)
                continue

        # After backtesting completes
        self._evaluate_performance()
        self._plot_results()
        self._cleanup_temp_files()
        self.logger.info("Backtesting completed successfully")

    def _prepare_data(self, current_quarter) -> Tuple[pd.DataFrame, pd.Series, float, float]:
        """
        Prepare data available at current quarter with preprocessing
        Returns:
            processed_indicators: Preprocessed indicator data
            gdp_train: Preprocessed GDP training data
            orig_gdp_mean: Mean of original GDP training data before preprocessing
            orig_gdp_std: Std dev of original GDP training data before preprocessing
        """
        try:
            # Preprocess GDP data (training data only - up to previous quarter)
            gdp_train, orig_gdp_mean, orig_gdp_std = self._preprocess_gdp(current_quarter)
            
            if len(gdp_train) < TRAINING_QUARTERS:
                raise ValueError(
                    f"Only {len(gdp_train)} GDP quarters available, "
                    f"need at least {TRAINING_QUARTERS} for training"
                )
            
            # Preprocess indicators (up to current quarter)
            indicator_subset = self.raw_indicators[self.raw_indicators.index <= current_quarter]
            processed_indicators = self._preprocess_indicators(indicator_subset)
            
            self.logger.info(f"Prepared data: {processed_indicators.shape[0]} indicator days, "
                            f"{len(gdp_train)} GDP quarters")
            
            return processed_indicators, gdp_train, orig_gdp_mean, orig_gdp_std
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

    def _run_full_pipeline(self, indicator_data: pd.DataFrame, 
                          gdp_train: pd.Series,
                          current_quarter: datetime,
                          orig_gdp_mean: float,
                          orig_gdp_std: float) -> Tuple[float, float, float, dict]:
        """
        Execute full nowcasting pipeline for a single quarter
        
        Args:
            orig_gdp_mean: Mean of original GDP training data before preprocessing
            orig_gdp_std: Std dev of original GDP training data before preprocessing
        """
        # Step 1: Factor Selection (Stage 4)
        self.logger.info("Running Stage 4: Factor Selection...")
        selector = FactorSelectionEngine(
            min_factors=MIN_FACTORS,
            max_factors=MAX_FACTORS,
            max_iterations=MAX_ITERATIONS,
            em_iterations=EM_ITERATIONS
        )
        selection_results = selector.run_factor_selection(indicator_data)
        optimal_factors = selection_results['selection_results']['optimal_factors']
        self.logger.info(f"Selected {optimal_factors} factors for {current_quarter}")

        # Step 2: Factor Extraction (Stage 5)
        self.logger.info("Running Stage 5: Factor Extraction...")
        extractor = RobustDynamicFactorExtractor(
            n_factors=optimal_factors,
            factor_order=1,
            max_iterations=MAX_ITERATIONS,
            em_iterations=EM_ITERATIONS,
            convergence_tol=CONVERGENCE_TOL
        )
        full_factors = extractor.fit_and_extract(indicator_data)
        
        # Get last date of GDP training period (previous quarter end)
        last_train_date = gdp_train.index[-1]
        
        # Use ONLY factors up to last training date for model training
        training_factors = full_factors[full_factors.index <= last_train_date]
        
        # Step 3: MIDAS Regression (Stage 6)
        self.logger.info("Running Stage 6: MIDAS Regression...")
        # Calculate stats for standardized GDP training data
        mean_train = gdp_train.mean()
        std_train = gdp_train.std()
        
        # Handle case where all GDP values are the same
        if std_train < 1e-6:
            self.logger.warning("GDP standard deviation is near zero. Using fallback value.")
            std_train = 1.0
            
        gdp_train_z = (gdp_train - mean_train) / std_train
        
        midas = RegularizedMIDASRegressor(
            poly_degree=2,
            horizon_days=HORIZON_DAYS,
            max_ar_lags=2
        )
        midas.fit(training_factors, gdp_train_z)
        
        # Nowcast using full factors (including current quarter)
        factors_nowcast = full_factors[full_factors.index <= current_quarter]
        nowcast_z, std_error_z = midas.nowcast(
            factors_nowcast, 
            gdp_train_z, 
            current_quarter
        )
        
        # Step 4: Conversion to YoY
        self.logger.info("Converting Z-score to YoY Growth...")
        # Use original GDP stats for conversion
        nowcast_yoy = nowcast_z * orig_gdp_std + orig_gdp_mean
        ci_lower_yoy = (nowcast_z - 1.96 * std_error_z) * orig_gdp_std + orig_gdp_mean
        ci_upper_yoy = (nowcast_z + 1.96 * std_error_z) * orig_gdp_std + orig_gdp_mean
        
        # Package diagnostics
        diagnostics = {
            'optimal_factors': optimal_factors,
            'midas_method': midas.diagnostics.get('regularization', 'ridge'),
            'midas_alpha': midas.diagnostics.get('alpha', 1.0),
            'gdp_train_mean': mean_train,
            'gdp_train_std': std_train,
            'orig_gdp_mean': orig_gdp_mean,
            'orig_gdp_std': orig_gdp_std
        }
        
        return nowcast_yoy, ci_lower_yoy, ci_upper_yoy, diagnostics

    def _store_results(self, date: datetime, nowcast: float, 
                      ci_lower: float, ci_upper: float,
                      actual: float, 
                      orig_gdp_mean: float,
                      orig_gdp_std: float,
                      diagnostics: dict) -> None:
        """Store backtesting results with diagnostics"""
        error = nowcast - actual
        variation_explained = 1 - (abs(error) / abs(actual)) if actual != 0 else 0
        
        self.results.append({
            'date': date,
            'nowcast': nowcast,
            'actual': actual,
            'error': error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'variation_explained': variation_explained,
            'optimal_factors': diagnostics['optimal_factors'],
            'midas_method': diagnostics['midas_method'],
            'midas_alpha': diagnostics['midas_alpha'],
            'gdp_train_mean': diagnostics['gdp_train_mean'],
            'gdp_train_std': diagnostics['gdp_train_std'],
            'orig_gdp_mean': orig_gdp_mean,
            'orig_gdp_std': orig_gdp_std
        })
        self.logger.info(f"Results stored for {date.strftime('%Y-%m-%d')}")

    def _store_error_results(self, date: datetime, error: Exception) -> None:
        """Store placeholder results when processing fails"""
        actual = self.raw_gdp.loc[date] if date in self.raw_gdp.index else np.nan
        self.results.append({
            'date': date,
            'nowcast': np.nan,
            'actual': actual,
            'error': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'variation_explained': np.nan,
            'optimal_factors': -1,
            'midas_method': "error",
            'midas_alpha': -1,
            'gdp_train_mean': np.nan,
            'gdp_train_std': np.nan,
            'orig_gdp_mean': np.nan,
            'orig_gdp_std': np.nan,
            'error_message': str(error)
        })
        self.logger.warning(f"Stored error placeholder for {date.strftime('%Y-%m-%d')}")

    def _evaluate_performance(self) -> None:
        """Calculate performance metrics from backtesting results"""
        if not self.results:
            self.logger.warning("No results to evaluate")
            return

        # Create results DataFrame
        df = pd.DataFrame(self.results)
        df = df.dropna(subset=['nowcast', 'actual'])
        
        if df.empty:
            self.logger.error("No valid results to evaluate")
            return
            
        df.set_index('date', inplace=True)

        # Calculate metrics
        metrics = {
            'RMSE': np.sqrt(np.mean(df['error']**2)),
            'MAE': np.mean(np.abs(df['error'])),
            'Bias': np.mean(df['error']),
            'Correlation': df[['nowcast', 'actual']].corr().iloc[0,1],
            'Coverage': np.mean((df['actual'] >= df['ci_lower']) & (df['actual'] <= df['ci_upper'])),
            'Variation_Explained': np.mean(df['variation_explained'])
        }

        # Save metrics
        os.makedirs("Backtest_Results", exist_ok=True)
        metrics_path = os.path.join("Backtest_Results", "performance_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Print metrics
        self.logger.info("\n" + "="*50)
        self.logger.info("BACKTESTING PERFORMANCE METRICS")
        self.logger.info("="*50)
        for k, v in metrics.items():
            self.logger.info(f"{k}: {v:.4f}")

        # Save full results
        results_path = os.path.join("Backtest_Results", "backtest_results.csv")
        df.to_csv(results_path)
        self.logger.info(f"Full results saved to {results_path}")

    def _plot_results(self) -> None:
        """Generate visualizations of backtesting performance"""
        if not self.results:
            return

        # Create results DataFrame
        df = pd.DataFrame(self.results)
        df = df.dropna(subset=['nowcast', 'actual'])
        
        if df.empty:
            self.logger.warning("No valid results to plot")
            return
            
        df.set_index('date', inplace=True)

        plt.figure(figsize=(15, 12))
        plt.suptitle("GDP Nowcasting Backtest Performance", fontsize=16)

        # Actual vs Nowcast
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df['actual'], 'o-', label='Actual', linewidth=2, markersize=6)
        plt.plot(df.index, df['nowcast'], 'x--', label='Nowcast', linewidth=2, markersize=6)
        plt.fill_between(
            df.index,
            df['ci_lower'],
            df['ci_upper'],
            alpha=0.2,
            label='95% CI'
        )
        plt.ylabel('YoY Growth (%)', fontsize=12)
        plt.title('Actual vs Nowcast', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Error histogram
        plt.subplot(2, 2, 2)
        sns.histplot(df['error'], kde=True, alpha=0.7)
        plt.axvline(0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Nowcast Error (%)', fontsize=12)
        plt.title('Error Distribution', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Error over time
        plt.subplot(2, 2, 3)
        plt.plot(df.index, df['error'], 'o-', linewidth=2, markersize=6)
        plt.axhline(0, color='r', linestyle='--', linewidth=2)
        plt.ylabel('Nowcast Error (%)', fontsize=12)
        plt.title('Error Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Variation explained
        plt.subplot(2, 2, 4)
        colors = ['green' if x > 0 else 'red' for x in df['variation_explained']]
        plt.bar(df.index, df['variation_explained'] * 100, color=colors, alpha=0.7)
        plt.ylabel('Variation Explained (%)', fontsize=12)
        plt.title('Proportion of Variation Explained', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_path = os.path.join("Backtest_Results", "backtest_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Performance plots saved to {plot_path}")

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary files: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup_temp_files()


if __name__ == "__main__":
    backtester = NowcastBacktester()
    try:
        backtester.load_data()
        backtester.run_backtest()
    except Exception as e:
        backtester.logger.error(f"Backtesting failed: {e}", exc_info=True)
        backtester._cleanup_temp_files()