import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import logging
import os
import json
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

class GDPDataProcessor:
    """GDP data preprocessing pipeline with comprehensive error handling and validation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_configuration(config_file)
        self._setup_logging()
        
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        default_config = {
            "date_formats": ["%m/%Y", "%Y-%m", "%m-%Y", "%Y/%m", "%B %Y", "%b %Y"],
            "winsorize_factor": 1.5,
            "stationarity_threshold": 0.05,
            "min_data_points": 10,
            "max_missing_ratio": 0.1,
            "outlier_detection_methods": ["iqr", "zscore"],
            "zscore_threshold": 3.0,
            "log_level": "INFO",
            "backup_original": True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logging.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config from {config_file}: {e}. Using defaults.")
        
        return default_config
    
    def _setup_logging(self):
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        
        log_dir = "GDP_processed"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'gdp_preprocessing.log')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file_path)
            ]
        )
        self.logger = logging.getLogger(__name__)

    
    def validate_input_parameters(self, file_path: str, date_col: str, value_col: str, output_file: str) -> bool:
        validation_errors = []
        
        if not os.path.exists(file_path):
            validation_errors.append(f"Input file not found: {file_path}")
        elif not os.access(file_path, os.R_OK):
            validation_errors.append(f"Input file not readable: {file_path}")
        
        output_dir = os.path.dirname(output_file) or '.'
        if not os.path.exists(output_dir):
            validation_errors.append(f"Output directory does not exist: {output_dir}")
        elif not os.access(output_dir, os.W_OK):
            validation_errors.append(f"Output directory not writable: {output_dir}")
        
        if not date_col or not isinstance(date_col, str):
            validation_errors.append("Date column name must be a non-empty string")
        
        if not value_col or not isinstance(value_col, str):
            validation_errors.append("Value column name must be a non-empty string")
        
        if validation_errors:
            for error in validation_errors:
                self.logger.error(error)
            return False
        
        return True
    
    def load_and_clean_data(self, file_path: str, date_col: str, value_col: str) -> Optional[pd.DataFrame]:
        self.logger.info(f"Loading data from '{file_path}'...")
        
        try:
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(f"Successfully loaded file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Unable to read file with any supported encoding")
            
            if date_col not in df.columns:
                raise KeyError(f"Date column '{date_col}' not found in data. Available columns: {list(df.columns)}")
            
            if value_col not in df.columns:
                raise KeyError(f"Value column '{value_col}' not found in data. Available columns: {list(df.columns)}")
            
            self.logger.info(f"Data shape: {df.shape}")
            
            df_cleaned = self._parse_dates_flexible(df, date_col)
            df_cleaned = self._clean_value_column(df_cleaned, value_col)
            
            if len(df_cleaned) < self.config["min_data_points"]:
                raise ValueError(f"Insufficient data points: {len(df_cleaned)} < {self.config['min_data_points']}")
            
            missing_ratio = df_cleaned[value_col].isna().sum() / len(df_cleaned)
            if missing_ratio > self.config["max_missing_ratio"]:
                raise ValueError(f"Too much missing data: {missing_ratio:.2%} > {self.config['max_missing_ratio']:.2%}")
            
            df_cleaned = df_cleaned.sort_index().drop_duplicates()
            
            self.logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Failed to load and clean data: {str(e)}")
            return None
    
    def _parse_dates_flexible(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        date_formats = self.config["date_formats"]
        
        for date_format in date_formats:
            try:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], format=date_format)
                df_copy = df_copy.set_index(date_col)
                self.logger.info(f"Successfully parsed dates using format: {date_format}")
                return df_copy
            except (ValueError, TypeError):
                continue
        
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], infer_datetime_format=True)
            df_copy = df_copy.set_index(date_col)
            self.logger.info("Successfully parsed dates using automatic inference")
            return df_copy
        except Exception as e:
            raise ValueError(f"Unable to parse dates in column '{date_col}' with any supported format. Error: {e}")
    
    def _clean_value_column(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        original_count = len(df)
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])
        
        cleaned_count = len(df)
        if cleaned_count < original_count:
            self.logger.warning(f"Removed {original_count - cleaned_count} rows with invalid numeric values")
        
        return df
    
    def detect_and_handle_outliers(self, series: pd.Series, method: str = "iqr") -> Tuple[pd.Series, pd.Series]:
        self.logger.info(f"Detecting outliers using {method.upper()} method...")
        
        if method == "iqr":
            return self._winsorize_iqr(series)
        elif method == "zscore":
            return self._winsorize_zscore(series)
        else:
            self.logger.warning(f"Unknown outlier detection method: {method}. Using IQR.")
            return self._winsorize_iqr(series)
    
    def _winsorize_iqr(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        factor = self.config["winsorize_factor"]
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        
        outlier_count = outlier_mask.sum()
        self.logger.info(f"IQR winsorization complete. {outlier_count} outliers detected and capped.")
        
        return winsorized, outlier_mask
    
    def _winsorize_zscore(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        threshold = self.config["zscore_threshold"]
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_mask = z_scores > threshold
        
        mean_val = series.mean()
        std_val = series.std()
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val
        
        winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        
        outlier_count = outlier_mask.sum()
        self.logger.info(f"Z-score winsorization complete. {outlier_count} outliers detected and capped.")
        
        return winsorized, outlier_mask
    
    def standardize_series(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        self.logger.info("Standardizing series (Z-score normalization)...")
        
        series_mean = series.mean()
        series_std = series.std()
        series_min = series.min()
        series_max = series.max()
        
        if series_std == 0:
            self.logger.error("Cannot standardize series with zero standard deviation")
            raise ValueError("Standard deviation is zero. Cannot standardize series.")
        
        standardized = (series - series_mean) / series_std
        
        standardization_params = {
            'mean': float(series_mean),
            'std': float(series_std),
            'min': float(series_min),
            'max': float(series_max),
            'count': int(len(series))
        }
        
        self.logger.info(f"Standardization complete. Mean: {series_mean:.6f}, Std: {series_std:.6f}")
        return standardized, standardization_params
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        self.logger.info("Performing stationarity tests...")
        
        try:
            adf_result = adfuller(series.dropna(), autolag='AIC')
            
            stationarity_results = {
                'adf_statistic': float(adf_result[0]),
                'adf_pvalue': float(adf_result[1]),
                'adf_critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < self.config["stationarity_threshold"],
                'test_threshold': self.config["stationarity_threshold"]
            }
            
            if stationarity_results['is_stationary']:
                self.logger.info(f"Series appears stationary (p-value: {adf_result[1]:.6f})")
            else:
                self.logger.warning(f"Series may be non-stationary (p-value: {adf_result[1]:.6f})")
            
            return stationarity_results
            
        except Exception as e:
            self.logger.error(f"Stationarity test failed: {str(e)}")
            return {'error': str(e)}
    
    def save_results(self, processed_data: pd.Series, output_file: str, 
                    standardization_params: Dict[str, float], 
                    processing_metadata: Dict[str, Any]) -> bool:
        try:
            output_df = pd.DataFrame({'gdp_processed': processed_data})
            output_df.to_csv(output_file)
            self.logger.info(f"Processed data saved to '{output_file}'")
            
            metadata_file = output_file.replace('.csv', '_metadata.json')
            full_metadata = {
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'standardization_parameters': standardization_params,
                'processing_metadata': processing_metadata,
                'config_used': self.config
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Metadata saved to '{metadata_file}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            return False
    
    def process_gdp_data(self, input_file: str, date_column: str, value_column: str, output_file: str) -> bool:
        try:
            if not self.validate_input_parameters(input_file, date_column, value_column, output_file):
                return False
            
            if self.config["backup_original"]:
                os.makedirs("GDP_processed", exist_ok=True)
                backup_path = os.path.join("GDP_processed", os.path.basename(input_file).replace('.csv', '_backup.csv'))
                if not os.path.exists(backup_path):
                    pd.read_csv(input_file).to_csv(backup_path, index=False)
                    self.logger.info(f"Backup created: {backup_path}")

            
            df = self.load_and_clean_data(input_file, date_column, value_column)
            if df is None:
                return False
            
            gdp_series = df[value_column]
            stationarity_results = self.test_stationarity(gdp_series)
            
            outlier_method = self.config["outlier_detection_methods"][0]
            gdp_cleaned, outlier_mask = self.detect_and_handle_outliers(gdp_series, outlier_method)
            
            gdp_standardized, standardization_params = self.standardize_series(gdp_cleaned)
            
            processing_metadata = {
                'original_data_points': len(df),
                'final_data_points': len(gdp_standardized),
                'outliers_detected': int(outlier_mask.sum()),
                'outlier_detection_method': outlier_method,
                'stationarity_test': stationarity_results,
                'date_range': {
                    'start': gdp_standardized.index.min().isoformat(),
                    'end': gdp_standardized.index.max().isoformat()
                }
            }
            
            success = self.save_results(gdp_standardized, output_file, standardization_params, processing_metadata)
            
            if success:
                self._display_processing_summary(gdp_standardized, standardization_params, processing_metadata)
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Processing pipeline failed: {str(e)}")
            return False
    
    def _display_processing_summary(self, processed_data: pd.Series, 
                                   standardization_params: Dict[str, float],
                                   metadata: Dict[str, Any]):
        print("\n" + "="*60)
        print("GDP DATA PREPROCESSING SUMMARY")
        print("="*60)
        
        print(f"Data Points Processed: {len(processed_data)}")
        print(f"Date Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        print(f"Outliers Detected: {metadata['outliers_detected']}")
        print(f"Outlier Method: {metadata['outlier_detection_method'].upper()}")
        
        if 'is_stationary' in metadata['stationarity_test']:
            stationarity_status = "STATIONARY" if metadata['stationarity_test']['is_stationary'] else "NON-STATIONARY"
            print(f"Stationarity Status: {stationarity_status}")
        
        print("\nStandardization Parameters (for inverse transformation):")
        for param, value in standardization_params.items():
            if isinstance(value, float):
                print(f"  {param.upper()}: {value:.6f}")
            else:
                print(f"  {param.upper()}: {value}")
        
        print(f"\nSample of Processed Data:")
        print(processed_data.head().to_string())
        if len(processed_data) > 5:
            print("...")
            print(processed_data.tail().to_string())
        
        print("="*60)


def main():
    INPUT_CSV_FILE = 'Input Files/Real GDP YoY Quarterly India-2.csv'
    DATE_COL_NAME = 'Date'
    VALUE_COL_NAME = 'Real GDP: YoY'
    OUTPUT_CSV_FILE = os.path.join('GDP_processed', 'target.csv')
    
    try:
        processor = GDPDataProcessor()
        
        success = processor.process_gdp_data(
            input_file=INPUT_CSV_FILE,
            date_column=DATE_COL_NAME,
            value_column=VALUE_COL_NAME,
            output_file=OUTPUT_CSV_FILE
        )
        
        if success:
            print("\nProcessing completed successfully!")
            return 0
        else:
            print("\nProcessing failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())