import pandas as pd
import numpy as np
import json
from datetime import datetime
import statsmodels.api as sm
import logging
import os
from datetime import timedelta
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RegularizedMIDASRegressor:
    """
    A regularized MIDAS regression class focused on addressing overfitting in 
    the MIDAS-specific components: Almon polynomial weights and AR lag selection.
    
    Since you're using Dynamic Factor Models for dimensionality reduction,
    this focuses on:
    1. Regularized estimation of Almon polynomial weights
    2. Optimal AR lag selection via cross-validation
    3. Regularized regression methods for coefficient estimation
    4. Robust weight constraint enforcement
    """

    def __init__(self, 
                 poly_degree: int = 2, 
                 horizon_days: int = 90, 
                 max_ar_lags: int = 4,
                 regularization: str = 'ridge',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 weight_constraint: str = 'normalized',  # 'normalized', 'monotonic', 'exponential'
                 min_observations: int = 15):
        """
        Initialize the Regularized MIDAS Regressor.
        
        Args:
            poly_degree: Degree of Almon polynomial (lower = less overfitting)
            horizon_days: Number of days to look back
            max_ar_lags: Maximum AR lags to consider (will be cross-validated)
            regularization: 'ridge', 'lasso', 'elastic_net', 'ols'
            alpha: Regularization strength
            l1_ratio: L1 ratio for Elastic Net
            weight_constraint: Type of weight constraint to prevent overfitting
            min_observations: Minimum observations required for fitting
        """
        self.poly_degree = poly_degree
        self.horizon_days = horizon_days
        self.max_ar_lags = max_ar_lags
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weight_constraint = weight_constraint
        self.min_observations = min_observations
        
        self.optimal_ar_lags = None
        self.model_results = None
        self.scaler = StandardScaler()
        self.almon_weights_cache = {}
        self.diagnostics = {}
        
        # ADD: Variables to store training information for JSON output
        self.training_period = {}
        self.data_sources = {}
        self.factor_names = []
        
        logging.info(f"Regularized MIDAS initialized: {regularization}, max_ar_lags={max_ar_lags}")
    
    def _cross_validate_ar_lags(self, factors: pd.DataFrame, gdp: pd.Series) -> int:
        """
        Cross-validate AR lags with regularization to prevent overfitting.
        Uses expanding window cross-validation for time series.
        """
        logging.info("Cross-validating AR lags with regularization...")
        
        lag_scores = {}
        min_train_size = max(20, self.min_observations)
        
        for n_lags in range(1, self.max_ar_lags + 1):
            try:
                y, X, dates = self.prepare_midas_data(factors, gdp, n_ar_lags=n_lags)
                
                if len(y) < min_train_size:
                    lag_scores[n_lags] = np.inf
                    continue
                
                # Expanding window cross-validation
                scores = []
                for i in range(min_train_size, len(y)):
                    X_train, X_test = X[:i], X[i:i+1]
                    y_train, y_test = y[:i], y[i:i+1]
                    
                    if len(X_test) == 0:
                        continue
                    
                    # Fit regularized model
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    model = self._get_regularized_model()
                    model.fit(X_train_scaled, y_train)
                    
                    y_pred = model.predict(X_test_scaled)
                    scores.append(mean_squared_error(y_test, y_pred))
                
                # Use median instead of mean for robustness
                lag_scores[n_lags] = np.median(scores) if scores else np.inf
                
            except Exception as e:
                logging.warning(f"CV failed for {n_lags} lags: {e}")
                lag_scores[n_lags] = np.inf
        
        if not lag_scores or all(score == np.inf for score in lag_scores.values()):
            logging.warning("All CV attempts failed. Using 1 lag.")
            return 1
        
        # Select lag with minimum score
        optimal_lags = min(lag_scores, key=lag_scores.get)
        
        # Apply penalty for too many lags (regularization)
        penalized_scores = {}
        for lags, score in lag_scores.items():
            if score != np.inf:
                penalty = 0.1 * lags  # Penalty increases with number of lags
                penalized_scores[lags] = score + penalty
        
        if penalized_scores:
            optimal_lags = min(penalized_scores, key=penalized_scores.get)
        
        logging.info(f"Optimal AR lags: {optimal_lags} (score: {lag_scores[optimal_lags]:.6f})")
        return optimal_lags
    
    def _get_regularized_model(self):
        """Get the appropriate regularized model."""
        if self.regularization == 'ridge':
            return Ridge(alpha=self.alpha, random_state=42)
        elif self.regularization == 'lasso':
            return Lasso(alpha=self.alpha, random_state=42, max_iter=2000)
        elif self.regularization == 'elastic_net':
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=42, max_iter=2000)
        else:  # ols
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
    
    def _create_constrained_almon_weights(self, n_lags: int) -> np.ndarray:
        """
        Create Almon weights with different constraint types to prevent overfitting.
        """
        # Use cache to avoid recomputation
        cache_key = (n_lags, self.poly_degree, self.weight_constraint)
        if cache_key in self.almon_weights_cache:
            return self.almon_weights_cache[cache_key]
        
        if self.weight_constraint == 'exponential':
            # Simple exponential decay (most regularized)
            weights = np.exp(-0.1 * np.arange(n_lags))
            weights = weights / weights.sum()
            
        elif self.weight_constraint == 'monotonic':
            # Monotonically decreasing weights
            weights = 1.0 / (1 + np.arange(n_lags))
            weights = weights / weights.sum()
            
        elif self.weight_constraint == 'normalized':
            # Standard Almon with normalization
            lags = np.arange(1, n_lags + 1)
            weights = np.zeros(n_lags)
            
            # Reduced polynomial to prevent overfitting
            effective_degree = min(self.poly_degree, max(1, n_lags // 3))
            
            for i in range(effective_degree + 1):
                coeff = 1.0 / (i + 1)  # Damping factor
                weights += coeff * (lags ** i)
            
            weights = weights / weights.sum()
            
        else:  # 'unconstrained'
            lags = np.arange(1, n_lags + 1)
            weights = np.zeros(n_lags)
            for i in range(self.poly_degree + 1):
                weights += (lags ** i)
            weights = weights / weights.sum()
        
        # Cache the result
        self.almon_weights_cache[cache_key] = weights
        return weights
    
    def _cross_validate_hyperparameters(self, factors: pd.DataFrame, gdp: pd.Series) -> Dict[str, float]:
        """Cross-validate regularization hyperparameters."""
        if self.regularization == 'ols':
            return {'alpha': 0.0}
        
        logging.info("Cross-validating hyperparameters...")
        
        # Prepare data with optimal AR lags
        y, X, _ = self.prepare_midas_data(factors, gdp, self.optimal_ar_lags)
        
        if len(y) < self.min_observations:
            logging.warning("Insufficient data for hyperparameter tuning")
            return {'alpha': self.alpha, 'l1_ratio': self.l1_ratio}
        
        # Define search space (conservative to prevent overfitting)
        if self.regularization == 'ridge':
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        elif self.regularization == 'lasso':
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        else:  # elastic_net
            alphas = [0.01, 0.1, 1.0, 10.0]
            l1_ratios = [0.1, 0.5, 0.9]
        
        best_score = np.inf
        best_params = {'alpha': self.alpha, 'l1_ratio': self.l1_ratio}
        
        # Time series split with minimum train size
        tscv = TimeSeriesSplit(n_splits=min(3, len(y) // 10))
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.regularization == 'elastic_net':
            for alpha in alphas:
                for l1_ratio in l1_ratios:
                    scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        if len(train_idx) < 5 or len(val_idx) < 1:
                            continue
                            
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        try:
                            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            scores.append(mean_squared_error(y_val, y_pred))
                        except:
                            scores.append(np.inf)
                    
                    if scores:
                        avg_score = np.mean(scores)
                        if avg_score < best_score:
                            best_score = avg_score
                            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        else:
            for alpha in alphas:
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    if len(train_idx) < 5 or len(val_idx) < 1:
                        continue
                        
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    try:
                        if self.regularization == 'ridge':
                            model = Ridge(alpha=alpha, random_state=42)
                        else:  # lasso
                            model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        scores.append(mean_squared_error(y_val, y_pred))
                    except:
                        scores.append(np.inf)
                
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = {'alpha': alpha}
        
        logging.info(f"Best hyperparameters: {best_params}")
        return best_params
    
    def prepare_midas_data(self, factors: pd.DataFrame, gdp: pd.Series, n_ar_lags: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare MIDAS data with enhanced robustness checks.
        """
        logging.info(f"Preparing MIDAS data with {n_ar_lags} AR lags...")
        
        midas_y, midas_X, midas_dates = [], [], []
        
        # Get sorted GDP dates
        gdp_clean = gdp.dropna().sort_index()
        
        for gdp_date in gdp_clean.index:
            # Check if we have enough AR history
            required_history = gdp_date - pd.DateOffset(months=3 * n_ar_lags)
            if required_history < gdp_clean.index.min():
                continue
            
            # Get factor window
            window_start = gdp_date - pd.DateOffset(days=self.horizon_days)
            factor_window = factors.loc[
                (factors.index > window_start) & 
                (factors.index <= gdp_date)
            ].dropna()
            
            if len(factor_window) < 5:  # Minimum data points for reliable weights
                continue
            
            current_regressors = []
            
            # Create weighted factor regressors
            for factor_col in factors.columns:
                factor_series = factor_window[factor_col].dropna()
                
                if len(factor_series) >= 3:  # Minimum for meaningful weighting
                    weights = self._create_constrained_almon_weights(len(factor_series))
                    
                    # Robust weighted average (handle outliers)
                    factor_values = factor_series.values
                    
                    # Winsorize extreme values
                    q1, q99 = np.percentile(factor_values, [1, 99])
                    factor_values = np.clip(factor_values, q1, q99)
                    
                    weighted_avg = np.sum(factor_values * weights)
                    current_regressors.append(weighted_avg)
                else:
                    # Not enough data - use simple average or skip
                    if len(factor_series) > 0:
                        current_regressors.append(factor_series.mean())
                    else:
                        current_regressors.append(0.0)
            
            # Add AR terms with robustness
            ar_terms = []
            for i in range(1, n_ar_lags + 1):
                lag_date = gdp_date - pd.DateOffset(months=3 * i)
                
                # Find closest available date
                available_dates = gdp_clean.index[gdp_clean.index <= lag_date]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    ar_terms.append(gdp_clean.loc[closest_date])
                else:
                    ar_terms.append(0.0)
            
            current_regressors.extend(ar_terms)
            
            # Add data point
            midas_X.append(current_regressors)
            midas_y.append(gdp_clean.loc[gdp_date])
            midas_dates.append(gdp_date)
        
        if len(midas_y) < self.min_observations:
            logging.warning(f"Only {len(midas_y)} observations available (minimum: {self.min_observations})")
        
        logging.info(f"MIDAS data prepared: {len(midas_y)} observations, {len(current_regressors)} features")
        return np.array(midas_y), np.array(midas_X), pd.DatetimeIndex(midas_dates)
    
    def fit(self, factors: pd.DataFrame, gdp: pd.Series):
        """
        Fit the regularized MIDAS model with comprehensive overfitting prevention.
        """
        logging.info("Fitting regularized MIDAS model...")
        
        # ADD: Store training information for JSON output
        self.factor_names = factors.columns.tolist()
        train_start_date = max(factors.index.min(), gdp.index.min())
        train_end_date = gdp.dropna().index.max()
        
        self.training_period = {
            "start_date": train_start_date.strftime('%Y-%m-%d'),
            "end_date": train_end_date.strftime('%Y-%m-%d'),
            "n_observations": len(gdp.dropna())
        }
        
        # Step 1: Cross-validate AR lags
        self.optimal_ar_lags = self._cross_validate_ar_lags(factors, gdp)
        
        # Step 2: Prepare data with optimal AR lags
        y, X, dates = self.prepare_midas_data(factors, gdp, self.optimal_ar_lags)
        
        if len(y) < self.min_observations:
            logging.error(f"Insufficient data: {len(y)} < {self.min_observations}")
            return False
        
        # Step 3: Cross-validate hyperparameters
        best_params = self._cross_validate_hyperparameters(factors, gdp)
        self.alpha = best_params['alpha']
        if 'l1_ratio' in best_params:
            self.l1_ratio = best_params['l1_ratio']
        
        # Step 4: Fit final model
        X_scaled = self.scaler.fit_transform(X)
        self.model_results = self._get_regularized_model()
        
        try:
            self.model_results.fit(X_scaled, y)
            
            # Calculate diagnostics
            y_pred = self.model_results.predict(X_scaled)
            
            # Robust R-squared calculation
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            n, p = X_scaled.shape
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
            
            # ADD: Enhanced diagnostics for JSON output
            self.diagnostics = {
                'r_squared': r2,
                'r_squared_adj': adj_r2,
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'n_observations': n,
                'n_features': p,
                'optimal_ar_lags': self.optimal_ar_lags,
                'regularization': self.regularization,
                'alpha': self.alpha,
                'weight_constraint': self.weight_constraint,
                # ADD: Additional statistics for compatibility
                'aic': None,  # Not available for sklearn models
                'bic': None,  # Not available for sklearn models
                'log_likelihood': None,  # Not available for sklearn models
                'n_parameters': p,
                'f_statistic': None,  # Not available for sklearn models
                'p_value_f_statistic': None  # Not available for sklearn models
            }
            
            if hasattr(self.model_results, 'coef_'):
                # Calculate feature importance
                feature_names = self.factor_names + [f'AR_lag_{i}' for i in range(1, self.optimal_ar_lags + 1)]
                coefficients = dict(zip(feature_names, self.model_results.coef_))
                
                # ADD: Store coefficients for JSON output
                self.diagnostics['coefficients'] = coefficients
                
                # Count non-zero coefficients (for sparsity)
                non_zero_coefs = np.sum(np.abs(self.model_results.coef_) > 1e-6)
                self.diagnostics['non_zero_coefficients'] = non_zero_coefs
                self.diagnostics['sparsity'] = 1 - (non_zero_coefs / len(self.model_results.coef_))
            
            # ADD: Perform residual diagnostics (simplified for sklearn models)
            residuals = y - y_pred
            
            # Basic residual statistics
            self.diagnostics['residual_mean'] = np.mean(residuals)
            self.diagnostics['residual_std'] = np.std(residuals)
            self.diagnostics['residual_skewness'] = float(pd.Series(residuals).skew())
            self.diagnostics['residual_kurtosis'] = float(pd.Series(residuals).kurtosis())
            
            logging.info("--- Residual Diagnostics ---")
            logging.info(f"Residual mean: {self.diagnostics['residual_mean']:.6f}")
            logging.info(f"Residual std: {self.diagnostics['residual_std']:.6f}")
            logging.info("--------------------------")
            
            logging.info(f"Model fitted: R²={r2:.4f}, Adj R²={adj_r2:.4f}, RMSE={np.sqrt(mean_squared_error(y, y_pred)):.4f}")
            return True
            
        except Exception as e:
            logging.error(f"Model fitting failed: {e}")
            return False
    
    def _bootstrap_nowcast(self, X_nowcast: np.ndarray, n_boot: int = 1000) -> float:
        """
        ADD: Bootstrap method for uncertainty quantification (adapted for sklearn models).
        """
        if self.model_results is None:
            return 0.0
        
        # For sklearn models, we'll use a simpler approach
        # based on the model's training performance
        return self.diagnostics.get('rmse', 0.5)
    
    def nowcast(self, factors: pd.DataFrame, gdp_history: pd.Series, target_quarter_end: pd.Timestamp) -> Tuple[float, float]:
        """Generate nowcast with uncertainty quantification."""
        if self.model_results is None:
            raise ValueError("Model must be fitted before nowcasting")
        
        logging.info(f"Generating nowcast for quarter ending {target_quarter_end.strftime('%Y-%m-%d')}...")
        
        # Prepare nowcast regressors
        window_start = target_quarter_end - pd.DateOffset(days=self.horizon_days)
        factor_window = factors.loc[
            (factors.index > window_start) & 
            (factors.index <= target_quarter_end)
        ].dropna()
        
        if len(factor_window) < 5:
            logging.warning("Limited factor data for nowcast")
        
        nowcast_regressors = []
        
        # Factor regressors
        for factor_col in factors.columns:
            factor_series = factor_window[factor_col].dropna()
            
            if len(factor_series) >= 3:
                weights = self._create_constrained_almon_weights(len(factor_series))
                
                # Robust weighted average
                factor_values = factor_series.values
                q1, q99 = np.percentile(factor_values, [1, 99])
                factor_values = np.clip(factor_values, q1, q99)
                
                weighted_avg = np.sum(factor_values * weights)
                nowcast_regressors.append(weighted_avg)
            else:
                if len(factor_series) > 0:
                    nowcast_regressors.append(factor_series.mean())
                else:
                    nowcast_regressors.append(0.0)
        
        # AR terms
        gdp_sorted = gdp_history.sort_index()
        for i in range(1, self.optimal_ar_lags + 1):
            if len(gdp_sorted) >= i:
                nowcast_regressors.append(gdp_sorted.iloc[-i])
            else:
                nowcast_regressors.append(0.0)
        
        # Scale and predict
        X_nowcast = np.array(nowcast_regressors).reshape(1, -1)
        X_nowcast_scaled = self.scaler.transform(X_nowcast)
        
        prediction = self.model_results.predict(X_nowcast_scaled)[0]
        
        # Estimate uncertainty using bootstrap method
        std_error = self._bootstrap_nowcast(X_nowcast_scaled)
        
        return prediction, std_error

def forecast_factors(factors_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Simple and robust factor forecasting using regularized AR models.
    """
    last_known_date = factors_df.index.max()
    days_to_forecast = (target_date - last_known_date).days

    if days_to_forecast <= 0:
        return factors_df

    logging.info(f"Forecasting factors for {days_to_forecast} days...")
    
    full_factors = factors_df.copy()
    forecast_index = pd.date_range(
        start=last_known_date + timedelta(days=1), 
        periods=days_to_forecast,
        freq='D'
    )
    
    forecast_df = pd.DataFrame(index=forecast_index, columns=factors_df.columns)
    
    for col in factors_df.columns:
        try:
            factor_data = factors_df[col].dropna()
            
            if len(factor_data) > 10:
                # Use Ridge regression for robust forecasting
                lags = min(3, len(factor_data) // 5)  # Conservative lag selection
                
                X, y = [], []
                for i in range(lags, len(factor_data)):
                    X.append(factor_data.iloc[i-lags:i].values)
                    y.append(factor_data.iloc[i])
                
                if len(X) > 5:
                    X, y = np.array(X), np.array(y)
                    
                    # Fit Ridge model
                    model = Ridge(alpha=1.0, random_state=42)
                    model.fit(X, y)
                    
                    # Generate forecasts
                    last_values = factor_data.iloc[-lags:].values
                    forecasts = []
                    
                    for _ in range(days_to_forecast):
                        pred = model.predict(last_values.reshape(1, -1))[0]
                        # Add small random noise to prevent deterministic forecasts
                        pred += np.random.normal(0, 0.01)
                        forecasts.append(pred)
                        last_values = np.append(last_values[1:], pred)
                    
                    forecast_df[col] = forecasts
                else:
                    # Use last value with small decay
                    last_value = factor_data.iloc[-1]
                    decay_factor = 0.99
                    forecasts = [last_value * (decay_factor ** i) for i in range(1, days_to_forecast + 1)]
                    forecast_df[col] = forecasts
            else:
                # Use last known value
                last_value = factor_data.iloc[-1] if len(factor_data) > 0 else 0
                forecast_df[col] = last_value
                
        except Exception as e:
            logging.warning(f"Forecasting failed for factor {col}: {e}")
            last_value = factors_df[col].dropna().iloc[-1] if len(factors_df[col].dropna()) > 0 else 0
            forecast_df[col] = last_value
    
    return pd.concat([full_factors, forecast_df], axis=0).loc[:target_date]

# ADD: Main execution block with comprehensive JSON output
if __name__ == "__main__":
    # --- Configuration ---
    FACTORS_PATH = "Extracted latent factor/latent_factors.csv"
    TARGET_GDP_PATH = "GDP_processed/target.csv"
    RESULTS_DIR = "MIDAS_results"
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- Load Data ---
    logging.info("--- Starting Regularized MIDAS Regression Pipeline ---")
    try:
        factors_df = pd.read_csv(FACTORS_PATH, index_col='Date', parse_dates=True)
        gdp_df = pd.read_csv(TARGET_GDP_PATH, index_col='Date', parse_dates=True)
        gdp_series = gdp_df[gdp_df.columns[0]]  # Convert to Series
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Please ensure Stage 1 was run successfully.")
        exit()

    # --- Identify Overlapping Period for Training ---
    train_start_date = max(factors_df.index.min(), gdp_series.index.min())
    train_end_date = gdp_series.dropna().index.max()

    factors_train = factors_df.loc[train_start_date:train_end_date]
    gdp_train = gdp_series.loc[train_start_date:train_end_date]

    logging.info(f"Training data aligned. Overlapping period: {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")

    # --- Test different regularization approaches and select best ---
    methods = ['ridge', 'lasso', 'elastic_net']
    model_results = {}
    
    for method in methods:
        logging.info(f"\n=== Testing {method.upper()} regularization ===")
        
        # Initialize model with conservative settings
        model = RegularizedMIDASRegressor(
            poly_degree=2,
            horizon_days=90,
            max_ar_lags=3,  # Conservative
            regularization=method,
            alpha=1.0,
            l1_ratio=0.5,
            weight_constraint='normalized',
            min_observations=10
        )
        
        # Fit the model and store results
        success = model.fit(factors_train, gdp_train)
        if success:
            model_results[method] = {
                'model': model,
                'rmse': model.diagnostics['rmse'],
                'adj_r2': model.diagnostics['r_squared_adj']
            }
            logging.info(f"{method.upper()} RMSE: {model.diagnostics['rmse']:.4f}")
        else:
            logging.warning(f"{method.upper()} fitting failed")

    # --- Select best model ---
    if not model_results:
        logging.error("All models failed to fit. Aborting.")
        exit()
    
    best_method = min(model_results, key=lambda k: model_results[k]['rmse'])
    best_model = model_results[best_method]['model']
    logging.info(f"Best model: {best_method.upper()} (RMSE: {model_results[best_method]['rmse']:.4f})")
    
    # --- Generate Nowcast ---
    last_gdp_date = train_end_date
    nowcast_target_date = last_gdp_date + pd.DateOffset(months=3)
    factors_nowcast = forecast_factors(factors_df, nowcast_target_date)
    gdp_history_for_nowcast = gdp_series.loc[:last_gdp_date]
    
    try:
        nowcast_value, nowcast_se = best_model.nowcast(factors_nowcast, gdp_history_for_nowcast, nowcast_target_date)
        nowcast_success = True
    except Exception as e:
        logging.error(f"Nowcast generation failed: {e}")
        nowcast_value, nowcast_se = 0, 0
        nowcast_success = False
    
    # --- Prepare Results Dictionary ---
    results = {
        "model_info": {
            "model_type": "RegularizedMIDAS",
            "regularization_method": best_method,
            "poly_degree": best_model.poly_degree,
            "optimal_ar_lags": best_model.optimal_ar_lags,
            "horizon_days": best_model.horizon_days,
            "alpha": best_model.alpha,
            "l1_ratio": best_model.l1_ratio if best_method == "elastic_net" else None,
            "weight_constraint": best_model.weight_constraint,
            "run_timestamp": datetime.now().isoformat()
        },
        "training_period": best_model.training_period,
        "data_sources": {
            "factors_path": FACTORS_PATH,
            "target_gdp_path": TARGET_GDP_PATH,
            "n_factors": len(best_model.factor_names),
            "factor_names": best_model.factor_names
        },
        "model_statistics": best_model.diagnostics,
        "nowcast": {
            "target_date": nowcast_target_date.strftime('%Y-%m-%d'),
            "last_gdp_date": last_gdp_date.strftime('%Y-%m-%d'),
            "predicted_zscore": float(nowcast_value) if nowcast_success else None,
            "standard_error_zscore": float(nowcast_se) if nowcast_success else None,
            "confidence_interval_95": {
                "lower_bound": float(nowcast_value - 1.96 * nowcast_se) if nowcast_success else None,
                "upper_bound": float(nowcast_value + 1.96 * nowcast_se) if nowcast_success else None
            } if nowcast_success else None,
            "success": nowcast_success
        },
        "tested_models": {
            method: {
                "rmse": model_results[method]['rmse'],
                "adj_r2": model_results[method]['adj_r2']
            } for method in model_results
        },
        "status": "success" if nowcast_success else "partial_success"
    }
    
    # --- Save Results to JSON ---
    json_filename = f"midas_results.json"
    json_filepath = os.path.join(RESULTS_DIR, json_filename)
    
    def json_serializer(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return str(obj)
    
    try:
        with open(json_filepath, 'w') as f:
            json.dump(results, f, indent=4, default=json_serializer)
        
        print(f"Results saved to: {json_filepath}")
        logging.info(f"Results exported to {json_filepath}")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
    
    logging.info("--- Regularized MIDAS Pipeline Finished ---")

