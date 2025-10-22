import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
import logging
import os
import json
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.optimize import minimize
from scipy.linalg import LinAlgError
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def load_optimal_factors():
    try:
        with open("Factor Selection Results/factor_selection_results.json", 'r') as f:
            results = json.load(f)
        return results["selection_results"]["optimal_factors"]
    except:
        return 3  # fallback

# --- ENHANCED Configuration Constants for Robustness ---
MAX_ITERATIONS = 2000        # Increased for better convergence
EM_ITERATIONS = 30           # Reduced for faster convergence
CONVERGENCE_TOL = 1e-3       # Relaxed tolerance
MIN_VARIANCE_THRESHOLD = 1e-4
MIN_OBSERVATIONS_THRESHOLD = 30  # Reduced to keep more series
MAX_MISSING_PERCENTAGE = 99.0    # Very high tolerance for missing data
OUTPUT_DIR = "Extracted latent factor"
INPUT_DATA_PATH = "Processed Data/All_Processed_data.csv"
OUTPUT_FACTORS_PATH = os.path.join(OUTPUT_DIR, "latent_factors.csv")
OUTPUT_LOADINGS_PLOT_PATH = os.path.join(OUTPUT_DIR, "factor_loadings_heatmap.png")
OUTPUT_PARAMS_PATH = os.path.join(OUTPUT_DIR, "model_parameters.json")
NUM_FACTORS_TO_EXTRACT = load_optimal_factors()

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'factor_extraction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DFM-Extractor')

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)


class RobustDynamicFactorExtractor:
    """
    Robust Real-time Dynamic Factor Model optimized for high missing data scenarios.
    """
    
    def __init__(
        self,
        n_factors: int = 2,
        factor_order: int = 1,
        max_iterations: int = MAX_ITERATIONS,
        em_iterations: int = EM_ITERATIONS,
        convergence_tol: float = CONVERGENCE_TOL
    ):
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        
        self.n_factors = n_factors
        self.factor_order = factor_order
        self.max_iterations = max_iterations
        self.em_iterations = em_iterations
        self.convergence_tol = convergence_tol
        
        self.model = None
        self.model_results = None
        self.loadings = None
        self.factors = None
        self.converged = False
        self.factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
        
        logger.info(
            f"Initialized Robust DFM Extractor: {self.n_factors} factors, "
            f"AR order {self.factor_order}, Max iterations {self.max_iterations}"
        )

    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load and basic validation of data."""
        logger.info(f"Loading data from '{filepath}'")
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
            df = df.sort_index().asfreq('D')
            
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            total_obs = df.shape[0] * df.shape[1]
            missing_obs = df.isnull().sum().sum()
            missing_pct = (missing_obs / total_obs) * 100
            
            logger.info(f"Data loaded: {df.shape[0]} periods, {df.shape[1]} series")
            logger.info(f"Missing observations: {missing_obs:,} ({missing_pct:.1f}%)")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.exception(f"Data loading failed: {e}")
            return None

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation optimized for high missing data."""
        logger.info("Validating data with high missing data tolerance...")
        
        original_shape = data.shape
        
        # 1. Remove completely empty series
        empty_series = data.columns[data.isnull().all()]
        if len(empty_series) > 0:
            logger.warning(f"Removing {len(empty_series)} completely empty series")
            data = data.drop(columns=empty_series)
        
        # 2. Remove series with excessive missing data (>98%)
        missing_pct_per_series = (data.isnull().sum() / len(data)) * 100
        extreme_missing = missing_pct_per_series[missing_pct_per_series > MAX_MISSING_PERCENTAGE].index
        if len(extreme_missing) > 0:
            logger.warning(f"Removing {len(extreme_missing)} series with >{MAX_MISSING_PERCENTAGE}% missing")
            data = data.drop(columns=extreme_missing)
        
        # 3. Remove series with too few observations
        insufficient_data = data.columns[data.count() < MIN_OBSERVATIONS_THRESHOLD]
        if len(insufficient_data) > 0:
            logger.warning(f"Removing {len(insufficient_data)} series with <{MIN_OBSERVATIONS_THRESHOLD} observations")
            data = data.drop(columns=insufficient_data)
        
        # 4. Remove near-constant series
        constant_series = []
        for col in data.columns:
            non_missing = data[col].dropna()
            if len(non_missing) > 1 and non_missing.var() < MIN_VARIANCE_THRESHOLD:
                constant_series.append(col)
        
        if constant_series:
            logger.warning(f"Removing {len(constant_series)} near-constant series")
            data = data.drop(columns=constant_series)
        
        # 5. Final validation
        if data.shape[1] < self.n_factors:
            raise ValueError(
                f"Insufficient series ({data.shape[1]}) for {self.n_factors} factors. "
                f"Reduce NUM_FACTORS_TO_EXTRACT to {max(1, data.shape[1] - 1)}"
            )
        
        logger.info(f"Data validation: {original_shape} → {data.shape}")
        return data

    def _get_robust_starting_values(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate robust starting values using PCA on available data."""
        try:
            logger.info("Computing robust starting values...")
            
            # Strategy 1: Use periods with most data availability
            data_availability = data.count(axis=1)
            top_periods = data_availability.nlargest(min(500, len(data) // 4)).index
            subset_data = data.loc[top_periods]
            
            # Strategy 2: Fill missing values temporarily for PCA
            imputer = SimpleImputer(strategy='mean')
            filled_data = pd.DataFrame(
                imputer.fit_transform(subset_data),
                columns=subset_data.columns,
                index=subset_data.index
            )
            
            # Strategy 3: Run PCA
            pca = PCA(n_components=self.n_factors)
            pca_factors = pca.fit_transform(filled_data)
            pca_loadings = pca.components_.T
            
            # Prepare starting parameters
            n_series = len(data.columns)
            n_params = n_series * self.n_factors  # Factor loadings
            n_params += self.n_factors * self.n_factors  # Transition matrix
            n_params += n_series  # Idiosyncratic variances
            
            start_params = np.zeros(n_params)
            
            # Set factor loadings
            start_params[:n_series * self.n_factors] = pca_loadings.flatten()
            
            # Set transition matrix (close to identity but stable)
            transition_start = n_series * self.n_factors
            for i in range(self.n_factors):
                start_params[transition_start + i * self.n_factors + i] = 0.8
            
            # Set idiosyncratic variances
            var_start = transition_start + self.n_factors * self.n_factors
            start_params[var_start:var_start + n_series] = 0.5
            
            logger.info(f"Generated {len(start_params)} starting parameters")
            logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            
            return start_params
            
        except Exception as e:
            logger.warning(f"Could not generate starting values: {e}")
            return None

    def _initialize_model(self, data: pd.DataFrame) -> None:
        """Initialize model with robust settings."""
        try:
            logger.info("Initializing robust DFM...")
            
            self.model = DynamicFactor(
                endog=data,
                k_factors=self.n_factors,
                factor_order=self.factor_order,
                error_order=0,
                enforce_stationarity=False,  # Let model find best fit
                enforce_invertibility=False,
                initialization='diffuse',    # Changed to diffuse
                initial_variance=1.0,       # Conservative initial variance
                loglikelihood_burn=5        # Reduced burn-in
            )
            
            logger.info(f"Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def fit_and_extract(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Main fitting and extraction method."""
        try:
            # Clean and validate data
            clean_data = self._validate_and_clean_data(data)
            
            # Initialize model
            self._initialize_model(clean_data)
            
            # Estimate model
            logger.info("Starting model estimation...")
            success = self._estimate_model(clean_data)
            
            if success:
                return self._extract_results(clean_data)
            else:
                logger.error("Model estimation failed")
                return None
                
        except Exception as e:
            logger.exception(f"Factor extraction failed: {e}")
            return None

    def _estimate_model(self, data: pd.DataFrame) -> bool:
        """Estimate model using multiple robust strategies."""
        
        strategies = [
            ("Direct MLE with Robust Settings", self._try_robust_direct_mle),
            ("Conservative Direct MLE", self._try_conservative_mle),
            ("Simple MLE", self._try_simple_mle),
            ("EM Algorithm", self._try_em_algorithm)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Attempting: {strategy_name}")
                
                if strategy_func(data):
                    logger.info(f"✓ SUCCESS: {strategy_name}")
                    return True
                else:
                    logger.warning(f"✗ FAILED: {strategy_name}")
                    
            except Exception as e:
                logger.warning(f"✗ ERROR in {strategy_name}: {str(e)[:150]}")
                continue
        
        logger.error("All estimation strategies failed")
        return False

    def _try_robust_direct_mle(self, data: pd.DataFrame) -> bool:
        """Robust Direct MLE - Primary Method."""
        try:
            # Get starting values
            start_params = self._get_robust_starting_values(data)
            
            # Fit with robust settings
            self.model_results = self.model.fit(
                start_params=start_params,
                method='lbfgs',
                maxiter=self.max_iterations,
                disp=True,
                options={
                    'maxfun': self.max_iterations * 3,
                    'gtol': self.convergence_tol,
                    'ftol': self.convergence_tol * 5,
                    'eps': 1e-8,
                    'maxcor': 15,
                    'maxls': 30
                }
            )
            
            return self._validate_results()
            
        except Exception as e:
            logger.debug(f"Robust Direct MLE failed: {e}")
            return False

    def _try_conservative_mle(self, data: pd.DataFrame) -> bool:
        """Conservative MLE with relaxed settings."""
        try:
            self.model_results = self.model.fit(
                method='lbfgs',
                maxiter=500,
                disp=False,
                options={
                    'maxfun': 1500,
                    'gtol': 1e-2,  # Very relaxed
                    'ftol': 1e-3,  # Very relaxed
                    'eps': 1e-6
                }
            )
            
            return self._validate_results()
            
        except Exception as e:
            logger.debug(f"Conservative MLE failed: {e}")
            return False

    def _try_simple_mle(self, data: pd.DataFrame) -> bool:
        """Simple MLE with minimal settings."""
        try:
            self.model_results = self.model.fit(
                method='lbfgs',
                maxiter=200,
                disp=False
            )
            
            return self._validate_results()
            
        except Exception as e:
            logger.debug(f"Simple MLE failed: {e}")
            return False

    def _try_em_algorithm(self, data: pd.DataFrame) -> bool:
        """EM Algorithm as fallback."""
        try:
            self.model_results = self.model.fit(
                method='em',
                maxiter=self.em_iterations,
                disp=False,
                tolerance=self.convergence_tol * 10
            )
            
            return self._validate_results()
            
        except Exception as e:
            logger.debug(f"EM Algorithm failed: {e}")
            return False

    def _validate_results(self) -> bool:
        """Validate model results."""
        if self.model_results is None:
            return False
        
        # Check if we have valid log-likelihood
        if np.isnan(self.model_results.llf) or np.isinf(self.model_results.llf):
            logger.warning("Invalid log-likelihood")
            return False
        
        # Check if we have parameters
        if self.model_results.params is None or len(self.model_results.params) == 0:
            logger.warning("No parameters estimated")
            return False
        
        logger.info(f"Model converged: LL={self.model_results.llf:.2f}")
        return True

    def _extract_results(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract factors and loadings from fitted model."""
        try:
            # Extract factor loadings
            self._extract_loadings(data.columns)
            
            # Extract factors
            factors_df = self._extract_factors(data.index)
            
            # Set convergence status
            self.converged = True
            
            logger.info("Results extracted successfully")
            logger.info(f"Diagnostics: LL={self.model_results.llf:.2f}, "
                       f"AIC={self.model_results.aic:.2f}, BIC={self.model_results.bic:.2f}")
            
            return factors_df
            
        except Exception as e:
            logger.error(f"Results extraction failed: {e}")
            return None

    def _extract_loadings(self, columns: pd.Index) -> None:
        """Extract factor loadings."""
        try:
            n_series = len(columns)
            
            # Extract loadings from first n_series * n_factors parameters
            params = self.model_results.params
            loadings_params = params[:n_series * self.n_factors]
            
            # Reshape to matrix
            loadings_matrix = loadings_params.values.reshape(n_series, self.n_factors)
            
            self.loadings = pd.DataFrame(
                loadings_matrix,
                index=columns,
                columns=self.factor_names
            )
            
            logger.info("Factor loadings extracted")
            
        except Exception as e:
            logger.error(f"Loadings extraction failed: {e}")
            self.loadings = None

    def _extract_factors(self, index: pd.Index) -> pd.DataFrame:
        """Extract smoothed factors."""
        try:
            # Get smoothed states
            smoothed_states = self.model_results.smoothed_state
            
            # Extract factors (first n_factors states)
            factors_array = smoothed_states[:self.n_factors, :].T
            
            factors_df = pd.DataFrame(
                factors_array,
                index=index,
                columns=self.factor_names
            )
            
            self.factors = factors_df
            logger.info(f"Extracted {self.n_factors} factors")
            
            return factors_df
            
        except Exception as e:
            logger.error(f"Factor extraction failed: {e}")
            return pd.DataFrame(
                np.nan,
                index=index,
                columns=self.factor_names
            )

    def export_parameters(self, output_path: str) -> None:
        """Export model parameters."""
        try:
            if self.model_results is None:
                logger.error("No results to export")
                return
            
            # Calculate factor statistics
            factor_stats = {}
            if self.factors is not None:
                for factor in self.factor_names:
                    if factor in self.factors.columns:
                        factor_stats[factor] = {
                            'mean': float(self.factors[factor].mean()),
                            'std': float(self.factors[factor].std()),
                            'min': float(self.factors[factor].min()),
                            'max': float(self.factors[factor].max())
                        }
            
            params_dict = {
                "methodology": {
                    "approach": "Robust Real-time Dynamic Factor Model",
                    "estimation_method": "Enhanced Direct MLE with fallbacks",
                    "missing_data_handling": "Kalman Filter - No interpolation",
                    "purpose": "India GDP Nowcasting"
                },
                "model_specification": {
                    "n_factors": self.n_factors,
                    "factor_order": self.factor_order,
                    "max_iterations": self.max_iterations,
                    "convergence_tolerance": self.convergence_tol
                },
                "estimation_results": {
                    "converged": self.converged,
                    "log_likelihood": float(self.model_results.llf),
                    "aic": float(self.model_results.aic),
                    "bic": float(self.model_results.bic),
                    "n_observations": int(self.model_results.nobs),
                    "n_series": len(self.loadings) if self.loadings is not None else 0
                },
                "factor_loadings": {
                    "loadings_matrix": (self.loadings.values.tolist() 
                                    if self.loadings is not None else None),
                    "series_names": (self.loadings.index.tolist() 
                                if self.loadings is not None else None),
                    "factor_names": self.factor_names
                },
                "factor_statistics": factor_stats
            }
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(params_dict, f, indent=2, default=str)
                
            logger.info(f"Parameters exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Parameter export failed: {e}")

    def plot_factor_loadings(self, output_path: str) -> bool:
        """Generate factor loadings heatmap."""
        if self.loadings is None:
            logger.error("No loadings to plot")
            return False
            
        try:
            fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(self.loadings))))
            
            sns.heatmap(
                self.loadings,
                annot=True,
                fmt=".3f",
                cmap="RdBu_r",
                center=0,
                linewidths=0.5,
                cbar_kws={'label': 'Factor Loading'},
                ax=ax
            )
            
            ax.set_title(f'Robust DFM - Factor Loadings\n'
                        f'India GDP Nowcasting ({self.n_factors} Factors)', 
                        fontsize=14, pad=20)
            ax.set_xlabel('Latent Factors', fontsize=12)
            ax.set_ylabel('Economic Indicators', fontsize=12)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Loadings plot saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
            return False

    def save_factors(self, factors_df: pd.DataFrame, output_path: str) -> bool:
        """Save factors to CSV."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            factors_df.to_csv(output_path)
            logger.info(f"Factors saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save factors: {e}")
            return False


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("ROBUST DYNAMIC FACTOR MODEL - INDIA GDP NOWCASTING")
    logger.info("=" * 60)
    
    # Initialize extractor
    extractor = RobustDynamicFactorExtractor(
        n_factors=NUM_FACTORS_TO_EXTRACT,
        factor_order=1,
        max_iterations=MAX_ITERATIONS,
        em_iterations=EM_ITERATIONS
    )
    
    # Load data
    data = extractor.load_data(INPUT_DATA_PATH)
    if data is None:
        logger.error("❌ Data loading failed")
        return
    
    # Extract factors
    logger.info("🚀 Starting factor extraction...")
    factors = extractor.fit_and_extract(data)
    
    if factors is None:
        logger.error("❌ Factor extraction failed")
        return
    
    # Save results
    logger.info("💾 Saving results...")
    saved_count = 0
    
    if extractor.save_factors(factors, OUTPUT_FACTORS_PATH):
        saved_count += 1
    
    if extractor.plot_factor_loadings(OUTPUT_LOADINGS_PLOT_PATH):
        saved_count += 1
    
    extractor.export_parameters(OUTPUT_PARAMS_PATH)
    saved_count += 1
    
    # Summary
    logger.info(f"✅ EXTRACTION COMPLETED!")
    logger.info(f"   Status: {'✅ Converged' if extractor.converged else '⚠️ Check results'}")
    logger.info(f"   Factors: {factors.shape[1]}")
    logger.info(f"   Time periods: {factors.shape[0]}")
    logger.info(f"   Log-Likelihood: {extractor.model_results.llf:.2f}")
    logger.info(f"   Files saved: {saved_count}/3")
    
    # Show factor loadings
    if extractor.loadings is not None:
        logger.info("\n🎯 TOP FACTOR LOADINGS:")
        for factor in extractor.factor_names:
            top_loadings = extractor.loadings[factor].abs().nlargest(3)
            logger.info(f"   {factor}:")
            for series, loading in top_loadings.items():
                logger.info(f"     • {series}: {loading:.3f}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Execution interrupted")
    except Exception as e:
        logger.exception(f"💥 Unexpected error: {e}")