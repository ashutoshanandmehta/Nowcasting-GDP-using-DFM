import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
import logging
import os
import json
from typing import Optional, Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.optimize import minimize
from scipy.linalg import LinAlgError
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import time
from datetime import datetime
import sys
from io import StringIO
import contextlib

# --- Configuration Constants ---
MAX_ITERATIONS = 2000
EM_ITERATIONS = 30
CONVERGENCE_TOL = 1e-3
MIN_VARIANCE_THRESHOLD = 1e-4
MIN_OBSERVATIONS_THRESHOLD = 30
MAX_MISSING_PERCENTAGE = 99
INPUT_DATA_PATH = "Processed Data/All_Processed_data.csv"
OUTPUT_DIR = "Factor Selection Results"
SELECTION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "factor_selection_results.json")
COMPARISON_PLOT_PATH = os.path.join(OUTPUT_DIR, "factor_selection_comparison.png")

# Factor range to test
MIN_FACTORS = 1
MAX_FACTORS = 5

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'factor_selection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DFM-Factor-Selection')

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)


class FactorSelectionEngine:
    """
    Automated factor selection engine for Dynamic Factor Models.
    Tests multiple factor configurations and selects optimal based on information criteria.
    """
    
    def __init__(
        self,
        min_factors: int = MIN_FACTORS,
        max_factors: int = MAX_FACTORS,
        factor_order: int = 1,
        max_iterations: int = MAX_ITERATIONS,
        em_iterations: int = EM_ITERATIONS,
        convergence_tol: float = CONVERGENCE_TOL
    ):
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.factor_order = factor_order
        self.max_iterations = max_iterations
        self.em_iterations = em_iterations
        self.convergence_tol = convergence_tol
        
        self.results = {}
        self.optimal_factors = None
        self.selection_criteria = 'AIC'  # Default selection criterion
        
        logger.info(f"Factor Selection Engine initialized")
        logger.info(f"Testing factors: {min_factors} to {max_factors}")
    
    @contextlib.contextmanager
    def _capture_optimizer_output(self):
        """Capture L-BFGS-B optimizer output for detailed logging."""
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            yield captured_output
        finally:
            sys.stdout = old_stdout
    
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
        
        # 2. Remove series with excessive missing data
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
        
        logger.info(f"Data validation: {original_shape} → {data.shape}")
        return data
    
    def _get_robust_starting_values(self, data: pd.DataFrame, n_factors: int) -> Optional[np.ndarray]:
        """Generate robust starting values using PCA."""
        try:
            logger.debug(f"Computing starting values for {n_factors} factors...")
            
            # Use periods with most data availability
            data_availability = data.count(axis=1)
            top_periods = data_availability.nlargest(min(500, len(data) // 4)).index
            subset_data = data.loc[top_periods]
            
            # Fill missing values temporarily for PCA
            imputer = SimpleImputer(strategy='mean')
            filled_data = pd.DataFrame(
                imputer.fit_transform(subset_data),
                columns=subset_data.columns,
                index=subset_data.index
            )
            
            # Run PCA
            pca = PCA(n_components=n_factors)
            pca_factors = pca.fit_transform(filled_data)
            pca_loadings = pca.components_.T
            
            # Log PCA explained variance
            logger.info(f"PCA explained variance for {n_factors} factors: {pca.explained_variance_ratio_[:n_factors].sum():.3f}")
            
            # Prepare starting parameters
            n_series = len(data.columns)
            n_params = n_series * n_factors  # Factor loadings
            n_params += n_factors * n_factors  # Transition matrix
            n_params += n_series  # Idiosyncratic variances
            
            start_params = np.zeros(n_params)
            
            # Set factor loadings
            start_params[:n_series * n_factors] = pca_loadings.flatten()
            
            # Set transition matrix (close to identity but stable)
            transition_start = n_series * n_factors
            for i in range(n_factors):
                start_params[transition_start + i * n_factors + i] = 0.8
            
            # Set idiosyncratic variances
            var_start = transition_start + n_factors * n_factors
            start_params[var_start:var_start + n_series] = 0.5
            
            return start_params
            
        except Exception as e:
            logger.debug(f"Could not generate starting values for {n_factors} factors: {e}")
            return None
    
    def _fit_model(self, data: pd.DataFrame, n_factors: int) -> Dict[str, Any]:
        """Fit DFM for specific number of factors."""
        logger.info(f"Fitting DFM with {n_factors} factors...")
        
        result = {
            'n_factors': n_factors,
            'converged': False,
            'log_likelihood': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'n_observations': 0,
            'n_series': len(data.columns),
            'estimation_time': 0.0,
            'method_used': None,
            'error_message': None
        }
        
        start_time = time.time()
        
        try:
            # Initialize model
            model = DynamicFactor(
                endog=data,
                k_factors=n_factors,
                factor_order=self.factor_order,
                error_order=0,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization='diffuse',
                initial_variance=1.0,
                loglikelihood_burn=5
            )
            
            # Try multiple estimation strategies
            strategies = [
                ("Robust MLE", self._try_robust_mle),
                ("Conservative MLE", self._try_conservative_mle),
                ("Simple MLE", self._try_simple_mle),
                ("EM Algorithm", self._try_em_algorithm)
            ]
            
            model_results = None
            method_used = None
            
            for strategy_name, strategy_func in strategies:
                try:
                    logger.debug(f"  Trying: {strategy_name}")
                    model_results = strategy_func(model, data, n_factors)
                    
                    if model_results is not None and self._validate_results(model_results):
                        method_used = strategy_name
                        logger.debug(f"  ✓ SUCCESS: {strategy_name}")
                        break
                    else:
                        logger.debug(f"  ✗ FAILED: {strategy_name}")
                        
                except Exception as e:
                    logger.debug(f"  ✗ ERROR in {strategy_name}: {str(e)[:100]}")
                    continue
            
            if model_results is not None and method_used is not None:
                result.update({
                    'converged': True,
                    'log_likelihood': float(model_results.llf),
                    'aic': float(model_results.aic),
                    'bic': float(model_results.bic),
                    'n_observations': int(model_results.nobs),
                    'method_used': method_used
                })
                
                logger.info(f"Model converged: LL={model_results.llf:.2f}")
                logger.info(f"✓ SUCCESS: {method_used}")
                logger.info(f"Diagnostics: LL={model_results.llf:.2f}, AIC={model_results.aic:.2f}, BIC={model_results.bic:.2f}")
                logger.info(f"  ## SUCCESS: {n_factors} factors | LL={model_results.llf:.2f} | AIC={model_results.aic:.2f} | BIC={model_results.bic:.2f}")
            else:
                result['error_message'] = "All estimation methods failed"
                logger.warning(f"  ## FAILED: {n_factors} factors - All methods failed")
                logger.warning(f"  Last attempt details: Check convergence tolerances and data quality")
                
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"  ## ERROR: {n_factors} factors - {str(e)[:100]}")
        
        result['estimation_time'] = time.time() - start_time
        return result
    
    def _try_robust_mle(self, model: DynamicFactor, data: pd.DataFrame, n_factors: int):
        """Robust MLE estimation with detailed logging."""
        start_params = self._get_robust_starting_values(data, n_factors)
        
        logger.info(f"Starting L-BFGS-B optimization for {n_factors} factors...")
        
        with self._capture_optimizer_output() as captured:
            result = model.fit(
                start_params=start_params,
                method='lbfgs',
                maxiter=self.max_iterations,
                disp=True,  # Enable optimizer output
                options={
                    'maxfun': self.max_iterations * 3,
                    'gtol': self.convergence_tol,
                    'ftol': self.convergence_tol * 5,
                    'eps': 1e-8,
                    'disp': True  # Enable detailed output
                }
            )
        
        # Log the captured L-BFGS-B output
        optimizer_output = captured.getvalue()
        if optimizer_output.strip():
            logger.info(f"L-BFGS-B Optimization Details for {n_factors} factors:")
            for line in optimizer_output.strip().split('\n'):
                logger.info(f"  {line}")
        
        return result
    
    def _try_conservative_mle(self, model: DynamicFactor, data: pd.DataFrame, n_factors: int):
        """Conservative MLE estimation with detailed logging."""
        logger.info(f"Starting Conservative L-BFGS-B optimization for {n_factors} factors...")
        
        with self._capture_optimizer_output() as captured:
            result = model.fit(
                method='lbfgs',
                maxiter=500,
                disp=True,
                options={
                    'maxfun': 1500,
                    'gtol': 1e-2,
                    'ftol': 1e-3,
                    'eps': 1e-6,
                    'disp': True
                }
            )
        
        # Log the captured L-BFGS-B output
        optimizer_output = captured.getvalue()
        if optimizer_output.strip():
            logger.info(f"Conservative L-BFGS-B Optimization Details for {n_factors} factors:")
            for line in optimizer_output.strip().split('\n'):
                logger.info(f"  {line}")
        
        return result
    
    def _try_simple_mle(self, model: DynamicFactor, data: pd.DataFrame, n_factors: int):
        """Simple MLE estimation with detailed logging."""
        logger.info(f"Starting Simple L-BFGS-B optimization for {n_factors} factors...")
        
        with self._capture_optimizer_output() as captured:
            result = model.fit(
                method='lbfgs',
                maxiter=200,
                disp=True
            )
        
        # Log the captured L-BFGS-B output
        optimizer_output = captured.getvalue()
        if optimizer_output.strip():
            logger.info(f"Simple L-BFGS-B Optimization Details for {n_factors} factors:")
            for line in optimizer_output.strip().split('\n'):
                logger.info(f"  {line}")
        
        return result
    
    def _try_em_algorithm(self, model: DynamicFactor, data: pd.DataFrame, n_factors: int):
        """EM algorithm estimation."""
        logger.info(f"Starting EM algorithm for {n_factors} factors...")
        return model.fit(
            method='em',
            maxiter=self.em_iterations,
            disp=False,
            tolerance=self.convergence_tol * 10
        )
    
    def _validate_results(self, model_results) -> bool:
        """Validate model results."""
        if model_results is None:
            return False
        
        # Check if we have valid log-likelihood
        if np.isnan(model_results.llf) or np.isinf(model_results.llf):
            return False
        
        # Check if we have parameters
        if model_results.params is None or len(model_results.params) == 0:
            return False
        
        return True
    
    def run_factor_selection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run factor selection across all specified factor counts."""
        logger.info("🚀 Starting automated factor selection...")
        
        # Clean data once
        clean_data = self._validate_and_clean_data(data)
        
        # Check if we have enough series for maximum factors
        max_testable = min(self.max_factors, clean_data.shape[1] - 1)
        if max_testable < self.max_factors:
            logger.warning(f"Reducing max factors from {self.max_factors} to {max_testable} due to data constraints")
            self.max_factors = max_testable
        
        # Fit models for each factor count
        for n_factors in range(self.min_factors, self.max_factors + 1):
            if n_factors >= clean_data.shape[1]:
                logger.warning(f"Skipping {n_factors} factors - not enough series")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"FITTING MODEL WITH {n_factors} FACTORS")
            logger.info(f"{'='*60}")
            
            self.results[n_factors] = self._fit_model(clean_data, n_factors)
        
        # Select optimal number of factors
        self._select_optimal_factors()
        
        # Prepare final results
        selection_results = {
            'selection_timestamp': datetime.now().isoformat(),
            'data_info': {
                'n_periods': len(clean_data),
                'n_series': len(clean_data.columns),
                'date_range': {
                    'start': clean_data.index.min().isoformat(),
                    'end': clean_data.index.max().isoformat()
                }
            },
            'factor_range_tested': {
                'min_factors': self.min_factors,
                'max_factors': self.max_factors
            },
            'model_results': self.results,
            'selection_results': {
                'optimal_factors': self.optimal_factors,
                'selection_criterion': self.selection_criteria,
                'selection_rationale': self._get_selection_rationale()
            },
            'convergence_summary': self._get_convergence_summary()
        }
        
        return selection_results
    
    def _select_optimal_factors(self) -> None:
        """Select optimal number of factors based on information criteria."""
        logger.info("## Selecting optimal number of factors...")
        
        # Filter converged results
        converged_results = {k: v for k, v in self.results.items() if v['converged']}
        
        if not converged_results:
            logger.error("No models converged successfully")
            self.optimal_factors = None
            return
        
        # Primary selection: AIC
        aic_values = {k: v['aic'] for k, v in converged_results.items()}
        optimal_aic = min(aic_values, key=aic_values.get)
        
        # Secondary selection: BIC
        bic_values = {k: v['bic'] for k, v in converged_results.items()}
        optimal_bic = min(bic_values, key=bic_values.get)
        
        # Use AIC as primary criterion
        self.optimal_factors = optimal_aic
        self.selection_criteria = 'AIC'
        
        logger.info(f"## OPTIMAL FACTORS SELECTED: {self.optimal_factors}")
        logger.info(f"   Based on AIC: {self.optimal_factors} factors (AIC={aic_values[optimal_aic]:.2f})")
        logger.info(f"   Based on BIC: {optimal_bic} factors (BIC={bic_values[optimal_bic]:.2f})")
        
        # Show comparison
        logger.info("\n## FACTOR COMPARISON:")
        for n_factors in sorted(converged_results.keys()):
            result = converged_results[n_factors]
            marker = "👑" if n_factors == self.optimal_factors else "  "
            logger.info(f"{marker} {n_factors} factors: LL={result['log_likelihood']:.2f}, "
                       f"AIC={result['aic']:.2f}, BIC={result['bic']:.2f}")
    
    def _get_selection_rationale(self) -> str:
        """Get rationale for factor selection."""
        if self.optimal_factors is None:
            return "No models converged successfully"
        
        converged_results = {k: v for k, v in self.results.items() if v['converged']}
        
        if len(converged_results) == 1:
            return f"Only {self.optimal_factors} factor model converged successfully"
        
        optimal_result = converged_results[self.optimal_factors]
        aic_value = optimal_result['aic']
        
        # Find second best
        other_aic = {k: v['aic'] for k, v in converged_results.items() if k != self.optimal_factors}
        if other_aic:
            second_best = min(other_aic, key=other_aic.get)
            improvement = other_aic[second_best] - aic_value
            return f"Selected {self.optimal_factors} factors with AIC={aic_value:.2f}, " \
                   f"improvement of {improvement:.2f} over {second_best} factors"
        
        return f"Selected {self.optimal_factors} factors with AIC={aic_value:.2f}"
    
    def _get_convergence_summary(self) -> Dict[str, Any]:
        """Get summary of convergence results."""
        total_tested = len(self.results)
        converged = sum(1 for r in self.results.values() if r['converged'])
        
        return {
            'total_models_tested': total_tested,
            'models_converged': converged,
            'convergence_rate': converged / total_tested if total_tested > 0 else 0,
            'failed_models': [k for k, v in self.results.items() if not v['converged']],
            'successful_models': [k for k, v in self.results.items() if v['converged']]
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """Save selection results to JSON."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"## Results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def plot_comparison(self, output_path: str) -> bool:
        """Create comparison plot of factor selection criteria."""
        try:
            converged_results = {k: v for k, v in self.results.items() if v['converged']}
            
            if len(converged_results) < 2:
                logger.warning("Not enough converged models for comparison plot")
                return False
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            factors = sorted(converged_results.keys())
            log_likelihoods = [converged_results[f]['log_likelihood'] for f in factors]
            aics = [converged_results[f]['aic'] for f in factors]
            bics = [converged_results[f]['bic'] for f in factors]
            estimation_times = [converged_results[f]['estimation_time'] for f in factors]
            
            # Plot 1: Log-Likelihood
            ax1.plot(factors, log_likelihoods, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Factors')
            ax1.set_ylabel('Log-Likelihood')
            ax1.set_title('Log-Likelihood by Number of Factors')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: AIC and BIC
            ax2.plot(factors, aics, 'o-', label='AIC', linewidth=2, markersize=8)
            ax2.plot(factors, bics, 's-', label='BIC', linewidth=2, markersize=8)
            ax2.axvline(x=self.optimal_factors, color='red', linestyle='--', alpha=0.7, label=f'Optimal ({self.optimal_factors})')
            ax2.set_xlabel('Number of Factors')
            ax2.set_ylabel('Information Criterion')
            ax2.set_title('AIC and BIC Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Estimation Time
            ax3.bar(factors, estimation_times, alpha=0.7)
            ax3.set_xlabel('Number of Factors')
            ax3.set_ylabel('Estimation Time (seconds)')
            ax3.set_title('Estimation Time by Number of Factors')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Model Comparison Table
            ax4.axis('tight')
            ax4.axis('off')
            
            table_data = []
            for f in factors:
                r = converged_results[f]
                marker = "✓" if f == self.optimal_factors else ""
                table_data.append([
                    f"{marker} {f}",
                    f"{r['log_likelihood']:.2f}",
                    f"{r['aic']:.2f}",
                    f"{r['bic']:.2f}",
                    f"{r['estimation_time']:.1f}s"
                ])
            
            table = ax4.table(
                cellText=table_data,
                colLabels=['Factors', 'Log-Likelihood', 'AIC', 'BIC', 'Time'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.set_title('Model Comparison Summary')
            
            plt.suptitle('Dynamic Factor Model - Factor Selection Analysis', fontsize=16, y=0.98)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"## Comparison plot saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create comparison plot: {e}")
            return False


def main() -> None:
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("DYNAMIC FACTOR MODEL - AUTOMATED FACTOR SELECTION")
    logger.info("=" * 70)
    
    # Initialize factor selection engine
    selector = FactorSelectionEngine(
        min_factors=MIN_FACTORS,
        max_factors=MAX_FACTORS,
        max_iterations=MAX_ITERATIONS,
        em_iterations=EM_ITERATIONS
    )
    
    # Load data
    logger.info("## Loading data...")
    data = selector.load_data(INPUT_DATA_PATH)
    if data is None:
        logger.error("## Data loading failed")
        return
    
    # Run factor selection
    logger.info("## Running factor selection analysis...")
    results = selector.run_factor_selection(data)
    
    # Save results
    logger.info("## Saving results...")
    saved_json = selector.save_results(results, SELECTION_RESULTS_PATH)
    saved_plot = selector.plot_comparison(COMPARISON_PLOT_PATH)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("## FACTOR SELECTION COMPLETED!")
    logger.info("=" * 70)
    
    if selector.optimal_factors is not None:
        logger.info(f"## OPTIMAL NUMBER OF FACTORS: {selector.optimal_factors}")
        logger.info(f"   Selection criterion: {selector.selection_criteria}")
        logger.info(f"   Selection rationale: {selector._get_selection_rationale()}")
    else:
        logger.error("## No optimal factors selected - check convergence issues")
    
    # Convergence summary
    summary = selector._get_convergence_summary()
    logger.info(f"\n## CONVERGENCE SUMMARY:")
    logger.info(f"   Models tested: {summary['total_models_tested']}")
    logger.info(f"   Models converged: {summary['models_converged']}")
    logger.info(f"   Convergence rate: {summary['convergence_rate']:.1%}")
    
    if summary['failed_models']:
        logger.info(f"   Failed models: {summary['failed_models']}")
    
    # File outputs
    logger.info(f"\n OUTPUT FILES:")
    logger.info(f"   Results JSON: {'✅' if saved_json else '❌'} {SELECTION_RESULTS_PATH}")
    logger.info(f"   Comparison plot: {'✅' if saved_plot else '❌'} {COMPARISON_PLOT_PATH}")
    
    # Integration note
    logger.info(f"\n INTEGRATION NOTE:")
    logger.info(f"   The results are saved in '{SELECTION_RESULTS_PATH}'")
    logger.info(f"   Your Stage 4 script can import this file to use the optimal factor count")
    logger.info(f"   Recommended: NUM_FACTORS_TO_EXTRACT = {selector.optimal_factors}")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️")