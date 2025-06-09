import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Optional

# --- Setup Logging ---
# Sets up a logger to monitor the script's execution.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === New Function for Optimal Factor Determination using BIC (Fully Adaptive) ===
def determine_optimal_factors_by_bic(
    df_panel: pd.DataFrame,
    max_k: int
) -> Tuple[Optional[int], Optional[int], Optional[int], Dict[int, float]]:
    """
    Determines the optimal model specification (k_factors, factor_order, error_order)
    by fitting multiple DFM models and selecting the one with the lowest BIC.

    This version is fully adaptive: it tries a sequence of models from most
    to least complex and selects the first one that converges for each 'k'.
    """
    logging.info(f"Starting search for optimal factors (max_k={max_k}) using BIC...")
    results_store = {} # Store results: {k: {'bic': val, 'spec': (k, fo, eo)}}

    max_k_adjusted = min(max_k, df_panel.shape[1] - 1)
    if max_k_adjusted < 1:
        logging.error("Cannot determine factors; panel has fewer than 2 variables.")
        return None, None, None, {}

    # Define model specifications to try, from most to least complex
    specs_to_try = [(1, 1), (1, 0), (0, 1), (0, 0)] # (factor_order, error_order)

    for k in range(1, max_k_adjusted + 1):
        logging.info(f"  Testing with k_factors = {k}...")
        
        for factor_order, error_order in specs_to_try:
            try:
                model = sm.tsa.DynamicFactor(
                    df_panel, k_factors=k, factor_order=factor_order, error_order=error_order
                )
                results = model.fit(disp=False, maxiter=500)
                
                if results.mle_retvals['converged']:
                    spec = (k, factor_order, error_order)
                    results_store[k] = {'bic': results.bic, 'spec': spec}
                    logging.info(f"    k={k}, spec=(fo={factor_order},eo={error_order}) | Converged. BIC = {results.bic:.2f}")
                    break # Success, move to the next k
            except (np.linalg.LinAlgError, ValueError) as e:
                logging.warning(f"    k={k}, spec=(fo={factor_order},eo={error_order}) | Failed: {type(e).__name__}. Retrying with simpler model.")
        else: # This 'else' belongs to the for loop, runs if 'break' was not hit
            logging.error(f"    k={k} | All model specifications failed to converge.")

    if not results_store:
        logging.error("Could not determine optimal factors. No candidate models converged.")
        return None, None, None, {}

    # Find the k that corresponds to the minimum BIC score among successful models
    optimal_k = min(results_store, key=lambda k: results_store[k]['bic'])
    _, optimal_factor_order, optimal_error_order = results_store[optimal_k]['spec']
    
    logging.info(f"Completed BIC search. Optimal spec: k={optimal_k}, factor_order={optimal_factor_order}, error_order={optimal_error_order} (lowest BIC).")
    return optimal_k, optimal_factor_order, optimal_error_order, results_store


# === Main DFM Estimation and Visualization Function (Rectified) ===
def fit_dfm_and_visualize(panel_filepath: str, panel_name: str, max_factors_to_test: int = 5) -> None:
    """
    Loads a panel, automatically determines the optimal model specifications using BIC
    on raw data, fits the final DFM, and visualizes the results.
    """
    logging.info(f"--- Starting DFM for '{panel_name}' Panel ---")
    
    try:
        df_panel = pd.read_csv(panel_filepath, index_col='Date', parse_dates=True)
        df_panel = df_panel.asfreq('D')
        logging.info(f"Loaded panel from '{panel_filepath}'. Shape: {df_panel.shape}")
    except Exception as e:
        logging.error(f"Failed to load '{panel_filepath}': {e}")
        return

    # Determine Optimal Model Specifications
    optimal_k, optimal_fo, optimal_eo, _ = determine_optimal_factors_by_bic(df_panel, max_k=max_factors_to_test)
    
    if optimal_k is None:
        logging.error(f"Halting process for '{panel_name}'.")
        return

    # Fit the final Dynamic Factor Model with the optimal specifications
    logging.info(f"Fitting final model for '{panel_name}' with k_factors={optimal_k}, factor_order={optimal_fo}, error_order={optimal_eo}...")
    try:
        final_model = sm.tsa.DynamicFactor(df_panel, k_factors=optimal_k, factor_order=optimal_fo, error_order=optimal_eo)
        final_results = final_model.fit(disp=True, maxiter=2000)
    except Exception as e:
        logging.error(f"Final DFM estimation failed for '{panel_name}': {e}")
        return

    # Extract, Export, and Visualize the Latent Factor(s)
    try:
        # **Correction for shape and type errors**: Ensure factors are a correctly shaped DataFrame
        smoothed_factors = final_results.factors.smoothed
        
        if isinstance(smoothed_factors, np.ndarray):
            # The shape from statsmodels is (k, T), we need (T, k)
            # We must TRANSPOSE the array with .T
            factor_cols = [f'factor_{i+1}' for i in range(smoothed_factors.shape[0])]
            smoothed_factors_df = pd.DataFrame(
                smoothed_factors.T,
                index=df_panel.index,
                columns=factor_cols
            )
        else: # Already a DataFrame
            smoothed_factors_df = smoothed_factors

        factor_filename = f"Extracted_Factors_{panel_name.replace(' ', '_')}.csv"
        smoothed_factors_df.to_csv(factor_filename)
        logging.info(f"Extracted {smoothed_factors_df.shape[1]} latent factor(s) saved to '{factor_filename}'")
        
        # Plotting the results
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in smoothed_factors_df.columns:
            smoothed_factors_df[col].plot(ax=ax, label=str(col).capitalize(), lw=2.5)
            
        ax.set_title(f'Latent Common Factor(s): {panel_name} (k={optimal_k})', fontsize=18, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Factor Value (Standardized)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=1.2)
        ax.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error in post-processing for '{panel_name}': {e}")

# === Main Execution Block ===
if __name__ == '__main__':
    panels_to_model = {
        "Agricultural": "Agricultural_Panel.csv",
        "Industrial Activity": "Industrial_Activity_Panel.csv",
        "Financial Sentiment": "Financial_Sentiment_Panel.csv",
        "Consumer Demand": "Consumer_Demand_Panel.csv"
    }
    
    for name, filepath in panels_to_model.items():
        try:
            fit_dfm_and_visualize(panel_filepath=filepath, panel_name=name)
            print("\n" + "="*80 + "\n")
        except FileNotFoundError:
            logging.warning(f"File '{filepath}' not found. Skipping panel '{name}'.")
