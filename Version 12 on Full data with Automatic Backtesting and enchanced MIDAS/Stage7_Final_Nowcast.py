import json
import os
from datetime import datetime
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_json_file(filepath: str) -> dict:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
        logging.info(f"Successfully loaded {filepath}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {filepath}: {e}")
        raise

def convert_zscore_to_yoy_growth(zscore: float, mean: float, std: float) -> float:
    """
    Convert standardized z-score back to original YoY growth rate.
    
    Args:
        zscore (float): The standardized z-score prediction
        mean (float): The mean used during standardization
        std (float): The standard deviation used during standardization
    
    Returns:
        float: The YoY growth rate as a percentage
    """
    # Convert z-score back to original scale
    original_value = zscore * std + mean
    return original_value

def calculate_confidence_interval_yoy(lower_zscore: float, upper_zscore: float, 
                                    mean: float, std: float) -> dict:
    """
    Convert confidence interval z-scores to YoY growth rates.
    
    Args:
        lower_zscore (float): Lower bound z-score
        upper_zscore (float): Upper bound z-score
        mean (float): The mean used during standardization
        std (float): The standard deviation used during standardization
    
    Returns:
        dict: Dictionary with lower and upper bounds in YoY growth rates
    """
    lower_yoy = convert_zscore_to_yoy_growth(lower_zscore, mean, std)
    upper_yoy = convert_zscore_to_yoy_growth(upper_zscore, mean, std)
    
    return {
        "lower_bound": lower_yoy,
        "upper_bound": upper_yoy
    }

def create_final_prediction_results(midas_results: dict, target_metadata: dict) -> dict:
    """
    Create the final prediction results by combining MIDAS results with GDP metadata.
    
    Args:
        midas_results (dict): Results from MIDAS regression
        target_metadata (dict): GDP processing metadata with standardization parameters
    
    Returns:
        dict: Final prediction results with YoY growth rates
    """
    # Extract standardization parameters
    standardization_params = target_metadata.get("standardization_parameters", {})
    mean = standardization_params.get("mean")
    std = standardization_params.get("std")
    
    if mean is None or std is None:
        raise ValueError("Standardization parameters (mean, std) not found in target metadata")
    
    # Extract nowcast results
    nowcast = midas_results.get("nowcast", {})
    if not nowcast:
        raise ValueError("Nowcast data not found in MIDAS results")
    
    predicted_zscore = nowcast.get("predicted_zscore")
    standard_error_zscore = nowcast.get("standard_error_zscore")
    confidence_interval = nowcast.get("confidence_interval_95", {})
    
    if predicted_zscore is None:
        raise ValueError("Predicted z-score not found in nowcast results")
    
    # Convert z-score to YoY growth rate
    predicted_yoy_growth = convert_zscore_to_yoy_growth(predicted_zscore, mean, std)
    
    # Convert standard error to YoY scale
    standard_error_yoy = standard_error_zscore * std if standard_error_zscore else None
    
    # Convert confidence interval to YoY growth rates
    confidence_interval_yoy = None
    if confidence_interval:
        lower_bound = confidence_interval.get("lower_bound")
        upper_bound = confidence_interval.get("upper_bound")
        if lower_bound is not None and upper_bound is not None:
            confidence_interval_yoy = calculate_confidence_interval_yoy(
                lower_bound, upper_bound, mean, std
            )
    
    # Create final results dictionary
    final_results = {
        "prediction_info": {
            "model_type": midas_results.get("model_info", {}).get("model_type", "MIDAS"),
            "target_date": nowcast.get("target_date"),
            "last_gdp_date": nowcast.get("last_gdp_date"),
            "prediction_timestamp": datetime.now().isoformat(),
            "status": midas_results.get("status", "unknown")
        },
        "nowcast_results": {
            "predicted_yoy_growth_percent": round(predicted_yoy_growth, 4),
            "standard_error_yoy_percent": round(standard_error_yoy, 4) if standard_error_yoy else None,
            "confidence_interval_95_percent": {
                "lower_bound": round(confidence_interval_yoy["lower_bound"], 4),
                "upper_bound": round(confidence_interval_yoy["upper_bound"], 4)
            } if confidence_interval_yoy else None
        },
        "raw_zscore_results": {
            "predicted_zscore": predicted_zscore,
            "standard_error_zscore": standard_error_zscore,
            "confidence_interval_95_zscore": confidence_interval
        },
        "standardization_info": {
            "mean": mean,
            "std": std,
            "min": standardization_params.get("min"),
            "max": standardization_params.get("max"),
            "count": standardization_params.get("count")
        },
        "model_performance": {
            "r_squared": midas_results.get("model_statistics", {}).get("r_squared"),
            "adj_r_squared": midas_results.get("model_statistics", {}).get("adj_r_squared"),
            "aic": midas_results.get("model_statistics", {}).get("aic"),
            "bic": midas_results.get("model_statistics", {}).get("bic"),
            "n_observations": midas_results.get("model_statistics", {}).get("n_observations")
        },
        "data_info": {
            "training_period": midas_results.get("training_period", {}),
            "n_factors": midas_results.get("data_sources", {}).get("n_factors"),
            "factor_names": midas_results.get("data_sources", {}).get("factor_names")
        }
    }
    
    return final_results

def main():
    """Main function to process MIDAS results and create final GDP nowcast."""
    
    # Configuration
    MIDAS_RESULTS_PATH = "MIDAS_results/midas_results.json"
    TARGET_METADATA_PATH = "GDP_processed/target_metadata.json"
    OUTPUT_DIR = "Final_Results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.info("--- Starting Final GDP Nowcast Conversion ---")
    
    try:
        # Load input files
        logging.info("Loading MIDAS results and GDP metadata...")
        midas_results = load_json_file(MIDAS_RESULTS_PATH)
        target_metadata = load_json_file(TARGET_METADATA_PATH)
        
        # Create final prediction results
        logging.info("Converting z-scores to YoY growth rates...")
        final_results = create_final_prediction_results(midas_results, target_metadata)
        
        # Display results
        print("\n" + "="*60)
        print("         FINAL GDP NOWCAST RESULTS")
        print("="*60)
        
        nowcast = final_results["nowcast_results"]
        prediction_info = final_results["prediction_info"]
        
        print(f"Target Quarter End Date: {prediction_info['target_date']}")
        print(f"Based on Data Through: {prediction_info['last_gdp_date']}")
        print(f"Model Type: {prediction_info['model_type']}")
        print("-" * 60)
        
        print(f"Predicted YoY GDP Growth: {nowcast['predicted_yoy_growth_percent']:.2f}%")
        
        if nowcast['standard_error_yoy_percent']:
            print(f"Standard Error: ±{nowcast['standard_error_yoy_percent']:.2f}%")
        
        if nowcast['confidence_interval_95_percent']:
            ci = nowcast['confidence_interval_95_percent']
            print(f"95% Confidence Interval: [{ci['lower_bound']:.2f}%, {ci['upper_bound']:.2f}%]")
        
        print("-" * 60)
        
        # Model performance
        performance = final_results["model_performance"]
        if performance.get('adj_r_squared'):
            print(f"Model Adjusted R²: {performance['adj_r_squared']:.4f}")
        if performance.get('n_observations'):
            print(f"Training Observations: {performance['n_observations']}")
        
        print("="*60)
        
        # Save final results
        output_filename = f"final_gdp_nowcast.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_filepath, 'w') as f:
            json.dump(final_results, f, indent=4)
        
        print(f"\nFinal results saved to: {output_filepath}")
        logging.info(f"Final results exported to {output_filepath}")
        
        # Also save a summary file
        summary_filename = f"gdp_nowcast_summary.json"
        summary_filepath = os.path.join(OUTPUT_DIR, summary_filename)
        
        summary = {
            "gdp_nowcast_summary": {
                "target_date": prediction_info['target_date'],
                "predicted_yoy_growth_percent": nowcast['predicted_yoy_growth_percent'],
                "confidence_interval_95_percent": nowcast['confidence_interval_95_percent'],
                "model_type": prediction_info['model_type'],
                "prediction_date": prediction_info['prediction_timestamp']
            }
        }
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Summary results saved to: {summary_filepath}")
        logging.info("--- Final GDP Nowcast Conversion Complete ---")
        
    except Exception as e:
        logging.error(f"Error in final prediction conversion: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()