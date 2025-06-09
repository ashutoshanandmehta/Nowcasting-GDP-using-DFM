import os
import logging
import pandas as pd
import numpy as np
from midaspy.model import MIDASRegressor

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NowcastPipeline:
    """
    A custom wrapper around the midaspy library to create a robust nowcasting pipeline,
    incorporating all the data cleaning and validation logic.
    """

    def __init__(self, target_file, factor_files, hf_lag=66, y_lag=1):
        """
        Initializes the pipeline with file paths and model parameters.
        """
        self.target_file = target_file
        self.factor_files = factor_files
        self.hf_lag = hf_lag
        self.y_lag = y_lag

        # These will be populated later
        self.X_df = None
        self.y_df = None
        self.model_results = None
        self.nowcast_value = None

    def _check_file_exists(self, file_path):
        if not os.path.isfile(file_path):
            logging.error(f"The file {file_path} does not exist!")
            raise FileNotFoundError(f"File {file_path} not found.")

    def _prepare_data(self):
        """
        Loads, merges, and cleans all data.
        """
        logging.info("Pipeline Step 1: Loading and Cleaning Data")

        # Validate factor files and load them
        list_of_factor_dfs = []
        for prefix, f_name in self.factor_files.items():
            self._check_file_exists(f_name)
            try:
                temp_df = pd.read_csv(f_name, parse_dates=['Date'], index_col='Date')
            except Exception as e:
                logging.error(f"Error reading {f_name}: {e}")
                raise

            # Ensure columns with prefix 'factor' exist before renaming
            rename_dict = {col: f"{prefix}_{col}" for col in temp_df.columns if col.startswith('factor')}
            if not rename_dict:
                logging.warning(f"No factor columns found in {f_name}.")
            temp_df.rename(columns=rename_dict, inplace=True)
            list_of_factor_dfs.append(temp_df)

        # Merge all factor dataframes
        self.X_df = pd.concat(list_of_factor_dfs, axis=1)
        self.X_df.sort_index(inplace=True)

        # Adjust for ragged edges: drop rows before the latest start date across all factor files
        latest_start_date = max(df.index.min() for df in list_of_factor_dfs)
        self.X_df = self.X_df[self.X_df.index >= latest_start_date].copy()
        logging.info(f"Daily data aligned, starting from {latest_start_date.date()}.")

        # Robustly fill missing values (first forward then backward)
        self.X_df.ffill(inplace=True)
        self.X_df.bfill(inplace=True)
        logging.info("Successfully cleaned and imputed all missing values.")

        # Load target variable data
        self._check_file_exists(self.target_file)
        try:
            self.y_df = pd.read_csv(self.target_file, parse_dates=['Date'])
            self.y_df.set_index('Date', inplace=True)
        except Exception as e:
            logging.error(f"Error reading target file {self.target_file}: {e}")
            raise

        # Find the first valid target observation where enough history is available
        valid_dates = self.y_df.index[self.y_df.index.map(lambda d: len(self.X_df[self.X_df.index < d]) >= self.hf_lag)]
        if valid_dates.empty:
            logging.error("Not enough history for any target observation.")
            raise ValueError("Not enough history for any target observation.")
        first_valid_y_date = valid_dates.min()

        self.y_df = self.y_df[self.y_df.index >= first_valid_y_date]
        logging.info(f"Quarterly data trimmed, starting from {first_valid_y_date.date()}.")

    def train(self):
        """
        Trains the nowcasting model.
        """
        logging.info("Pipeline Step 2: Training Nowcasting Model")
        try:
            model = MIDASRegressor(
                endog=self.y_df['QSAAR_GDP_Growth'],
                exog=self.X_df,
                ylag=self.y_lag,
                xlag=self.hf_lag,
                horizon=0,  # Crucial for nowcasting
                poly='beta'
            )
            self.model_results = model.fit()
            logging.info("Model training complete.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def nowcast(self):
        """
        Generates and stores the nowcast value.
        """
        logging.info("Pipeline Step 3: Generating Nowcast")
        if self.model_results is None:
            logging.error("Model is not trained. Please run train() first.")
            raise RuntimeError("Model not trained")
        try:
            prediction = self.model_results.predict()
            # Ensure prediction has at least one value
            if prediction.empty:
                logging.error("Prediction failed: No predicted values obtained.")
                raise ValueError("Empty prediction")
            self.nowcast_value = prediction.values[0].item()
            logging.info("Nowcast generation complete.")
        except Exception as e:
            logging.error(f"Error during nowcast prediction: {e}")
            raise

    def get_summary(self):
        """
        Presents a summary of the model's performance.
        """
        if self.model_results is None:
            logging.info("Model has not been trained yet. Please run the pipeline first.")
            return

        try:
            logging.info("\n--- Model Summary ---")
            adj_r2 = self.model_results.adj_r_squared()
            logging.info(f"Adjusted R-squared: {adj_r2:.4f}")
            interpretation = (
                "Interpretation: The model has some explanatory power."
                if adj_r2 >= 0 else "Interpretation: The model is a poor fit for the data."
            )
            logging.info(interpretation)
            logging.info("\nModel Parameters:")
            logging.info(self.model_results.params)
        except Exception as e:
            logging.error(f"Error generating model summary: {e}")
            raise

    def run(self):
        """
        Executes the full pipeline in order.
        """
        try:
            self._prepare_data()
            self.train()
            self.nowcast()
            return self.nowcast_value
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise


# --- How to Use Our New Custom Pipeline ---
if __name__ == "__main__":
    TARGET_FILE = 'target.csv'
    FACTOR_FILES = {
        'Cons': 'Extracted_Factors_Consumer_Demand.csv',
        'Fin': 'Extracted_Factors_Financial_Sentiment.csv',
        'Ind': 'Extracted_Factors_Industrial_Activity.csv'
    }

    try:
        midas_pipeline = NowcastPipeline(target_file=TARGET_FILE, factor_files=FACTOR_FILES)
        final_nowcast = midas_pipeline.run()
        logging.info(f"\nFINAL NOWCAST FOR LATEST QUARTER: {final_nowcast:.4f}")
        midas_pipeline.get_summary()
    except Exception as pipeline_error:
        logging.error(f"Pipeline terminated with an error: {pipeline_error}")
