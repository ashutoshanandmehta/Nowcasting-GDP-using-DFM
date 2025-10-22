import pandas as pd
import os
import logging

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.gdp = self.load_gdp()
        self.indicators = self.load_indicators()
        self.metadata = self.load_metadata()

    def load_gdp(self):
        path = self.config['gdp_path']
        if not os.path.exists(path):
            raise FileNotFoundError(f"GDP file not found: {path}")
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        if df.isnull().any().any():
            self.logger.warning("GDP data contains missing values.")
        return df.squeeze()

    def load_indicators(self):
        path = self.config['indicator_path']
        if not os.path.exists(path):
            raise FileNotFoundError(f"Indicator file not found: {path}")
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        if df.isnull().any().any():
            self.logger.warning("Indicator data contains missing values.")
        return df

    def load_metadata(self):
        path = self.config['metadata_path']
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path)

    def get_training_data(self, current_quarter):
        # Return GDP and indicators up to but not including current_quarter
        gdp_train = self.gdp[self.gdp.index < current_quarter]
        indicators_train = self.indicators[self.indicators.index < current_quarter]
        return gdp_train, indicators_train

    def get_actual_gdp(self, current_quarter):
        return self.gdp.loc[current_quarter] if current_quarter in self.gdp.index else None 