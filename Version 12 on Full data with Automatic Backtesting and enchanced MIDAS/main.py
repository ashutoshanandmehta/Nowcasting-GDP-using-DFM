import argparse
import logging
import yaml
from data_loader import DataLoader
from nowcast_pipeline import NowcastPipeline
from backtester import Backtester

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

def main():
    parser = argparse.ArgumentParser(description="GDP Nowcasting Backtest")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config['log_path'])

    data_loader = DataLoader(config)
    gdp = data_loader.gdp
    evaluation_quarters = gdp.index[config['training_quarters']:]

    pipeline = NowcastPipeline(config)
    backtester = Backtester(config)
    backtester.data_loader = data_loader
    backtester.pipeline = pipeline
    backtester.run(evaluation_quarters)
    backtester.evaluate_and_plot()

if __name__ == "__main__":
    main() 