"""
Run walk-forward backtest on historical data.

Usage:
    python -m scripts.run_backtest
"""
import logging
import json

from model.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    result = run_backtest()
    print("\n=== BACKTEST RESULTS ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
