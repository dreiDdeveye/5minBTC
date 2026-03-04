"""
BTC Mecha Predictor - Main Entry Point
Runs: Binance WebSocket collector + FastAPI dashboard concurrently.
"""
import asyncio
import logging

import uvicorn

from dashboard.app import app
from ingestion.collector import run as run_collector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting BTC Mecha Predictor...")

    # Dashboard server config
    uvi_config = uvicorn.Config(
        app, host="0.0.0.0", port=8000, log_level="info"
    )
    server = uvicorn.Server(uvi_config)

    # Run collector and dashboard concurrently
    await asyncio.gather(
        run_collector(),
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
