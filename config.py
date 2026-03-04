import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Binance WebSocket
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
KLINE_STREAM = "btcusdt@kline_1m"
DEPTH_STREAM = "btcusdt@depth20@100ms"

# Binance REST
BINANCE_REST_BASE = "https://api.binance.com"
BINANCE_FUTURES_REST_BASE = "https://fapi.binance.com"

# Binance aggTrade stream
AGGTRADE_STREAM = "btcusdt@aggTrade"

# Whale detection
WHALE_THRESHOLD_BTC = 1.0  # trades >= 1 BTC considered "large"

# Feature engineering
FEATURE_WINDOW = 5          # compute features every 5 closed candles
LOOKBACK_EMA_LONG = 21
LOOKBACK_ATR = 7
LOOKBACK_RSI = 7
LOOKBACK_VOL_AVG = 20
LOOKBACK_BOLLINGER = 20

# Model
SIGNAL_THRESHOLD = 0.58
MODEL_DIR = Path(__file__).parent / "model" / "artifacts"
WALK_FORWARD_TRAIN_SIZE = 2016   # ~7 days of 5-min blocks
WALK_FORWARD_TEST_SIZE = 288     # ~1 day of 5-min blocks

# Feature columns used by the model
FEATURE_COLS = [
    "return_1m", "return_3m", "return_5m",
    "volume_ratio_20", "volume_spike",
    "atr_7", "bollinger_width",
    "ema_cross", "rsi_7",
    "order_flow_imbalance",
    # Binance Futures features
    "funding_rate",
    "oi_change_pct",
    "long_short_ratio",
    "taker_buy_ratio",
    "whale_ratio",
    "whale_net_flow",
]
