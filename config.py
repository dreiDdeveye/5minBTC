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

# Polymarket
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_SECRET = os.getenv("POLYMARKET_SECRET", "")
POLYMARKET_FUNDER = os.getenv("POLYMARKET_FUNDER", "")
POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Prediction horizon
PREDICTION_HORIZON_MIN = 15  # predict 15 minutes ahead (longer horizon = stronger signal)

# Feature engineering
FEATURE_WINDOW = 5          # compute features every 5 closed candles (entry timing)
LOOKBACK_EMA_LONG = 21
LOOKBACK_ATR = 7
LOOKBACK_RSI = 7
LOOKBACK_VOL_AVG = 20
LOOKBACK_BOLLINGER = 20

# Model — binary OVER/UNDER threshold
SIGNAL_THRESHOLD = 0.50       # P(up) > this => OVER, else => UNDER
MODEL_DIR = Path(__file__).parent / "model" / "artifacts"
WALK_FORWARD_TRAIN_SIZE = 2016   # ~7 days of 5-min blocks
WALK_FORWARD_TEST_SIZE = 288     # ~1 day of 5-min blocks

# Feature columns used by the model
FEATURE_COLS = [
    "return_1m", "return_3m", "return_5m",
    "volume_ratio_20", "volume_spike",
    "atr_7", "atr_pct", "bollinger_width", "bollinger_pos",
    "ema_cross", "rsi_7",
    "order_flow_imbalance",
    "price_acceleration",
    # Binance Futures features
    "funding_rate",
    "oi_change_pct",
    "long_short_ratio",
    "taker_buy_ratio",
    "whale_ratio",
    "whale_net_flow",
    # Polymarket features
    "poly_up_price",
    "poly_spread",
]

# Model hyperparameters
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_MIN_CHILD_WEIGHT = 3
XGB_EARLY_STOPPING = 20

# Auto-retrain after this many new feature rows with targets
AUTO_RETRAIN_THRESHOLD = 288  # ~1 day of 5-min rows
