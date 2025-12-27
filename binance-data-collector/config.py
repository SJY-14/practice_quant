from datetime import datetime, timedelta, UTC

# API Endpoints
FUTURES_BASE_URL = "https://fapi.binance.com"
SPOT_BASE_URL = "https://api.binance.com"

# Symbols
SYMBOLS = {
    "futures": "BTCUSDT",  # Perpetual
    "spot": "BTCUSDT"
}

# Timeframe
INTERVAL = "5m"

# Data period
END_TIME = datetime.now(UTC).replace(tzinfo=None)
START_TIME = END_TIME - timedelta(days=365)  # 1 year

# OI period (max 30 days, use 28 for safety margin)
OI_START_TIME = END_TIME - timedelta(days=28)

# API limits
FUTURES_KLINES_LIMIT = 1500
SPOT_KLINES_LIMIT = 1000
OI_LIMIT = 500

# Rate limiting (requests per second)
REQUEST_DELAY = 0.1  # 100ms between requests

# Output
OUTPUT_DIR = "data"
