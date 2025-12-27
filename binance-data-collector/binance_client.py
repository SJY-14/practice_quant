import requests
import time
from typing import List, Dict, Optional
from config import (
    FUTURES_BASE_URL, SPOT_BASE_URL,
    FUTURES_KLINES_LIMIT, SPOT_KLINES_LIMIT, OI_LIMIT,
    REQUEST_DELAY
)


class BinanceClient:
    """Binance API client for Spot and Futures data"""

    def __init__(self):
        self.session = requests.Session()

    def _request(self, url: str, params: dict, raise_on_error: bool = True) -> List:
        """Make API request with rate limiting"""
        time.sleep(REQUEST_DELAY)
        response = self.session.get(url, params=params)
        if raise_on_error:
            response.raise_for_status()
        elif response.status_code != 200:
            print(f"  Warning: API returned {response.status_code}: {response.text[:200]}")
            return []
        return response.json()

    def get_futures_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = FUTURES_KLINES_LIMIT
    ) -> List:
        """
        Fetch futures klines data

        Returns list of:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        url = f"{FUTURES_BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        return self._request(url, params)

    def get_spot_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = SPOT_KLINES_LIMIT
    ) -> List:
        """
        Fetch spot klines data

        Returns list of:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        url = f"{SPOT_BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        return self._request(url, params)

    def get_open_interest_hist(
        self,
        symbol: str,
        period: str,
        start_time: int,
        end_time: int,
        limit: int = OI_LIMIT
    ) -> List[Dict]:
        """
        Fetch historical open interest data (futures only, max 30 days)

        Returns list of:
        {"symbol", "sumOpenInterest", "sumOpenInterestValue", "timestamp"}
        """
        url = f"{FUTURES_BASE_URL}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        return self._request(url, params, raise_on_error=False)
