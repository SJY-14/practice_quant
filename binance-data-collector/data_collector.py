import pandas as pd
from datetime import datetime
from typing import List, Optional
from binance_client import BinanceClient
from config import (
    FUTURES_KLINES_LIMIT, SPOT_KLINES_LIMIT, OI_LIMIT,
    OUTPUT_DIR
)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp"""
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime"""
    return datetime.utcfromtimestamp(ms / 1000)


class DataCollector:
    def __init__(self):
        self.client = BinanceClient()

    def collect_klines(
        self,
        market_type: str,  # "futures" or "spot"
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Collect all klines data with pagination"""
        all_data = []
        current_start = datetime_to_ms(start_time)
        end_ms = datetime_to_ms(end_time)

        if market_type == "futures":
            limit = FUTURES_KLINES_LIMIT
            fetch_func = self.client.get_futures_klines
        else:
            limit = SPOT_KLINES_LIMIT
            fetch_func = self.client.get_spot_klines

        request_count = 0
        while current_start < end_ms:
            print(f"  Fetching {market_type} klines from {ms_to_datetime(current_start)}...")

            data = fetch_func(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
                limit=limit
            )

            if not data:
                break

            all_data.extend(data)
            request_count += 1

            # Move to next batch (last candle's close time + 1)
            current_start = data[-1][6] + 1

            if len(data) < limit:
                break

        print(f"  Collected {len(all_data)} candles in {request_count} requests")

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        df['trades'] = df['trades'].astype(int)

        return df

    def collect_open_interest(
        self,
        symbol: str,
        period: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Collect open interest history (max 30 days)"""
        all_data = []
        current_start = datetime_to_ms(start_time)
        end_ms = datetime_to_ms(end_time)

        request_count = 0
        while current_start < end_ms:
            print(f"  Fetching OI from {ms_to_datetime(current_start)}...")

            data = self.client.get_open_interest_hist(
                symbol=symbol,
                period=period,
                start_time=current_start,
                end_time=end_ms,
                limit=OI_LIMIT
            )

            if not data:
                break

            all_data.extend(data)
            request_count += 1

            # Move to next batch
            current_start = data[-1]['timestamp'] + 1

            if len(data) < OI_LIMIT:
                break

        print(f"  Collected {len(all_data)} OI records in {request_count} requests")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
        df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)

        return df

    def calculate_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CVD from klines data"""
        # Buy volume = taker_buy_base (already in data)
        df['buy_volume'] = df['taker_buy_base']

        # Sell volume = total - buy
        df['sell_volume'] = df['volume'] - df['buy_volume']

        # Volume delta = buy - sell
        df['volume_delta'] = df['buy_volume'] - df['sell_volume']

        # CVD = cumulative sum of volume delta
        df['cvd'] = df['volume_delta'].cumsum()

        return df

    def merge_with_oi(
        self,
        klines_df: pd.DataFrame,
        oi_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge klines with OI data"""
        if oi_df.empty:
            klines_df['open_interest'] = None
            return klines_df

        # Rename for merge
        oi_df = oi_df.rename(columns={
            'timestamp': 'open_time',
            'sumOpenInterest': 'open_interest'
        })

        # Merge on open_time
        merged = pd.merge(
            klines_df,
            oi_df[['open_time', 'open_interest']],
            on='open_time',
            how='left'
        )

        return merged

    def collect_futures_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        oi_start_time: datetime
    ) -> pd.DataFrame:
        """Collect complete futures data (klines + CVD + OI)"""
        print(f"\n=== Collecting Futures Data ({symbol} Perpetual) ===")

        # Collect klines
        print("\n1. Collecting Klines...")
        klines_df = self.collect_klines(
            market_type="futures",
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )

        # Calculate CVD
        print("\n2. Calculating CVD...")
        klines_df = self.calculate_cvd(klines_df)

        # Collect OI (30 days only)
        print("\n3. Collecting Open Interest (30 days)...")
        oi_df = self.collect_open_interest(
            symbol=symbol,
            period=interval,
            start_time=oi_start_time,
            end_time=end_time
        )

        # Merge
        print("\n4. Merging data...")
        result_df = self.merge_with_oi(klines_df, oi_df)

        return result_df

    def collect_spot_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Collect complete spot data (klines + CVD, no OI)"""
        print(f"\n=== Collecting Spot Data ({symbol}) ===")

        # Collect klines
        print("\n1. Collecting Klines...")
        klines_df = self.collect_klines(
            market_type="spot",
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )

        # Calculate CVD
        print("\n2. Calculating CVD...")
        klines_df = self.calculate_cvd(klines_df)

        return klines_df

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV"""
        filepath = f"{OUTPUT_DIR}/{filename}"

        # Select columns for output
        output_columns = [
            'open_time', 'open', 'high', 'low', 'close',
            'volume', 'buy_volume', 'sell_volume',
            'volume_delta', 'cvd'
        ]

        if 'open_interest' in df.columns:
            output_columns.append('open_interest')

        df[output_columns].to_csv(filepath, index=False)
        print(f"\nSaved to {filepath}")
        print(f"Total rows: {len(df)}")
