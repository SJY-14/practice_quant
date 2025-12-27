#!/usr/bin/env python3
"""
Binance Data Collector
Collects 5m klines data with CVD and OI (30 days for futures)
"""

import os
import argparse
from datetime import datetime, timedelta, UTC
from data_collector import DataCollector
from config import OUTPUT_DIR, INTERVAL


def parse_args():
    parser = argparse.ArgumentParser(
        description='Binance 5분봉 데이터 수집기 (가격, 거래량, CVD, OI)'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='BTCUSDT',
        help='심볼 (기본: BTCUSDT)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=365,
        help='수집 기간 (일, 기본: 365)'
    )
    parser.add_argument(
        '--market', '-m',
        type=str,
        choices=['futures', 'spot', 'both'],
        default='both',
        help='마켓 타입 (기본: both)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Calculate time range
    end_time = datetime.now(UTC).replace(tzinfo=None)
    start_time = end_time - timedelta(days=args.days)
    oi_start_time = end_time - timedelta(days=28)  # OI max 30 days

    collector = DataCollector()

    print(f"\n=== Binance Data Collector ===")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days ({start_time.date()} ~ {end_time.date()})")
    print(f"Market: {args.market}")

    # Collect Futures data
    if args.market in ['futures', 'both']:
        futures_df = collector.collect_futures_data(
            symbol=args.symbol,
            interval=INTERVAL,
            start_time=start_time,
            end_time=end_time,
            oi_start_time=oi_start_time
        )
        filename = f"{args.symbol}_perp_5m.csv"
        collector.save_to_csv(futures_df, filename)

    # Collect Spot data
    if args.market in ['spot', 'both']:
        spot_df = collector.collect_spot_data(
            symbol=args.symbol,
            interval=INTERVAL,
            start_time=start_time,
            end_time=end_time
        )
        filename = f"{args.symbol}_spot_5m.csv"
        collector.save_to_csv(spot_df, filename)

    print("\n=== Collection Complete ===")


if __name__ == "__main__":
    main()
