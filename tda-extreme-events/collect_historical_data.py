"""
바이낸스 과거 데이터 수집 스크립트
Binance Historical Data Collector

Usage:
    python collect_historical_data.py [--days 365] [--symbol BTCUSDT]
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse
import os

def fetch_binance_klines(symbol='BTCUSDT', interval='5m', market='futures', limit=1500, end_time=None):
    """
    바이낸스 캔들 데이터 수집

    Parameters:
    -----------
    symbol : str
        거래 쌍 (기본값: BTCUSDT)
    interval : str
        캔들 간격 (기본값: 5m)
    market : str
        시장 종류 'futures' 또는 'spot'
    limit : int
        한 번에 가져올 캔들 수 (최대 1500)
    end_time : datetime
        끝 시간 (None이면 현재)

    Returns:
    --------
    list : 캔들 데이터 리스트
    """
    if market == 'futures':
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ API request failed: {e}")
        return []

def collect_data(symbol='BTCUSDT', interval='5m', market='futures', days=365):
    """
    과거 데이터를 수집하여 DataFrame으로 반환

    Parameters:
    -----------
    symbol : str
        거래 쌍
    interval : str
        캔들 간격
    market : str
        시장 종류
    days : int
        수집할 과거 일수

    Returns:
    --------
    pd.DataFrame : 수집된 데이터
    """
    print(f"📥 Collecting {market} data for {symbol}...")
    print(f"  Interval: {interval}")
    print(f"  Period: {days} days")

    all_data = []
    end_time = datetime.now()

    # 필요한 반복 횟수 계산
    # 5분 간격 → 1일 = 288 캔들, 1500개씩 가져오면 약 5.2일치
    candles_per_day = (24 * 60) // 5  # 5분 간격
    total_candles_needed = days * candles_per_day
    iterations = (total_candles_needed // 1500) + 1

    print(f"  Total candles needed: ~{total_candles_needed}")
    print(f"  Fetching in {iterations} batches...\n")

    for i in range(iterations):
        print(f"  Batch {i+1}/{iterations}...", end=' ')

        klines = fetch_binance_klines(symbol, interval, market, 1500, end_time)

        if not klines:
            print("No data")
            break

        # 데이터 처리
        batch_data = []
        for k in klines:
            candle = {
                'open_time': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'buy_volume': float(k[9]),  # taker_buy_base_volume
            }

            # Buy/Sell volume 계산
            candle['sell_volume'] = candle['volume'] - candle['buy_volume']
            candle['volume_delta'] = candle['buy_volume'] - candle['sell_volume']

            batch_data.append(candle)

        all_data.extend(batch_data)
        print(f"✅ {len(batch_data)} candles")

        # 다음 배치를 위해 end_time 업데이트
        if batch_data:
            end_time = batch_data[0]['open_time']

        # Rate limiting (바이낸스 API 제한 준수)
        time.sleep(0.5)

        # 목표 일수만큼 수집했으면 종료
        if len(all_data) >= total_candles_needed:
            print(f"  Target reached ({len(all_data)} candles)")
            break

    # DataFrame 생성 및 정렬
    df = pd.DataFrame(all_data)

    if len(df) == 0:
        print("  ❌ No data collected!")
        return pd.DataFrame()

    df = df.sort_values('open_time').reset_index(drop=True)

    # 중복 제거
    df = df.drop_duplicates(subset='open_time', keep='first')

    # CVD (Cumulative Volume Delta) 계산
    df['cvd'] = df['volume_delta'].cumsum()

    print(f"\n  ✅ Collected {len(df)} candles")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"  Duration: {(df['open_time'].max() - df['open_time'].min()).days} days")

    return df

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Binance Historical Data Collector')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days to collect (default: 365)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading pair symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='5m',
                       help='Candle interval (default: 5m)')
    parser.add_argument('--output-dir', type=str, default='../binance-data-collector',
                       help='Output directory (default: ../binance-data-collector)')

    args = parser.parse_args()

    print("="*80)
    print("📊 BINANCE HISTORICAL DATA COLLECTOR")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Period: {args.days} days")
    print(f"Output: {args.output_dir}")
    print("="*80)
    print()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Futures 데이터 수집
        print("\n1️⃣ FUTURES DATA")
        print("-"*80)
        df_futures = collect_data(args.symbol, args.interval, 'futures', args.days)

        if len(df_futures) > 0:
            output_path = os.path.join(args.output_dir, f'{args.symbol}_perp_{args.interval}.csv')
            df_futures.to_csv(output_path, index=False)
            print(f"💾 Futures data saved to: {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        else:
            print("⚠️ No futures data collected")

        print()

        # Spot 데이터 수집
        print("2️⃣ SPOT DATA")
        print("-"*80)
        df_spot = collect_data(args.symbol, args.interval, 'spot', args.days)

        if len(df_spot) > 0:
            output_path = os.path.join(args.output_dir, f'{args.symbol}_spot_{args.interval}.csv')
            df_spot.to_csv(output_path, index=False)
            print(f"💾 Spot data saved to: {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        else:
            print("⚠️ No spot data collected")

        print()
        print("="*80)
        print("✅ DATA COLLECTION COMPLETE!")
        print("="*80)

        # 요약 출력
        if len(df_futures) > 0 and len(df_spot) > 0:
            print("\n📊 Summary:")
            print(f"  Futures candles: {len(df_futures):,}")
            print(f"  Spot candles:    {len(df_spot):,}")
            print(f"  Futures range:   {df_futures['open_time'].min()} to {df_futures['open_time'].max()}")
            print(f"  Spot range:      {df_spot['open_time'].min()} to {df_spot['open_time'].max()}")
            print(f"\n🎯 Next step: python realtime_predictor.py")

    except KeyboardInterrupt:
        print("\n\n⚠️ Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
