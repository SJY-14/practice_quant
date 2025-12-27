"""
Real-time Binance Data Fetcher
Fetches live BTCUSDT data from Binance and makes predictions
"""

import sys
sys.path.append('/notebooks/binance-data-collector')

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque
import joblib

from binance_client import BinanceClient
from realtime_predictor import TDAPricePredictor


class LiveDataFetcher:
    """
    Fetches real-time data from Binance and maintains a rolling window
    """

    def __init__(self, symbol='BTCUSDT', interval='5m', window_size=200):
        """
        Parameters:
        -----------
        symbol : str
            Trading pair (default: BTCUSDT)
        interval : str
            Candle interval (default: 5m)
        window_size : int
            Number of recent candles to keep
        """
        self.symbol = symbol
        self.interval = interval
        self.window_size = window_size
        self.client = BinanceClient()

        # Data buffers
        self.futures_data = deque(maxlen=window_size)
        self.spot_data = deque(maxlen=window_size)

        # CVD trackers
        self.cvd_futures = 0
        self.cvd_spot = 0

    def fetch_historical_klines(self, market_type='futures', limit=200):
        """
        Fetch recent historical candles to initialize the window.
        """
        print(f"📥 Fetching last {limit} {market_type} candles...")

        if market_type == 'futures':
            url = "https://fapi.binance.com/fapi/v1/klines"
        else:
            url = "https://api.binance.com/api/v3/klines"

        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        klines = response.json()

        # Process klines
        processed_data = []
        for k in klines:
            candle = {
                'open_time': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'taker_buy_base': float(k[9]),  # buy volume
            }

            # Calculate buy/sell volumes
            candle['buy_volume'] = candle['taker_buy_base']
            candle['sell_volume'] = candle['volume'] - candle['buy_volume']
            candle['volume_delta'] = candle['buy_volume'] - candle['sell_volume']

            # Update CVD
            if market_type == 'futures':
                self.cvd_futures += candle['volume_delta']
                candle['cvd'] = self.cvd_futures
            else:
                self.cvd_spot += candle['volume_delta']
                candle['cvd'] = self.cvd_spot

            processed_data.append(candle)

        print(f"  ✅ Fetched {len(processed_data)} candles")
        print(f"  Latest: {processed_data[-1]['open_time']} - Price: ${processed_data[-1]['close']:,.2f}")

        return processed_data

    def initialize(self):
        """Initialize data buffers with historical data."""
        print("\n🔄 Initializing data buffers...")

        # Fetch futures data
        futures_candles = self.fetch_historical_klines('futures', self.window_size)
        self.futures_data.extend(futures_candles)

        # Fetch spot data
        spot_candles = self.fetch_historical_klines('spot', self.window_size)
        self.spot_data.extend(spot_candles)

        print(f"✅ Initialized with {len(self.futures_data)} futures and {len(self.spot_data)} spot candles")

    def fetch_latest_candle(self, market_type='futures'):
        """Fetch the most recent completed candle."""
        if market_type == 'futures':
            url = "https://fapi.binance.com/fapi/v1/klines"
        else:
            url = "https://api.binance.com/api/v3/klines"

        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': 1
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        kline = response.json()[0]

        candle = {
            'open_time': pd.to_datetime(kline[0], unit='ms'),
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5]),
            'taker_buy_base': float(kline[9]),
        }

        candle['buy_volume'] = candle['taker_buy_base']
        candle['sell_volume'] = candle['volume'] - candle['buy_volume']
        candle['volume_delta'] = candle['buy_volume'] - candle['sell_volume']

        # Update CVD
        if market_type == 'futures':
            self.cvd_futures += candle['volume_delta']
            candle['cvd'] = self.cvd_futures
        else:
            self.cvd_spot += candle['volume_delta']
            candle['cvd'] = self.cvd_spot

        return candle

    def update(self):
        """Fetch latest candles and update buffers."""
        # Fetch latest candles
        futures_candle = self.fetch_latest_candle('futures')
        spot_candle = self.fetch_latest_candle('spot')

        # Add to buffers
        self.futures_data.append(futures_candle)
        self.spot_data.append(spot_candle)

        return futures_candle, spot_candle

    def get_dataframe(self):
        """
        Convert buffers to DataFrame format for prediction.
        """
        # Convert to DataFrames
        df_futures = pd.DataFrame(list(self.futures_data))
        df_futures = df_futures.add_suffix('_futures')
        df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

        df_spot = pd.DataFrame(list(self.spot_data))
        df_spot = df_spot.add_suffix('_spot')
        df_spot.rename(columns={'open_time_spot': 'open_time'}, inplace=True)

        # Merge
        df = pd.merge(df_futures, df_spot, on='open_time', how='inner')

        return df

    def get_current_price(self):
        """Get latest price from both markets."""
        return {
            'futures': self.futures_data[-1]['close'] if self.futures_data else None,
            'spot': self.spot_data[-1]['close'] if self.spot_data else None,
            'timestamp': self.futures_data[-1]['open_time'] if self.futures_data else None
        }


class LivePredictor:
    """
    Makes real-time predictions using trained model
    """

    def __init__(self, model_path='tda_prediction_model.pkl'):
        """Load trained model."""
        print(f"🤖 Loading model from {model_path}...")
        self.predictor = TDAPricePredictor()
        self.predictor.load_model(model_path)
        print(f"  Window size: {self.predictor.window_size}")
        print(f"  Forecast horizon: {self.predictor.forecast_horizon} steps ({self.predictor.forecast_horizon * 5} minutes)")
        print(f"  Features: {len(self.predictor.feature_names)}")

    def prepare_features_from_df(self, df):
        """
        Extract features from live data (same as training).
        """
        # Extract TDA features
        tda_features = self.predictor.extract_tda_features(df, use_columns_prefix='futures')

        # Create ML features
        df_features = self.predictor.create_features(df, tda_features)

        return df_features

    def predict(self, df):
        """
        Make prediction on latest data.

        Returns:
        --------
        prediction : dict
            Predicted price, current price, and metadata
        """
        try:
            # Prepare features
            df_features = self.prepare_features_from_df(df)

            if len(df_features) == 0:
                return None

            # Get latest features
            latest_features = df_features.iloc[-1:][self.predictor.feature_names].values

            # Predict
            predicted_price = self.predictor.predict(latest_features)[0]

            # Get current info
            current_price = df_features['close_futures'].iloc[-1]
            current_time = df_features['open_time'].iloc[-1]

            # Calculate prediction time
            prediction_time = current_time + timedelta(minutes=self.predictor.forecast_horizon * 5)

            # TDA status
            tda_l1 = df_features['tda_l1'].iloc[-1]
            tda_l2 = df_features['tda_l2'].iloc[-1]
            tda_wd = df_features['tda_wd'].iloc[-1]

            prediction = {
                'current_time': current_time,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'prediction_time': prediction_time,
                'predicted_change': predicted_price - current_price,
                'predicted_change_pct': (predicted_price - current_price) / current_price * 100,
                'forecast_horizon_minutes': self.predictor.forecast_horizon * 5,
                'tda_l1_norm': tda_l1,
                'tda_l2_norm': tda_l2,
                'tda_wasserstein': tda_wd
            }

            return prediction

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None


def monitor_and_predict(update_interval=300, model_path='tda_prediction_model.pkl'):
    """
    Continuous monitoring and prediction loop.

    Parameters:
    -----------
    update_interval : int
        Seconds between updates (default: 300 = 5 minutes)
    """
    print("="*80)
    print("🚀 REAL-TIME BITCOIN PREDICTION SYSTEM")
    print("="*80)

    # Initialize fetcher
    fetcher = LiveDataFetcher(symbol='BTCUSDT', interval='5m', window_size=200)
    fetcher.initialize()

    # Initialize predictor
    predictor = LivePredictor(model_path=model_path)

    # Save status function
    def save_status(prediction, current_prices):
        status = {
            'timestamp': datetime.now().isoformat(),
            'current_prices': current_prices,
            'prediction': prediction,
            'model_info': {
                'forecast_horizon_minutes': predictor.predictor.forecast_horizon * 5,
                'features_count': len(predictor.predictor.feature_names)
            }
        }

        with open('live_prediction_status.json', 'w') as f:
            json.dump(status, f, indent=2, default=str)

    print(f"\n⏰ Update interval: {update_interval} seconds ({update_interval/60:.1f} minutes)")
    print("📊 Starting monitoring loop...\n")

    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"🔄 Update #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            # Fetch latest data
            print("📥 Fetching latest candles...")
            futures_candle, spot_candle = fetcher.update()

            # Get current prices
            current_prices = fetcher.get_current_price()
            print(f"💰 Current Prices:")
            print(f"  Futures: ${current_prices['futures']:,.2f}")
            print(f"  Spot:    ${current_prices['spot']:,.2f}")
            print(f"  Spread:  ${current_prices['futures'] - current_prices['spot']:,.2f}")

            # Get DataFrame
            df = fetcher.get_dataframe()
            print(f"📊 Data buffer: {len(df)} candles")

            # Make prediction
            print("\n🤖 Making prediction...")
            prediction = predictor.predict(df)

            if prediction:
                print(f"\n🎯 PREDICTION:")
                print(f"  Current Price:    ${prediction['current_price']:,.2f}")
                print(f"  Predicted Price:  ${prediction['predicted_price']:,.2f}")
                print(f"  Expected Change:  ${prediction['predicted_change']:+,.2f} ({prediction['predicted_change_pct']:+.2f}%)")
                print(f"  Prediction Time:  {prediction['prediction_time'].strftime('%H:%M:%S')} ({prediction['forecast_horizon_minutes']} min ahead)")

                print(f"\n📊 TDA Status:")
                print(f"  L¹ Norm:  {prediction['tda_l1_norm']:.4f}")
                print(f"  L² Norm:  {prediction['tda_l2_norm']:.4f}")
                print(f"  WD:       {prediction['tda_wasserstein']:.4f}")

                # Save status
                save_status(prediction, current_prices)
                print(f"\n💾 Status saved to: live_prediction_status.json")
            else:
                print("❌ Prediction failed")

            # Wait
            print(f"\n⏱️  Next update in {update_interval} seconds...")
            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Bitcoin Prediction')
    parser.add_argument('--interval', type=int, default=300,
                       help='Update interval in seconds (default: 300)')
    parser.add_argument('--model', type=str, default='tda_prediction_model.pkl',
                       help='Path to trained model')

    args = parser.parse_args()

    monitor_and_predict(
        update_interval=args.interval,
        model_path=args.model
    )
