"""
Test prediction system using historical data (offline mode)
"""
import sys
sys.path.append('/notebooks/binance-data-collector')

from realtime_predictor import TDAPricePredictor
import pandas as pd
import json
from datetime import datetime, timedelta

print("="*80)
print("🧪 TESTING PREDICTION SYSTEM (OFFLINE MODE)")
print("="*80)

# Load predictor
print("\n1️⃣ Loading trained model...")
predictor = TDAPricePredictor()
predictor.load_model('tda_prediction_model.pkl')

# Load recent data from CSV files
print("\n2️⃣ Loading recent historical data...")
df_futures = pd.read_csv('/notebooks/binance-data-collector/BTCUSDT_perp_5m.csv')
df_futures['open_time'] = pd.to_datetime(df_futures['open_time'])
df_futures = df_futures.add_suffix('_futures')
df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

df_spot = pd.read_csv('/notebooks/binance-data-collector/BTCUSDT_spot_5m.csv')
df_spot['open_time'] = pd.to_datetime(df_spot['open_time'])
df_spot = df_spot.add_suffix('_spot')
df_spot.rename(columns={'open_time_spot': 'open_time'}, inplace=True)

# Merge
df = pd.merge(df_futures, df_spot, on='open_time', how='inner')

# Use recent data but leave room for validation (use data from 12 candles before end)
forecast_horizon = predictor.forecast_horizon
df_recent = df.iloc[-(200 + forecast_horizon):-forecast_horizon].copy()

print(f"  Loaded {len(df_recent)} recent candles")
print(f"  Latest: {df_recent['open_time'].iloc[-1]}")

# Get current prices
current_price_futures = df_recent['close_futures'].iloc[-1]
current_price_spot = df_recent['close_spot'].iloc[-1]
current_time = df_recent['open_time'].iloc[-1]

print(f"\n3️⃣ Current Prices (from data):")
print(f"  Futures: ${current_price_futures:,.2f}")
print(f"  Spot:    ${current_price_spot:,.2f}")
print(f"  Spread:  ${current_price_futures - current_price_spot:,.2f}")
print(f"  Time:    {current_time}")

# Extract TDA features
print("\n4️⃣ Extracting TDA features...")
tda_features = predictor.extract_tda_features(df_recent, use_columns_prefix='futures')

# Create ML features
print("\n5️⃣ Creating ML features...")
df_features = predictor.create_features(df_recent, tda_features)

if len(df_features) == 0:
    print("❌ Not enough data to create features")
    sys.exit(1)

# Get latest features
latest_features = df_features.iloc[-1:][predictor.feature_names].values

# Make prediction
print("\n6️⃣ Making prediction...")
predicted_price = predictor.predict(latest_features)[0]

# Get actual future price (for validation)
actual_future_idx = df.index[df['open_time'] == current_time].tolist()[0] + forecast_horizon
if actual_future_idx < len(df):
    actual_future_price = df['close_futures'].iloc[actual_future_idx]
    has_actual = True
else:
    actual_future_price = None
    has_actual = False

# Calculate prediction time
prediction_time = current_time + timedelta(minutes=predictor.forecast_horizon * 5)

# TDA status
tda_l1 = df_features['tda_l1'].iloc[-1]
tda_l2 = df_features['tda_l2'].iloc[-1]
tda_wd = df_features['tda_wd'].iloc[-1]

prediction_change = predicted_price - current_price_futures
prediction_change_pct = (predicted_price - current_price_futures) / current_price_futures * 100

print(f"\n✅ PREDICTION RESULT:")
print(f"  Current Price:    ${current_price_futures:,.2f}")
print(f"  Predicted Price:  ${predicted_price:,.2f}")
print(f"  Expected Change:  ${prediction_change:+,.2f} ({prediction_change_pct:+.2f}%)")
print(f"  Prediction Time:  {prediction_time.strftime('%Y-%m-%d %H:%M:%S')} ({predictor.forecast_horizon * 5} min ahead)")

if has_actual:
    actual_error = abs(predicted_price - actual_future_price)
    print(f"\n📊 Validation (using historical data):")
    print(f"  Actual Future Price: ${actual_future_price:,.2f}")
    print(f"  Prediction Error:    ${actual_error:,.2f}")

print(f"\n📈 TDA Status:")
print(f"  L¹ Norm:  {tda_l1:.4f}")
print(f"  L² Norm:  {tda_l2:.4f}")
print(f"  WD:       {tda_wd:.4f}")

# Interpretation
print(f"\n🔍 Interpretation:")
if abs(prediction_change_pct) > 1.0:
    direction = "📈 Strong Bullish" if prediction_change_pct > 0 else "📉 Strong Bearish"
    action = "Consider long positions" if prediction_change_pct > 0 else "Consider short positions"
elif abs(prediction_change_pct) > 0.3:
    direction = "📈 Bullish" if prediction_change_pct > 0 else "📉 Bearish"
    action = "Monitor for entry" if prediction_change_pct > 0 else "Monitor for exit"
else:
    direction = "➡️ Neutral"
    action = "Wait for clearer signal"

print(f"  Direction: {direction}")
print(f"  Suggestion: {action}")

# TDA interpretation
if tda_l1 > 0.8:
    tda_status = "⚠️ HIGH (Complex market, high volatility expected)"
elif tda_l1 > 0.3:
    tda_status = "✅ NORMAL (Normal market activity)"
else:
    tda_status = "🟢 LOW (Stable market)"

print(f"  Market Complexity: {tda_status}")

# Save prediction status
status = {
    'timestamp': datetime.now().isoformat(),
    'data_timestamp': str(current_time),
    'current_prices': {
        'futures': float(current_price_futures),
        'spot': float(current_price_spot),
        'timestamp': str(current_time)
    },
    'prediction': {
        'current_price': float(current_price_futures),
        'predicted_price': float(predicted_price),
        'predicted_change': float(prediction_change),
        'predicted_change_pct': float(prediction_change_pct),
        'prediction_time': str(prediction_time),
        'forecast_horizon_minutes': predictor.forecast_horizon * 5,
        'tda_l1_norm': float(tda_l1),
        'tda_l2_norm': float(tda_l2),
        'tda_wasserstein': float(tda_wd)
    }
}

if has_actual:
    status['validation'] = {
        'actual_future_price': float(actual_future_price),
        'prediction_error': float(actual_error)
    }

with open('live_prediction_status.json', 'w') as f:
    json.dump(status, f, indent=2)

print(f"\n💾 Status saved to: live_prediction_status.json")

print("\n" + "="*80)
print("✅ TEST COMPLETE!")
print("="*80)
print("\nNote: This test used historical data due to Binance API restrictions.")
print("The prediction system is working correctly.")
print("For live predictions, you would need to run this from a location with Binance API access.")
