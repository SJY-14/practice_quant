"""
Quick test of live prediction system
"""
import sys
sys.path.append('/notebooks/binance-data-collector')

from live_data_fetcher import LiveDataFetcher, LivePredictor
import json
from datetime import datetime

print("="*80)
print("🧪 TESTING LIVE PREDICTION SYSTEM")
print("="*80)

# Initialize fetcher
print("\n1️⃣ Initializing data fetcher...")
fetcher = LiveDataFetcher(symbol='BTCUSDT', interval='5m', window_size=200)
fetcher.initialize()

# Initialize predictor
print("\n2️⃣ Loading trained model...")
predictor = LivePredictor(model_path='tda_prediction_model.pkl')

# Get current prices
current_prices = fetcher.get_current_price()
print(f"\n3️⃣ Current Prices:")
print(f"  Futures: ${current_prices['futures']:,.2f}")
print(f"  Spot:    ${current_prices['spot']:,.2f}")
print(f"  Spread:  ${current_prices['futures'] - current_prices['spot']:,.2f}")

# Get DataFrame
df = fetcher.get_dataframe()
print(f"\n4️⃣ Data buffer: {len(df)} candles")

# Make prediction
print("\n5️⃣ Making prediction...")
prediction = predictor.predict(df)

if prediction:
    print(f"\n✅ PREDICTION SUCCESS:")
    print(f"  Current Price:    ${prediction['current_price']:,.2f}")
    print(f"  Predicted Price:  ${prediction['predicted_price']:,.2f}")
    print(f"  Expected Change:  ${prediction['predicted_change']:+,.2f} ({prediction['predicted_change_pct']:+.2f}%)")
    print(f"  Prediction Time:  {prediction['prediction_time'].strftime('%H:%M:%S')} ({prediction['forecast_horizon_minutes']} min ahead)")

    print(f"\n📊 TDA Status:")
    print(f"  L¹ Norm:  {prediction['tda_l1_norm']:.4f}")
    print(f"  L² Norm:  {prediction['tda_l2_norm']:.4f}")
    print(f"  WD:       {prediction['tda_wasserstein']:.4f}")

    # Save status
    status = {
        'timestamp': datetime.now().isoformat(),
        'current_prices': current_prices,
        'prediction': prediction
    }

    with open('live_prediction_status.json', 'w') as f:
        json.dump(status, f, indent=2, default=str)

    print(f"\n💾 Status saved to: live_prediction_status.json")
else:
    print("❌ Prediction failed")

print("\n" + "="*80)
print("✅ TEST COMPLETE!")
print("="*80)
