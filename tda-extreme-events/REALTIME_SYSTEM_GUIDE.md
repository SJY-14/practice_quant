# 🚀 Real-time Bitcoin Prediction System - Complete Guide

## 📋 System Overview

이 시스템은 **TDA (Topological Data Analysis) + Machine Learning**을 사용하여 비트코인 가격을 실시간으로 예측합니다.

### 🔧 Components

1. **realtime_predictor.py** - TDA + XGBoost 모델 학습
2. **live_data_fetcher.py** - 실시간 바이낸스 데이터 수집 & 예측
3. **live_dashboard.ipynb** - Jupyter 대시보드 (시각화)

### 📊 Data Sources

- **Futures**: `/notebooks/binance-data-collector/BTCUSDT_perp_5m.csv` (105,122 rows)
- **Spot**: `/notebooks/binance-data-collector/BTCUSDT_spot_5m.csv` (105,121 rows)

---

## 🎯 Features

### TDA Features
- **L¹ Norm**: 시장의 위상학적 복잡도
- **L² Norm**: 구조적 안정성
- **Wasserstein Distance**: 토폴로지 변화율

### Price Features
- Futures & Spot 가격
- Price spread (선물-현물 차이)
- Returns (1, 5, 12 steps)

### Volume Features
- Trading volume (Futures & Spot)
- CVD (Cumulative Volume Delta)
- Buy/Sell volume분류

### Technical Indicators
- Moving Averages (5, 12, 24 periods)
- Volatility (rolling std)
- High-Low range

---

## 🚀 Quick Start

### Step 1: Train Model (First Time Only)

```bash
cd /notebooks/tda-extreme-events
python realtime_predictor.py
```

**Training Time**: ~10-15 minutes (105,000 samples with TDA computation)

**Output Files**:
- `tda_prediction_model.pkl` - Trained XGBoost model
- `training_metrics.json` - Performance metrics

### Step 2: Start Real-time Monitoring

```bash
python live_data_fetcher.py --interval 300
```

**Parameters**:
- `--interval`: Update interval in seconds (default: 300 = 5 minutes)
- `--model`: Path to model file (default: tda_prediction_model.pkl)

**What it does**:
1. Fetches last 200 candles from Binance (initialize buffer)
2. Every 5 minutes:
   - Fetches latest candle (Futures + Spot)
   - Extracts TDA features
   - Makes price prediction
   - Saves to `live_prediction_status.json`

### Step 3: View Dashboard

Open `live_dashboard.ipynb` in Jupyter and run cells:
- First cells: Show current prediction
- Last cell: Auto-refresh every 60 seconds

---

## 📊 Model Details

### Architecture
- **Algorithm**: XGBoost Regressor
- **Window Size**: 60 candles (5 hours)
- **Forecast Horizon**: 12 candles (60 minutes ahead)
- **Features**: ~30 features (TDA + Price + Volume + Indicators)

### Training Configuration
```python
XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Expected Performance
- **Test MAE**: ~$100-200 (varies with market volatility)
- **Test R²**: 0.85-0.95
- **Prediction Accuracy**: Better during stable markets

---

## 🔍 Usage Examples

### Example 1: One-time Prediction

```bash
# Start monitor (will run once and update status file)
python live_data_fetcher.py --interval 300 &

# Check result
cat live_prediction_status.json
```

### Example 2: Continuous Monitoring

```bash
# Terminal 1: Run predictor
python live_data_fetcher.py --interval 300

# Terminal 2: Watch updates
watch -n 60 cat live_prediction_status.json
```

### Example 3: Dashboard View

```jupyter
# In live_dashboard.ipynb
# Run all cells to see current prediction
# Last cell provides auto-refresh
```

---

## 📁 File Outputs

### training_metrics.json
```json
{
  "train_mae": 156.23,
  "test_mae": 178.45,
  "train_r2": 0.9234,
  "test_r2": 0.8956,
  "forecast_horizon_minutes": 60,
  "window_size_minutes": 300,
  "trained_at": "2025-12-27T02:30:00"
}
```

### live_prediction_status.json
```json
{
  "timestamp": "2025-12-27T02:35:00",
  "current_prices": {
    "futures": 95432.50,
    "spot": 95428.30,
    "timestamp": "2025-12-27T02:30:00"
  },
  "prediction": {
    "current_price": 95432.50,
    "predicted_price": 95678.20,
    "predicted_change": 245.70,
    "predicted_change_pct": 0.26,
    "forecast_horizon_minutes": 60,
    "tda_l1_norm": 0.4231,
    "tda_l2_norm": 0.0687,
    "tda_wasserstein": 0.0053
  }
}
```

---

## 🛠️ Advanced Usage

### Customize Prediction Horizon

Edit `realtime_predictor.py`:
```python
predictor = TDAPricePredictor(
    window_size=60,  # 5 hours
    forecast_horizon=24  # 2 hours ahead (instead of 1 hour)
)
```

Then retrain:
```bash
python realtime_predictor.py
```

### Use Different Data

```python
df = predictor.load_data(
    futures_path='/path/to/your/futures_data.csv',
    spot_path='/path/to/your/spot_data.csv'
)
```

### Modify Update Interval

```bash
# Update every 1 minute
python live_data_fetcher.py --interval 60

# Update every 15 minutes
python live_data_fetcher.py --interval 900
```

---

## 📈 Interpretation Guide

### Prediction Interpretation

| Predicted Change | Interpretation | Action |
|-----------------|----------------|--------|
| > +1% | Strong bullish | Consider long positions |
| +0.3% to +1% | Bullish | Monitor for entry |
| -0.3% to +0.3% | Neutral | Wait for clearer signal |
| -1% to -0.3% | Bearish | Monitor for exit |
| < -1% | Strong bearish | Consider short positions |

### TDA Metrics Interpretation

**L¹ Norm**:
- **High (> 0.8)**: Complex market structure → High volatility expected
- **Medium (0.3-0.8)**: Normal market activity
- **Low (< 0.3)**: Simple structure → Stable market

**L² Norm**:
- Follows similar pattern to L¹ but more sensitive to outliers

**Wasserstein Distance**:
- **High (> 0.01)**: Rapid topology changes → Potential trend reversal
- **Low (< 0.005)**: Stable topology → Continuation likely

### Confidence Indicators

**High Confidence Predictions**:
- Low TDA norms (< 0.5)
- Recent model training (< 1 week old)
- Stable market conditions
- Small predicted changes (< 1%)

**Low Confidence Predictions**:
- High TDA norms (> 0.8)
- Old model (> 1 month)
- High volatility periods
- Large predicted changes (> 2%)

---

## ⚠️ Important Notes

### Limitations

1. **Not Financial Advice**: This is a research tool, not investment advice
2. **Market Conditions**: Performance degrades during extreme events
3. **Latency**: 5-minute candles mean 5-minute delay
4. **Overfitting Risk**: Retrain regularly with fresh data

### Best Practices

1. **Regular Retraining**: Retrain model weekly with latest data
2. **Validation**: Compare predictions with actual outcomes
3. **Risk Management**: Never risk more than you can afford to lose
4. **Multiple Signals**: Combine with other analysis methods
5. **Paper Trading**: Test strategies before using real money

### Error Handling

**If prediction fails**:
```python
# Check data buffer
df = fetcher.get_dataframe()
print(len(df))  # Should be >= 200

# Check model
predictor.predictor.model  # Should not be None
```

**If TDA computation is slow**:
- Reduce `window_size` to 30 (faster but less accurate)
- Use fewer features
- Sample data (every 2nd candle)

---

## 🔄 Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ 1. TRAINING (One-time / Weekly)                        │
│    python realtime_predictor.py                         │
│    ├─ Load Futures + Spot data (105K rows)              │
│    ├─ Extract TDA features (L1, L2, WD)                 │
│    ├─ Create ML features (~30 features)                 │
│    ├─ Train XGBoost model                               │
│    └─ Save model & metrics                              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. REAL-TIME PREDICTION (Continuous)                   │
│    python live_data_fetcher.py --interval 300           │
│    ├─ Fetch last 200 candles (initialize)               │
│    └─ Every 5 minutes:                                  │
│       ├─ Fetch latest candle (Futures + Spot)           │
│       ├─ Update rolling buffer                          │
│       ├─ Extract features                               │
│       ├─ Make prediction                                │
│       └─ Save to JSON                                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. VISUALIZATION (Optional)                            │
│    live_dashboard.ipynb                                 │
│    ├─ Load live_prediction_status.json                  │
│    ├─ Display current prediction                        │
│    ├─ Show TDA metrics                                  │
│    └─ Auto-refresh                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Problem: Training too slow

**Solution**:
```python
# Reduce data size
df = df.tail(50000)  # Use only recent 50K candles

# Or reduce window size
predictor = TDAPricePredictor(window_size=30)  # Faster
```

### Problem: API rate limit

**Solution**:
```python
# Increase delay in binance_client.py
REQUEST_DELAY = 0.5  # Increase from 0.1 to 0.5
```

### Problem: Memory error

**Solution**:
```bash
# Process in chunks or use smaller window
# Reduce features or downsample data
```

### Problem: Poor predictions

**Solution**:
1. Retrain with more recent data
2. Adjust forecast_horizon (try shorter)
3. Check if market conditions changed significantly
4. Add more features or use different model

---

## 📚 Technical Details

### TDA Computation Pipeline

```
Price Data (Futures + Spot)
    ↓
Normalize features to [0, 1]
    ↓
Create multivariate point cloud (4D: close, volume, volume_delta, cvd)
    ↓
Sliding window (size=60)
    ↓
For each window:
    ├─ Construct Vietoris-Rips complex
    ├─ Compute persistence diagram (H₀, H₁)
    ├─ Convert to persistence landscape
    ├─ Calculate L¹ and L² norms
    └─ Compute Wasserstein distance with previous
    ↓
TDA Feature Vector [l1_norm, l2_norm, wasserstein_dist]
```

### Feature Engineering

**Raw Features** (from CSV):
- Futures: close, volume, cvd, volume_delta
- Spot: close, volume, cvd, volume_delta

**Computed Features**:
- TDA: l1_norm, l2_norm, wasserstein_dist
- Price: spread, spread_pct, returns (1,5,12)
- Volume: ratio, buy/sell split
- Technical: MA(5,12,24), volatility, HL range

**Final Feature Set**: ~30 features per sample

---

## 🎓 Learning Resources

### Understanding TDA
- Original Paper: arXiv:2405.16052
- Persistence Homology: https://en.wikipedia.org/wiki/Persistent_homology
- Wasserstein Distance: Measures "distance" between topological shapes

### Understanding XGBoost
- Official Docs: https://xgboost.readthedocs.io/
- XGBoost for Time Series: https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

### Binance API
- Futures: https://binance-docs.github.io/apidocs/futures/en/
- Spot: https://binance-docs.github.io/apidocs/spot/en/

---

## 📞 Support & Issues

### Common Questions

**Q: How accurate are the predictions?**
A: Test MAE is typically $100-200 (0.1-0.2% for BTC at $100K). Accuracy varies with market conditions.

**Q: Can I use this for other coins?**
A: Yes! Just collect data for other pairs and retrain. Works best with high-liquidity pairs.

**Q: How often should I retrain?**
A: Weekly is recommended. Daily if market is very volatile.

**Q: Why use both Futures and Spot?**
A: Futures-Spot spread provides valuable information about market sentiment and arbitrage opportunities.

---

## 🔮 Future Enhancements

Potential improvements:
1. Multi-timeframe analysis (1m, 15m, 1h)
2. Ensemble models (XGB + LSTM + RF)
3. Sentiment analysis integration
4. Order book depth features
5. Funding rate analysis
6. On-chain metrics

---

**Last Updated**: 2025-12-27
**Version**: 1.0.0
**Author**: TDA Prediction System
