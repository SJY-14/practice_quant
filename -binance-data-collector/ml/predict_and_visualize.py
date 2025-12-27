"""
Prediction and Visualization Script
Shows what the trained model can predict
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.tda_features import TDAFeatureExtractor
from ml.models.nbeats import NBeatsWithExogenous


def load_model_and_data(model_path, data_path, lookback=96, horizon=12, tda_window=50):
    """Load trained model and prepare data"""

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Extract features
    prices = df['close'].values.astype(np.float32)

    # Normalize prices
    scaler = StandardScaler()
    prices_normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Extract TDA features
    print("Extracting TDA features...")
    tda_extractor = TDAFeatureExtractor(window_size=tda_window)
    tda_features = tda_extractor.extract_features(prices_normalized)

    # Pad TDA features
    padding = np.zeros((tda_window - 1, 3))
    tda_features = np.vstack([padding, tda_features])

    # Normalize TDA
    tda_scaler = StandardScaler()
    tda_features = tda_scaler.fit_transform(tda_features)

    # Extract exogenous features
    exog_cols = ['volume', 'buy_volume', 'sell_volume', 'volume_delta', 'cvd']
    available_cols = [c for c in exog_cols if c in df.columns]

    if available_cols:
        exog_features = df[available_cols].values.astype(np.float32)
        exog_scaler = StandardScaler()
        exog_features = exog_scaler.fit_transform(exog_features)
    else:
        exog_features = None

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_exog = 3 + (exog_features.shape[1] if exog_features is not None else 0)

    model = NBeatsWithExogenous(
        lookback=lookback,
        horizon=horizon,
        n_exog_features=n_exog
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Device: {device}")

    return model, df, prices_normalized, tda_features, exog_features, scaler, device


def predict_future(model, prices, tda_features, exog_features, lookback, horizon, device, use_exog=True):
    """Make prediction for the most recent data"""

    # Use last lookback period
    x = torch.FloatTensor(prices[-lookback:]).unsqueeze(0).to(device)
    tda = torch.FloatTensor(tda_features[-1]).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_exog and exog_features is not None:
            exog = torch.FloatTensor(exog_features[-1]).unsqueeze(0).to(device)
            combined_exog = torch.cat([tda, exog], dim=1)
            pred = model(x, combined_exog)
        else:
            pred = model(x, tda)

    return pred.cpu().numpy()[0]


def predict_multiple_points(model, prices, tda_features, exog_features, lookback, horizon, device, n_points=50):
    """Make predictions at multiple historical points"""

    predictions = []
    actuals = []
    timestamps = []

    # Start from lookback position and predict every 12 steps (1 hour intervals)
    start_idx = lookback
    end_idx = len(prices) - horizon
    step = horizon  # Predict every hour

    indices = range(start_idx, end_idx, step)
    if len(indices) > n_points:
        # Sample evenly if too many points
        indices = [indices[i] for i in range(0, len(indices), len(indices)//n_points)]

    for i in indices[:n_points]:
        # Input
        x = torch.FloatTensor(prices[i-lookback:i]).unsqueeze(0).to(device)
        tda = torch.FloatTensor(tda_features[i-1]).unsqueeze(0).to(device)

        # Actual future
        actual = prices[i:i+horizon]

        # Predict
        with torch.no_grad():
            if exog_features is not None:
                exog = torch.FloatTensor(exog_features[i-1]).unsqueeze(0).to(device)
                combined_exog = torch.cat([tda, exog], dim=1)
                pred = model(x, combined_exog)
            else:
                pred = model(x, tda)

        predictions.append(pred.cpu().numpy()[0])
        actuals.append(actual)
        timestamps.append(i)

    return np.array(predictions), np.array(actuals), timestamps


def visualize_predictions(df, prices_normalized, predictions, actuals, timestamps,
                          future_pred, scaler, save_path='checkpoints'):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Recent Price History + Future Prediction
    ax1 = fig.add_subplot(gs[0, :])

    # Denormalize prices
    prices_original = scaler.inverse_transform(prices_normalized.reshape(-1, 1)).flatten()
    future_denorm = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()

    # Plot last 500 points (about 42 hours)
    plot_len = min(500, len(prices_original))
    x_range = range(len(prices_original) - plot_len, len(prices_original))

    ax1.plot(x_range, prices_original[-plot_len:], label='Historical Price', linewidth=2, color='blue')

    # Plot future prediction
    future_x = range(len(prices_original), len(prices_original) + len(future_pred))
    ax1.plot(future_x, future_denorm, label='Predicted Price (Next 1 Hour)',
             linewidth=2, color='red', linestyle='--', marker='o', markersize=4)

    ax1.axvline(len(prices_original)-1, color='green', linestyle=':', alpha=0.7, label='Current Time')
    ax1.set_xlabel('Time (5-min intervals)')
    ax1.set_ylabel('BTC Price (USDT)')
    ax1.set_title('BTC Price: Historical + Predicted Future (Next 1 Hour = 12 steps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add price change annotation
    current_price = prices_original[-1]
    predicted_price = future_denorm[-1]
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100

    ax1.text(0.02, 0.95,
             f'Current: ${current_price:,.2f}\nPredicted (1h): ${predicted_price:,.2f}\nChange: {price_change:+.2f} ({price_change_pct:+.2f}%)',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. Prediction vs Actual (Multiple Points)
    ax2 = fig.add_subplot(gs[1, 0])

    # Show only 1-step ahead for clarity
    pred_1step = predictions[:, 0]
    actual_1step = actuals[:, 0]

    pred_1step_denorm = scaler.inverse_transform(pred_1step.reshape(-1, 1)).flatten()
    actual_1step_denorm = scaler.inverse_transform(actual_1step.reshape(-1, 1)).flatten()

    ax2.plot(actual_1step_denorm, label='Actual', alpha=0.7, linewidth=2)
    ax2.plot(pred_1step_denorm, label='Predicted', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Prediction Point')
    ax2.set_ylabel('Price (USDT)')
    ax2.set_title('1-Step Ahead Predictions vs Actual (Sampled)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Prediction Accuracy
    ax3 = fig.add_subplot(gs[1, 1])

    # Calculate direction accuracy for each point
    pred_directions = np.sign(predictions[:, -1] - predictions[:, 0])
    actual_directions = np.sign(actuals[:, -1] - actuals[:, 0])

    correct = (pred_directions == actual_directions).astype(int)

    colors = ['green' if c == 1 else 'red' for c in correct]
    ax3.bar(range(len(correct)), correct, color=colors, alpha=0.7)
    ax3.set_xlabel('Prediction Point')
    ax3.set_ylabel('Correct (1) / Wrong (0)')
    ax3.set_title(f'Direction Prediction Accuracy: {correct.mean()*100:.1f}%')
    ax3.set_ylim([-0.1, 1.1])
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Price Change Distribution
    ax4 = fig.add_subplot(gs[2, 0])

    # Predicted vs actual price changes
    pred_changes = predictions[:, -1] - predictions[:, 0]
    actual_changes = actuals[:, -1] - actuals[:, 0]

    ax4.scatter(actual_changes, pred_changes, alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(actual_changes.min(), pred_changes.min())
    max_val = max(actual_changes.max(), pred_changes.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    ax4.set_xlabel('Actual Price Change (Normalized)')
    ax4.set_ylabel('Predicted Price Change (Normalized)')
    ax4.set_title('Price Change: Predicted vs Actual')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Trading Signals
    ax5 = fig.add_subplot(gs[2, 1])

    # Show recent predictions as trading signals
    recent_n = min(20, len(timestamps))
    recent_times = timestamps[-recent_n:]
    recent_pred_changes = pred_changes[-recent_n:]
    recent_actual_changes = actual_changes[-recent_n:]

    x_pos = np.arange(recent_n)
    width = 0.35

    ax5.bar(x_pos - width/2, recent_actual_changes, width, label='Actual', alpha=0.7)
    ax5.bar(x_pos + width/2, recent_pred_changes, width, label='Predicted', alpha=0.7)

    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.set_xlabel('Recent Prediction Points')
    ax5.set_ylabel('Price Change (Normalized)')
    ax5.set_title('Recent Trading Signals (BUY=↑ / SELL=↓)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    plt.savefig(f'{save_path}/prediction_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}/prediction_analysis.png")

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nCurrent BTC Price: ${current_price:,.2f}")
    print(f"Predicted Price (1 hour): ${predicted_price:,.2f}")
    print(f"Expected Change: {price_change:+.2f} USDT ({price_change_pct:+.2f}%)")
    print(f"\nSignal: {'📈 BUY (Price expected to rise)' if price_change > 0 else '📉 SELL (Price expected to fall)'}")
    print(f"\nHistorical Direction Accuracy: {correct.mean()*100:.1f}%")
    print("="*60)


def main():
    model_path = 'checkpoints/best_model.pt'
    data_path = 'BTCUSDT_5m.csv'

    # Load model and data
    model, df, prices_norm, tda_features, exog_features, scaler, device = load_model_and_data(
        model_path, data_path
    )

    # Make future prediction
    print("\nMaking future prediction...")
    future_pred = predict_future(
        model, prices_norm, tda_features, exog_features,
        lookback=96, horizon=12, device=device
    )

    # Make multiple historical predictions for analysis
    print("Analyzing historical predictions...")
    predictions, actuals, timestamps = predict_multiple_points(
        model, prices_norm, tda_features, exog_features,
        lookback=96, horizon=12, device=device, n_points=50
    )

    # Visualize
    print("Creating visualization...")
    visualize_predictions(
        df, prices_norm, predictions, actuals, timestamps,
        future_pred, scaler
    )


if __name__ == "__main__":
    main()
