"""
Enhanced Training Pipeline with Futures-Spot Correlation
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.tda_features import TDAFeatureExtractor
from ml.models.nbeats_enhanced import NBeatsEnhanced


class EnhancedCryptoDataset(Dataset):
    """Dataset with futures-spot correlation features"""

    def __init__(
        self,
        prices: np.ndarray,
        correlation_features: np.ndarray,
        tda_features: np.ndarray,
        lookback: int = 96,
        horizon: int = 12
    ):
        self.prices = prices
        self.correlation_features = correlation_features
        self.tda_features = tda_features
        self.lookback = lookback
        self.horizon = horizon

        self.valid_start = lookback
        self.valid_end = len(prices) - horizon

    def __len__(self):
        return self.valid_end - self.valid_start

    def __getitem__(self, idx):
        i = self.valid_start + idx

        # Price history
        x = self.prices[i - self.lookback:i]

        # Target
        y = self.prices[i:i + self.horizon]

        # Correlation features (most recent)
        corr_feat = self.correlation_features[i - 1]

        # TDA features
        tda_idx = min(i - 1, len(self.tda_features) - 1)
        tda = self.tda_features[tda_idx]

        return {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y),
            'correlation': torch.FloatTensor(corr_feat),
            'tda': torch.FloatTensor(tda)
        }


def load_and_prepare_data(data_path: str, tda_window: int = 50):
    """Load prepared data with correlation features"""

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Extract prices (target: perpetual futures close)
    prices = df['close'].values.astype(np.float32)

    # Normalize prices
    price_scaler = StandardScaler()
    prices_normalized = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Extract correlation features
    correlation_cols = [
        'price_spread',
        'price_spread_pct',
        'volume_ratio',
        'cvd_diff',
        'volume_delta_diff',
        'return_correlation',
        'spread_change',
        'buy_ratio_diff'
    ]

    available_corr_cols = [c for c in correlation_cols if c in df.columns]
    print(f"\n  Correlation features ({len(available_corr_cols)}): {available_corr_cols}")

    correlation_features = df[available_corr_cols].values.astype(np.float32)

    # Normalize correlation features
    corr_scaler = StandardScaler()
    correlation_features = corr_scaler.fit_transform(correlation_features)

    # Extract TDA features
    print("\nExtracting TDA features...")
    tda_extractor = TDAFeatureExtractor(window_size=tda_window)
    tda_features = tda_extractor.extract_features(prices_normalized)

    # Pad TDA features
    padding = np.zeros((tda_window - 1, 3))
    tda_features = np.vstack([padding, tda_features])

    # Normalize TDA
    tda_scaler = StandardScaler()
    tda_features = tda_scaler.fit_transform(tda_features)

    print(f"\nData shapes:")
    print(f"  Prices: {prices_normalized.shape}")
    print(f"  Correlation features: {correlation_features.shape}")
    print(f"  TDA features: {tda_features.shape}")

    return prices_normalized, correlation_features, tda_features, price_scaler


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        corr = batch['correlation'].to(device)
        tda = batch['tda'].to(device)

        optimizer.zero_grad()

        # Forward pass with correlation and TDA features
        pred = model(x, correlation_features=corr, exog_features=tda)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            corr = batch['correlation'].to(device)
            tda = batch['tda'].to(device)

            pred = model(x, correlation_features=corr, exog_features=tda)

            loss = criterion(pred, y)
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return avg_loss, all_preds, all_targets


def calculate_metrics(preds, targets):
    """Calculate evaluation metrics"""
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100

    # Direction accuracy
    pred_direction = np.sign(preds[:, -1] - preds[:, 0])
    true_direction = np.sign(targets[:, -1] - targets[:, 0])
    direction_acc = np.mean(pred_direction == true_direction) * 100

    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_acc
    }


def train(
    data_path: str,
    lookback: int = 96,
    horizon: int = 12,
    tda_window: int = 50,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    save_path: str = 'checkpoints_enhanced'
):
    """Main training function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ENHANCED N-BEATS TRAINING - Futures-Spot Correlation")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Load data
    prices, correlation_features, tda_features, scaler = load_and_prepare_data(
        data_path, tda_window=tda_window
    )

    # Create dataset
    dataset = EnhancedCryptoDataset(
        prices=prices,
        correlation_features=correlation_features,
        tda_features=tda_features,
        lookback=lookback,
        horizon=horizon
    )

    # Train/val/test split
    n_samples = len(dataset)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, n_samples))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Create enhanced model
    model = NBeatsEnhanced(
        lookback=lookback,
        horizon=horizon,
        n_correlation_features=correlation_features.shape[1],
        n_exog_features=3,  # TDA features
        use_attention=True
    ).to(device)

    print(f"\n{'='*60}")
    print(f"Model Architecture:")
    print(f"  Correlation features: {correlation_features.shape[1]}")
    print(f"  TDA features: 3")
    print(f"  With attention mechanism: Yes")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'epoch': epoch,
                'val_loss': val_loss
            }, f'{save_path}/best_model_enhanced.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Load best model and test
    checkpoint = torch.load(f'{save_path}/best_model_enhanced.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, preds, targets = evaluate(model, test_loader, criterion, device)
    metrics = calculate_metrics(preds, targets)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
    print(f"{'='*60}")

    # Plot results
    plot_results(train_losses, val_losses, preds, targets, save_path)

    return model, metrics


def plot_results(train_losses, val_losses, preds, targets, save_path):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(train_losses, label='Train')
    axes[0, 0].plot(val_losses, label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Predictions vs Actual
    n_show = min(200, len(preds))
    axes[0, 1].plot(targets[:n_show, 0], label='Actual', alpha=0.7)
    axes[0, 1].plot(preds[:n_show, 0], label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Price (normalized)')
    axes[0, 1].set_title('Predictions vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Scatter plot
    axes[1, 0].scatter(targets[:, 0], preds[:, 0], alpha=0.3, s=10)
    axes[1, 0].plot([targets.min(), targets.max()],
                     [targets.min(), targets.max()], 'r--')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title('Prediction Scatter')
    axes[1, 0].grid(True)

    # Error distribution
    errors = preds[:, 0] - targets[:, 0]
    axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_path}/results_enhanced.png', dpi=150)
    plt.close()
    print(f"\nResults saved to {save_path}/results_enhanced.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Enhanced N-BEATS with Futures-Spot Correlation')
    parser.add_argument('--data', type=str, default='../BTCUSDT_merged_features.csv',
                        help='Path to merged features CSV')
    parser.add_argument('--lookback', type=int, default=96,
                        help='Lookback window (default: 96 = 8 hours)')
    parser.add_argument('--horizon', type=int, default=12,
                        help='Forecast horizon (default: 12 = 1 hour)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()

    train(
        data_path=args.data,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
