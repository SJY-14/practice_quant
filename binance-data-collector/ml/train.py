"""
Training Pipeline for N-BEATS with TDA Features
Cryptocurrency Price Prediction
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.tda_features import TDAFeatureExtractor
from ml.models.nbeats import NBeatsWithTDA, NBeatsWithExogenous


class CryptoDataset(Dataset):
    """
    Dataset for cryptocurrency price prediction with TDA features
    """

    def __init__(
        self,
        prices: np.ndarray,
        tda_features: np.ndarray,
        exog_features: Optional[np.ndarray] = None,
        lookback: int = 96,
        horizon: int = 12
    ):
        self.prices = prices
        self.tda_features = tda_features
        self.exog_features = exog_features
        self.lookback = lookback
        self.horizon = horizon

        # Valid indices (need lookback history + horizon future)
        self.valid_start = lookback
        self.valid_end = len(prices) - horizon

    def __len__(self):
        return self.valid_end - self.valid_start

    def __getitem__(self, idx):
        i = self.valid_start + idx

        # Input sequence
        x = self.prices[i - self.lookback:i]

        # Target
        y = self.prices[i:i + self.horizon]

        # TDA features (use the most recent available)
        tda_idx = min(i - 1, len(self.tda_features) - 1)
        tda = self.tda_features[tda_idx]

        result = {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y),
            'tda': torch.FloatTensor(tda)
        }

        if self.exog_features is not None:
            exog = self.exog_features[i - 1]
            result['exog'] = torch.FloatTensor(exog)

        return result


def load_and_preprocess_data(
    data_path: str,
    tda_window: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load data and extract TDA features

    Args:
        data_path: Path to CSV file
        tda_window: Window size for TDA feature extraction

    Returns:
        prices, tda_features, exog_features, scaler
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Extract price (close)
    prices = df['close'].values.astype(np.float32)

    # Normalize prices
    scaler = StandardScaler()
    prices_normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Extract TDA features
    print("Extracting TDA features...")
    tda_extractor = TDAFeatureExtractor(window_size=tda_window)
    tda_features = tda_extractor.extract_features(prices_normalized)

    # Pad TDA features to match price length
    padding = np.zeros((tda_window - 1, 3))
    tda_features = np.vstack([padding, tda_features])

    # Normalize TDA features
    tda_scaler = StandardScaler()
    tda_features = tda_scaler.fit_transform(tda_features)

    # Extract exogenous features if available
    exog_cols = ['volume', 'buy_volume', 'sell_volume', 'volume_delta', 'cvd']
    available_cols = [c for c in exog_cols if c in df.columns]

    if available_cols:
        exog_features = df[available_cols].values.astype(np.float32)
        exog_scaler = StandardScaler()
        exog_features = exog_scaler.fit_transform(exog_features)
    else:
        exog_features = None

    print(f"Data loaded: {len(prices)} samples")
    print(f"TDA features shape: {tda_features.shape}")
    if exog_features is not None:
        print(f"Exogenous features shape: {exog_features.shape}")

    return prices_normalized, tda_features, exog_features, scaler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_exog: bool = False
) -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        tda = batch['tda'].to(device)

        optimizer.zero_grad()

        if use_exog and 'exog' in batch:
            exog = batch['exog'].to(device)
            # Combine TDA with other exogenous features
            combined_exog = torch.cat([tda, exog], dim=1)
            pred = model(x, combined_exog)
        else:
            pred = model(x, tda)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_exog: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            tda = batch['tda'].to(device)

            if use_exog and 'exog' in batch:
                exog = batch['exog'].to(device)
                combined_exog = torch.cat([tda, exog], dim=1)
                pred = model(x, combined_exog)
            else:
                pred = model(x, tda)

            loss = criterion(pred, y)
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return avg_loss, all_preds, all_targets


def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    # MSE
    mse = np.mean((preds - targets) ** 2)

    # MAE
    mae = np.mean(np.abs(preds - targets))

    # MAPE
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
    lookback: int = 96,      # 8 hours
    horizon: int = 12,        # 1 hour
    tda_window: int = 50,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    use_tda: bool = True,
    use_exog: bool = True,
    save_path: str = 'checkpoints'
):
    """
    Main training function
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    prices, tda_features, exog_features, scaler = load_and_preprocess_data(
        data_path, tda_window=tda_window
    )

    # Create dataset
    dataset = CryptoDataset(
        prices=prices,
        tda_features=tda_features,
        exog_features=exog_features if use_exog else None,
        lookback=lookback,
        horizon=horizon
    )

    # Train/val/test split
    n_samples = len(dataset)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    # Time series split (no shuffle)
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

    # Create model
    if use_exog and exog_features is not None:
        n_exog = 3 + exog_features.shape[1]  # TDA + other exog
        model = NBeatsWithExogenous(
            lookback=lookback,
            horizon=horizon,
            n_exog_features=n_exog
        )
    else:
        model = NBeatsWithTDA(
            lookback=lookback,
            horizon=horizon,
            use_tda=use_tda
        )

    model = model.to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_exog=use_exog and exog_features is not None
        )
        val_loss, _, _ = evaluate(
            model, val_loader, criterion, device,
            use_exog=use_exog and exog_features is not None
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f'{save_path}/best_model.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'{save_path}/best_model.pt'))
    test_loss, preds, targets = evaluate(
        model, test_loader, criterion, device,
        use_exog=use_exog and exog_features is not None
    )

    metrics = calculate_metrics(preds, targets)

    print("\n=== Test Results ===")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

    # Plot results
    plot_results(train_losses, val_losses, preds, targets, save_path)

    return model, metrics


def plot_results(
    train_losses: list,
    val_losses: list,
    preds: np.ndarray,
    targets: np.ndarray,
    save_path: str
):
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

    # Prediction vs Actual (first horizon step)
    n_show = min(200, len(preds))
    axes[0, 1].plot(targets[:n_show, 0], label='Actual', alpha=0.7)
    axes[0, 1].plot(preds[:n_show, 0], label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Price (normalized)')
    axes[0, 1].set_title('Predictions vs Actual (1-step ahead)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Scatter plot
    axes[1, 0].scatter(targets[:, 0], preds[:, 0], alpha=0.3, s=10)
    axes[1, 0].plot([targets.min(), targets.max()],
                     [targets.min(), targets.max()], 'r--')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title('Prediction Scatter Plot')
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
    plt.savefig(f'{save_path}/results.png', dpi=150)
    plt.close()
    print(f"\nResults saved to {save_path}/results.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train N-BEATS with TDA features')
    parser.add_argument('--data', type=str, default='data/BTCUSDT_perp_5m.csv',
                        help='Path to data CSV')
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
    parser.add_argument('--no-tda', action='store_true',
                        help='Disable TDA features')
    parser.add_argument('--no-exog', action='store_true',
                        help='Disable exogenous features')

    args = parser.parse_args()

    train(
        data_path=args.data,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_tda=not args.no_tda,
        use_exog=not args.no_exog
    )
