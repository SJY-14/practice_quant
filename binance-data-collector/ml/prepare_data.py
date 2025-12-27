"""
Data Preparation: Merge Futures and Spot data with correlation features
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_merge_data(perp_path: str, spot_path: str) -> pd.DataFrame:
    """
    Load futures and spot data, merge them, and create correlation features

    Args:
        perp_path: Path to perpetual futures CSV
        spot_path: Path to spot CSV

    Returns:
        Merged DataFrame with correlation features
    """
    print("Loading data...")
    df_perp = pd.read_csv(perp_path)
    df_spot = pd.read_csv(spot_path)

    print(f"  Perp rows: {len(df_perp)}")
    print(f"  Spot rows: {len(df_spot)}")

    # Convert timestamp
    df_perp['open_time'] = pd.to_datetime(df_perp['open_time'])
    df_spot['open_time'] = pd.to_datetime(df_spot['open_time'])

    # Merge on timestamp (inner join to ensure same timestamps)
    print("\nMerging datasets...")
    df_merged = pd.merge(
        df_perp,
        df_spot,
        on='open_time',
        suffixes=('_perp', '_spot'),
        how='inner'
    )

    print(f"  Merged rows: {len(df_merged)}")

    # Create correlation features
    print("\nCreating correlation features...")

    # 1. Price Spread (Futures premium/discount)
    df_merged['price_spread'] = df_merged['close_perp'] - df_merged['close_spot']
    df_merged['price_spread_pct'] = (df_merged['price_spread'] / df_merged['close_spot']) * 100

    # 2. Volume Ratio (Futures vs Spot activity)
    df_merged['volume_ratio'] = df_merged['volume_perp'] / (df_merged['volume_spot'] + 1e-8)

    # 3. CVD Difference (Buying/Selling pressure difference)
    df_merged['cvd_diff'] = df_merged['cvd_perp'] - df_merged['cvd_spot']

    # 4. Volume Delta Difference
    df_merged['volume_delta_diff'] = df_merged['volume_delta_perp'] - df_merged['volume_delta_spot']

    # 5. Price momentum alignment (both rising/falling together)
    df_merged['perp_return'] = df_merged['close_perp'].pct_change()
    df_merged['spot_return'] = df_merged['close_spot'].pct_change()
    df_merged['return_correlation'] = df_merged['perp_return'] * df_merged['spot_return']  # Positive when aligned

    # 6. Spread momentum (is spread increasing or decreasing?)
    df_merged['spread_change'] = df_merged['price_spread'].diff()

    # 7. Buy/Sell volume ratio difference
    df_merged['buy_ratio_perp'] = df_merged['buy_volume_perp'] / (df_merged['volume_perp'] + 1e-8)
    df_merged['buy_ratio_spot'] = df_merged['buy_volume_spot'] / (df_merged['volume_spot'] + 1e-8)
    df_merged['buy_ratio_diff'] = df_merged['buy_ratio_perp'] - df_merged['buy_ratio_spot']

    # Fill NaN values from diff/pct_change
    df_merged = df_merged.fillna(method='bfill').fillna(0)

    print("\n" + "="*60)
    print("CORRELATION FEATURES CREATED:")
    print("="*60)
    print("  1. price_spread: Futures - Spot price")
    print("  2. price_spread_pct: Spread as % of spot price")
    print("  3. volume_ratio: Futures/Spot volume ratio")
    print("  4. cvd_diff: CVD difference")
    print("  5. volume_delta_diff: Volume delta difference")
    print("  6. return_correlation: Price movement alignment")
    print("  7. spread_change: Spread momentum")
    print("  8. buy_ratio_diff: Buy pressure difference")
    print("="*60)

    # Statistics
    print("\nSpread Statistics:")
    print(f"  Mean spread: ${df_merged['price_spread'].mean():.2f}")
    print(f"  Std spread: ${df_merged['price_spread'].std():.2f}")
    print(f"  Mean spread %: {df_merged['price_spread_pct'].mean():.4f}%")

    print("\nVolume Statistics:")
    print(f"  Mean volume ratio: {df_merged['volume_ratio'].mean():.2f}")
    print(f"  Perp avg volume: {df_merged['volume_perp'].mean():.2f}")
    print(f"  Spot avg volume: {df_merged['volume_spot'].mean():.2f}")

    return df_merged


def create_feature_dataframe(df_merged: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Create final feature DataFrame for model training

    Args:
        df_merged: Merged DataFrame
        output_path: Optional path to save CSV

    Returns:
        Feature DataFrame ready for training
    """
    print("\nCreating feature DataFrame...")

    # Select features for model
    features = pd.DataFrame({
        'open_time': df_merged['open_time'],

        # Primary target: Perpetual futures price
        'close': df_merged['close_perp'],

        # OHLCV from perpetual
        'open': df_merged['open_perp'],
        'high': df_merged['high_perp'],
        'low': df_merged['low_perp'],
        'volume': df_merged['volume_perp'],

        # Perpetual-specific
        'buy_volume': df_merged['buy_volume_perp'],
        'sell_volume': df_merged['sell_volume_perp'],
        'volume_delta': df_merged['volume_delta_perp'],
        'cvd': df_merged['cvd_perp'],

        # Spot reference
        'spot_close': df_merged['close_spot'],
        'spot_volume': df_merged['volume_spot'],
        'spot_cvd': df_merged['cvd_spot'],

        # Correlation features (NEW!)
        'price_spread': df_merged['price_spread'],
        'price_spread_pct': df_merged['price_spread_pct'],
        'volume_ratio': df_merged['volume_ratio'],
        'cvd_diff': df_merged['cvd_diff'],
        'volume_delta_diff': df_merged['volume_delta_diff'],
        'return_correlation': df_merged['return_correlation'],
        'spread_change': df_merged['spread_change'],
        'buy_ratio_diff': df_merged['buy_ratio_diff'],
    })

    # Add Open Interest if available
    if 'open_interest' in df_merged.columns:
        features['open_interest'] = df_merged['open_interest'].fillna(method='ffill').fillna(0)

    print(f"  Features shape: {features.shape}")
    print(f"  Feature columns: {len(features.columns)}")

    if output_path:
        features.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")

    return features


if __name__ == "__main__":
    # Paths
    perp_path = '../BTCUSDT_perp_5m.csv'
    spot_path = '../BTCUSDT_spot_5m.csv'
    output_path = '../BTCUSDT_merged_features.csv'

    # Load and merge
    df_merged = load_and_merge_data(perp_path, spot_path)

    # Create feature DataFrame
    df_features = create_feature_dataframe(df_merged, output_path)

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Total samples: {len(df_features)}")
    print(f"Date range: {df_features['open_time'].min()} to {df_features['open_time'].max()}")
    print(f"Output file: {output_path}")
    print("="*60)
