"""
Main script to reproduce the TDA extreme event detection paper
using Bitcoin data from the binance-data-collector
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Import TDA analyzer
from tda_analysis import TDAExtremeEventDetector

def load_bitcoin_data(file_path):
    """Load Bitcoin data from CSV file."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df

def prepare_data_for_tda(df, use_multivariate=True):
    """
    Prepare data for TDA analysis.

    Two approaches:
    1. Multivariate: Use multiple features (close, volume, cvd, etc.) as different dimensions
    2. Univariate with Takens embedding: Use only price with time-delay embedding
    """
    print(f"\nPreparing data for TDA (multivariate={use_multivariate})...")

    if use_multivariate:
        # Use multiple features to create point cloud
        # This simulates analyzing multiple stocks simultaneously (as in the paper)
        features = {
            'close': df['close'].values,
            'volume': df['volume'].values,
            'volume_delta': df['volume_delta'].values,
            'cvd': df['cvd'].values
        }

        print(f"Using multivariate approach with features: {list(features.keys())}")

        # Normalize each feature to [0, 1] for better TDA performance
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        for key in features:
            features[key] = scaler.fit_transform(features[key].reshape(-1, 1)).flatten()

        return features, 'multivariate'
    else:
        # Use only close price with Takens embedding
        print("Using univariate approach with Takens embedding")
        close_prices = df['close'].values
        return close_prices, 'univariate'

def main():
    """Main execution function."""
    print("="*80)
    print("TOPOLOGICAL DATA ANALYSIS FOR BITCOIN EXTREME EVENT DETECTION")
    print("Based on: Rai et al. (2024) - arXiv:2405.16052")
    print("="*80)

    # Load Bitcoin data
    data_path = '../-binance-data-collector/BTCUSDT_5m.csv'
    df = load_bitcoin_data(data_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timestamps = df['timestamp'].values
    close_prices = df['close'].values

    # Initialize TDA detector
    # Window size = 60 (as in paper for daily data, here we use 60 5-min candles = 5 hours)
    detector = TDAExtremeEventDetector(
        window_size=60,
        maxdim=1,  # Compute H0 and H1
        threshold_sigma=4  # μ + 4σ threshold
    )

    # ===== Analysis 1: Multivariate Approach =====
    print("\n" + "="*80)
    print("ANALYSIS 1: MULTIVARIATE APPROACH")
    print("Using multiple features (close, volume, volume_delta, cvd)")
    print("This simulates the paper's approach of analyzing multiple stocks together")
    print("="*80)

    data_multi, approach_type = prepare_data_for_tda(df, use_multivariate=True)

    # Create point cloud
    point_cloud_multi = detector.create_multivariate_point_cloud(data_multi)
    print(f"Point cloud shape: {point_cloud_multi.shape}")

    # Run TDA analysis
    print("\nRunning TDA analysis on multivariate data...")
    results_multi = detector.sliding_window_analysis(point_cloud_multi, homology_dim=1)

    # Print summary
    detector.print_summary(results_multi, timestamps)

    # Plot results
    print("\nGenerating plots...")
    fig1 = detector.plot_results(
        timestamps,
        results_multi,
        price_data=close_prices,
        title_prefix="Bitcoin (Multivariate)"
    )
    plt.savefig('results_multivariate.png', dpi=300, bbox_inches='tight')
    print("Saved: results_multivariate.png")

    # ===== Analysis 2: Univariate with Takens Embedding =====
    print("\n" + "="*80)
    print("ANALYSIS 2: UNIVARIATE APPROACH WITH TAKENS EMBEDDING")
    print("Using only close price with time-delay embedding")
    print("="*80)

    # Compute log returns
    log_returns = detector.compute_log_returns(close_prices)
    print(f"Log returns shape: {log_returns.shape}")

    # Create point cloud using Takens embedding
    embedding_dim = 3
    delay = 1
    point_cloud_uni = detector.create_point_cloud_takens(
        log_returns,
        embedding_dim=embedding_dim,
        delay=delay
    )
    print(f"Point cloud shape (Takens embedding): {point_cloud_uni.shape}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Time delay: {delay}")

    # Run TDA analysis
    print("\nRunning TDA analysis on univariate data...")
    results_uni = detector.sliding_window_analysis(point_cloud_uni, homology_dim=1)

    # Adjust timestamps for log returns
    timestamps_lr = timestamps[1:]  # Log returns start from index 1

    # Print summary
    detector.print_summary(results_uni, timestamps_lr)

    # Plot results
    print("\nGenerating plots...")
    fig2 = detector.plot_results(
        timestamps_lr,
        results_uni,
        price_data=close_prices[1:],
        title_prefix="Bitcoin (Takens Embedding)"
    )
    plt.savefig('results_takens.png', dpi=300, bbox_inches='tight')
    print("Saved: results_takens.png")

    # ===== Comparison Plot =====
    print("\nCreating comparison plot...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plot multivariate L1 norm
    ax = axes[0]
    plot_timestamps_multi = timestamps[detector.window_size-1:]
    l1_events_multi = detector.detect_extreme_events(results_multi['l1_norms'])

    ax.plot(plot_timestamps_multi, results_multi['l1_norms'], 'b-',
            linewidth=1.0, label='Multivariate', alpha=0.7)
    ax.axhline(l1_events_multi['threshold'], color='red', linestyle='--',
               linewidth=1.5, label=f'Threshold (μ + {detector.threshold_sigma}σ)')

    if len(l1_events_multi['event_indices']) > 0:
        event_times = plot_timestamps_multi[l1_events_multi['event_indices']]
        event_vals = l1_events_multi['event_values']
        ax.scatter(event_times, event_vals, color='red', s=100, zorder=5,
                  label='Extreme Events', marker='o')

    ax.set_ylabel('L¹ Norm', fontsize=12)
    ax.set_title('L¹ Norm - Multivariate Approach', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot Takens L1 norm
    ax = axes[1]
    # Adjust timestamps for Takens (log returns + embedding)
    offset = 1 + (embedding_dim - 1) * delay
    plot_timestamps_uni = timestamps[offset + detector.window_size - 1:]
    l1_events_uni = detector.detect_extreme_events(results_uni['l1_norms'])

    ax.plot(plot_timestamps_uni, results_uni['l1_norms'], 'g-',
            linewidth=1.0, label='Takens Embedding', alpha=0.7)
    ax.axhline(l1_events_uni['threshold'], color='red', linestyle='--',
               linewidth=1.5, label=f'Threshold (μ + {detector.threshold_sigma}σ)')

    if len(l1_events_uni['event_indices']) > 0:
        event_times = plot_timestamps_uni[l1_events_uni['event_indices']]
        event_vals = l1_events_uni['event_values']
        ax.scatter(event_times, event_vals, color='red', s=100, zorder=5,
                  label='Extreme Events', marker='o')

    ax.set_ylabel('L¹ Norm', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('L¹ Norm - Takens Embedding Approach', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: comparison.png")

    # Save numerical results
    print("\nSaving numerical results...")

    # Multivariate results
    results_df_multi = pd.DataFrame({
        'timestamp': plot_timestamps_multi,
        'l1_norm': results_multi['l1_norms'],
        'l2_norm': results_multi['l2_norms'],
        'wasserstein_distance': results_multi['wasserstein_distances']
    })
    results_df_multi.to_csv('results_multivariate.csv', index=False)
    print("Saved: results_multivariate.csv")

    # Takens results
    results_df_uni = pd.DataFrame({
        'timestamp': plot_timestamps_uni,
        'l1_norm': results_uni['l1_norms'],
        'l2_norm': results_uni['l2_norms'],
        'wasserstein_distance': results_uni['wasserstein_distances']
    })
    results_df_uni.to_csv('results_takens.csv', index=False)
    print("Saved: results_takens.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results_multivariate.png: Multivariate analysis plots")
    print("  - results_takens.png: Takens embedding analysis plots")
    print("  - comparison.png: Comparison of both methods")
    print("  - results_multivariate.csv: Numerical results (multivariate)")
    print("  - results_takens.csv: Numerical results (Takens)")
    print("\nPaper reference: arXiv:2405.16052")
    print("="*80)

if __name__ == '__main__':
    main()
