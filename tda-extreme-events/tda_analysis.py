"""
Topological Data Analysis for Extreme Event Detection in Bitcoin Market
Based on: "Identifying Extreme Events in the Stock Market: A Topological Data Analysis"
by Anish Rai et al. (arXiv:2405.16052)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from persim import wasserstein
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class TDAExtremeEventDetector:
    """
    Detect extreme events in financial time series using Topological Data Analysis.

    The method uses:
    - L1 and L2 norms of persistence landscapes
    - Wasserstein distance between consecutive persistence diagrams
    - Threshold: μ + 4σ for extreme event detection
    """

    def __init__(self, window_size=60, maxdim=1, threshold_sigma=4):
        """
        Parameters:
        -----------
        window_size : int
            Size of the sliding window
        maxdim : int
            Maximum homology dimension to compute (0 or 1)
        threshold_sigma : int
            Number of standard deviations for threshold (default: 4)
        """
        self.window_size = window_size
        self.maxdim = maxdim
        self.threshold_sigma = threshold_sigma

    def compute_log_returns(self, prices):
        """
        Compute log returns from price series.

        Parameters:
        -----------
        prices : array-like
            Price time series

        Returns:
        --------
        log_returns : np.array
            Log returns
        """
        prices = np.array(prices)
        log_returns = np.log(prices[1:] / prices[:-1])
        return log_returns

    def create_point_cloud_takens(self, time_series, embedding_dim=3, delay=1):
        """
        Create point cloud using Takens' embedding theorem.
        Converts 1D time series to multi-dimensional point cloud.

        Parameters:
        -----------
        time_series : array-like
            1D time series data
        embedding_dim : int
            Embedding dimension
        delay : int
            Time delay

        Returns:
        --------
        point_cloud : np.array
            Embedded point cloud
        """
        n = len(time_series)
        m = n - (embedding_dim - 1) * delay

        if m <= 0:
            raise ValueError(f"Time series too short for embedding_dim={embedding_dim}, delay={delay}")

        point_cloud = np.zeros((m, embedding_dim))
        for i in range(embedding_dim):
            point_cloud[:, i] = time_series[i * delay:i * delay + m]

        return point_cloud

    def create_multivariate_point_cloud(self, data_dict):
        """
        Create point cloud from multiple time series (like multiple stocks).

        Parameters:
        -----------
        data_dict : dict
            Dictionary of time series {name: series}

        Returns:
        --------
        point_cloud : np.array
            Multi-dimensional point cloud
        """
        series_list = [np.array(v) for v in data_dict.values()]
        min_len = min(len(s) for s in series_list)

        # Truncate all series to same length
        series_list = [s[:min_len] for s in series_list]

        # Stack into point cloud
        point_cloud = np.column_stack(series_list)
        return point_cloud

    def compute_persistence_diagram(self, point_cloud):
        """
        Compute persistence diagram using Vietoris-Rips complex.

        Parameters:
        -----------
        point_cloud : np.array
            Point cloud data

        Returns:
        --------
        diagrams : dict
            Persistence diagrams for each dimension
        """
        result = ripser(point_cloud, maxdim=self.maxdim)
        return result['dgms']

    def compute_persistence_landscape(self, diagram, num_landscapes=5, resolution=100):
        """
        Compute persistence landscape from persistence diagram.

        Parameters:
        -----------
        diagram : np.array
            Persistence diagram (birth-death pairs)
        num_landscapes : int
            Number of landscape functions
        resolution : int
            Number of points for discretization

        Returns:
        --------
        landscapes : np.array
            Persistence landscapes (num_landscapes x resolution)
        """
        # Remove infinite points
        diagram = diagram[diagram[:, 1] < np.inf]

        if len(diagram) == 0:
            return np.zeros((num_landscapes, resolution))

        # Create grid
        min_val = diagram[:, 0].min()
        max_val = diagram[:, 1].max()
        grid = np.linspace(min_val, max_val, resolution)

        # Compute landscape functions
        landscapes = np.zeros((num_landscapes, resolution))

        for i, x in enumerate(grid):
            # For each point x, compute heights of all tents
            heights = []
            for birth, death in diagram:
                mid = (birth + death) / 2
                if birth <= x <= death:
                    if x <= mid:
                        height = x - birth
                    else:
                        height = death - x
                    heights.append(height)

            # Sort heights in descending order
            heights = sorted(heights, reverse=True)

            # Assign to landscapes
            for k in range(min(num_landscapes, len(heights))):
                landscapes[k, i] = heights[k]

        return landscapes

    def compute_lp_norm(self, landscape, p=1):
        """
        Compute Lp norm of persistence landscape.

        Parameters:
        -----------
        landscape : np.array
            Persistence landscape
        p : int
            Norm order (1 or 2)

        Returns:
        --------
        norm : float
            Lp norm value
        """
        if p == 1:
            # L1 norm: sum of absolute values
            norm = np.sum(np.abs(landscape))
        elif p == 2:
            # L2 norm: square root of sum of squares
            norm = np.sqrt(np.sum(landscape ** 2))
        else:
            norm = np.sum(np.abs(landscape) ** p) ** (1/p)

        return norm

    def compute_wasserstein_distance(self, dgm1, dgm2, p=2):
        """
        Compute Wasserstein distance between two persistence diagrams.

        Parameters:
        -----------
        dgm1, dgm2 : np.array
            Persistence diagrams
        p : int
            Wasserstein order

        Returns:
        --------
        distance : float
            Wasserstein distance
        """
        # Remove infinite points
        dgm1 = dgm1[dgm1[:, 1] < np.inf]
        dgm2 = dgm2[dgm2[:, 1] < np.inf]

        if len(dgm1) == 0 or len(dgm2) == 0:
            return 0.0

        try:
            distance = wasserstein(dgm1, dgm2, matching=False)
            return distance
        except:
            return 0.0

    def sliding_window_analysis(self, point_cloud, homology_dim=1):
        """
        Perform TDA analysis using sliding window.

        Parameters:
        -----------
        point_cloud : np.array
            Full point cloud data
        homology_dim : int
            Which homology dimension to analyze (0 or 1)

        Returns:
        --------
        results : dict
            Dictionary containing L1 norms, L2 norms, and Wasserstein distances
        """
        n = len(point_cloud)
        num_windows = n - self.window_size + 1

        l1_norms = []
        l2_norms = []
        wasserstein_distances = []
        persistence_diagrams = []

        print(f"Computing TDA for {num_windows} windows...")

        for i in range(num_windows):
            if i % 500 == 0:
                print(f"  Window {i}/{num_windows}")

            # Extract window
            window = point_cloud[i:i + self.window_size]

            # Compute persistence diagram
            diagrams = self.compute_persistence_diagram(window)
            dgm = diagrams[homology_dim]
            persistence_diagrams.append(dgm)

            # Compute persistence landscape
            landscape = self.compute_persistence_landscape(dgm)

            # Compute norms
            l1 = self.compute_lp_norm(landscape, p=1)
            l2 = self.compute_lp_norm(landscape, p=2)

            l1_norms.append(l1)
            l2_norms.append(l2)

            # Compute Wasserstein distance with previous diagram
            if i > 0:
                wd = self.compute_wasserstein_distance(
                    persistence_diagrams[i-1],
                    persistence_diagrams[i]
                )
                wasserstein_distances.append(wd)
            else:
                wasserstein_distances.append(0.0)

        results = {
            'l1_norms': np.array(l1_norms),
            'l2_norms': np.array(l2_norms),
            'wasserstein_distances': np.array(wasserstein_distances),
            'persistence_diagrams': persistence_diagrams
        }

        return results

    def detect_extreme_events(self, values):
        """
        Detect extreme events using μ + 4σ threshold.

        Parameters:
        -----------
        values : np.array
            Time series of norms or distances

        Returns:
        --------
        extreme_events : dict
            Dictionary with threshold, mean, std, and event indices
        """
        mean = np.mean(values)
        std = np.std(values)
        threshold = mean + self.threshold_sigma * std

        # Find indices where values exceed threshold
        event_indices = np.where(values > threshold)[0]

        return {
            'threshold': threshold,
            'mean': mean,
            'std': std,
            'event_indices': event_indices,
            'event_values': values[event_indices] if len(event_indices) > 0 else np.array([])
        }

    def plot_results(self, timestamps, results, price_data=None, title_prefix="Bitcoin"):
        """
        Plot TDA analysis results.

        Parameters:
        -----------
        timestamps : array-like
            Time stamps
        results : dict
            Results from sliding_window_analysis
        price_data : array-like, optional
            Price data for reference
        title_prefix : str
            Prefix for plot titles
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        # Adjust timestamps to match window results
        # Ensure timestamps match the exact length of results
        num_results = len(results['l1_norms'])
        plot_timestamps = timestamps[self.window_size-1:self.window_size-1+num_results]

        # Plot 1: L1 Norm
        ax = axes[0]
        l1_events = self.detect_extreme_events(results['l1_norms'])
        ax.plot(plot_timestamps, results['l1_norms'], 'k-', linewidth=0.8, label='L1 Norm')
        ax.axhline(l1_events['mean'], color='blue', linestyle='--', label=f'Mean (μ)')
        ax.axhline(l1_events['threshold'], color='red', linestyle='-',
                   label=f'Threshold (μ + {self.threshold_sigma}σ)')

        # Highlight extreme events
        if len(l1_events['event_indices']) > 0:
            event_times = plot_timestamps[l1_events['event_indices']]
            event_vals = l1_events['event_values']
            ax.scatter(event_times, event_vals, color='green', s=50, zorder=5,
                      label='Extreme Events', marker='o')

        ax.set_ylabel('L¹ Norm', fontsize=12)
        ax.set_title(f'{title_prefix} - L¹ Norm', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 2: L2 Norm
        ax = axes[1]
        l2_events = self.detect_extreme_events(results['l2_norms'])
        ax.plot(plot_timestamps, results['l2_norms'], 'k-', linewidth=0.8, label='L2 Norm')
        ax.axhline(l2_events['mean'], color='blue', linestyle='--', label=f'Mean (μ)')
        ax.axhline(l2_events['threshold'], color='red', linestyle='-',
                   label=f'Threshold (μ + {self.threshold_sigma}σ)')

        if len(l2_events['event_indices']) > 0:
            event_times = plot_timestamps[l2_events['event_indices']]
            event_vals = l2_events['event_values']
            ax.scatter(event_times, event_vals, color='green', s=50, zorder=5,
                      label='Extreme Events', marker='o')

        ax.set_ylabel('L² Norm', fontsize=12)
        ax.set_title(f'{title_prefix} - L² Norm', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 3: Wasserstein Distance
        ax = axes[2]
        wd_events = self.detect_extreme_events(results['wasserstein_distances'])
        ax.plot(plot_timestamps, results['wasserstein_distances'], 'k-',
                linewidth=0.8, label='Wasserstein Distance')
        ax.axhline(wd_events['mean'], color='blue', linestyle='--', label=f'Mean (μ)')
        ax.axhline(wd_events['threshold'], color='red', linestyle='-',
                   label=f'Threshold (μ + {self.threshold_sigma}σ)')

        if len(wd_events['event_indices']) > 0:
            event_times = plot_timestamps[wd_events['event_indices']]
            event_vals = wd_events['event_values']
            ax.scatter(event_times, event_vals, color='green', s=50, zorder=5,
                      label='Extreme Events', marker='o')

        ax.set_ylabel('Wasserstein Distance', fontsize=12)
        ax.set_title(f'{title_prefix} - Wasserstein Distance', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 4: Price (if provided)
        if price_data is not None:
            ax = axes[3]
            ax.plot(timestamps, price_data, 'b-', linewidth=1.0)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title(f'{title_prefix} Price', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Mark extreme events on price chart
            all_events = set()
            for events in [l1_events, l2_events, wd_events]:
                all_events.update(events['event_indices'])

            if len(all_events) > 0:
                all_events = sorted(list(all_events))
                event_times = plot_timestamps[all_events]
                # Find corresponding prices
                event_prices = []
                for et in event_times:
                    idx = np.argmin(np.abs(timestamps - et))
                    event_prices.append(price_data[idx])

                ax.scatter(event_times, event_prices, color='red', s=100,
                          zorder=5, label='Detected Extreme Events', marker='v')
                ax.legend(loc='upper left')

        plt.tight_layout()
        return fig

    def print_summary(self, results, timestamps):
        """Print summary of extreme events detected."""
        print("\n" + "="*80)
        print("EXTREME EVENT DETECTION SUMMARY")
        print("="*80)

        plot_timestamps = timestamps[self.window_size-1:]

        for metric_name, values in [('L¹ Norm', results['l1_norms']),
                                     ('L² Norm', results['l2_norms']),
                                     ('Wasserstein Distance', results['wasserstein_distances'])]:
            events = self.detect_extreme_events(values)
            print(f"\n{metric_name}:")
            print(f"  Mean (μ): {events['mean']:.6f}")
            print(f"  Std (σ): {events['std']:.6f}")
            print(f"  Threshold (μ + {self.threshold_sigma}σ): {events['threshold']:.6f}")
            print(f"  Number of Extreme Events: {len(events['event_indices'])}")

            if len(events['event_indices']) > 0:
                print(f"  Extreme Event Dates:")
                for idx in events['event_indices'][:10]:  # Show first 10
                    date = plot_timestamps[idx]
                    value = values[idx]
                    print(f"    - {date}: {value:.6f}")
                if len(events['event_indices']) > 10:
                    print(f"    ... and {len(events['event_indices']) - 10} more")

        print("\n" + "="*80)
