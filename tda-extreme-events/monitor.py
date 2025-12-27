"""
Real-time TDA Monitoring System for Bitcoin
Monitors current market conditions and alerts for extreme events
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from tda_analysis import TDAExtremeEventDetector
import warnings
warnings.filterwarnings('ignore')


class TDAMonitor:
    """Real-time TDA monitoring system."""

    def __init__(self, historical_data_path, window_size=60, lookback_days=30):
        """
        Initialize monitor.

        Parameters:
        -----------
        historical_data_path : str
            Path to historical data for threshold calculation
        window_size : int
            Size of sliding window
        lookback_days : int
            Days of history to use for threshold calculation
        """
        self.window_size = window_size
        self.lookback_days = lookback_days
        self.detector = TDAExtremeEventDetector(window_size=window_size)

        # Load historical data
        print(f"Loading historical data from: {historical_data_path}")
        self.df_history = pd.read_csv(historical_data_path)
        self.df_history['timestamp'] = pd.to_datetime(self.df_history['timestamp'])

        # Calculate baseline thresholds
        print("Calculating baseline thresholds from historical data...")
        self.calculate_baseline_thresholds()

        # Status file
        self.status_file = 'monitoring_status.json'

    def calculate_baseline_thresholds(self):
        """Calculate thresholds from historical data."""
        # Use last N days for threshold calculation
        cutoff_date = self.df_history['timestamp'].max() - timedelta(days=self.lookback_days)
        df_recent = self.df_history[self.df_history['timestamp'] >= cutoff_date].copy()

        print(f"Using {len(df_recent)} data points from last {self.lookback_days} days")

        # Prepare data
        features = {
            'close': df_recent['close'].values,
            'volume': df_recent['volume'].values,
            'volume_delta': df_recent['volume_delta'].values,
            'cvd': df_recent['cvd'].values
        }

        # Normalize
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        for key in features:
            features[key] = scaler.fit_transform(features[key].reshape(-1, 1)).flatten()

        point_cloud = self.detector.create_multivariate_point_cloud(features)

        # Run TDA analysis
        print("Running baseline TDA analysis...")
        results = self.detector.sliding_window_analysis(point_cloud, homology_dim=1)

        # Calculate thresholds
        self.baseline_stats = {
            'l1_norm': {
                'mean': np.mean(results['l1_norms']),
                'std': np.std(results['l1_norms']),
                'threshold': np.mean(results['l1_norms']) + 4 * np.std(results['l1_norms']),
                'threshold_70': np.mean(results['l1_norms']) + 2.8 * np.std(results['l1_norms']),  # 70%
                'threshold_90': np.mean(results['l1_norms']) + 3.6 * np.std(results['l1_norms']),  # 90%
            },
            'l2_norm': {
                'mean': np.mean(results['l2_norms']),
                'std': np.std(results['l2_norms']),
                'threshold': np.mean(results['l2_norms']) + 4 * np.std(results['l2_norms']),
                'threshold_70': np.mean(results['l2_norms']) + 2.8 * np.std(results['l2_norms']),
                'threshold_90': np.mean(results['l2_norms']) + 3.6 * np.std(results['l2_norms']),
            },
            'wasserstein': {
                'mean': np.mean(results['wasserstein_distances']),
                'std': np.std(results['wasserstein_distances']),
                'threshold': np.mean(results['wasserstein_distances']) + 4 * np.std(results['wasserstein_distances']),
                'threshold_70': np.mean(results['wasserstein_distances']) + 2.8 * np.std(results['wasserstein_distances']),
                'threshold_90': np.mean(results['wasserstein_distances']) + 3.6 * np.std(results['wasserstein_distances']),
            }
        }

        print("\n" + "="*60)
        print("BASELINE THRESHOLDS CALCULATED")
        print("="*60)
        for metric, stats in self.baseline_stats.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Threshold (μ + 4σ): {stats['threshold']:.6f}")
        print("="*60 + "\n")

    def analyze_current_window(self, recent_data):
        """
        Analyze current market window.

        Parameters:
        -----------
        recent_data : DataFrame
            Most recent window_size data points

        Returns:
        --------
        analysis : dict
            Current metrics and alert levels
        """
        if len(recent_data) < self.window_size:
            return None

        # Get last window_size points
        window = recent_data.tail(self.window_size).copy()

        # Prepare features
        features = {
            'close': window['close'].values,
            'volume': window['volume'].values,
            'volume_delta': window['volume_delta'].values,
            'cvd': window['cvd'].values
        }

        # Normalize
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        for key in features:
            features[key] = scaler.fit_transform(features[key].reshape(-1, 1)).flatten()

        point_cloud = self.detector.create_multivariate_point_cloud(features)

        # Compute persistence diagram
        diagrams = self.detector.compute_persistence_diagram(point_cloud)
        dgm = diagrams[1]  # H1

        # Compute landscape and norms
        landscape = self.detector.compute_persistence_landscape(dgm)
        l1_norm = self.detector.compute_lp_norm(landscape, p=1)
        l2_norm = self.detector.compute_lp_norm(landscape, p=2)

        # Calculate percentages
        l1_pct = (l1_norm / self.baseline_stats['l1_norm']['threshold']) * 100
        l2_pct = (l2_norm / self.baseline_stats['l2_norm']['threshold']) * 100

        # Determine alert level
        alert_level = self.determine_alert_level(l1_norm, l2_norm)

        analysis = {
            'timestamp': window['timestamp'].iloc[-1],
            'price': window['close'].iloc[-1],
            'metrics': {
                'l1_norm': {
                    'value': l1_norm,
                    'percent_of_threshold': l1_pct,
                    'threshold': self.baseline_stats['l1_norm']['threshold']
                },
                'l2_norm': {
                    'value': l2_norm,
                    'percent_of_threshold': l2_pct,
                    'threshold': self.baseline_stats['l2_norm']['threshold']
                }
            },
            'alert_level': alert_level,
            'persistence_diagram': dgm
        }

        return analysis

    def determine_alert_level(self, l1_norm, l2_norm):
        """
        Determine alert level based on metrics.

        Returns:
        --------
        alert : dict
            Alert level and message
        """
        l1_stats = self.baseline_stats['l1_norm']
        l2_stats = self.baseline_stats['l2_norm']

        # Check both L1 and L2
        alerts = []

        # L1 Norm alerts
        if l1_norm >= l1_stats['threshold']:
            alerts.append(('CRITICAL', 'L1 Norm exceeds threshold'))
        elif l1_norm >= l1_stats['threshold_90']:
            alerts.append(('SEVERE', 'L1 Norm at 90% of threshold'))
        elif l1_norm >= l1_stats['threshold_70']:
            alerts.append(('WARNING', 'L1 Norm at 70% of threshold'))

        # L2 Norm alerts
        if l2_norm >= l2_stats['threshold']:
            alerts.append(('CRITICAL', 'L2 Norm exceeds threshold'))
        elif l2_norm >= l2_stats['threshold_90']:
            alerts.append(('SEVERE', 'L2 Norm at 90% of threshold'))
        elif l2_norm >= l2_stats['threshold_70']:
            alerts.append(('WARNING', 'L2 Norm at 70% of threshold'))

        if not alerts:
            return {
                'level': 'NORMAL',
                'symbol': '✅',
                'color': 'green',
                'message': 'Market conditions normal'
            }

        # Get highest severity alert
        severity_order = {'CRITICAL': 3, 'SEVERE': 2, 'WARNING': 1}
        alerts.sort(key=lambda x: severity_order[x[0]], reverse=True)
        highest = alerts[0]

        alert_configs = {
            'CRITICAL': {'symbol': '🚨', 'color': 'red'},
            'SEVERE': {'symbol': '⚠️', 'color': 'orange'},
            'WARNING': {'symbol': '⚡', 'color': 'yellow'}
        }

        config = alert_configs[highest[0]]

        return {
            'level': highest[0],
            'symbol': config['symbol'],
            'color': config['color'],
            'message': '; '.join([a[1] for a in alerts])
        }

    def print_status(self, analysis):
        """Print current status to console."""
        timestamp = analysis['timestamp']
        price = analysis['price']
        metrics = analysis['metrics']
        alert = analysis['alert_level']

        # Create progress bars
        def progress_bar(percent, width=30):
            filled = int(width * percent / 100)
            bar = '█' * filled + '░' * (width - filled)
            return bar

        print("\n" + "="*70)
        print(f"🔍 BITCOIN TDA MONITOR - {timestamp}")
        print("="*70)
        print(f"\n💰 Current Price: ${price:,.2f}")

        print(f"\n📊 TOPOLOGICAL METRICS:\n")

        # L1 Norm
        l1 = metrics['l1_norm']
        l1_bar = progress_bar(l1['percent_of_threshold'])
        print(f"  L¹ Norm:  [{l1_bar}] {l1['percent_of_threshold']:.1f}%")
        print(f"            Value: {l1['value']:.6f} | Threshold: {l1['threshold']:.6f}")

        # L2 Norm
        l2 = metrics['l2_norm']
        l2_bar = progress_bar(l2['percent_of_threshold'])
        print(f"\n  L² Norm:  [{l2_bar}] {l2['percent_of_threshold']:.1f}%")
        print(f"            Value: {l2['value']:.6f} | Threshold: {l2['threshold']:.6f}")

        # Alert status
        print(f"\n🎯 STATUS: {alert['symbol']} {alert['level']}")
        print(f"   {alert['message']}")

        print("\n" + "="*70)

    def save_status(self, analysis):
        """Save current status to JSON file for web dashboard."""
        data = {
            'timestamp': str(analysis['timestamp']),
            'price': float(analysis['price']),
            'l1_norm': {
                'value': float(analysis['metrics']['l1_norm']['value']),
                'percent': float(analysis['metrics']['l1_norm']['percent_of_threshold']),
                'threshold': float(analysis['metrics']['l1_norm']['threshold'])
            },
            'l2_norm': {
                'value': float(analysis['metrics']['l2_norm']['value']),
                'percent': float(analysis['metrics']['l2_norm']['percent_of_threshold']),
                'threshold': float(analysis['metrics']['l2_norm']['threshold'])
            },
            'alert': analysis['alert_level'],
            'baseline_stats': {
                'l1_norm': {
                    'mean': float(self.baseline_stats['l1_norm']['mean']),
                    'std': float(self.baseline_stats['l1_norm']['std']),
                    'threshold': float(self.baseline_stats['l1_norm']['threshold'])
                },
                'l2_norm': {
                    'mean': float(self.baseline_stats['l2_norm']['mean']),
                    'std': float(self.baseline_stats['l2_norm']['std']),
                    'threshold': float(self.baseline_stats['l2_norm']['threshold'])
                }
            }
        }

        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)

    def run_once(self):
        """Run monitoring once with current data."""
        print("Analyzing current market condition...")

        # Use all data for current analysis
        analysis = self.analyze_current_window(self.df_history)

        if analysis:
            self.print_status(analysis)
            self.save_status(analysis)
            return analysis
        else:
            print("❌ Insufficient data for analysis")
            return None

    def run_continuous(self, interval_minutes=5):
        """Run monitoring continuously."""
        print(f"\n🚀 Starting continuous monitoring (checking every {interval_minutes} minutes)...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                self.run_once()
                print(f"\n⏱️  Next check in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Bitcoin TDA Monitor')
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous mode')
    parser.add_argument('--interval', type=int, default=5,
                       help='Check interval in minutes (default: 5)')
    parser.add_argument('--data', type=str,
                       default='../-binance-data-collector/BTCUSDT_5m.csv',
                       help='Path to Bitcoin data')

    args = parser.parse_args()

    # Initialize monitor
    monitor = TDAMonitor(
        historical_data_path=args.data,
        window_size=60,
        lookback_days=30
    )

    if args.continuous:
        monitor.run_continuous(interval_minutes=args.interval)
    else:
        monitor.run_once()


if __name__ == '__main__':
    main()
