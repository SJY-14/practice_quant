"""
Real-time Bitcoin Price Prediction using TDA + Machine Learning
Combines Futures and Spot data from Binance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from tda_analysis import TDAExtremeEventDetector


class TDAPricePredictor:
    """
    Bitcoin price predictor using TDA features + Machine Learning
    """

    def __init__(self, window_size=60, forecast_horizon=12):
        """
        Parameters:
        -----------
        window_size : int
            TDA window size (default: 60 = 5 hours for 5min data)
        forecast_horizon : int
            How many steps ahead to predict (default: 12 = 1 hour)
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.tda_detector = TDAExtremeEventDetector(window_size=window_size)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.feature_names = []

    def load_data(self, futures_path, spot_path):
        """Load and merge futures + spot data."""
        print("📊 Loading data...")

        # Load futures
        df_futures = pd.read_csv(futures_path)
        df_futures['open_time'] = pd.to_datetime(df_futures['open_time'])
        df_futures = df_futures.add_suffix('_futures')
        df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

        # Load spot
        df_spot = pd.read_csv(spot_path)
        df_spot['open_time'] = pd.to_datetime(df_spot['open_time'])
        df_spot = df_spot.add_suffix('_spot')
        df_spot.rename(columns={'open_time_spot': 'open_time'}, inplace=True)

        # Merge on timestamp
        df = pd.merge(df_futures, df_spot, on='open_time', how='inner')

        print(f"  Loaded {len(df)} rows")
        print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")
        print(f"  Columns: {df.columns.tolist()}")

        return df

    def extract_tda_features(self, df, use_columns_prefix='futures'):
        """
        Extract TDA features from price data.

        Returns DataFrame with TDA features for each window.
        """
        print("\n🔬 Extracting TDA features...")

        # Select columns for TDA
        if use_columns_prefix == 'futures':
            price_col = 'close_futures'
            volume_col = 'volume_futures'
            cvd_col = 'cvd_futures'
            vd_col = 'volume_delta_futures'
        else:
            price_col = 'close_spot'
            volume_col = 'volume_spot'
            cvd_col = 'cvd_spot'
            vd_col = 'volume_delta_spot'

        # Prepare multivariate point cloud
        features = {
            'close': df[price_col].values,
            'volume': df[volume_col].values,
            'cvd': df[cvd_col].values,
            'volume_delta': df[vd_col].values
        }

        # Normalize
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        for key in features:
            features[key] = scaler.fit_transform(features[key].reshape(-1, 1)).flatten()

        # Create point cloud
        point_cloud = self.tda_detector.create_multivariate_point_cloud(features)

        # Run TDA analysis
        print("  Computing persistence diagrams...")
        results = self.tda_detector.sliding_window_analysis(point_cloud, homology_dim=1)

        # Create features DataFrame
        tda_features = pd.DataFrame({
            'l1_norm': results['l1_norms'],
            'l2_norm': results['l2_norms'],
            'wasserstein_dist': results['wasserstein_distances']
        })

        # Align with original dataframe
        tda_features.index = df.index[self.window_size-1:self.window_size-1+len(tda_features)]

        print(f"  Extracted {len(tda_features)} TDA feature vectors")

        return tda_features

    def create_features(self, df, tda_features):
        """
        Create comprehensive feature set for ML.

        Combines:
        - TDA features (L1, L2, WD)
        - Price features (futures & spot)
        - Volume features
        - Technical indicators
        """
        print("\n📈 Creating ML features...")

        # Start from where TDA features start
        df_aligned = df.iloc[self.window_size-1:self.window_size-1+len(tda_features)].copy()
        df_aligned.index = tda_features.index

        # Add TDA features
        df_aligned['tda_l1'] = tda_features['l1_norm'].values
        df_aligned['tda_l2'] = tda_features['l2_norm'].values
        df_aligned['tda_wd'] = tda_features['wasserstein_dist'].values

        # Price features
        df_aligned['price_futures'] = df_aligned['close_futures']
        df_aligned['price_spot'] = df_aligned['close_spot']
        df_aligned['price_spread'] = df_aligned['close_futures'] - df_aligned['close_spot']
        df_aligned['price_spread_pct'] = (df_aligned['close_futures'] - df_aligned['close_spot']) / df_aligned['close_spot'] * 100

        # Returns
        df_aligned['return_futures_1'] = df_aligned['close_futures'].pct_change(1)
        df_aligned['return_futures_5'] = df_aligned['close_futures'].pct_change(5)
        df_aligned['return_futures_12'] = df_aligned['close_futures'].pct_change(12)
        df_aligned['return_spot_1'] = df_aligned['close_spot'].pct_change(1)

        # Volume features
        df_aligned['volume_futures'] = df_aligned['volume_futures']
        df_aligned['volume_spot'] = df_aligned['volume_spot']
        df_aligned['volume_ratio'] = df_aligned['volume_futures'] / (df_aligned['volume_spot'] + 1e-8)
        df_aligned['cvd_futures'] = df_aligned['cvd_futures']
        df_aligned['cvd_spot'] = df_aligned['cvd_spot']

        # Technical indicators
        # Moving averages
        df_aligned['ma_5'] = df_aligned['close_futures'].rolling(5).mean()
        df_aligned['ma_12'] = df_aligned['close_futures'].rolling(12).mean()
        df_aligned['ma_24'] = df_aligned['close_futures'].rolling(24).mean()

        # Volatility
        df_aligned['volatility_5'] = df_aligned['close_futures'].rolling(5).std()
        df_aligned['volatility_12'] = df_aligned['close_futures'].rolling(12).std()

        # High-Low range
        df_aligned['hl_range_futures'] = df_aligned['high_futures'] - df_aligned['low_futures']
        df_aligned['hl_range_spot'] = df_aligned['high_spot'] - df_aligned['low_spot']

        # Drop NaN
        df_aligned = df_aligned.dropna()

        print(f"  Created {df_aligned.shape[1]} features for {len(df_aligned)} samples")

        return df_aligned

    def prepare_ml_dataset(self, df_features):
        """
        Prepare X, y for machine learning.

        X: Current features
        y: Future price (forecast_horizon steps ahead)
        """
        print(f"\n🎯 Preparing ML dataset (predicting {self.forecast_horizon} steps ahead)...")

        # Target: future price
        df_features['target'] = df_features['close_futures'].shift(-self.forecast_horizon)

        # Remove last forecast_horizon rows (no target)
        df_ml = df_features[:-self.forecast_horizon].copy()

        # Select feature columns
        exclude_cols = ['open_time', 'target'] + [col for col in df_ml.columns if col.endswith('_futures') or col.endswith('_spot')]
        exclude_cols = list(set(exclude_cols))  # Remove duplicates

        feature_cols = [col for col in df_ml.columns if col not in exclude_cols]

        # Remove columns that are just renamed originals
        feature_cols = [col for col in feature_cols if not any(x in col for x in ['open_', 'high_', 'low_', 'buy_', 'sell_'])]

        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        timestamps = df_ml['open_time'].values

        self.feature_names = feature_cols

        print(f"  Features: {len(feature_cols)}")
        print(f"  Feature names: {feature_cols[:10]}... (showing first 10)")
        print(f"  Samples: {len(X)}")
        print(f"  Target shape: {y.shape}")

        return X, y, timestamps

    def train_model(self, X, y, test_size=0.2):
        """Train prediction model."""
        print(f"\n🤖 Training model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series!
        )

        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # Train XGBoost
        print("  Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(
            X_train_scaled, y_train_scaled,
            eval_set=[(X_test_scaled, y_test_scaled)],
            early_stopping_rounds=20,
            verbose=False
        )

        # Predictions
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_test_pred_scaled = self.model.predict(X_test_scaled)

        # Inverse scale
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

        # Evaluate
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\n📊 Model Performance:")
        print(f"  Train MAE: ${train_mae:.2f}")
        print(f"  Test MAE:  ${test_mae:.2f}")
        print(f"  Train RMSE: ${train_rmse:.2f}")
        print(f"  Test RMSE:  ${test_rmse:.2f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🎯 Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }

    def predict(self, X):
        """Make prediction on new data."""
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_pred

    def save_model(self, filepath='tda_prediction_model.pkl'):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'forecast_horizon': self.forecast_horizon
        }
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to: {filepath}")

    def load_model(self, filepath='tda_prediction_model.pkl'):
        """Load trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_names = model_data['feature_names']
        self.window_size = model_data['window_size']
        self.forecast_horizon = model_data['forecast_horizon']
        print(f"✅ Model loaded from: {filepath}")


def main():
    """Main training pipeline."""
    print("="*80)
    print("BITCOIN PRICE PREDICTION - TDA + MACHINE LEARNING")
    print("="*80)

    # Initialize predictor
    predictor = TDAPricePredictor(
        window_size=60,  # 5 hours
        forecast_horizon=12  # Predict 1 hour ahead
    )

    # Load data
    df = predictor.load_data(
        futures_path='/notebooks/binance-data-collector/BTCUSDT_perp_5m.csv',
        spot_path='/notebooks/binance-data-collector/BTCUSDT_spot_5m.csv'
    )

    # Extract TDA features
    tda_features = predictor.extract_tda_features(df, use_columns_prefix='futures')

    # Create ML features
    df_features = predictor.create_features(df, tda_features)

    # Prepare dataset
    X, y, timestamps = predictor.prepare_ml_dataset(df_features)

    # Train model
    metrics = predictor.train_model(X, y, test_size=0.2)

    # Save model
    predictor.save_model('tda_prediction_model.pkl')

    # Save metrics
    with open('training_metrics.json', 'w') as f:
        json.dump({
            'train_mae': float(metrics['train_mae']),
            'test_mae': float(metrics['test_mae']),
            'train_rmse': float(metrics['train_rmse']),
            'test_rmse': float(metrics['test_rmse']),
            'train_r2': float(metrics['train_r2']),
            'test_r2': float(metrics['test_r2']),
            'forecast_horizon_minutes': predictor.forecast_horizon * 5,
            'window_size_minutes': predictor.window_size * 5,
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved: tda_prediction_model.pkl")
    print(f"Metrics saved: training_metrics.json")
    print(f"\nPrediction horizon: {predictor.forecast_horizon * 5} minutes ({predictor.forecast_horizon} steps)")
    print(f"Test MAE: ${metrics['test_mae']:.2f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")


if __name__ == '__main__':
    main()
