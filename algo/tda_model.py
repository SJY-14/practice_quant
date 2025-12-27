"""
TDA + Machine Learning Model with K-Fold Cross-Validation
거래 알고리즘용 TDA 모델
"""
import sys
sys.path.append('/notebooks/tda-extreme-events')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from tda_analysis import TDAExtremeEventDetector
from config import ModelConfig


class TDATradingModel:
    """
    TDA + XGBoost 거래 모델 with K-Fold CV

    주요 기능:
    - TDA 특징 추출 (L1, L2, WD)
    - 시계열 K-Fold Cross-Validation
    - 가격 예측 및 변화율 예측
    """

    def __init__(self, config: ModelConfig):
        """
        Parameters:
        -----------
        config : ModelConfig
            모델 설정
        """
        self.config = config
        self.window_size = config.window_size
        self.forecast_horizon = 12  # 고정: 60분 예측

        # TDA detector
        self.tda_detector = TDAExtremeEventDetector(window_size=config.window_size)

        # Scalers
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Models (K-fold 결과 앙상블)
        self.models = []  # 각 fold의 모델 저장
        self.feature_names = []

        # 성능 지표
        self.cv_scores = []

    def extract_tda_features(self, df: pd.DataFrame, use_columns_prefix='futures') -> pd.DataFrame:
        """TDA 특징 추출"""
        print("\n🔬 Extracting TDA features...")

        # Select columns
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

    def create_features(self, df: pd.DataFrame, tda_features: pd.DataFrame) -> pd.DataFrame:
        """ML 특징 생성"""
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
        df_aligned['ma_5'] = df_aligned['close_futures'].rolling(5).mean()
        df_aligned['ma_12'] = df_aligned['close_futures'].rolling(12).mean()
        df_aligned['ma_24'] = df_aligned['close_futures'].rolling(24).mean()
        df_aligned['volatility_5'] = df_aligned['close_futures'].rolling(5).std()
        df_aligned['volatility_12'] = df_aligned['close_futures'].rolling(12).std()
        df_aligned['hl_range_futures'] = df_aligned['high_futures'] - df_aligned['low_futures']
        df_aligned['hl_range_spot'] = df_aligned['high_spot'] - df_aligned['low_spot']

        # Drop NaN
        df_aligned = df_aligned.dropna()

        print(f"  Created {df_aligned.shape[1]} features for {len(df_aligned)} samples")

        return df_aligned

    def prepare_ml_dataset(self, df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ML 데이터셋 준비"""
        print(f"\n🎯 Preparing ML dataset (predicting {self.forecast_horizon} steps ahead)...")

        # Target: future price
        df_features['target'] = df_features['close_futures'].shift(-self.forecast_horizon)

        # Remove last forecast_horizon rows (no target)
        df_ml = df_features[:-self.forecast_horizon].copy()

        # Select feature columns
        exclude_cols = ['open_time', 'target'] + [col for col in df_ml.columns if col.endswith('_futures') or col.endswith('_spot')]
        exclude_cols = list(set(exclude_cols))

        feature_cols = [col for col in df_ml.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if not any(x in col for x in ['open_', 'high_', 'low_', 'buy_', 'sell_'])]

        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        timestamps = df_ml['open_time'].values

        self.feature_names = feature_cols

        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(X)}")

        return X, y, timestamps

    def train_with_kfold(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        K-Fold Cross-Validation으로 모델 학습

        Returns:
        --------
        results : Dict
            CV 결과 및 성능 지표
        """
        print(f"\n🤖 Training with {self.config.n_folds}-Fold Time Series CV...")

        # TimeSeriesSplit (시계열 전용)
        tscv = TimeSeriesSplit(n_splits=self.config.n_folds)

        cv_scores = {
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': []
        }

        self.models = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n  Fold {fold+1}/{self.config.n_folds}")
            print(f"    Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

            # Split
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)

            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()

            # Train XGBoost
            model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=-1
            )

            model.fit(
                X_train_scaled, y_train_scaled,
                eval_set=[(X_val_scaled, y_val_scaled)],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )

            # Predictions
            y_train_pred_scaled = model.predict(X_train_scaled)
            y_val_pred_scaled = model.predict(X_val_scaled)

            # Inverse scale
            y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
            y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()

            # Evaluate
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            cv_scores['train_mae'].append(train_mae)
            cv_scores['val_mae'].append(val_mae)
            cv_scores['train_r2'].append(train_r2)
            cv_scores['val_r2'].append(val_r2)

            print(f"    Train MAE: ${train_mae:.2f}, R²: {train_r2:.4f}")
            print(f"    Val MAE:   ${val_mae:.2f}, R²: {val_r2:.4f}")

            # Save model
            self.models.append(model)

        # 평균 성능
        print(f"\n📊 Cross-Validation Results:")
        print(f"  Average Train MAE: ${np.mean(cv_scores['train_mae']):.2f} ± ${np.std(cv_scores['train_mae']):.2f}")
        print(f"  Average Val MAE:   ${np.mean(cv_scores['val_mae']):.2f} ± ${np.std(cv_scores['val_mae']):.2f}")
        print(f"  Average Train R²:  {np.mean(cv_scores['train_r2']):.4f} ± {np.std(cv_scores['train_r2']):.4f}")
        print(f"  Average Val R²:    {np.mean(cv_scores['val_r2']):.4f} ± {np.std(cv_scores['val_r2']):.4f}")

        self.cv_scores = cv_scores

        return {
            'cv_scores': cv_scores,
            'avg_train_mae': np.mean(cv_scores['train_mae']),
            'avg_val_mae': np.mean(cv_scores['val_mae']),
            'avg_train_r2': np.mean(cv_scores['train_r2']),
            'avg_val_r2': np.mean(cv_scores['val_r2']),
            'feature_names': self.feature_names,
            'n_folds': self.config.n_folds
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        앙상블 예측 (모든 fold 모델의 평균)

        Parameters:
        -----------
        X : np.ndarray
            입력 특징

        Returns:
        --------
        predictions : np.ndarray
            예측 가격
        """
        if len(self.models) == 0:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler_X.transform(X)

        # 모든 fold 모델로 예측
        predictions_scaled = np.array([model.predict(X_scaled) for model in self.models])

        # 평균
        avg_prediction_scaled = predictions_scaled.mean(axis=0)

        # Inverse scale
        predictions = self.scaler_y.inverse_transform(avg_prediction_scaled.reshape(-1, 1)).flatten()

        return predictions

    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'models': self.models,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'forecast_horizon': self.forecast_horizon,
            'cv_scores': self.cv_scores,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to: {filepath}")

    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_names = model_data['feature_names']
        self.window_size = model_data['window_size']
        self.forecast_horizon = model_data['forecast_horizon']
        self.cv_scores = model_data.get('cv_scores', [])
        print(f"✅ Model loaded from: {filepath}")


if __name__ == '__main__':
    # 테스트
    print("Testing TDATradingModel...")
    # (실제 테스트는 train.py에서 수행)
