"""
Train on Post-Trump Data
post_trump 데이터로 모델 훈련

Usage:
    python train_trump.py
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config_trump import trump_config
from tda_model_gpu import TDATradingModelGPU


def load_trump_train_data():
    """Trump 훈련 데이터 로드 (post_trump)"""
    print("="*80)
    print("📥 LOADING POST-TRUMP TRAINING DATA")
    print("="*80)

    # Futures
    df_futures = pd.read_csv(trump_config.data.futures_path)
    df_futures['open_time'] = pd.to_datetime(df_futures['open_time'])
    df_futures = df_futures.add_suffix('_futures')
    df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

    # Spot
    df_spot = pd.read_csv(trump_config.data.spot_path)
    df_spot['open_time'] = pd.to_datetime(df_spot['open_time'])
    df_spot = df_spot.add_suffix('_spot')
    df_spot.rename(columns={'open_time_spot': 'open_time'}, inplace=True)

    # Merge
    df = pd.merge(df_futures, df_spot, on='open_time', how='inner')
    df = df.sort_values('open_time').reset_index(drop=True)

    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Period: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"   Duration: {(df['open_time'].max() - df['open_time'].min()).days} days")

    return df


def train_trump_model():
    """트럼프 데이터로 모델 훈련"""
    print("\n" + "="*80)
    print("🚀 TRAINING ON POST-TRUMP DATA")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(trump_config.model_save_path), exist_ok=True)
    os.makedirs(trump_config.output_dir, exist_ok=True)

    # 1. 데이터 로드
    train_df = load_trump_train_data()

    # 2. TDA 모델 초기화 (GPU)
    print("\n" + "="*80)
    print("STEP 1: MODEL INITIALIZATION (GPU)")
    print("="*80)

    model = TDATradingModelGPU(trump_config.model, use_gpu=True)
    print(f"✅ Model initialized with {trump_config.model.n_folds}-fold CV (GPU-accelerated)")

    # 3. TDA 특징 추출
    print("\n" + "="*80)
    print("STEP 2: TDA FEATURE EXTRACTION")
    print("="*80)

    tda_features = model.extract_tda_features(train_df, use_columns_prefix='futures')
    print(f"✅ TDA features extracted: {len(tda_features)} windows")

    # 4. ML 특징 생성
    print("\n" + "="*80)
    print("STEP 3: MACHINE LEARNING FEATURES")
    print("="*80)

    df_features = model.create_features(train_df, tda_features)
    print(f"✅ ML features created: {len(df_features)} samples, {df_features.shape[1]} features")

    # 5. 데이터셋 준비
    print("\n" + "="*80)
    print("STEP 4: DATASET PREPARATION")
    print("="*80)

    X, y, timestamps = model.prepare_ml_dataset(df_features)
    print(f"✅ Dataset ready: X={X.shape}, y={y.shape}")

    # 6. K-Fold Cross-Validation 학습
    print("\n" + "="*80)
    print("STEP 5: K-FOLD CROSS-VALIDATION TRAINING")
    print("="*80)

    import numpy as np
    cv_results = model.train_with_kfold(X, y)
    print(f"✅ K-Fold CV complete!")

    # 7. 모델 저장
    print("\n" + "="*80)
    print("STEP 6: MODEL SAVING")
    print("="*80)

    model.save_model(trump_config.model_save_path)
    print(f"✅ Model saved to: {trump_config.model_save_path}")

    # 8. 메트릭 저장
    print("\n" + "="*80)
    print("STEP 7: SAVING METRICS")
    print("="*80)

    metrics = {
        'trained_at': datetime.now().isoformat(),
        'dataset': 'post_trump',
        'data_period': {
            'train_start': train_df['open_time'].min().isoformat(),
            'train_end': train_df['open_time'].max().isoformat(),
            'num_samples': len(train_df),
            'num_days': (train_df['open_time'].max() - train_df['open_time'].min()).days
        },
        'model_config': {
            'n_folds': trump_config.model.n_folds,
            'window_size': trump_config.model.window_size,
            'n_estimators': trump_config.model.n_estimators,
            'max_depth': trump_config.model.max_depth,
            'learning_rate': trump_config.model.learning_rate
        },
        'cv_results': {
            'avg_train_mae': float(cv_results['avg_train_mae']),
            'avg_val_mae': float(cv_results['avg_val_mae']),
            'avg_train_r2': float(cv_results['avg_train_r2']),
            'avg_val_r2': float(cv_results['avg_val_r2']),
            'train_mae_std': float(np.std(cv_results['cv_scores']['train_mae'])),
            'val_mae_std': float(np.std(cv_results['cv_scores']['val_mae']))
        },
        'feature_names': cv_results['feature_names']
    }

    metrics_path = os.path.join(trump_config.output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics saved to: {metrics_path}")

    # 9. 완료
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📊 Summary:")
    print(f"  Dataset: Post-Trump ({len(train_df):,} rows)")
    print(f"  Period: {train_df['open_time'].min().date()} to {train_df['open_time'].max().date()}")
    print(f"  Average Val MAE:  ${cv_results['avg_val_mae']:.2f}")
    print(f"  Average Val R²:   {cv_results['avg_val_r2']:.4f}")
    print()
    print("🎯 Next step:")
    print("  Run backtest: python backtest_trump.py")
    print()

    return model, metrics


if __name__ == '__main__':
    try:
        model, metrics = train_trump_model()
        print("✅ Training successful!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
