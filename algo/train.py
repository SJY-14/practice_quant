"""
Training Pipeline
학습 파이프라인: 365일전~30일전 데이터로 K-fold CV

Usage:
    python train.py [--config conservative|aggressive]
"""
import os
import sys
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config, get_conservative_config, get_aggressive_config
from data_loader import TradingDataLoader
from tda_model import TDATradingModel


def train_model(cfg=None):
    """
    모델 학습

    Parameters:
    -----------
    cfg : Config, optional
        설정 (기본값: config)
    """
    if cfg is None:
        cfg = config

    print("="*80)
    print("🚀 TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. 데이터 로드
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)

    loader = TradingDataLoader(cfg.data)
    train_df, test_df = loader.load_and_split_data()

    print(f"\n✅ Training data: {len(train_df):,} rows")
    print(f"   ({cfg.data.train_days_before} days ago → {cfg.data.train_days_until} days ago)")

    # 2. TDA 모델 초기화
    print("\n" + "="*80)
    print("STEP 2: MODEL INITIALIZATION")
    print("="*80)

    model = TDATradingModel(cfg.model)
    print(f"✅ Model initialized with {cfg.model.n_folds}-fold CV")

    # 3. TDA 특징 추출
    print("\n" + "="*80)
    print("STEP 3: TDA FEATURE EXTRACTION")
    print("="*80)

    tda_features = model.extract_tda_features(train_df, use_columns_prefix='futures')
    print(f"✅ TDA features extracted: {len(tda_features)} windows")

    # 4. ML 특징 생성
    print("\n" + "="*80)
    print("STEP 4: MACHINE LEARNING FEATURES")
    print("="*80)

    df_features = model.create_features(train_df, tda_features)
    print(f"✅ ML features created: {len(df_features)} samples, {df_features.shape[1]} features")

    # 5. 데이터셋 준비
    print("\n" + "="*80)
    print("STEP 5: DATASET PREPARATION")
    print("="*80)

    X, y, timestamps = model.prepare_ml_dataset(df_features)
    print(f"✅ Dataset ready: X={X.shape}, y={y.shape}")

    # 6. K-Fold Cross-Validation 학습
    print("\n" + "="*80)
    print("STEP 6: K-FOLD CROSS-VALIDATION TRAINING")
    print("="*80)

    cv_results = model.train_with_kfold(X, y)
    print(f"✅ K-Fold CV complete!")

    # 7. 모델 저장
    print("\n" + "="*80)
    print("STEP 7: MODEL SAVING")
    print("="*80)

    model.save_model(cfg.model_save_path)
    print(f"✅ Model saved to: {cfg.model_save_path}")

    # 8. 메트릭 저장
    print("\n" + "="*80)
    print("STEP 8: SAVING METRICS")
    print("="*80)

    metrics = {
        'trained_at': datetime.now().isoformat(),
        'data_period': {
            'train_start': train_df['open_time'].min().isoformat(),
            'train_end': train_df['open_time'].max().isoformat(),
            'num_samples': len(train_df)
        },
        'model_config': {
            'n_folds': cfg.model.n_folds,
            'window_size': cfg.model.window_size,
            'n_estimators': cfg.model.n_estimators,
            'max_depth': cfg.model.max_depth,
            'learning_rate': cfg.model.learning_rate
        },
        'cv_results': {
            'avg_train_mae': cv_results['avg_train_mae'],
            'avg_val_mae': cv_results['avg_val_mae'],
            'avg_train_r2': cv_results['avg_train_r2'],
            'avg_val_r2': cv_results['avg_val_r2'],
            'train_mae_std': float(np.std(cv_results['cv_scores']['train_mae'])),
            'val_mae_std': float(np.std(cv_results['cv_scores']['val_mae']))
        },
        'feature_names': cv_results['feature_names']
    }

    metrics_path = os.path.join(cfg.output_dir, 'training_metrics.json')
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
    print(f"  Average Val MAE:  ${cv_results['avg_val_mae']:.2f}")
    print(f"  Average Val R²:   {cv_results['avg_val_r2']:.4f}")
    print()
    print("🎯 Next steps:")
    print("  1. Review training metrics")
    print("  2. Run backtest: python backtest.py")
    print()

    return model, metrics


if __name__ == '__main__':
    import numpy as np  # train_with_kfold에서 사용

    parser = argparse.ArgumentParser(description='Train TDA Trading Model')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'conservative', 'aggressive'],
                       help='Configuration preset')
    args = parser.parse_args()

    # 설정 로드
    if args.config == 'conservative':
        cfg = get_conservative_config()
        print("🛡️  Using CONSERVATIVE configuration")
    elif args.config == 'aggressive':
        cfg = get_aggressive_config()
        print("⚡ Using AGGRESSIVE configuration")
    else:
        cfg = config
        print("⚙️  Using DEFAULT configuration")

    # 학습 실행
    try:
        model, metrics = train_model(cfg)
        print("✅ Training successful!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
