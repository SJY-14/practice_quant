"""
Backtesting Pipeline
백테스팅 파이프라인: 30일전~현재 데이터로 백테스트

Usage:
    python backtest.py [--config conservative|aggressive] [--model path/to/model.pkl]
"""
import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config, get_conservative_config, get_aggressive_config
from data_loader import TradingDataLoader
from tda_model import TDATradingModel
from backtester import BacktestEngine


def run_backtest(cfg=None, model_path=None):
    """
    백테스트 실행

    Parameters:
    -----------
    cfg : Config, optional
        설정 (기본값: config)
    model_path : str, optional
        모델 경로 (기본값: cfg.model_save_path)
    """
    if cfg is None:
        cfg = config

    if model_path is None:
        model_path = cfg.model_save_path

    print("="*80)
    print("🔬 BACKTESTING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 출력 디렉토리 생성
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. 데이터 로드
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)

    loader = TradingDataLoader(cfg.data)
    train_df, test_df = loader.load_and_split_data()

    print(f"\n✅ Test data: {len(test_df):,} rows")
    print(f"   ({cfg.data.test_days_before} days ago → present)")

    # 2. 모델 로드
    print("\n" + "="*80)
    print("STEP 2: MODEL LOADING")
    print("="*80)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Run training first: python train.py")
        sys.exit(1)

    model = TDATradingModel(cfg.model)
    model.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")

    # 3. TDA 특징 추출 (테스트 데이터)
    print("\n" + "="*80)
    print("STEP 3: TDA FEATURE EXTRACTION (TEST DATA)")
    print("="*80)

    tda_features_test = model.extract_tda_features(test_df, use_columns_prefix='futures')
    print(f"✅ TDA features extracted: {len(tda_features_test)} windows")

    # 4. ML 특징 생성 (테스트 데이터)
    print("\n" + "="*80)
    print("STEP 4: MACHINE LEARNING FEATURES (TEST DATA)")
    print("="*80)

    df_features_test = model.create_features(test_df, tda_features_test)
    print(f"✅ ML features created: {len(df_features_test)} samples")

    # 5. 예측 생성
    print("\n" + "="*80)
    print("STEP 5: GENERATING PREDICTIONS")
    print("="*80)

    # ⚠️  CRITICAL: Look-ahead bias 방지
    # 각 시점에서 forecast_horizon만큼 미래를 예측하므로,
    # 마지막 forecast_horizon개 샘플은 예측 불가
    forecast_horizon = 12

    X_test = df_features_test[model.feature_names].values
    predictions = model.predict(X_test)

    print(f"✅ Predictions generated: {len(predictions)} predictions")
    print(f"   Forecast horizon: {forecast_horizon} candles (60 minutes)")

    # 6. 백테스팅 실행
    print("\n" + "="*80)
    print("STEP 6: RUNNING BACKTEST")
    print("="*80)

    # 백테스팅을 위해 데이터와 예측을 정렬
    # df_features_test와 predictions의 길이가 같으므로 직접 사용
    backtest_df = df_features_test.copy()

    # 백테스트 엔진 초기화
    engine = BacktestEngine(cfg.backtest, cfg.trading)

    # 백테스트 실행
    backtest_results = engine.run_backtest(backtest_df, predictions)

    # 7. 결과 저장
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)

    # 백테스트 결과 저장
    results_summary = {
        'backtest_date': datetime.now().isoformat(),
        'test_period': {
            'start': test_df['open_time'].min().isoformat(),
            'end': test_df['open_time'].max().isoformat(),
            'num_candles': len(test_df)
        },
        'performance': {
            'initial_capital': backtest_results['initial_capital'],
            'final_equity': backtest_results['final_equity'],
            'total_return': backtest_results['total_return'],
            'max_drawdown': backtest_results['max_drawdown'],
            'sharpe_ratio': backtest_results['sharpe_ratio']
        },
        'trading_stats': {
            'total_trades': backtest_results['total_trades'],
            'winning_trades': backtest_results['winning_trades'],
            'losing_trades': backtest_results['losing_trades'],
            'win_rate': backtest_results['win_rate'],
            'max_consecutive_losses': backtest_results['max_consecutive_losses']
        },
        'config': {
            'leverage': cfg.trading.leverage,
            'entry_threshold_long': cfg.trading.entry_threshold_long,
            'entry_threshold_short': cfg.trading.entry_threshold_short,
            'stop_loss_pct': cfg.trading.stop_loss_pct,
            'take_profit_pct': cfg.trading.take_profit_pct,
            'maker_fee': cfg.backtest.maker_fee,
            'taker_fee': cfg.backtest.taker_fee,
            'slippage_pct': cfg.backtest.slippage_pct
        }
    }

    # JSON 저장
    results_path = cfg.backtest_results_path
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✅ Results saved to: {results_path}")

    # 거래 로그 저장
    if cfg.backtest.save_trades and len(backtest_results['trades']) > 0:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_df.to_csv(cfg.trades_log_path, index=False)
        print(f"✅ Trades log saved to: {cfg.trades_log_path}")

    # Equity curve 저장
    if cfg.backtest.save_equity_curve and len(backtest_results['equity_curve']) > 0:
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        equity_path = os.path.join(cfg.output_dir, 'equity_curve.csv')
        equity_df.to_csv(equity_path, index=False)
        print(f"✅ Equity curve saved to: {equity_path}")

    # 8. 성능 분석
    print("\n" + "="*80)
    print("STEP 8: PERFORMANCE ANALYSIS")
    print("="*80)

    analyze_performance(backtest_results, cfg)

    # 9. 완료
    print("\n" + "="*80)
    print("✅ BACKTESTING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📊 Key Metrics:")
    print(f"  Total Return:     {backtest_results['total_return']:+.2f}%")
    print(f"  Max Drawdown:     {backtest_results['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:     {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Win Rate:         {backtest_results['win_rate']:.2f}%")
    print()
    print("📁 Output files:")
    print(f"  Results: {results_path}")
    print(f"  Trades:  {cfg.trades_log_path}")
    print()

    return backtest_results


def analyze_performance(results: dict, cfg):
    """성능 분석 및 평가"""

    print("\n📊 Detailed Performance Analysis:")

    # 손익 분석
    if results['total_trades'] > 0:
        avg_win = results['winning_trades'] / results['total_trades'] * 100 if results['total_trades'] > 0 else 0
        avg_loss = results['losing_trades'] / results['total_trades'] * 100 if results['total_trades'] > 0 else 0

        print(f"\n  Trading Activity:")
        print(f"    Winning trades: {results['winning_trades']} ({avg_win:.1f}%)")
        print(f"    Losing trades:  {results['losing_trades']} ({avg_loss:.1f}%)")

    # 리스크 평가
    print(f"\n  Risk Assessment:")
    risk_score = "UNKNOWN"
    if results['max_drawdown'] < 10:
        risk_score = "🟢 LOW"
    elif results['max_drawdown'] < 20:
        risk_score = "🟡 MEDIUM"
    else:
        risk_score = "🔴 HIGH"

    print(f"    Risk Level: {risk_score}")
    print(f"    Max Drawdown: {results['max_drawdown']:.2f}%")

    # 수익성 평가
    print(f"\n  Profitability:")
    profit_score = "UNPROFITABLE"
    if results['total_return'] > 10:
        profit_score = "🟢 HIGHLY PROFITABLE"
    elif results['total_return'] > 0:
        profit_score = "🟡 PROFITABLE"
    else:
        profit_score = "🔴 UNPROFITABLE"

    print(f"    Status: {profit_score}")
    print(f"    Total Return: {results['total_return']:+.2f}%")

    # 추천사항
    print(f"\n  Recommendations:")
    if results['total_return'] > 0 and results['max_drawdown'] < 20:
        print("    ✅ Strategy shows promise!")
        print("    ✅ Consider paper trading before live deployment")
    elif results['total_return'] > 0:
        print("    ⚠️  Profitable but high risk")
        print("    ⚠️  Improve risk management before live trading")
    else:
        print("    ❌ Strategy needs improvement")
        print("    ❌ Review parameters and retrain model")

    # 거래 빈도
    if 'equity_curve' in results and len(results['equity_curve']) > 0:
        test_duration_hours = len(results['equity_curve']) * 5 / 60  # 5분봉
        trades_per_day = results['total_trades'] / (test_duration_hours / 24)
        print(f"\n  Trading Frequency:")
        print(f"    Trades per day: {trades_per_day:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest TDA Trading Model')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'conservative', 'aggressive'],
                       help='Configuration preset')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: from config)')
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

    # 백테스트 실행
    try:
        results = run_backtest(cfg, args.model)
        print("\n✅ Backtesting successful!")
    except Exception as e:
        print(f"\n❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
