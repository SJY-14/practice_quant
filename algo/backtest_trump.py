"""
Backtest on Trump 30 Days Data
trump_30days 데이터로 백테스트

Usage:
    python backtest_trump.py
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config_trump import trump_config
from tda_model import TDATradingModel
from backtester import BacktestEngine


def load_trump_test_data():
    """Trump 백테스트 데이터 로드 (trump_30days)"""
    print("="*80)
    print("📥 LOADING TRUMP 30-DAYS TEST DATA")
    print("="*80)

    # Futures
    futures_path = '/notebooks/algo/data/BTCUSDT_perp_trump_30days.csv'
    df_futures = pd.read_csv(futures_path)
    df_futures['open_time'] = pd.to_datetime(df_futures['open_time'])
    df_futures = df_futures.add_suffix('_futures')
    df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

    # Spot
    spot_path = '/notebooks/algo/data/BTCUSDT_spot_trump_30days.csv'
    df_spot = pd.read_csv(spot_path)
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


def visualize_results(backtest_results, metrics, output_dir):
    """백테스트 결과 시각화"""
    print("\n" + "="*80)
    print("📊 VISUALIZING RESULTS")
    print("="*80)

    # 스타일 설정
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Figure 생성
    fig = plt.figure(figsize=(20, 12))

    # 1. Equity Curve
    ax1 = plt.subplot(3, 2, 1)
    if len(backtest_results['equity_curve']) > 0:
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

        ax1.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2, label='Equity', color='#2E86DE')
        ax1.axhline(y=backtest_results['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(equity_df['timestamp'], backtest_results['initial_capital'], equity_df['equity'],
                         where=(equity_df['equity'] >= backtest_results['initial_capital']),
                         alpha=0.3, color='green', label='Profit')
        ax1.fill_between(equity_df['timestamp'], backtest_results['initial_capital'], equity_df['equity'],
                         where=(equity_df['equity'] < backtest_results['initial_capital']),
                         alpha=0.3, color='red', label='Loss')

        ax1.set_title('📈 Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Returns Distribution
    ax2 = plt.subplot(3, 2, 2)
    if len(backtest_results['trades']) > 0:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_with_pnl = trades_df[trades_df['action'] == 'CLOSE']

        if len(trades_with_pnl) > 0:
            ax2.hist(trades_with_pnl['pnl'], bins=30, alpha=0.7, color='#0984E3', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
            ax2.set_title('📊 Trade Returns Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('PnL ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    # 3. Win/Loss Pie Chart
    ax3 = plt.subplot(3, 2, 3)
    winning = backtest_results['winning_trades']
    losing = backtest_results['losing_trades']

    if winning + losing > 0:
        colors = ['#00B894', '#D63031']
        explode = (0.05, 0.05)
        ax3.pie([winning, losing], labels=['Winning Trades', 'Losing Trades'],
                autopct='%1.1f%%', colors=colors, explode=explode, shadow=True, startangle=90)
        ax3.set_title(f'🎯 Win Rate: {backtest_results["win_rate"]:.2f}%', fontsize=14, fontweight='bold')

    # 4. Performance Metrics Bar Chart
    ax4 = plt.subplot(3, 2, 4)
    metrics_data = {
        'Total Return (%)': backtest_results['total_return'],
        'Max Drawdown (%)': -backtest_results['max_drawdown'],
        'Sharpe Ratio': backtest_results['sharpe_ratio'] * 10,  # Scale for visibility
        'Win Rate (%)': backtest_results['win_rate']
    }

    colors_bar = ['#00B894' if v > 0 else '#D63031' for v in metrics_data.values()]
    bars = ax4.bar(range(len(metrics_data)), list(metrics_data.values()), color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(metrics_data)))
    ax4.set_xticklabels(metrics_data.keys(), rotation=45, ha='right')
    ax4.set_title('📊 Performance Metrics', fontsize=14, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

    # 5. Drawdown Chart
    ax5 = plt.subplot(3, 2, 5)
    if len(backtest_results['equity_curve']) > 0:
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100

        ax5.fill_between(equity_df['timestamp'], 0, equity_df['drawdown'],
                         color='#D63031', alpha=0.5)
        ax5.plot(equity_df['timestamp'], equity_df['drawdown'], color='#D63031', linewidth=2)
        ax5.set_title('📉 Drawdown Over Time', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Drawdown (%)')
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()

    # 6. Trade Timeline
    ax6 = plt.subplot(3, 2, 6)
    if len(backtest_results['trades']) > 0:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Separate open and close trades
        open_trades = trades_df[trades_df['action'] == 'OPEN']
        close_trades = trades_df[trades_df['action'] == 'CLOSE']

        if len(open_trades) > 0:
            ax6.scatter(open_trades['timestamp'], open_trades['price'],
                       color='green', marker='^', s=100, alpha=0.6, label='Open', edgecolor='black')
        if len(close_trades) > 0:
            # Color by profitability
            profitable = close_trades[close_trades['pnl'] > 0]
            unprofitable = close_trades[close_trades['pnl'] <= 0]

            if len(profitable) > 0:
                ax6.scatter(profitable['timestamp'], profitable['price'],
                           color='blue', marker='v', s=100, alpha=0.6, label='Close (Profit)', edgecolor='black')
            if len(unprofitable) > 0:
                ax6.scatter(unprofitable['timestamp'], unprofitable['price'],
                           color='red', marker='v', s=100, alpha=0.6, label='Close (Loss)', edgecolor='black')

        ax6.set_title('📍 Trade Timeline', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Price ($)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    viz_path = os.path.join(output_dir, 'backtest_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {viz_path}")

    plt.close()


def run_trump_backtest():
    """트럼프 30일 데이터로 백테스트"""
    print("\n" + "="*80)
    print("🔬 BACKTESTING ON TRUMP 30-DAYS DATA")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 출력 디렉토리 생성
    os.makedirs(trump_config.output_dir, exist_ok=True)

    # 1. 테스트 데이터 로드
    test_df = load_trump_test_data()

    # 2. 모델 로드
    print("\n" + "="*80)
    print("STEP 1: MODEL LOADING")
    print("="*80)

    if not os.path.exists(trump_config.model_save_path):
        print(f"❌ Model not found: {trump_config.model_save_path}")
        print("   Run training first: python train_trump.py")
        sys.exit(1)

    model = TDATradingModel(trump_config.model)
    model.load_model(trump_config.model_save_path)
    print(f"✅ Model loaded from: {trump_config.model_save_path}")

    # 3. TDA 특징 추출 (테스트 데이터)
    print("\n" + "="*80)
    print("STEP 2: TDA FEATURE EXTRACTION (TEST DATA)")
    print("="*80)

    tda_features_test = model.extract_tda_features(test_df, use_columns_prefix='futures')
    print(f"✅ TDA features extracted: {len(tda_features_test)} windows")

    # 4. ML 특징 생성 (테스트 데이터)
    print("\n" + "="*80)
    print("STEP 3: MACHINE LEARNING FEATURES (TEST DATA)")
    print("="*80)

    df_features_test = model.create_features(test_df, tda_features_test)
    print(f"✅ ML features created: {len(df_features_test)} samples")

    # 5. 예측 생성
    print("\n" + "="*80)
    print("STEP 4: GENERATING PREDICTIONS")
    print("="*80)

    X_test = df_features_test[model.feature_names].values
    predictions = model.predict(X_test)

    print(f"✅ Predictions generated: {len(predictions)} predictions")

    # 6. 백테스팅 실행
    print("\n" + "="*80)
    print("STEP 5: RUNNING BACKTEST")
    print("="*80)

    backtest_df = df_features_test.copy()

    # 백테스트 엔진 초기화
    engine = BacktestEngine(trump_config.backtest, trump_config.trading)

    # 백테스트 실행
    backtest_results = engine.run_backtest(backtest_df, predictions)

    # 7. 결과 저장
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)

    # 백테스트 결과 저장
    results_summary = {
        'backtest_date': datetime.now().isoformat(),
        'dataset': 'trump_30days',
        'test_period': {
            'start': test_df['open_time'].min().isoformat(),
            'end': test_df['open_time'].max().isoformat(),
            'num_candles': len(test_df),
            'num_days': (test_df['open_time'].max() - test_df['open_time'].min()).days
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
            'leverage': trump_config.trading.leverage,
            'entry_threshold_long': trump_config.trading.entry_threshold_long,
            'entry_threshold_short': trump_config.trading.entry_threshold_short,
            'stop_loss_pct': trump_config.trading.stop_loss_pct,
            'take_profit_pct': trump_config.trading.take_profit_pct,
            'maker_fee': trump_config.backtest.maker_fee,
            'taker_fee': trump_config.backtest.taker_fee,
            'slippage_pct': trump_config.backtest.slippage_pct
        }
    }

    # JSON 저장
    results_path = trump_config.backtest_results_path
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✅ Results saved to: {results_path}")

    # 거래 로그 저장
    if trump_config.backtest.save_trades and len(backtest_results['trades']) > 0:
        trades_df = pd.DataFrame(backtest_results['trades'])
        trades_df.to_csv(trump_config.trades_log_path, index=False)
        print(f"✅ Trades log saved to: {trump_config.trades_log_path}")

    # Equity curve 저장
    if trump_config.backtest.save_equity_curve and len(backtest_results['equity_curve']) > 0:
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        equity_path = os.path.join(trump_config.output_dir, 'equity_curve.csv')
        equity_df.to_csv(equity_path, index=False)
        print(f"✅ Equity curve saved to: {equity_path}")

    # 8. 시각화
    print("\n" + "="*80)
    print("STEP 7: VISUALIZATION")
    print("="*80)

    # Load training metrics
    metrics_path = os.path.join(trump_config.output_dir, 'training_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            training_metrics = json.load(f)
    else:
        training_metrics = {}

    visualize_results(backtest_results, training_metrics, trump_config.output_dir)

    # 9. 완료
    print("\n" + "="*80)
    print("✅ BACKTESTING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📊 Key Metrics:")
    print(f"  Dataset: Trump 30-Days ({len(test_df):,} rows)")
    print(f"  Period: {test_df['open_time'].min().date()} to {test_df['open_time'].max().date()}")
    print(f"  Total Return:     {backtest_results['total_return']:+.2f}%")
    print(f"  Max Drawdown:     {backtest_results['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:     {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Win Rate:         {backtest_results['win_rate']:.2f}%")
    print()
    print("📁 Output files:")
    print(f"  Results: {results_path}")
    print(f"  Trades:  {trump_config.trades_log_path}")
    print(f"  Visualization: {trump_config.output_dir}/backtest_visualization.png")
    print()

    return backtest_results


if __name__ == '__main__':
    try:
        results = run_trump_backtest()
        print("✅ Backtesting successful!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
