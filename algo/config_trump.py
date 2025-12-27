"""
Trump Data Configuration
트럼프 데이터 전용 설정
"""
from config import Config, DataConfig, ModelConfig, TradingConfig, BacktestConfig


# Trump 데이터 설정
trump_config = Config()

# 데이터 경로 업데이트
trump_config.data = DataConfig()
trump_config.data.futures_path = '/notebooks/algo/data/BTCUSDT_perp_post_trump.csv'
trump_config.data.spot_path = '/notebooks/algo/data/BTCUSDT_spot_post_trump.csv'

# Trump 데이터는 이미 분할되어 있으므로 전체 사용
trump_config.data.train_days_before = 99999  # 전체 데이터 사용
trump_config.data.train_days_until = 0
trump_config.data.test_days_before = 99999
trump_config.data.test_days_until = 0

# 모델 설정 (기본값 사용)
trump_config.model = ModelConfig()

# 거래 설정 (기본값 사용)
trump_config.trading = TradingConfig()

# 백테스트 설정 (기본값 사용)
trump_config.backtest = BacktestConfig()

# 출력 경로
trump_config.output_dir = '/notebooks/algo/results_trump'
trump_config.model_save_path = '/notebooks/algo/models/tda_trump_model.pkl'
trump_config.backtest_results_path = '/notebooks/algo/results_trump/backtest_results.json'
trump_config.trades_log_path = '/notebooks/algo/results_trump/trades_log.csv'


if __name__ == '__main__':
    print("="*80)
    print("TRUMP DATA CONFIGURATION")
    print("="*80)

    print("\n📊 Data Paths:")
    print(f"  Futures (train): {trump_config.data.futures_path}")
    print(f"  Spot (train):    {trump_config.data.spot_path}")

    print("\n🤖 Model:")
    print(f"  K-Fold CV: {trump_config.model.n_folds} folds")

    print("\n📈 Trading:")
    print(f"  Leverage: {trump_config.trading.leverage}x")
    print(f"  Entry: Long >= {trump_config.trading.entry_threshold_long}%, Short <= {trump_config.trading.entry_threshold_short}%")

    print("\n💰 Backtest:")
    print(f"  Initial capital: ${trump_config.backtest.initial_capital:,.2f}")

    print("\n✅ Configuration ready!")
