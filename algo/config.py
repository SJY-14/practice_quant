"""
Trading Algorithm Configuration
백테스팅 및 실전 거래 설정
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    """데이터 설정"""
    # 데이터 경로
    futures_path: str = '/notebooks/binance-data-collector/BTCUSDT_perp_5m.csv'
    spot_path: str = '/notebooks/binance-data-collector/BTCUSDT_spot_5m.csv'

    # 데이터 분할
    train_days_before: int = 365  # 365일 전부터
    train_days_until: int = 30    # 30일 전까지 (학습용)
    test_days_before: int = 30    # 30일 전부터
    test_days_until: int = 0      # 현재까지 (백테스트용)

    # 특징 생성
    window_size: int = 60         # TDA 윈도우 크기 (5시간)
    forecast_horizon: int = 12    # 예측 구간 (60분)


@dataclass
class ModelConfig:
    """모델 설정"""
    # K-Fold Cross-Validation
    n_folds: int = 5
    shuffle: bool = False  # 시계열이므로 shuffle 금지

    # TDA 파라미터
    window_size: int = 60
    maxdim: int = 1  # H0, H1까지 계산

    # XGBoost 파라미터
    n_estimators: int = 200
    max_depth: int = 7
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

    # 학습 설정
    early_stopping_rounds: int = 20
    test_size: float = 0.2  # 각 fold 내 validation split


@dataclass
class TradingConfig:
    """거래 전략 설정"""
    # 시그널 생성
    entry_threshold_long: float = 0.3   # 상승 예측 >= 0.3% → 롱 진입
    entry_threshold_short: float = -0.3  # 하락 예측 <= -0.3% → 숏 진입
    exit_threshold: float = 0.1          # ±0.1% 이내 → 포지션 청산

    # 리스크 관리
    max_position_size: float = 1.0       # 최대 포지션 크기 (계정의 100%)
    stop_loss_pct: float = 2.0           # 손절 (2% 손실 시)
    take_profit_pct: float = 3.0         # 익절 (3% 수익 시)

    # 레버리지
    leverage: int = 1                    # 레버리지 배수 (1 = 현물 수준)
    max_leverage: int = 10               # 최대 레버리지

    # TDA 필터
    use_tda_filter: bool = True
    tda_l1_threshold: float = 0.8        # L1 Norm > 0.8 → 변동성 높음, 거래 제한

    # 포지션 관리
    allow_position_flip: bool = False    # 롱→숏, 숏→롱 즉시 전환 허용 여부
    min_hold_candles: int = 1            # 최소 보유 캔들 수


@dataclass
class BacktestConfig:
    """백테스팅 설정"""
    # 초기 자본
    initial_capital: float = 10000.0     # $10,000 시작

    # 거래 비용
    maker_fee: float = 0.0002            # 0.02% (바이낸스 선물 Maker)
    taker_fee: float = 0.0004            # 0.04% (바이낸스 선물 Taker)
    use_taker_fee: bool = True           # 보수적으로 Taker fee 사용

    # 슬리피지
    slippage_pct: float = 0.01           # 0.01% 슬리피지

    # 펀딩비 (바이낸스 선물)
    funding_rate: float = 0.0001         # 0.01% (8시간마다)
    funding_interval_hours: int = 8

    # Look-ahead bias 방지
    use_close_price_only: bool = True    # 캔들 종가만 사용 (현실적)
    order_execution_delay: int = 0       # 주문 실행 지연 (캔들 수)

    # 리스크 제한
    max_drawdown_stop: float = 20.0      # 20% 최대 낙폭 시 거래 중지
    max_consecutive_losses: int = 5      # 연속 5회 손실 시 거래 중지

    # 로깅
    save_trades: bool = True
    save_equity_curve: bool = True
    verbose: bool = True


@dataclass
class Config:
    """전체 설정"""
    data: DataConfig = None
    model: ModelConfig = None
    trading: TradingConfig = None
    backtest: BacktestConfig = None

    # 출력 경로
    output_dir: str = '/notebooks/algo/results'
    model_save_path: str = '/notebooks/algo/models/tda_trading_model.pkl'
    backtest_results_path: str = '/notebooks/algo/results/backtest_results.json'
    trades_log_path: str = '/notebooks/algo/results/trades_log.csv'

    def __post_init__(self):
        """초기화 후 검증"""
        # 기본값 설정
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()

        # 레버리지 제한
        if self.trading.leverage > self.trading.max_leverage:
            raise ValueError(f"Leverage {self.trading.leverage} exceeds max {self.trading.max_leverage}")

        # 데이터 기간 검증
        if self.data.train_days_before <= self.data.train_days_until:
            raise ValueError("train_days_before must be > train_days_until")

        if self.data.test_days_before <= self.data.test_days_until:
            raise ValueError("test_days_before must be > test_days_until")


# 기본 설정 인스턴스
config = Config()


# 보수적 설정 (실전 거래용)
def get_conservative_config() -> Config:
    """보수적 설정 반환"""
    cfg = Config()
    cfg.trading.leverage = 1  # 레버리지 사용 안함
    cfg.trading.entry_threshold_long = 0.5  # 더 높은 확신
    cfg.trading.entry_threshold_short = -0.5
    cfg.trading.stop_loss_pct = 1.5  # 더 타이트한 손절
    cfg.trading.use_tda_filter = True
    cfg.backtest.max_drawdown_stop = 10.0  # 10% 낙폭 제한
    return cfg


# 공격적 설정 (백테스팅용)
def get_aggressive_config() -> Config:
    """공격적 설정 반환"""
    cfg = Config()
    cfg.trading.leverage = 3
    cfg.trading.entry_threshold_long = 0.2
    cfg.trading.entry_threshold_short = -0.2
    cfg.trading.stop_loss_pct = 3.0
    cfg.trading.take_profit_pct = 5.0
    cfg.trading.allow_position_flip = True
    return cfg


if __name__ == '__main__':
    # 설정 확인
    print("="*80)
    print("TRADING ALGORITHM CONFIGURATION")
    print("="*80)

    cfg = config

    print("\n📊 Data Config:")
    print(f"  Train period: {cfg.data.train_days_before} days ago → {cfg.data.train_days_until} days ago")
    print(f"  Test period:  {cfg.data.test_days_before} days ago → {cfg.data.test_days_until} days ago (present)")

    print("\n🤖 Model Config:")
    print(f"  K-Fold CV: {cfg.model.n_folds} folds")
    print(f"  TDA window: {cfg.model.window_size} candles")

    print("\n📈 Trading Config:")
    print(f"  Entry thresholds: Long >= {cfg.trading.entry_threshold_long}%, Short <= {cfg.trading.entry_threshold_short}%")
    print(f"  Leverage: {cfg.trading.leverage}x")
    print(f"  Stop Loss: {cfg.trading.stop_loss_pct}%")

    print("\n💰 Backtest Config:")
    print(f"  Initial capital: ${cfg.backtest.initial_capital:,.2f}")
    print(f"  Fees: Maker {cfg.backtest.maker_fee*100:.3f}%, Taker {cfg.backtest.taker_fee*100:.3f}%")
    print(f"  Slippage: {cfg.backtest.slippage_pct}%")
    print(f"  Max drawdown stop: {cfg.backtest.max_drawdown_stop}%")

    print("\n✅ Configuration validated successfully!")
