"""
Realistic Backtesting Engine
현실적인 백테스팅 엔진 with 모든 주의사항 반영

주의사항:
1. Look-ahead bias 방지 (미래 데이터 사용 금지)
2. 거래 비용 (수수료, 슬리피지)
3. 펀딩비 (바이낸스 선물)
4. 현실적인 주문 실행
5. 레버리지 및 청산 처리
6. 리스크 관리
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from config import BacktestConfig, TradingConfig


class PositionSide(Enum):
    """포지션 방향"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Order:
    """주문 클래스"""
    def __init__(self, timestamp, side, price, size, leverage=1):
        self.timestamp = timestamp
        self.side = side  # PositionSide
        self.price = price
        self.size = size
        self.leverage = leverage


class Position:
    """포지션 클래스"""
    def __init__(self, side: PositionSide, entry_price: float, size: float,
                 leverage: int, entry_time: datetime):
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.leverage = leverage
        self.entry_time = entry_time

        # 청산가 계산
        self.liquidation_price = self._calculate_liquidation_price()

    def _calculate_liquidation_price(self) -> float:
        """
        청산가 계산 (간소화 버전)

        레버리지 청산:
        - 롱: entry_price * (1 - 1/leverage * 0.9)
        - 숏: entry_price * (1 + 1/leverage * 0.9)
        """
        if self.side == PositionSide.LONG:
            return self.entry_price * (1 - 1/self.leverage * 0.9)
        elif self.side == PositionSide.SHORT:
            return self.entry_price * (1 + 1/self.leverage * 0.9)
        else:
            return 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size * self.leverage
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) * self.size * self.leverage
        else:
            return 0.0

    def is_liquidated(self, current_price: float) -> bool:
        """청산 여부 확인"""
        if self.side == PositionSide.LONG:
            return current_price <= self.liquidation_price
        elif self.side == PositionSide.SHORT:
            return current_price >= self.liquidation_price
        else:
            return False


class BacktestEngine:
    """
    백테스팅 엔진

    핵심 원칙:
    - 각 시점에서 사용 가능한 데이터만 사용 (Look-ahead bias 방지)
    - 모든 거래 비용 반영
    - 현실적인 주문 실행
    """

    def __init__(self, backtest_config: BacktestConfig, trading_config: TradingConfig):
        """
        Parameters:
        -----------
        backtest_config : BacktestConfig
            백테스팅 설정
        trading_config : TradingConfig
            거래 전략 설정
        """
        self.bt_config = backtest_config
        self.tr_config = trading_config

        # 계정 상태
        self.initial_capital = backtest_config.initial_capital
        self.capital = backtest_config.initial_capital
        self.equity = backtest_config.initial_capital

        # 포지션
        self.position: Optional[Position] = None

        # 거래 기록
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # 통계
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_equity = backtest_config.initial_capital

        # 펀딩비 추적
        self.last_funding_time = None

    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        백테스팅 실행

        Parameters:
        -----------
        df : pd.DataFrame
            테스트 데이터 (시간 순서 보장됨)
        predictions : np.ndarray
            각 시점의 예측 가격 (같은 길이)

        Returns:
        --------
        results : Dict
            백테스팅 결과
        """
        print("="*80)
        print("🔬 RUNNING BACKTEST")
        print("="*80)

        # 검증
        if len(df) != len(predictions):
            raise ValueError(f"Data length mismatch: df={len(df)}, predictions={len(predictions)}")

        print(f"\n📊 Backtest period: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
        print(f"  Total candles: {len(df)}")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")

        # 시작
        for i in range(len(df)):
            # ⚠️  CRITICAL: 현재 시점 데이터만 사용
            current_row = df.iloc[i]
            current_time = current_row['open_time']
            current_price = current_row['close_futures']  # 종가 사용 (현실적)
            predicted_price = predictions[i]

            # 예측 변화율 계산
            predicted_change_pct = (predicted_price - current_price) / current_price * 100

            # TDA 필터 (옵션)
            if self.tr_config.use_tda_filter:
                # TDA 특징이 있다면
                if 'tda_l1' in current_row:
                    tda_l1 = current_row['tda_l1']
                    if tda_l1 > self.tr_config.tda_l1_threshold:
                        # 변동성 높음 → 거래 제한
                        self._update_equity(current_time, current_price)
                        continue

            # 펀딩비 지불 (8시간마다)
            self._pay_funding_fee(current_time, current_price)

            # 청산 확인
            if self.position and self.position.is_liquidated(current_price):
                self._liquidate_position(current_time, current_price)
                continue

            # 손절/익절 확인
            if self.position:
                if self._check_stop_loss(current_price) or self._check_take_profit(current_price):
                    self._close_position(current_time, current_price, reason="SL/TP")

            # 거래 시그널 생성
            signal = self._generate_signal(predicted_change_pct)

            # 거래 실행
            if signal == PositionSide.LONG:
                if not self.position or self.position.side != PositionSide.LONG:
                    # 롱 포지션 진입
                    self._open_position(current_time, current_price, PositionSide.LONG)
            elif signal == PositionSide.SHORT:
                if not self.position or self.position.side != PositionSide.SHORT:
                    # 숏 포지션 진입
                    self._open_position(current_time, current_price, PositionSide.SHORT)
            elif signal == PositionSide.NEUTRAL:
                # 중립 → 포지션 청산
                if self.position:
                    self._close_position(current_time, current_price, reason="NEUTRAL")

            # Equity curve 업데이트
            self._update_equity(current_time, current_price)

            # Max drawdown 업데이트
            self._update_max_drawdown()

            # 리스크 관리
            if self._should_stop_trading():
                print(f"\n⚠️  Risk limit reached at {current_time}")
                break

        # 마지막 포지션 청산
        if self.position:
            final_price = df['close_futures'].iloc[-1]
            final_time = df['open_time'].iloc[-1]
            self._close_position(final_time, final_price, reason="END")

        # 결과 계산
        results = self._calculate_results(df)

        return results

    def _generate_signal(self, predicted_change_pct: float) -> PositionSide:
        """거래 시그널 생성"""
        if predicted_change_pct >= self.tr_config.entry_threshold_long:
            return PositionSide.LONG
        elif predicted_change_pct <= self.tr_config.entry_threshold_short:
            return PositionSide.SHORT
        elif abs(predicted_change_pct) <= self.tr_config.exit_threshold:
            return PositionSide.NEUTRAL
        else:
            # 기존 포지션 유지
            if self.position:
                return self.position.side
            else:
                return PositionSide.NEUTRAL

    def _open_position(self, timestamp, price, side):
        """포지션 진입"""
        # 기존 포지션 청산
        if self.position:
            if self.tr_config.allow_position_flip:
                self._close_position(timestamp, price, reason="FLIP")
            else:
                return  # 포지션 전환 불가

        # 포지션 크기 계산
        position_size = self.capital * self.tr_config.max_position_size

        # 수수료 차감
        fee = self._calculate_fee(position_size)
        self.capital -= fee

        # 포지션 생성
        self.position = Position(
            side=side,
            entry_price=price,
            size=position_size / price,  # BTC 수량
            leverage=self.tr_config.leverage,
            entry_time=timestamp
        )

        # 거래 기록
        if self.bt_config.save_trades:
            self.trades.append({
                'timestamp': timestamp,
                'action': 'OPEN',
                'side': side.value,
                'price': price,
                'size': position_size,
                'fee': fee,
                'capital': self.capital
            })

    def _close_position(self, timestamp, price, reason=""):
        """포지션 청산"""
        if not self.position:
            return

        # 손익 계산
        pnl = self.position.unrealized_pnl(price)

        # 수수료 차감
        position_value = self.position.size * price
        fee = self._calculate_fee(position_value)

        # 슬리피지 차감
        slippage = position_value * self.bt_config.slippage_pct / 100

        # 자본 업데이트
        self.capital += pnl - fee - slippage

        # 통계 업데이트
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

        # 거래 기록
        if self.bt_config.save_trades:
            self.trades.append({
                'timestamp': timestamp,
                'action': 'CLOSE',
                'side': self.position.side.value,
                'price': price,
                'entry_price': self.position.entry_price,
                'pnl': pnl,
                'fee': fee,
                'slippage': slippage,
                'capital': self.capital,
                'reason': reason
            })

        # 포지션 제거
        self.position = None

    def _liquidate_position(self, timestamp, price):
        """청산 (강제)"""
        if not self.position:
            return

        print(f"  ⚠️  LIQUIDATION at {timestamp}: Price ${price:,.2f}, Liq Price ${self.position.liquidation_price:,.2f}")

        # 청산 시 자본 0
        self.capital = 0
        self.equity = 0

        # 거래 기록
        if self.bt_config.save_trades:
            self.trades.append({
                'timestamp': timestamp,
                'action': 'LIQUIDATION',
                'side': self.position.side.value,
                'price': price,
                'entry_price': self.position.entry_price,
                'pnl': -self.position.size * self.position.entry_price,  # 전액 손실
                'capital': 0
            })

        self.position = None

    def _check_stop_loss(self, current_price) -> bool:
        """손절 확인"""
        if not self.position:
            return False

        pnl_pct = self.position.unrealized_pnl(current_price) / (self.position.size * self.position.entry_price) * 100

        return pnl_pct <= -self.tr_config.stop_loss_pct

    def _check_take_profit(self, current_price) -> bool:
        """익절 확인"""
        if not self.position:
            return False

        pnl_pct = self.position.unrealized_pnl(current_price) / (self.position.size * self.position.entry_price) * 100

        return pnl_pct >= self.tr_config.take_profit_pct

    def _pay_funding_fee(self, current_time, current_price):
        """펀딩비 지불 (8시간마다)"""
        if not self.position:
            return

        if self.last_funding_time is None:
            self.last_funding_time = current_time
            return

        # 8시간 경과 확인
        hours_elapsed = (current_time - self.last_funding_time).total_seconds() / 3600

        if hours_elapsed >= self.bt_config.funding_interval_hours:
            # 펀딩비 계산
            position_value = self.position.size * current_price
            funding_fee = position_value * self.bt_config.funding_rate

            # 차감
            self.capital -= funding_fee

            # 업데이트
            self.last_funding_time = current_time

    def _calculate_fee(self, amount) -> float:
        """거래 수수료 계산"""
        if self.bt_config.use_taker_fee:
            return amount * self.bt_config.taker_fee
        else:
            return amount * self.bt_config.maker_fee

    def _update_equity(self, timestamp, current_price):
        """Equity 업데이트"""
        unrealized_pnl = 0
        if self.position:
            unrealized_pnl = self.position.unrealized_pnl(current_price)

        self.equity = self.capital + unrealized_pnl

        if self.bt_config.save_equity_curve:
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'capital': self.capital,
                'unrealized_pnl': unrealized_pnl
            })

    def _update_max_drawdown(self):
        """Max drawdown 업데이트"""
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def _should_stop_trading(self) -> bool:
        """거래 중지 여부 확인"""
        # Max drawdown 초과
        if self.max_drawdown >= self.bt_config.max_drawdown_stop:
            return True

        # 연속 손실 초과
        if self.consecutive_losses >= self.bt_config.max_consecutive_losses:
            return True

        # 자본 소진
        if self.capital <= 0:
            return True

        return False

    def _calculate_results(self, df) -> Dict:
        """백테스팅 결과 계산"""
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)

        # 수익률
        total_return = (self.equity - self.initial_capital) / self.initial_capital * 100

        # 승률
        win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0

        # Sharpe Ratio (간소화)
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 288) if returns.std() > 0 else 0  # 5분봉 연율화
        else:
            sharpe_ratio = 0

        results = {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'max_consecutive_losses': self.max_consecutive_losses,
            'sharpe_ratio': sharpe_ratio,
            'trades': self.trades if self.bt_config.save_trades else [],
            'equity_curve': self.equity_curve if self.bt_config.save_equity_curve else []
        }

        # 출력
        print(f"\n💰 Performance:")
        print(f"  Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"  Final Equity:     ${self.equity:,.2f}")
        print(f"  Total Return:     {total_return:+.2f}%")

        print(f"\n📈 Trading Statistics:")
        print(f"  Total Trades:     {self.total_trades}")
        print(f"  Winning Trades:   {self.winning_trades}")
        print(f"  Losing Trades:    {self.losing_trades}")
        print(f"  Win Rate:         {win_rate:.2f}%")

        print(f"\n📉 Risk Metrics:")
        print(f"  Max Drawdown:     {self.max_drawdown:.2f}%")
        print(f"  Max Consecutive Losses: {self.max_consecutive_losses}")
        print(f"  Sharpe Ratio:     {sharpe_ratio:.2f}")

        print("\n" + "="*80)

        return results


if __name__ == '__main__':
    # 테스트
    print("Backtesting Engine Ready")
