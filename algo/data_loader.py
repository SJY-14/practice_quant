"""
Data Loader for Trading Algorithm
시간 순서를 엄격히 보존하는 데이터 로더
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import DataConfig


class TradingDataLoader:
    """
    거래 알고리즘용 데이터 로더

    주요 기능:
    - 시간 기반 데이터 분할 (Look-ahead bias 방지)
    - 선물 + 현물 데이터 병합
    - 데이터 품질 검증
    """

    def __init__(self, config: DataConfig):
        """
        Parameters:
        -----------
        config : DataConfig
            데이터 설정
        """
        self.config = config

    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        데이터 로드 및 학습/테스트 분할

        Returns:
        --------
        train_df : pd.DataFrame
            학습 데이터 (365일전 ~ 30일전)
        test_df : pd.DataFrame
            테스트 데이터 (30일전 ~ 현재)
        """
        print("="*80)
        print("📥 LOADING DATA")
        print("="*80)

        # 1. 데이터 로드
        df_futures, df_spot = self._load_raw_data()

        # 2. 데이터 병합
        df = self._merge_data(df_futures, df_spot)

        # 3. 데이터 품질 검증
        df = self._validate_data(df)

        # 4. 시간 기반 분할
        train_df, test_df = self._time_based_split(df)

        # 5. 분할 결과 출력
        self._print_split_summary(train_df, test_df)

        return train_df, test_df

    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """선물 및 현물 데이터 로드"""
        print("\n1️⃣ Loading raw data...")

        # Futures
        df_futures = pd.read_csv(self.config.futures_path)
        df_futures['open_time'] = pd.to_datetime(df_futures['open_time'])
        df_futures = df_futures.add_suffix('_futures')
        df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

        # Spot
        df_spot = pd.read_csv(self.config.spot_path)
        df_spot['open_time'] = pd.to_datetime(df_spot['open_time'])
        df_spot = df_spot.add_suffix('_spot')
        df_spot.rename(columns={'open_time_spot': 'open_time'}, inplace=True)

        print(f"  Futures: {len(df_futures):,} rows")
        print(f"  Spot:    {len(df_spot):,} rows")

        return df_futures, df_spot

    def _merge_data(self, df_futures: pd.DataFrame, df_spot: pd.DataFrame) -> pd.DataFrame:
        """선물 + 현물 데이터 병합"""
        print("\n2️⃣ Merging futures + spot data...")

        df = pd.merge(df_futures, df_spot, on='open_time', how='inner')

        # 시간 순서로 정렬 (중요!)
        df = df.sort_values('open_time').reset_index(drop=True)

        print(f"  Merged: {len(df):,} rows")
        print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 품질 검증 및 정제"""
        print("\n3️⃣ Validating data quality...")

        initial_len = len(df)

        # 중복 제거
        df = df.drop_duplicates(subset='open_time', keep='first')
        if len(df) < initial_len:
            print(f"  ⚠️  Removed {initial_len - len(df)} duplicate rows")

        # 결측치 확인
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("  ⚠️  Missing values detected:")
            for col in missing_counts[missing_counts > 0].index:
                print(f"    {col}: {missing_counts[col]}")

            # 결측치 제거 (보수적 접근)
            df = df.dropna()
            print(f"  Removed rows with missing values: {initial_len - len(df)}")

        # 음수 가격/거래량 확인
        price_cols = [col for col in df.columns if 'close' in col or 'open' in col or 'high' in col or 'low' in col]
        volume_cols = [col for col in df.columns if 'volume' in col]

        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"  ⚠️  Found non-positive values in {col}")
                df = df[df[col] > 0]

        for col in volume_cols:
            if (df[col] < 0).any():
                print(f"  ⚠️  Found negative values in {col}")
                df = df[df[col] >= 0]

        # 극단적 가격 변동 확인 (플래시 크래시 등)
        df['price_change_pct'] = df['close_futures'].pct_change() * 100
        extreme_moves = df[abs(df['price_change_pct']) > 10]  # 10% 이상 변동

        if len(extreme_moves) > 0:
            print(f"  ⚠️  Found {len(extreme_moves)} extreme price movements (>10%)")
            print(f"    Keeping them (potential real events)")

        # 시간 간격 확인 (5분봉이 맞는지)
        df['time_diff'] = df['open_time'].diff().dt.total_seconds() / 60
        irregular_intervals = df[df['time_diff'] != 5]

        if len(irregular_intervals) > 1:  # 첫 번째 행 제외
            print(f"  ⚠️  Found {len(irregular_intervals)-1} irregular time intervals")
            print(f"    This is normal for data gaps")

        df = df.drop(columns=['price_change_pct', 'time_diff'])

        print(f"  ✅ Validated: {len(df):,} rows")

        return df

    def _time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        시간 기반 데이터 분할

        Look-ahead bias 방지:
        - 과거 데이터로만 학습
        - 미래 데이터는 테스트에만 사용
        """
        print("\n4️⃣ Time-based split (preventing look-ahead bias)...")

        # 현재 시간 기준
        latest_time = df['open_time'].max()

        # 분할 경계 계산
        train_start = latest_time - timedelta(days=self.config.train_days_before)
        train_end = latest_time - timedelta(days=self.config.train_days_until)
        test_start = latest_time - timedelta(days=self.config.test_days_before)
        test_end = latest_time - timedelta(days=self.config.test_days_until)

        print(f"  Train period: {train_start.date()} to {train_end.date()}")
        print(f"  Test period:  {test_start.date()} to {test_end.date()}")

        # 분할
        train_df = df[(df['open_time'] >= train_start) & (df['open_time'] < train_end)].copy()
        test_df = df[(df['open_time'] >= test_start) & (df['open_time'] <= test_end)].copy()

        # 인덱스 리셋 (중요!)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, test_df

    def _print_split_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """데이터 분할 요약 출력"""
        print("\n5️⃣ Split summary:")
        print(f"  Train set: {len(train_df):,} rows")
        print(f"    Period: {train_df['open_time'].min().date()} to {train_df['open_time'].max().date()}")
        print(f"    Duration: {(train_df['open_time'].max() - train_df['open_time'].min()).days} days")

        print(f"\n  Test set:  {len(test_df):,} rows")
        print(f"    Period: {test_df['open_time'].min().date()} to {test_df['open_time'].max().date()}")
        print(f"    Duration: {(test_df['open_time'].max() - test_df['open_time'].min()).days} days")

        # 시간 갭 확인
        train_end = train_df['open_time'].max()
        test_start = test_df['open_time'].min()
        gap = (test_start - train_end).total_seconds() / 60 / 60 / 24  # days

        if gap > 1:
            print(f"\n  ⚠️  Gap between train and test: {gap:.1f} days")
        else:
            print(f"\n  ✅ No significant gap between train and test")

        # 가격 통계
        train_price_range = (train_df['close_futures'].min(), train_df['close_futures'].max())
        test_price_range = (test_df['close_futures'].min(), test_df['close_futures'].max())

        print(f"\n  Train price range: ${train_price_range[0]:,.2f} - ${train_price_range[1]:,.2f}")
        print(f"  Test price range:  ${test_price_range[0]:,.2f} - ${test_price_range[1]:,.2f}")

        print("\n" + "="*80)
        print("✅ DATA LOADING COMPLETE")
        print("="*80)


def load_specific_period(
    futures_path: str,
    spot_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    특정 기간 데이터 로드 (유틸리티 함수)

    Parameters:
    -----------
    futures_path : str
        선물 데이터 경로
    spot_path : str
        현물 데이터 경로
    start_date : str, optional
        시작일 (YYYY-MM-DD)
    end_date : str, optional
        종료일 (YYYY-MM-DD)

    Returns:
    --------
    df : pd.DataFrame
        병합된 데이터
    """
    # Futures
    df_futures = pd.read_csv(futures_path)
    df_futures['open_time'] = pd.to_datetime(df_futures['open_time'])
    df_futures = df_futures.add_suffix('_futures')
    df_futures.rename(columns={'open_time_futures': 'open_time'}, inplace=True)

    # Spot
    df_spot = pd.read_csv(spot_path)
    df_spot['open_time'] = pd.to_datetime(df_spot['open_time'])
    df_spot = df_spot.add_suffix('_spot')
    df_spot.rename(columns={'open_time_spot': 'open_time'}, inplace=True)

    # Merge
    df = pd.merge(df_futures, df_spot, on='open_time', how='inner')
    df = df.sort_values('open_time').reset_index(drop=True)

    # Filter by date
    if start_date:
        df = df[df['open_time'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['open_time'] <= pd.to_datetime(end_date)]

    return df


if __name__ == '__main__':
    # 테스트
    from config import config

    loader = TradingDataLoader(config.data)
    train_df, test_df = loader.load_and_split_data()

    print("\n📊 Sample data (train):")
    print(train_df.head())

    print("\n📊 Sample data (test):")
    print(test_df.head())
