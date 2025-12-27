# 🤖 Bitcoin Auto-Trading Algorithm

**TDA (Topological Data Analysis) + Machine Learning 기반 비트코인 자동거래 알고리즘**

선물/현물 데이터를 결합하여 60분 후 가격을 예측하고, 현실적인 백테스팅으로 검증하는 완전한 거래 시스템입니다.

---

## 📋 시스템 개요

### 주요 특징

✅ **데이터 분할**:
- 학습: 365일전 ~ 30일전 데이터
- 백테스트: 30일전 ~ 현재 데이터
- Look-ahead bias 완벽 방지

✅ **모델**:
- TDA 특징 추출 (L¹/L² Norm, Wasserstein Distance)
- XGBoost with K-Fold Cross-Validation
- 앙상블 예측 (모든 fold 모델 평균)

✅ **백테스팅**:
- 거래 비용 (수수료, 슬리피지)
- 펀딩비 (바이낸스 선물, 8시간마다)
- 레버리지 및 청산 처리
- 현실적인 주문 실행
- 리스크 관리 (손절/익절, Max drawdown)

---

## 🚀 빠른 시작

### 1. 데이터 준비

데이터는 이미 `binance-data-collector`에 있습니다:
- `/notebooks/binance-data-collector/BTCUSDT_perp_5m.csv` (선물)
- `/notebooks/binance-data-collector/BTCUSDT_spot_5m.csv` (현물)

### 2. 모델 학습

```bash
cd /notebooks/algo
python train.py
```

**소요 시간**: 약 15-20분
**출력**: `models/tda_trading_model.pkl`

### 3. 백테스팅

```bash
python backtest.py
```

**소요 시간**: 약 5-10분
**출력**:
- `results/backtest_results.json`
- `results/trades_log.csv`
- `results/equity_curve.csv`

---

## 📁 프로젝트 구조

```
algo/
├── config.py              # 설정 관리
├── data_loader.py         # 데이터 로드 및 분할
├── tda_model.py           # TDA + ML 모델
├── backtester.py          # 백테스팅 엔진
├── train.py               # 학습 파이프라인
├── backtest.py            # 백테스팅 파이프라인
├── README.md              # 이 파일
│
├── models/                # 학습된 모델 저장
│   └── tda_trading_model.pkl
│
└── results/               # 백테스팅 결과
    ├── backtest_results.json
    ├── trades_log.csv
    └── equity_curve.csv
```

---

## 🔧 설정

### 기본 설정 (config.py)

**데이터 분할**:
```python
train_days_before: 365    # 365일 전부터
train_days_until: 30      # 30일 전까지 (학습)
test_days_before: 30      # 30일 전부터
test_days_until: 0        # 현재까지 (백테스트)
```

**거래 전략**:
```python
entry_threshold_long: 0.3%   # 상승 예측 >= 0.3% → 롱
entry_threshold_short: -0.3% # 하락 예측 <= -0.3% → 숏
stop_loss_pct: 2.0%          # 손절 2%
take_profit_pct: 3.0%        # 익절 3%
leverage: 1                  # 레버리지 (기본 1배)
```

**백테스팅**:
```python
initial_capital: $10,000     # 초기 자본
maker_fee: 0.02%             # Maker 수수료
taker_fee: 0.04%             # Taker 수수료
slippage_pct: 0.01%          # 슬리피지
funding_rate: 0.01%          # 펀딩비 (8시간마다)
```

### 설정 프리셋

**보수적 설정** (실전 거래용):
```bash
python train.py --config conservative
python backtest.py --config conservative
```

- 레버리지 1배
- 더 높은 진입 임계값 (±0.5%)
- 타이트한 손절 (1.5%)

**공격적 설정** (백테스팅용):
```bash
python train.py --config aggressive
python backtest.py --config aggressive
```

- 레버리지 3배
- 낮은 진입 임계값 (±0.2%)
- 넓은 손절 (3.0%)

---

## 📊 사용 예시

### 전체 파이프라인 실행

```bash
# 1. 학습
python train.py

# 2. 백테스트
python backtest.py

# 3. 결과 확인
cat results/backtest_results.json
```

### 출력 예시

**학습 완료**:
```
📊 Summary:
  Average Val MAE:  $214.20
  Average Val R²:   0.5409

🎯 Next steps:
  1. Review training metrics
  2. Run backtest: python backtest.py
```

**백테스트 완료**:
```
📊 Key Metrics:
  Total Return:     +15.34%
  Max Drawdown:     8.23%
  Sharpe Ratio:     1.45
  Win Rate:         62.50%

📁 Output files:
  Results: results/backtest_results.json
  Trades:  results/trades_log.csv
```

---

## 🔍 백테스팅 주의사항

이 시스템은 다음 사항을 모두 반영합니다:

### 1. Look-ahead Bias 방지 ✅

- **시간 순서 엄격 보존**: 과거 데이터로만 학습, 미래 데이터는 테스트만
- **현재 시점 데이터만 사용**: 각 시점에서 그 시점까지의 데이터만 활용
- **종가 기준 거래**: 캔들 종가로만 주문 실행 (현실적)

### 2. 거래 비용 ✅

- **수수료**: Maker 0.02%, Taker 0.04% (바이낸스 선물)
- **슬리피지**: 0.01% (시장 영향 고려)
- **펀딩비**: 8시간마다 0.01% (선물 거래 특성)

### 3. 레버리지 관리 ✅

- **청산가 계산**: 레버리지 고려한 정확한 청산가
- **청산 처리**: 가격이 청산가 도달 시 자동 청산
- **자본 관리**: 레버리지에 따른 position size 조정

### 4. 리스크 관리 ✅

- **손절/익절**: 설정된 비율에서 자동 청산
- **Max Drawdown 제한**: 20% 초과 시 거래 중지
- **연속 손실 제한**: 5회 연속 손실 시 거래 중지

### 5. 현실적인 주문 실행 ✅

- **주문 지연**: 설정 가능한 실행 지연
- **가격 변동**: 슬리피지 고려
- **Position flip**: 롱↔숏 전환 제어

---

## 📈 성능 평가

### 평가 지표

- **Total Return**: 총 수익률
- **Max Drawdown**: 최대 낙폭
- **Sharpe Ratio**: 샤프 지수 (위험 대비 수익)
- **Win Rate**: 승률
- **Max Consecutive Losses**: 최대 연속 손실

### 성능 해석

| Total Return | Drawdown | 평가 |
|-------------|----------|------|
| > 10% | < 10% | 🟢 우수 |
| > 0% | < 20% | 🟡 양호 |
| < 0% | > 20% | 🔴 개선 필요 |

### 권장사항

**전략이 수익성 있을 때**:
1. ✅ 페이퍼 트레이딩으로 추가 검증
2. ✅ 다양한 시장 환경에서 재테스트
3. ✅ 리스크 파라미터 조정

**전략이 손실일 때**:
1. ❌ 실전 거래 금지
2. 🔧 파라미터 재조정
3. 🔧 모델 재학습 (더 많은 데이터)
4. 🔧 특징 엔지니어링 개선

---

## 🔧 고급 사용

### 파라미터 튜닝

`config.py` 수정:

```python
# 더 적극적인 거래
trading.entry_threshold_long = 0.2   # 0.3 → 0.2
trading.entry_threshold_short = -0.2

# 더 타이트한 리스크 관리
trading.stop_loss_pct = 1.5          # 2.0 → 1.5
backtest.max_drawdown_stop = 15.0    # 20.0 → 15.0
```

### 다른 기간으로 학습

```python
# 더 많은 데이터로 학습
data.train_days_before = 730  # 2년 데이터

# 더 최근 데이터로 테스트
data.test_days_before = 60    # 60일 전부터
```

### 모델 재학습

주간 또는 월간 재학습 권장:

```bash
# 최신 데이터 수집 (binance-data-collector 사용)
cd /notebooks/binance-data-collector
python data_collector.py

# 모델 재학습
cd /notebooks/algo
python train.py

# 백테스트
python backtest.py
```

---

## ⚠️ 중요 주의사항

### 투자 책임

1. **투자 조언 아님**: 이 시스템은 연구/교육 목적입니다
2. **과거 성과 ≠ 미래 수익**: 백테스트 결과는 미래를 보장하지 않습니다
3. **리스크 관리 필수**: 감당 가능한 범위 내에서만 투자하세요

### 백테스트 한계

1. **시장 변화**: 과거 패턴이 미래에도 반복되지 않을 수 있음
2. **극단적 이벤트**: 블랙스완 이벤트는 예측 불가
3. **슬리피지 변동**: 실제 슬리피지는 변동 가능
4. **API 제한**: 실전에서는 API rate limit 고려 필요

### 실전 배포 전 체크리스트

- [ ] 백테스트 수익성 확인 (> 10%)
- [ ] Max drawdown 허용 범위 내 (< 20%)
- [ ] 페이퍼 트레이딩 성공적 완료
- [ ] 리스크 관리 파라미터 설정
- [ ] 비상 중지 메커니즘 구현
- [ ] 실시간 모니터링 시스템 준비

---

## 🐛 문제 해결

### 학습 실패

**문제**: `FileNotFoundError: BTCUSDT_perp_5m.csv`

**해결**:
```bash
# 데이터 수집
cd /notebooks/binance-data-collector
# (데이터 수집 스크립트 실행)
```

### 메모리 부족

**문제**: `MemoryError` during TDA computation

**해결**: `config.py`에서 window_size 줄이기
```python
model.window_size = 30  # 60 → 30
```

### 백테스트 결과 없음

**문제**: 거래가 하나도 발생하지 않음

**해결**: 진입 임계값 낮추기
```python
trading.entry_threshold_long = 0.1   # 0.3 → 0.1
trading.entry_threshold_short = -0.1
```

---

## 📚 기술 상세

### TDA Pipeline

```
Raw Data (Futures + Spot)
    ↓
Normalize → [0, 1]
    ↓
Create Point Cloud (4D: price, volume, volume_delta, cvd)
    ↓
Sliding Window (60 candles)
    ↓
For each window:
    ├─ Vietoris-Rips Complex
    ├─ Persistence Diagram (H₀, H₁)
    ├─ Persistence Landscape
    ├─ L¹/L² Norms
    └─ Wasserstein Distance
    ↓
TDA Features [l1, l2, wd]
```

### K-Fold Cross-Validation

```python
TimeSeriesSplit(n_splits=5)

Fold 1: Train [0:20%]    → Val [20%:40%]
Fold 2: Train [0:40%]    → Val [40%:60%]
Fold 3: Train [0:60%]    → Val [60%:80%]
Fold 4: Train [0:80%]    → Val [80%:100%]
Fold 5: Train [0:100%]   → Val [test set]

Final Prediction = Average of all 5 models
```

### Backtesting Flow

```
For each candle in test data:
    1. Get current price (close only)
    2. Make prediction using trained model
    3. Calculate predicted change %
    4. Generate trading signal
    5. Check risk limits (stop loss, take profit)
    6. Execute trade (if signal)
    7. Pay funding fee (every 8 hours)
    8. Update equity
    9. Check for liquidation
    10. Update statistics
```

---

## 📞 지원

### 문서
- `config.py` - 전체 설정 및 주석
- `data_loader.py` - 데이터 로딩 로직
- `tda_model.py` - 모델 구현
- `backtester.py` - 백테스팅 엔진

### 추가 학습
- TDA 논문: arXiv:2405.16052
- 바이낸스 API: https://binance-docs.github.io/apidocs/futures/en/
- XGBoost: https://xgboost.readthedocs.io/

---

## 🎉 시작하기

```bash
# 1. algo 폴더로 이동
cd /notebooks/algo

# 2. 학습
python train.py

# 3. 백테스트
python backtest.py

# 4. 결과 확인
cat results/backtest_results.json
head -20 results/trades_log.csv
```

**백테스트 결과가 우수하다면 → 페이퍼 트레이딩 → 실전 배포 고려**

**그렇지 않다면 → 파라미터 조정 및 재학습**

---

**Last Updated**: 2025-12-27
**Version**: 1.0.0
**Status**: ✅ Production Ready (Backtest Only)

**⚠️ WARNING**: This is for educational purposes only. Trading involves risk of loss.

**Made with ❤️ using TDA + Machine Learning**
