# 🎯 Trump Data Experiment

**Post-Trump 데이터로 학습하고 Trump 30-Days 데이터로 백테스팅**

트럼프 당선 이후 비트코인 시장 데이터를 사용한 TDA + Machine Learning 기반 자동거래 알고리즘 실험입니다.

---

## 📋 실험 개요

### 데이터셋

| 데이터셋 | 용도 | 파일 | 행 수 | 기간 |
|---------|------|------|-------|------|
| **Post-Trump** | 학습 | BTCUSDT_perp/spot_post_trump.csv | ~119,847 | 트럼프 당선 이후 |
| **Trump 30-Days** | 백테스트 | BTCUSDT_perp/spot_trump_30days.csv | ~8,642 | 최근 30일 |

### 실험 목적

1. **트럼프 당선 이후 시장 패턴 학습**: Post-Trump 데이터로 TDA 특징과 가격 패턴 학습
2. **최근 30일 성능 검증**: 학습된 모델이 최근 시장에서도 작동하는지 백테스팅
3. **현실적인 거래 시뮬레이션**: 모든 거래 비용과 리스크 반영

---

## 🔬 방법론

### 1. 데이터 처리

```
Post-Trump Data (학습)
├── Futures: BTCUSDT_perp_post_trump.csv
└── Spot: BTCUSDT_spot_post_trump.csv
    ↓
Merge on timestamp
    ↓
119,847 rows × 5min candles
    ↓
Feature Engineering
    ↓
ML Dataset

Trump 30-Days Data (백테스트)
├── Futures: BTCUSDT_perp_trump_30days.csv
└── Spot: BTCUSDT_spot_trump_30days.csv
    ↓
Merge on timestamp
    ↓
8,642 rows × 5min candles
    ↓
Same Feature Engineering
    ↓
Backtest Dataset
```

### 2. TDA 특징 추출

**Topological Data Analysis** 를 사용하여 시장의 위상학적 구조를 분석합니다.

```python
# 4D Point Cloud 생성
point_cloud = [close, volume, volume_delta, cvd]

# Sliding Window (60 candles = 5 hours)
for each window:
    ├─ Vietoris-Rips Complex 구성
    ├─ Persistence Diagram 계산 (H₀, H₁)
    ├─ Persistence Landscape 변환
    ├─ L¹ Norm 계산 (시장 복잡도)
    ├─ L² Norm 계산 (구조적 안정성)
    └─ Wasserstein Distance (토폴로지 변화율)

Output: [l1_norm, l2_norm, wasserstein_dist]
```

**TDA 특징의 의미**:

- **L¹ Norm**: 시장의 위상학적 복잡도
  - 높음 (> 0.8): 복잡한 시장 구조 → 높은 변동성
  - 낮음 (< 0.3): 단순한 구조 → 안정적 시장

- **L² Norm**: 구조적 안정성 (L¹과 유사하지만 이상치에 더 민감)

- **Wasserstein Distance**: 연속된 시점 간 토폴로지 변화율
  - 높음 (> 0.01): 빠른 구조 변화 → 추세 전환 가능
  - 낮음 (< 0.005): 안정적 토폴로지 → 추세 지속

### 3. Machine Learning 특징

TDA 특징 외에도 다음 특징들을 사용합니다:

**가격 특징**:
```python
- price_futures, price_spot
- price_spread (선물 - 현물)
- price_spread_pct (%)
- return_futures_1, return_futures_5, return_futures_12
- return_spot_1
```

**거래량 특징**:
```python
- volume_futures, volume_spot
- volume_ratio (선물/현물)
- cvd_futures, cvd_spot (Cumulative Volume Delta)
```

**기술적 지표**:
```python
- ma_5, ma_12, ma_24 (이동평균)
- volatility_5, volatility_12 (변동성)
- hl_range_futures, hl_range_spot (고가-저가 범위)
```

**총 특징 수**: ~30개

### 4. K-Fold Cross-Validation

시계열 데이터의 특성을 고려한 **TimeSeriesSplit**을 사용합니다:

```
Fold 1: Train [0%-20%]  → Validate [20%-40%]
Fold 2: Train [0%-40%]  → Validate [40%-60%]
Fold 3: Train [0%-60%]  → Validate [60%-80%]
Fold 4: Train [0%-80%]  → Validate [80%-100%]
Fold 5: Train [0%-100%] → Validate [Hold-out set]

각 Fold마다 XGBoost 모델 학습
    ↓
최종 예측 = 모든 Fold 모델의 평균 (앙상블)
```

**XGBoost 하이퍼파라미터**:
```python
n_estimators=200
max_depth=7
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
```

### 5. 거래 전략

**예측 기반 시그널 생성**:

```python
predicted_change_pct = (predicted_price - current_price) / current_price * 100

if predicted_change_pct >= +0.3%:
    → LONG 진입

elif predicted_change_pct <= -0.3%:
    → SHORT 진입

elif abs(predicted_change_pct) <= 0.1%:
    → NEUTRAL (포지션 청산)

else:
    → 기존 포지션 유지
```

**리스크 관리**:
```python
- Stop Loss: 2% 손실 시 자동 청산
- Take Profit: 3% 수익 시 자동 청산
- Max Drawdown: 20% 초과 시 거래 중지
- Max Consecutive Losses: 5회 연속 손실 시 중지
```

---

## 🔬 백테스팅 방법론

### Look-ahead Bias 방지

**절대 금지사항**:
```python
# ❌ 잘못된 예시 (미래 데이터 사용)
prediction = model.predict(future_data)  # 미래 정보 사용

# ✅ 올바른 예시 (현재 시점 데이터만 사용)
for i in range(len(data)):
    current_data = data[:i+1]  # 현재 시점까지만
    prediction = model.predict(current_data[-1])
```

**구현**:
```python
for i in range(len(test_data)):
    # 1. 현재 시점 데이터만 사용
    current_row = test_data.iloc[i]
    current_price = current_row['close_futures']  # 종가만 사용

    # 2. 예측 생성 (과거 데이터 기반)
    predicted_price = predictions[i]

    # 3. 거래 시그널 생성
    signal = generate_signal(predicted_price, current_price)

    # 4. 거래 실행
    execute_trade(signal, current_price)
```

### 거래 비용 반영

**수수료** (바이낸스 선물):
```python
Maker Fee: 0.02%
Taker Fee: 0.04% (보수적으로 Taker 사용)

거래 비용 = position_value × 0.0004
```

**슬리피지**:
```python
Slippage: 0.01%

실제 체결 가격 = 주문 가격 × (1 ± 0.0001)
```

**펀딩비** (선물 거래 특성):
```python
Funding Rate: 0.01% (8시간마다)

포지션 보유 시 8시간마다 0.01% 차감
```

### 레버리지 & 청산

**청산가 계산**:
```python
# 롱 포지션
liquidation_price = entry_price × (1 - 1/leverage × 0.9)

# 숏 포지션
liquidation_price = entry_price × (1 + 1/leverage × 0.9)

# 예시: 레버리지 1배 (현물 수준)
# 롱: entry × 0.1 = 거의 청산 없음
# 레버리지 3배일 경우:
# 롱: entry × 0.7 = 30% 하락 시 청산
```

**청산 처리**:
```python
if current_price <= liquidation_price:  # 롱
    capital = 0  # 전액 손실
    close_position()
```

### 현실적인 주문 실행

**종가 기준 거래**:
```python
# ✅ 현실적: 캔들 종가로만 거래
entry_price = candle['close']

# ❌ 비현실적: 캔들 내 최적가 사용
entry_price = candle['low']  # 매수 시 최저가는 불가능
```

**주문 지연**:
```python
order_execution_delay = 0  # 기본값: 즉시 실행

# 옵션: 1 candle 지연 (더 보수적)
order_execution_delay = 1
```

---

## 📊 평가 지표

### 수익성 지표

**Total Return**:
```
총 수익률 (%) = (최종 자본 - 초기 자본) / 초기 자본 × 100
```

**Sharpe Ratio**:
```
샤프 지수 = (평균 수익률 - 무위험 수익률) / 수익률 표준편차

높을수록 좋음 (> 1.0 양호, > 2.0 우수)
```

### 리스크 지표

**Max Drawdown**:
```
최대 낙폭 (%) = (최고점 - 최저점) / 최고점 × 100

낮을수록 좋음 (< 10% 우수, < 20% 양호)
```

**Win Rate**:
```
승률 (%) = 수익 거래 수 / 전체 거래 수 × 100

높을수록 좋음 (> 50% 양호, > 60% 우수)
```

### 거래 통계

- **Total Trades**: 전체 거래 횟수
- **Winning Trades**: 수익 거래 수
- **Losing Trades**: 손실 거래 수
- **Max Consecutive Losses**: 최대 연속 손실 (낮을수록 좋음)

---

## 🎯 예상 결과

### 학습 성능 (Post-Trump Data)

K-Fold CV 결과:
```
Average Val MAE: $150-250 (비트코인 가격 대비 0.2-0.3%)
Average Val R²: 0.45-0.65
```

### 백테스트 성능 (Trump 30-Days)

**낙관적 시나리오**:
```
Total Return: +10% ~ +20%
Max Drawdown: < 15%
Win Rate: > 55%
Sharpe Ratio: > 1.0
```

**현실적 시나리오**:
```
Total Return: +3% ~ +10%
Max Drawdown: < 20%
Win Rate: 50% ~ 55%
Sharpe Ratio: 0.5 ~ 1.0
```

**비관적 시나리오**:
```
Total Return: -5% ~ +3%
Max Drawdown: > 20%
Win Rate: < 50%
Sharpe Ratio: < 0.5
```

---

## 🚀 실행 방법

### 1. 데이터 준비

데이터는 이미 `algo/data/` 폴더에 다운로드되어 있습니다:
```bash
cd /notebooks/algo/data
ls -lh
# BTCUSDT_perp_post_trump.csv
# BTCUSDT_perp_trump_30days.csv
# BTCUSDT_spot_post_trump.csv
# BTCUSDT_spot_trump_30days.csv
```

### 2. 모델 학습

```bash
cd /notebooks/algo
python train_trump.py
```

**소요 시간**: 약 20-30분 (119,847 샘플, K-fold CV)

**출력**:
- `models/tda_trump_model.pkl` - 학습된 모델
- `results_trump/training_metrics.json` - 학습 성능 지표

### 3. 백테스팅

```bash
python backtest_trump.py
```

**소요 시간**: 약 5-10분

**출력**:
- `results_trump/backtest_results.json` - 백테스트 결과
- `results_trump/trades_log.csv` - 거래 내역
- `results_trump/equity_curve.csv` - Equity curve
- `results_trump/backtest_visualization.png` - 시각화

---

## 📊 결과 해석

### 시각화 차트 설명

생성되는 `backtest_visualization.png`에는 6개의 차트가 포함됩니다:

**1. Equity Curve** (좌상단):
- 시간에 따른 자본 변화
- 초기 자본 대비 수익/손실 영역 색상 표시
- 상승 추세 → 수익성 있음
- 하락 추세 → 손실

**2. Returns Distribution** (우상단):
- 거래별 손익 분포
- 정규분포 형태 → 안정적 거래
- 오른쪽 치우침 → 수익 편향
- 왼쪽 치우침 → 손실 편향

**3. Win/Loss Pie Chart** (좌중단):
- 수익 거래 vs 손실 거래 비율
- 50% 이상 → 양호한 승률

**4. Performance Metrics** (우중단):
- 주요 성능 지표 막대 그래프
- 양수 (녹색) → 좋음
- 음수 (빨강) → 나쁨

**5. Drawdown Over Time** (좌하단):
- 시간에 따른 낙폭 변화
- 낮을수록 좋음
- 급격한 상승 → 큰 손실 발생

**6. Trade Timeline** (우하단):
- 거래 시점과 가격 표시
- 녹색 ▲: 진입
- 파랑 ▼: 수익 청산
- 빨강 ▼: 손실 청산

### 결과 판단 기준

**우수한 결과**:
- ✅ Total Return > 10%
- ✅ Max Drawdown < 10%
- ✅ Win Rate > 60%
- ✅ Sharpe Ratio > 1.5
→ **실전 고려 가능** (페이퍼 트레이딩 후)

**양호한 결과**:
- ✅ Total Return > 3%
- ✅ Max Drawdown < 20%
- ✅ Win Rate > 50%
- ✅ Sharpe Ratio > 0.5
→ **파라미터 조정 후 재시도**

**불량한 결과**:
- ❌ Total Return < 0%
- ❌ Max Drawdown > 20%
- ❌ Win Rate < 45%
- ❌ Sharpe Ratio < 0
→ **전략 재검토 필요**

---

## 🔧 파라미터 조정

성능 개선을 위한 파라미터 조정:

### 더 보수적으로 (리스크 감소)

`config_trump.py` 수정:
```python
# 진입 임계값 상승
trading.entry_threshold_long = 0.5  # 0.3 → 0.5
trading.entry_threshold_short = -0.5

# 손절 타이트하게
trading.stop_loss_pct = 1.5  # 2.0 → 1.5

# 레버리지 감소
trading.leverage = 1  # 유지 또는 감소
```

### 더 공격적으로 (수익 증대)

```python
# 진입 임계값 하락
trading.entry_threshold_long = 0.2  # 0.3 → 0.2
trading.entry_threshold_short = -0.2

# 익절 목표 상승
trading.take_profit_pct = 5.0  # 3.0 → 5.0

# 레버리지 증가
trading.leverage = 2  # 1 → 2 (주의!)
```

---

## ⚠️ 주의사항

### 데이터 특성

**Post-Trump 데이터**:
- 트럼프 당선 이후 시장 (특정 시기)
- 해당 기간의 패턴에 과적합 가능
- 다른 시장 환경에서는 성능 저하 가능

**Trump 30-Days 데이터**:
- 최근 30일 데이터
- 시장 환경이 학습 데이터와 다를 수 있음
- Out-of-sample 성능 확인

### 백테스트 한계

1. **과거 성과 ≠ 미래 수익**: 백테스트 결과가 실전을 보장하지 않음
2. **시장 변화**: 새로운 시장 환경에서는 패턴이 달라질 수 있음
3. **슬리피지 변동**: 실제 슬리피지는 변동 가능
4. **유동성**: 대량 거래 시 체결 어려움 미반영

### 실전 배포 전 필수 확인

- [ ] 백테스트 수익성 확인 (> 10%)
- [ ] 리스크 지표 확인 (Drawdown < 20%)
- [ ] 여러 시장 환경에서 재테스트
- [ ] 페이퍼 트레이딩 1개월 이상
- [ ] 비상 중지 메커니즘 구현
- [ ] 실시간 모니터링 시스템 준비

---

## 📚 추가 분석

### TDA 특징 중요도 분석

학습 후 `results_trump/training_metrics.json`에서 확인:
```json
"feature_names": [
  "tda_l1",     // TDA L¹ Norm
  "tda_l2",     // TDA L² Norm
  "tda_wd",     // Wasserstein Distance
  "ma_12",      // 12-period MA
  "ma_24",      // 24-period MA
  ...
]
```

XGBoost 특징 중요도를 확인하여 어떤 특징이 예측에 가장 기여하는지 분석 가능

### 실패 케이스 분석

손실 거래가 많을 경우:
1. `results_trump/trades_log.csv`에서 손실 거래 확인
2. 손실 발생 시점의 TDA 지표 확인
3. 특정 패턴에서 실패하는지 분석
4. 필터 추가 (예: TDA L1 > 0.8일 때 거래 제한)

---

## 🎉 기대 효과

이 실험을 통해:

1. **TDA의 유효성 검증**: 위상학적 특징이 비트코인 가격 예측에 도움이 되는지 확인
2. **현실적인 백테스팅**: 모든 거래 비용을 반영한 실전과 유사한 환경
3. **시장 적응성 평가**: 학습 기간과 테스트 기간의 성능 차이 확인
4. **리스크 관리 효과**: 손절/익절이 실제로 작동하는지 검증

---

**Last Updated**: 2025-12-27
**Experiment Status**: ⏳ Running
**Next Steps**:
1. Complete training
2. Run backtest
3. Analyze results
4. Adjust parameters if needed

**Made with ❤️ using TDA + Machine Learning**
