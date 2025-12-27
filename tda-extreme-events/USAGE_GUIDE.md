# 📖 Bitcoin TDA Monitoring System - 사용 가이드

## 🎯 시스템 개요

이 시스템은 **Topological Data Analysis (TDA)**를 사용하여 비트코인 시장의 극단적 이벤트를 실시간으로 탐지합니다.

## 📚 L1 Norm 이해하기

### 🔬 L1 Norm이란?

**L1 Norm = 시장의 위상학적 복잡도를 숫자로 표현한 값**

#### 쉬운 비유:
- **L1 Norm = 0.2** → 🏞️ 평온한 호수 (정상)
- **L1 Norm = 0.6** → 🌊 파도 치는 바다 (주의)
- **L1 Norm = 1.2** → 🌪️ 폭풍우 치는 바다 (극단 이벤트!)

#### 수학적 의미:
```
L1 Norm = Persistence Landscape의 모든 높이를 더한 값

Persistence Landscape란?
→ 가격 데이터의 "위상학적 형태"를 함수로 나타낸 것
→ 시장 구조의 "구멍"이나 "연결"이 얼마나 오래 유지되는지 측정

L1 Norm ↑ = 복잡한 구조 많음 = 시장 불안정
L1 Norm ↓ = 단순한 구조 = 시장 안정
```

### 📊 실제 해석:

| L1 Norm 값 | Threshold 대비 | 상태 | 의미 |
|-----------|---------------|------|------|
| < 0.93 | < 70% | ✅ 정상 | 안정적 시장 |
| 0.93 - 1.20 | 70-90% | ⚡ 주의 | 변동성 증가 |
| 1.20 - 1.33 | 90-100% | ⚠️ 경고 | 극단 이벤트 임박 |
| > 1.33 | > 100% | 🚨 극단 | 극단 이벤트 발생! |

현재 Threshold = **1.332091** (최근 30일 데이터 기준)

---

## 🖥️ 1. 실시간 모니터링 스크립트

### 한 번만 실행 (현재 상태 체크):

```bash
cd /notebooks/tda-extreme-events
python monitor.py --data /notebooks/binance-data-collector/BTCUSDT_5m.csv
```

### 연속 모니터링 (5분마다 자동 체크):

```bash
python monitor.py --data /notebooks/binance-data-collector/BTCUSDT_5m.csv --continuous --interval 5
```

### 출력 예시:

```
======================================================================
🔍 BITCOIN TDA MONITOR - 2025-12-27 10:12:31
======================================================================

💰 Current Price: $9,253.24

📊 TOPOLOGICAL METRICS:

  L¹ Norm:  [█████████░░░░░░░░░░░░░░░░░░░░░] 32.1%
            Value: 0.428168 | Threshold: 1.332091

  L² Norm:  [████████████████░░░░░░░░░░░░░░] 55.6%
            Value: 0.069451 | Threshold: 0.124929

🎯 STATUS: ✅ NORMAL
   Market conditions normal

======================================================================
```

---

## 🌐 2. 웹 대시보드

### 대시보드 실행:

```bash
cd /notebooks/tda-extreme-events
./start_dashboard.sh
```

또는:

```bash
streamlit run dashboard.py --server.port=8501
```

### 접속:

브라우저에서 **http://localhost:8501** 열기

### 대시보드 기능:

#### 📊 메인 화면:
- **현재 가격** 실시간 표시
- **L¹ Norm, L² Norm** 현재 값과 threshold 대비 비율
- **Alert 상태** (✅ NORMAL / ⚡ WARNING / ⚠️ SEVERE / 🚨 CRITICAL)

#### 📈 게이지 차트:
- L¹ Norm, L² Norm의 현재 상태를 시각적으로 표시
- 색상 구분:
  - 🟢 녹색: 정상 (< 70%)
  - 🟡 노란색: 주의 (70-90%)
  - 🟠 주황색: 경고 (90-100%)
  - 🔴 빨간색: 극단 (> 100%)

#### 📉 시계열 차트:
- 비트코인 가격 추이
- L¹ Norm 변화 (threshold 표시)
- L² Norm 변화 (threshold 표시)
- 과거 1000개 데이터 포인트 시각화

#### 🔬 Persistence Diagram:
- 현재 윈도우의 위상학적 특징 시각화
- H₁ (1차원 homology) 특징 표시
- Birth-Death 좌표로 표현

#### ⚙️ 사이드바:
- **Auto-refresh**: 자동 새로고침 설정
- **Refresh interval**: 갱신 주기 조정 (10-300초)
- **Force Refresh**: 수동 새로고침

---

## 🎯 3. 사용 시나리오

### 시나리오 1: 트레이딩 리스크 관리

```
상황: L¹ Norm이 85%에 도달

해석: 시장 구조가 복잡해지고 있음
      극단 이벤트 가능성 증가

조치: 포지션 크기 줄이기
      Stop-loss 설정 강화
```

### 시나리오 2: 극단 이벤트 탐지

```
상황: L¹ Norm이 threshold 초과 (> 100%)
      L² Norm도 90% 이상

해석: 🚨 극단 이벤트 발생 중!
      시장이 급격하게 변화하고 있음

조치: 신규 진입 자제
      기존 포지션 재검토
```

### 시나리오 3: 정상 시장 확인

```
상황: L¹ Norm 40%, L² Norm 50%

해석: ✅ 안정적인 시장 환경
      정상 범위 내 변동

조치: 일반적인 트레이딩 전략 유지
```

---

## 📁 4. 생성되는 파일들

### `monitoring_status.json`
실시간 모니터링 결과가 JSON 형식으로 저장됩니다.

```json
{
  "timestamp": "2025-12-27 10:12:31",
  "price": 9253.24,
  "l1_norm": {
    "value": 0.428168,
    "percent": 32.1,
    "threshold": 1.332091
  },
  "alert": {
    "level": "NORMAL",
    "symbol": "✅",
    "message": "Market conditions normal"
  }
}
```

---

## ⚙️ 5. 고급 설정

### Window Size 조정:

```python
monitor = TDAMonitor(
    historical_data_path='...',
    window_size=60,      # 60개 데이터 포인트 (5분봉 → 5시간)
    lookback_days=30     # 최근 30일로 threshold 계산
)
```

- `window_size`: 분석할 데이터 윈도우 크기
  - 60 = 5시간 (5분봉 기준)
  - 120 = 10시간
  - 288 = 24시간

- `lookback_days`: Baseline threshold 계산에 사용할 과거 데이터 기간
  - 30일 = 최근 한 달 패턴 기준
  - 7일 = 최근 일주일 (더 민감)
  - 90일 = 최근 분기 (덜 민감)

### Threshold Sensitivity 조정:

기본값: **μ + 4σ** (매우 보수적)

더 민감하게:
- **μ + 3σ**: 더 많은 이벤트 탐지
- **μ + 2σ**: 매우 민감

덜 민감하게:
- **μ + 5σ**: 극히 드문 이벤트만 탐지

---

## 🔧 6. 문제 해결

### 문제: "FileNotFoundError"

```bash
# 데이터 파일 경로 확인
ls /notebooks/binance-data-collector/BTCUSDT_5m.csv

# 절대 경로 사용
python monitor.py --data /notebooks/binance-data-collector/BTCUSDT_5m.csv
```

### 문제: "Insufficient data"

```
원인: 데이터 포인트가 window_size보다 적음
해결: window_size 줄이기 또는 더 많은 데이터 수집
```

### 문제: 대시보드가 느림

```
해결: lookback_days 줄이기 (30 → 7)
      또는 데이터 샘플링 (5분봉 → 15분봉)
```

---

## 📊 7. 모니터링 권장 설정

### 단기 트레이더:
```
window_size = 60      # 5시간
lookback_days = 7     # 1주일
interval = 5분
```

### 중기 트레이더:
```
window_size = 120     # 10시간
lookback_days = 30    # 1개월
interval = 15분
```

### 장기 투자자:
```
window_size = 288     # 24시간
lookback_days = 90    # 3개월
interval = 1시간
```

---

## 📖 8. 참고 자료

- **원본 논문**: [arXiv:2405.16052](https://arxiv.org/abs/2405.16052)
- **Persistence Homology**: [Wikipedia](https://en.wikipedia.org/wiki/Persistent_homology)
- **TDA 입문**: Carlsson, G. (2009). "Topology and data." Bull. Amer. Math. Soc.

---

## 🚀 빠른 시작 요약

```bash
# 1. 현재 상태 확인
cd /notebooks/tda-extreme-events
python monitor.py --data /notebooks/binance-data-collector/BTCUSDT_5m.csv

# 2. 연속 모니터링 시작
python monitor.py --data /notebooks/binance-data-collector/BTCUSDT_5m.csv --continuous

# 3. 웹 대시보드 실행
./start_dashboard.sh

# 4. 브라우저에서 접속
# http://localhost:8501
```

---

## ⚡ 핵심 포인트

1. **L1 Norm ↑ = 시장 복잡도 ↑ = 위험도 ↑**
2. **Threshold 대비 70% 이상 → 주의**
3. **Threshold 초과 → 극단 이벤트!**
4. **실시간 모니터링 + 알림 = 리스크 관리**

Happy Trading! 📈🚀
