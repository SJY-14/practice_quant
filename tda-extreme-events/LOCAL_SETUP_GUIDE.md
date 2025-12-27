# 🏠 로컬 환경 설정 및 실행 가이드

## 📋 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [설치 및 설정](#설치-및-설정)
3. [실행 플로우](#실행-플로우)
4. [문제 해결](#문제-해결)

---

## 시스템 요구사항

### 필수 소프트웨어
- **Python**: 3.8 이상
- **Git**: 최신 버전
- **메모리**: 최소 8GB RAM (16GB 권장)
- **저장공간**: 최소 2GB

### 필수 Python 라이브러리
```
numpy
pandas
scikit-learn
xgboost
joblib
requests
gudhi (TDA 라이브러리)
plotly (대시보드용)
jupyter (대시보드용)
```

---

## 설치 및 설정

### Step 1: Git Repository Clone

```bash
# Repository clone (또는 pull)
git clone <your-repo-url>
cd tda-extreme-events

# 이미 clone했다면
git pull origin main
```

### Step 2: Python 가상환경 생성 (권장)

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: 필수 라이브러리 설치

```bash
# requirements.txt 생성 (아래 내용 복사)
cat > requirements.txt << EOF
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0
requests>=2.26.0
gudhi>=3.5.0
plotly>=5.0.0
jupyter>=1.0.0
ipywidgets>=7.6.0
EOF

# 라이브러리 설치
pip install -r requirements.txt
```

**GUDHI 설치 문제 시**:
```bash
# conda 사용 (권장)
conda install -c conda-forge gudhi

# 또는 소스 빌드
pip install --upgrade pip
pip install gudhi
```

### Step 4: 디렉토리 구조 확인

설치 후 디렉토리 구조:
```
tda-extreme-events/
├── realtime_predictor.py          # 모델 학습 스크립트
├── live_data_fetcher.py           # 실시간 데이터 수집 & 예측
├── tda_analysis.py                # TDA 핵심 로직
├── live_dashboard.ipynb           # Jupyter 대시보드
├── test_prediction_offline.py     # 오프라인 테스트
├── LOCAL_SETUP_GUIDE.md           # 이 파일
├── REALTIME_SYSTEM_GUIDE.md       # 상세 가이드
├── FINAL_SUMMARY.md               # 시스템 요약
└── binance_client.py              # 바이낸스 API 클라이언트
```

---

## 실행 플로우

### 🎯 전체 플로우 요약

```
1. 데이터 수집 (바이낸스 API)
   ↓
2. 모델 학습 (일회성)
   ↓
3. 실시간 예측 실행
   ↓
4. 대시보드 확인
```

---

### 플로우 1: 초기 데이터 수집 (최초 1회)

바이낸스에서 과거 데이터를 수집합니다.

#### Option A: 별도 데이터 수집기 사용

```bash
# binance-data-collector 폴더가 없다면 생성
mkdir -p ../binance-data-collector
cd ../binance-data-collector

# 데이터 수집 스크립트 실행 (예시)
# 실제로는 별도 수집기를 사용하거나 아래 스크립트 사용
```

#### Option B: 간단한 수집 스크립트 사용

`collect_historical_data.py` 생성:

```python
"""
바이낸스 과거 데이터 수집 스크립트
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_binance_klines(symbol='BTCUSDT', interval='5m', market='futures', limit=1500, end_time=None):
    """바이낸스 캔들 데이터 수집"""
    if market == 'futures':
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def collect_data(symbol='BTCUSDT', interval='5m', market='futures', days=365):
    """과거 데이터 수집"""
    print(f"📥 Collecting {market} data for {symbol}...")

    all_data = []
    end_time = datetime.now()

    # 1500개씩 역순으로 수집
    iterations = (days * 24 * 60) // (5 * 1500) + 1  # 5분 간격

    for i in range(iterations):
        print(f"  Fetching batch {i+1}/{iterations}...")

        klines = fetch_binance_klines(symbol, interval, market, 1500, end_time)

        if not klines:
            break

        # 데이터 처리
        for k in klines:
            candle = {
                'open_time': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'buy_volume': float(k[9]),
            }
            candle['sell_volume'] = candle['volume'] - candle['buy_volume']
            candle['volume_delta'] = candle['buy_volume'] - candle['sell_volume']
            all_data.append(candle)

        # 다음 배치를 위해 end_time 업데이트
        end_time = pd.to_datetime(klines[0][0], unit='ms')

        # Rate limiting
        time.sleep(0.5)

    # DataFrame 생성 및 정렬
    df = pd.DataFrame(all_data)
    df = df.sort_values('open_time').reset_index(drop=True)

    # CVD 계산
    df['cvd'] = df['volume_delta'].cumsum()

    print(f"  ✅ Collected {len(df)} candles")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")

    return df

# 실행
if __name__ == '__main__':
    # Futures 데이터 수집
    df_futures = collect_data('BTCUSDT', '5m', 'futures', days=365)
    df_futures.to_csv('../binance-data-collector/BTCUSDT_perp_5m.csv', index=False)
    print("💾 Futures data saved to BTCUSDT_perp_5m.csv")

    # Spot 데이터 수집
    df_spot = collect_data('BTCUSDT', '5m', 'spot', days=365)
    df_spot.to_csv('../binance-data-collector/BTCUSDT_spot_5m.csv', index=False)
    print("💾 Spot data saved to BTCUSDT_spot_5m.csv")

    print("\n✅ Data collection complete!")
```

실행:
```bash
cd tda-extreme-events
python collect_historical_data.py
```

**소요 시간**: 약 10-20분 (365일 데이터)

---

### 플로우 2: 모델 학습 (최초 1회, 이후 주 1회)

과거 데이터로 예측 모델을 학습합니다.

```bash
cd tda-extreme-events
python realtime_predictor.py
```

**출력**:
```
================================================================================
BITCOIN PRICE PREDICTION - TDA + MACHINE LEARNING
================================================================================
📊 Loading data...
  Loaded 105121 rows

🔬 Extracting TDA features...
  Computing persistence diagrams...
  Extracted 105062 TDA feature vectors

📈 Creating ML features...
  Created 39 features for 501 samples

🤖 Training model...
  Training XGBoost model...

📊 Model Performance:
  Train MAE: $6.97
  Test MAE:  $214.20
  Test R²:   0.5409

💾 Model saved to: tda_prediction_model.pkl
✅ TRAINING COMPLETE!
```

**소요 시간**: 약 10-15분

**생성 파일**:
- `tda_prediction_model.pkl` - 학습된 모델
- `training_metrics.json` - 성능 지표

---

### 플로우 3: 실시간 예측 실행

실시간으로 바이낸스 데이터를 받아 예측합니다.

```bash
python live_data_fetcher.py --interval 300
```

**파라미터**:
- `--interval`: 업데이트 간격 (초) - 기본값: 300 (5분)
- `--model`: 모델 파일 경로 - 기본값: tda_prediction_model.pkl

**출력**:
```
================================================================================
🚀 REAL-TIME BITCOIN PREDICTION SYSTEM
================================================================================

🔄 Initializing data buffers...
📥 Fetching last 200 futures candles...
  ✅ Fetched 200 candles
📥 Fetching last 200 spot candles...
  ✅ Fetched 200 candles

🤖 Loading model from tda_prediction_model.pkl...
  Window size: 60
  Forecast horizon: 12 steps (60 minutes)

⏰ Update interval: 300 seconds (5.0 minutes)
📊 Starting monitoring loop...

================================================================================
🔄 Update #1 - 2025-12-27 14:30:00
================================================================================
📥 Fetching latest candles...
💰 Current Prices:
  Futures: $95,432.50
  Spot:    $95,428.30
  Spread:  $4.20

🤖 Making prediction...

🎯 PREDICTION:
  Current Price:    $95,432.50
  Predicted Price:  $95,678.20
  Expected Change:  $+245.70 (+0.26%)
  Prediction Time:  15:30:00 (60 min ahead)

📊 TDA Status:
  L¹ Norm:  0.4231
  L² Norm:  0.0687
  WD:       0.0053

💾 Status saved to: live_prediction_status.json

⏱️  Next update in 300 seconds...
```

**생성 파일**:
- `live_prediction_status.json` - 최신 예측 결과

**백그라운드 실행**:
```bash
# Linux/Mac
nohup python live_data_fetcher.py --interval 300 > prediction.log 2>&1 &

# 로그 확인
tail -f prediction.log

# 프로세스 종료
pkill -f live_data_fetcher.py
```

---

### 플로우 4: 대시보드 확인

Jupyter 노트북에서 예측 결과를 시각화합니다.

```bash
# Jupyter 실행
jupyter notebook

# 브라우저에서 live_dashboard.ipynb 열기
```

**대시보드에서**:
1. 모든 셀 실행 (Cell → Run All)
2. 현재 예측 결과 확인
3. 마지막 셀 실행 → 60초마다 자동 갱신

**대시보드 미리보기**:
```
🚀 REAL-TIME BITCOIN PREDICTION
================================

📈 UP
$95,432.50 → $95,678.20
+0.26% (+$245.70 USDT)

Prediction for 60 minutes ahead
Last update: 2025-12-27 14:30:00

📊 TDA Market Analysis
L¹ Norm:  0.4231
L² Norm:  0.0687
Wasserstein: 0.0053
```

---

## 🔄 일일 사용 플로우

### 아침에 시스템 시작

```bash
# 1. 가상환경 활성화 (필요시)
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 2. 실시간 예측 시작 (백그라운드)
nohup python live_data_fetcher.py --interval 300 > prediction.log 2>&1 &

# 3. Jupyter 대시보드 실행
jupyter notebook
# 브라우저에서 live_dashboard.ipynb 열고 자동 갱신 셀 실행
```

### 저녁에 시스템 종료

```bash
# 1. 실시간 예측 프로세스 종료
pkill -f live_data_fetcher.py

# 2. Jupyter 종료 (Ctrl+C)

# 3. 가상환경 비활성화
deactivate
```

---

## 🔧 고급 설정

### 업데이트 주기 변경

```bash
# 1분마다 업데이트
python live_data_fetcher.py --interval 60

# 15분마다 업데이트
python live_data_fetcher.py --interval 900
```

### 모델 재학습 (주 1회 권장)

```bash
# 1. 최신 데이터 수집
python collect_historical_data.py

# 2. 모델 재학습
python realtime_predictor.py

# 3. 실시간 예측 재시작
pkill -f live_data_fetcher.py
python live_data_fetcher.py --interval 300 &
```

### 예측 구간 변경

`realtime_predictor.py` 수정:
```python
# 30분 후 예측 (기본: 60분)
predictor = TDAPricePredictor(
    window_size=60,
    forecast_horizon=6  # 6 * 5min = 30분
)
```

재학습:
```bash
python realtime_predictor.py
```

---

## 문제 해결

### 1. GUDHI 설치 실패

**문제**: `pip install gudhi` 실패

**해결**:
```bash
# conda 사용 (권장)
conda install -c conda-forge gudhi

# 또는 빌드 도구 설치 후
# Windows: Visual Studio Build Tools 설치
# Linux: sudo apt-get install build-essential cmake
# macOS: xcode-select --install

pip install gudhi
```

### 2. 바이낸스 API 오류 (HTTP 451)

**문제**: `requests.exceptions.HTTPError: 451`

**원인**: 지역 제한

**해결**:
```bash
# VPN 사용
# 또는 프록시 설정

# requests에 프록시 추가 (live_data_fetcher.py 수정)
proxies = {
    'http': 'http://proxy-server:port',
    'https': 'http://proxy-server:port'
}
response = requests.get(url, params=params, proxies=proxies)
```

### 3. 메모리 부족

**문제**: TDA 계산 중 메모리 부족

**해결**:
```python
# realtime_predictor.py에서 window_size 감소
predictor = TDAPricePredictor(
    window_size=30,  # 60 → 30 (메모리 절약)
    forecast_horizon=12
)
```

### 4. 예측 정확도 낮음

**해결**:
1. 더 많은 데이터로 재학습 (365일 → 730일)
2. 예측 구간 단축 (60분 → 30분)
3. 모델 재학습 빈도 증가 (주 1회 → 3일 1회)

### 5. live_prediction_status.json 생성 안됨

**원인**: 예측 실패 또는 권한 문제

**확인**:
```bash
# 로그 확인
tail -f prediction.log

# 권한 확인
chmod 644 live_prediction_status.json

# 수동 테스트
python test_prediction_offline.py
```

---

## 📊 성능 모니터링

### 예측 정확도 추적

`monitor_accuracy.py` 생성:
```python
"""
예측 정확도 모니터링
"""
import json
import pandas as pd
from datetime import datetime, timedelta

# 과거 예측 기록 로드
predictions = []
with open('prediction_history.jsonl', 'r') as f:
    for line in f:
        predictions.append(json.loads(line))

df = pd.DataFrame(predictions)

# 정확도 계산
df['error'] = abs(df['predicted_price'] - df['actual_price'])
df['error_pct'] = df['error'] / df['actual_price'] * 100

print(f"Average Error: ${df['error'].mean():.2f}")
print(f"Average Error %: {df['error_pct'].mean():.2f}%")
print(f"Median Error: ${df['error'].median():.2f}")
```

### 실시간 로그 모니터링

```bash
# 실시간 로그 확인
tail -f prediction.log

# 특정 패턴 검색
grep "PREDICTION:" prediction.log

# 최근 10개 예측 확인
grep "Predicted Price:" prediction.log | tail -10
```

---

## 🔒 보안 및 주의사항

### 1. API 키 사용 시

바이낸스 API 키가 필요한 경우:

```bash
# .env 파일 생성
cat > .env << EOF
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
EOF

# .gitignore에 추가
echo ".env" >> .gitignore
```

### 2. 백업

```bash
# 모델 및 설정 백업
mkdir -p backups
cp tda_prediction_model.pkl backups/model_$(date +%Y%m%d).pkl
cp training_metrics.json backups/metrics_$(date +%Y%m%d).json
```

### 3. 자동 재학습 스크립트

`auto_retrain.sh`:
```bash
#!/bin/bash
# 주간 자동 재학습 스크립트

echo "Starting weekly retrain..."

# 1. 데이터 수집
python collect_historical_data.py

# 2. 모델 백업
cp tda_prediction_model.pkl backups/model_$(date +%Y%m%d).pkl

# 3. 재학습
python realtime_predictor.py

# 4. 예측 재시작
pkill -f live_data_fetcher.py
sleep 5
nohup python live_data_fetcher.py --interval 300 > prediction.log 2>&1 &

echo "Retrain complete!"
```

실행 권한 부여 및 크론 등록:
```bash
chmod +x auto_retrain.sh

# 매주 일요일 새벽 2시 실행
crontab -e
# 추가: 0 2 * * 0 /path/to/auto_retrain.sh
```

---

## ✅ 체크리스트

### 초기 설정
- [ ] Python 3.8+ 설치 확인
- [ ] Git repository clone/pull
- [ ] 가상환경 생성 및 활성화
- [ ] 필수 라이브러리 설치 (`pip install -r requirements.txt`)
- [ ] GUDHI 설치 확인

### 데이터 수집
- [ ] 바이낸스 API 접근 확인
- [ ] 과거 데이터 수집 (`collect_historical_data.py`)
- [ ] 데이터 파일 확인 (BTCUSDT_perp_5m.csv, BTCUSDT_spot_5m.csv)

### 모델 학습
- [ ] 모델 학습 실행 (`python realtime_predictor.py`)
- [ ] 모델 파일 생성 확인 (`tda_prediction_model.pkl`)
- [ ] 성능 지표 확인 (`training_metrics.json`)

### 실시간 예측
- [ ] 실시간 예측 스크립트 실행 (`live_data_fetcher.py`)
- [ ] 예측 결과 파일 생성 확인 (`live_prediction_status.json`)
- [ ] 대시보드 실행 확인 (`live_dashboard.ipynb`)

### 일상 운영
- [ ] 자동 재학습 스크립트 설정 (선택)
- [ ] 백업 시스템 구축 (선택)
- [ ] 성능 모니터링 설정 (선택)

---

## 📞 추가 도움말

### 공식 문서
- `REALTIME_SYSTEM_GUIDE.md` - 상세 시스템 가이드
- `FINAL_SUMMARY.md` - 시스템 요약 및 성능 보고서

### 문의
- GitHub Issues
- 또는 프로젝트 문서 참조

---

**Last Updated**: 2025-12-27
**Version**: 1.0.0
**Status**: ✅ Ready for local deployment
