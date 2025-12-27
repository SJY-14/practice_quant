# 🤖 로컬 Claude에게 줄 프롬프트

## 📋 사용 방법

이 프롬프트를 로컬 Claude Code 또는 Claude Desktop에 **그대로 복사해서 붙여넣기**하면 자동으로 시스템을 설정하고 실행합니다.

---

## 🎯 프롬프트 (복사해서 사용)

```
안녕! 비트코인 가격 예측 시스템을 로컬 환경에서 설정하고 실행해줘.

### 시스템 개요
- **목적**: TDA + XGBoost를 사용한 비트코인 실시간 가격 예측
- **예측 구간**: 60분 후 가격
- **데이터**: 바이낸스 선물 + 현물 5분봉
- **모델 성능**: Test MAE $214 (0.2% 오차율)

### 작업 순서

**1단계: 환경 확인 및 설정**

먼저 현재 디렉토리를 확인하고 tda-extreme-events 폴더로 이동해줘:
```bash
pwd
ls -la
cd tda-extreme-events  # 또는 적절한 경로
```

필요한 Python 라이브러리가 설치되어 있는지 확인해줘:
- numpy, pandas, scikit-learn, xgboost
- joblib, requests
- gudhi (TDA 라이브러리)
- plotly, jupyter (대시보드용)

없으면 설치:
```bash
pip install numpy pandas scikit-learn xgboost joblib requests gudhi plotly jupyter ipywidgets
```

GUDHI 설치 실패하면:
```bash
conda install -c conda-forge gudhi
```

**2단계: 데이터 수집 스크립트 생성**

다음 스크립트를 `collect_historical_data.py`로 저장해줘:

```python
"""바이낸스 과거 데이터 수집"""
import requests
import pandas as pd
from datetime import datetime
import time
import os

def fetch_klines(symbol='BTCUSDT', interval='5m', market='futures', limit=1500, end_time=None):
    if market == 'futures':
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        url = "https://api.binance.com/api/v3/klines"

    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def collect_data(symbol='BTCUSDT', interval='5m', market='futures', days=365):
    print(f"📥 Collecting {market} data...")
    all_data = []
    end_time = datetime.now()
    iterations = (days * 24 * 60) // (5 * 1500) + 1

    for i in range(iterations):
        print(f"  Batch {i+1}/{iterations}...")
        klines = fetch_klines(symbol, interval, market, 1500, end_time)
        if not klines:
            break

        for k in klines:
            candle = {
                'open_time': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]),
                'volume': float(k[5]), 'buy_volume': float(k[9])
            }
            candle['sell_volume'] = candle['volume'] - candle['buy_volume']
            candle['volume_delta'] = candle['buy_volume'] - candle['sell_volume']
            all_data.append(candle)

        end_time = pd.to_datetime(klines[0][0], unit='ms')
        time.sleep(0.5)

    df = pd.DataFrame(all_data).sort_values('open_time').reset_index(drop=True)
    df['cvd'] = df['volume_delta'].cumsum()
    print(f"  ✅ {len(df)} candles collected")
    return df

if __name__ == '__main__':
    os.makedirs('../binance-data-collector', exist_ok=True)

    df_futures = collect_data('BTCUSDT', '5m', 'futures', days=365)
    df_futures.to_csv('../binance-data-collector/BTCUSDT_perp_5m.csv', index=False)
    print("💾 Futures saved")

    df_spot = collect_data('BTCUSDT', '5m', 'spot', days=365)
    df_spot.to_csv('../binance-data-collector/BTCUSDT_spot_5m.csv', index=False)
    print("💾 Spot saved")
    print("✅ Data collection complete!")
```

**3단계: 데이터 수집 실행**

```bash
python collect_historical_data.py
```

이 작업은 10-20분 정도 걸려. 완료될 때까지 기다려줘.

만약 바이낸스 API가 지역 제한(HTTP 451)으로 차단되면 알려줘. 대안을 찾아야 해.

**4단계: 모델 학습**

데이터 수집이 완료되면 모델을 학습해줘:

```bash
python realtime_predictor.py
```

이 작업은 10-15분 정도 걸려. 완료되면:
- `tda_prediction_model.pkl` 파일 생성 확인
- `training_metrics.json` 파일 열어서 성능 지표 보여줘

**5단계: 실시간 예측 테스트**

먼저 오프라인 모드로 테스트해줘:

```bash
python test_prediction_offline.py
```

결과를 보여주고, 예측이 정상적으로 작동하는지 확인해줘.

**6단계: 실시간 모니터링 시작**

바이낸스 API 접근이 가능하면 실시간 모니터링을 시작해줘:

```bash
python live_data_fetcher.py --interval 300
```

백그라운드로 실행하려면:
```bash
nohup python live_data_fetcher.py --interval 300 > prediction.log 2>&1 &
```

**7단계: 대시보드 실행**

Jupyter 대시보드를 열어줘:

```bash
jupyter notebook live_dashboard.ipynb
```

브라우저에서 열리면:
1. 모든 셀 실행 (Cell → Run All)
2. 현재 예측 결과 확인
3. 마지막 셀 실행하면 60초마다 자동 갱신

### 오류 처리

각 단계에서 오류가 발생하면:
1. 오류 메시지를 자세히 보여줘
2. 가능한 해결책을 제시해줘
3. 필요하면 대안을 찾아줘

특히 다음 오류들은 흔해:
- GUDHI 설치 실패 → conda 사용
- 바이낸스 API 451 오류 → VPN 필요 또는 오프라인 모드 사용
- 메모리 부족 → window_size 줄이기

### 최종 확인

모든 단계가 완료되면:
1. `live_prediction_status.json` 파일 내용을 보여줘
2. 현재 예측 결과를 해석해줘 (상승/하락, TDA 지표 의미)
3. 시스템이 정상 작동 중인지 확인해줘

### 추가 요청

시스템이 정상 작동하면:
1. 예측 결과를 주기적으로 모니터링하는 방법 알려줘
2. 모델을 재학습하는 방법 알려줘
3. 백업 및 유지보수 팁 알려줘

작업을 시작해줘!
```

---

## 🎯 간단 버전 프롬프트 (빠른 시작용)

시간이 없고 빠르게 시작하고 싶다면 이 간단 버전을 사용하세요:

```
비트코인 TDA 예측 시스템을 로컬에서 실행해줘.

현재 디렉토리: tda-extreme-events

작업:
1. Python 환경 확인 (numpy, pandas, sklearn, xgboost, gudhi, joblib, requests 필요)
2. 없으면 설치: pip install numpy pandas scikit-learn xgboost joblib requests gudhi
3. binance-data-collector 폴더 확인, 없으면 데이터 수집 스크립트 만들고 실행
4. python realtime_predictor.py 실행 (모델 학습)
5. python test_prediction_offline.py 실행 (테스트)
6. python live_data_fetcher.py --interval 300 실행 (실시간 예측)
7. 결과 확인 및 해석

바이낸스 API 오류(451)가 나면 오프라인 모드로 전환해줘.
각 단계마다 결과를 보고하고, 오류가 있으면 해결책 제시해줘.
```

---

## 🔧 고급 사용자용 프롬프트

이미 환경이 설정되어 있고 특정 작업만 필요한 경우:

### 모델 재학습만 하기
```
tda-extreme-events 폴더에서 비트코인 예측 모델을 재학습해줘.

1. 기존 모델 백업: cp tda_prediction_model.pkl backups/model_$(date +%Y%m%d).pkl
2. 새 데이터 수집: python collect_historical_data.py (365일)
3. 모델 재학습: python realtime_predictor.py
4. 성능 비교: 이전 vs 새 모델 metrics 비교
5. 실시간 예측 재시작

완료되면 성능 개선 여부 알려줘.
```

### 예측만 실행하기
```
tda-extreme-events에서 비트코인 가격 예측만 실행해줘.

이미 모델(tda_prediction_model.pkl)이 있어.

1. python test_prediction_offline.py 실행
2. 결과 해석해줘:
   - 예측 가격과 현재 가격 비교
   - 상승/하락 방향
   - TDA L1 Norm 의미 (높으면 변동성 높음)
   - 신뢰도 평가

간단명료하게 보고해줘.
```

### 대시보드만 띄우기
```
비트코인 예측 대시보드를 실행해줘.

1. live_prediction_status.json 파일 확인
2. jupyter notebook live_dashboard.ipynb 실행
3. 브라우저가 열리면 모든 셀 실행하는 방법 알려줘
4. 자동 갱신 셀 위치 알려줘

대시보드 사용법 간단히 설명해줘.
```

---

## 📊 프롬프트 사용 팁

### 1. 단계별 확인
Claude가 각 단계를 완료할 때마다 결과를 확인하세요. 문제가 있으면 즉시 알려줍니다.

### 2. 오류 처리
오류가 발생하면 Claude가 자동으로 해결책을 제시합니다. 필요하면 추가 질문하세요.

### 3. 커스터마이징
프롬프트를 수정해서 사용하세요:
- `days=365` → `days=180` (더 빠른 데이터 수집)
- `--interval 300` → `--interval 60` (1분마다 업데이트)
- `forecast_horizon=12` → `forecast_horizon=6` (30분 예측)

### 4. 후속 질문 예시
```
"예측 정확도를 개선하려면?"
"TDA L1 Norm이 0.8 이상이면 어떻게 해석해야 해?"
"모델을 주간 자동 재학습하도록 cron 설정해줘"
"예측 결과를 슬랙으로 보내는 스크립트 만들어줘"
```

---

## ✅ 체크리스트

프롬프트 사용 전 확인:

- [ ] Python 3.8+ 설치됨
- [ ] Git repository clone/pull 완료
- [ ] tda-extreme-events 폴더로 이동
- [ ] 인터넷 연결 확인 (바이낸스 API 접근용)
- [ ] 충분한 디스크 공간 (최소 2GB)
- [ ] 충분한 메모리 (최소 8GB RAM)

프롬프트 사용 후 확인:

- [ ] 데이터 수집 완료 (BTCUSDT_perp_5m.csv, BTCUSDT_spot_5m.csv)
- [ ] 모델 학습 완료 (tda_prediction_model.pkl)
- [ ] 테스트 성공 (test_prediction_offline.py)
- [ ] 실시간 예측 실행 중 (live_data_fetcher.py)
- [ ] 예측 결과 파일 생성 (live_prediction_status.json)

---

## 🎓 추가 학습 자료

프롬프트 실행 후 더 알고 싶다면:

```
"TDA가 어떻게 비트코인 가격 예측에 사용되는지 설명해줘"
"L1 Norm, L2 Norm, Wasserstein Distance의 차이점은?"
"XGBoost의 주요 하이퍼파라미터 튜닝 방법 알려줘"
"이 시스템을 알트코인에도 적용할 수 있어?"
"실전 트레이딩에 사용할 때 주의사항은?"
```

---

**Last Updated**: 2025-12-27
**Version**: 1.0.0
**사용 난이도**: ⭐⭐☆☆☆ (중급)

**참고**: 프롬프트를 그대로 복사해서 사용하거나, 필요에 맞게 수정해서 사용하세요!
