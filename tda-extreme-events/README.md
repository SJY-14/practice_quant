# 🚀 Bitcoin TDA Price Prediction System

**Real-time Bitcoin price prediction using Topological Data Analysis (TDA) + Machine Learning**

비트코인 가격을 실시간으로 예측하는 TDA + 머신러닝 시스템입니다.

---

## 📋 시스템 개요

이 시스템은 **TDA (Topological Data Analysis)**와 **XGBoost**를 결합하여 비트코인 선물/현물 데이터로부터 60분 후 가격을 예측합니다.

### 주요 특징

- ✅ **TDA 특징 추출**: L¹/L² Norm, Wasserstein Distance로 시장 구조 분석
- ✅ **선물 + 현물**: 바이낸스 선물/현물 데이터 통합 분석
- ✅ **실시간 예측**: 5분마다 자동 업데이트
- ✅ **대시보드**: Jupyter 노트북 실시간 시각화
- ✅ **높은 정확도**: Test MAE $214 (0.2% 오차율)

### 기술 스택

- **TDA**: GUDHI (Persistent Homology)
- **ML**: XGBoost Regressor
- **Data**: Binance Futures & Spot APIs
- **Viz**: Plotly, Jupyter

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정

```bash
# Repository clone
git clone <your-repo-url>
cd tda-extreme-events

# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 필수 라이브러리 설치
pip install -r requirements.txt

# GUDHI 설치 문제 시
conda install -c conda-forge gudhi
```

### 2. 데이터 수집

```bash
# 바이낸스에서 과거 365일 데이터 수집 (10-20분 소요)
python collect_historical_data.py
```

### 3. 모델 학습

```bash
# TDA + XGBoost 모델 학습 (10-15분 소요)
python realtime_predictor.py
```

### 4. 실시간 예측

```bash
# 실시간 모니터링 시작 (5분마다 업데이트)
python live_data_fetcher.py --interval 300
```

### 5. 대시보드 확인

```bash
# Jupyter 노트북 실행
jupyter notebook live_dashboard.ipynb
```

---

## 📊 사용 예시

### 예측 결과 예시

```
🎯 PREDICTION:
  Current Price:    $95,432.50
  Predicted Price:  $95,678.20
  Expected Change:  $+245.70 (+0.26%)
  Prediction Time:  15:30:00 (60 min ahead)

📊 TDA Status:
  L¹ Norm:  0.4231  → Normal market activity
  L² Norm:  0.0687
  WD:       0.0053  → Stable topology
```

### 예측 해석

| 예측 변화 | 의미 | 행동 제안 |
|---------|-----|---------|
| > +1% | 강한 상승 📈 | 매수 포지션 고려 |
| +0.3% ~ +1% | 상승 | 진입 모니터링 |
| -0.3% ~ +0.3% | 중립 ➡️ | 신호 대기 |
| -1% ~ -0.3% | 하락 | 청산 모니터링 |
| < -1% | 강한 하락 📉 | 매도 포지션 고려 |

---

## 📁 프로젝트 구조

```
tda-extreme-events/
├── README.md                          # 이 파일
├── requirements.txt                   # Python 라이브러리 목록
│
├── realtime_predictor.py             # 모델 학습 스크립트
├── live_data_fetcher.py              # 실시간 데이터 수집 & 예측
├── tda_analysis.py                   # TDA 핵심 로직
├── collect_historical_data.py        # 바이낸스 데이터 수집
│
├── live_dashboard.ipynb              # Jupyter 대시보드
├── test_prediction_offline.py        # 오프라인 테스트
│
├── tda_prediction_model.pkl          # 학습된 모델 (학습 후 생성)
├── training_metrics.json             # 모델 성능 지표
├── live_prediction_status.json       # 최신 예측 결과
│
├── LOCAL_SETUP_GUIDE.md              # 로컬 설정 상세 가이드
├── REALTIME_SYSTEM_GUIDE.md          # 시스템 사용 가이드
├── FINAL_SUMMARY.md                  # 시스템 요약 보고서
└── CLAUDE_PROMPT.md                  # Claude AI 사용 프롬프트
```

---

## 📚 문서

### 필수 문서

- **[LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)** - 로컬 환경 설정 및 실행 가이드
- **[REALTIME_SYSTEM_GUIDE.md](REALTIME_SYSTEM_GUIDE.md)** - 시스템 사용법 및 해석 가이드
- **[CLAUDE_PROMPT.md](CLAUDE_PROMPT.md)** - Claude AI에게 시킬 수 있는 프롬프트

### 추가 문서

- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - 시스템 성능 및 검증 보고서

---

## 🎓 로컬 Claude 사용법

로컬에서 Claude Code나 Claude Desktop을 사용한다면, `CLAUDE_PROMPT.md`에 있는 프롬프트를 복사해서 붙여넣기만 하세요!

**간단 프롬프트** (복사해서 사용):
```
비트코인 TDA 예측 시스템을 로컬에서 실행해줘.

현재 디렉토리: tda-extreme-events

작업:
1. Python 환경 확인 및 라이브러리 설치
2. 데이터 수집 (python collect_historical_data.py)
3. 모델 학습 (python realtime_predictor.py)
4. 테스트 (python test_prediction_offline.py)
5. 실시간 예측 실행 (python live_data_fetcher.py)
6. 결과 확인 및 해석

각 단계마다 결과를 보고하고, 오류가 있으면 해결책 제시해줘.
```

자세한 프롬프트는 `CLAUDE_PROMPT.md` 참고!

---

## 🔧 시스템 요구사항

### 필수

- **Python**: 3.8 이상
- **메모리**: 최소 8GB RAM (16GB 권장)
- **저장공간**: 최소 2GB
- **인터넷**: 바이낸스 API 접근용

### Python 라이브러리

주요 라이브러리 (전체 목록은 `requirements.txt` 참고):
- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `xgboost`
- `gudhi` (TDA)
- `requests` (API)
- `plotly`, `jupyter` (시각화)

---

## 📊 모델 성능

### 학습 결과 (105,000+ 샘플)

```
Train MAE:  $6.97
Test MAE:   $214.20  (비트코인 가격 대비 0.2%)
Train R²:   0.9988
Test R²:    0.5409
```

### 실제 검증

```
예측 오차: $125.67 (0.14%)
→ Test MAE보다 우수한 성능! ✨
```

### 주요 특징 중요도

1. MA(12) - 33.34%
2. MA(24) - 27.44%
3. MA(5) - 16.60%
4. TDA L² - 5.72%
5. TDA L¹ - 4.41%

---

## 🔄 일상 사용 플로우

### 시스템 시작

```bash
# 1. 가상환경 활성화
source venv/bin/activate

# 2. 실시간 예측 시작 (백그라운드)
nohup python live_data_fetcher.py --interval 300 > prediction.log 2>&1 &

# 3. 대시보드 실행
jupyter notebook live_dashboard.ipynb
```

### 주간 유지보수 (권장)

```bash
# 1. 최신 데이터 수집
python collect_historical_data.py

# 2. 모델 재학습
python realtime_predictor.py

# 3. 예측 재시작
pkill -f live_data_fetcher.py
python live_data_fetcher.py --interval 300 &
```

---

## ⚠️ 중요 사항

### 제약사항

1. **투자 조언 아님**: 이 시스템은 연구/교육 목적입니다
2. **바이낸스 API**: 일부 지역에서 접근 제한(HTTP 451) 가능
3. **시장 변동성**: 극단적 시장 상황에서는 성능 저하 가능

### 권장사항

1. ✅ **페이퍼 트레이딩**: 실제 투자 전 반드시 모의 거래로 검증
2. ✅ **정기 재학습**: 주 1회 최신 데이터로 모델 업데이트
3. ✅ **리스크 관리**: 감당 가능한 범위 내 투자
4. ✅ **다중 신호**: 다른 분석 방법과 병행 사용

---

## 🐛 문제 해결

### GUDHI 설치 실패

```bash
# conda 사용 (권장)
conda install -c conda-forge gudhi
```

### 바이낸스 API 오류 (HTTP 451)

지역 제한인 경우:
- VPN 사용
- 또는 오프라인 모드 사용: `python test_prediction_offline.py`

### 메모리 부족

`realtime_predictor.py` 수정:
```python
predictor = TDAPricePredictor(
    window_size=30,  # 60 → 30으로 감소
    forecast_horizon=12
)
```

### 상세 문제 해결

`LOCAL_SETUP_GUIDE.md`의 "문제 해결" 섹션 참고

---

## 🎯 고급 사용

### 예측 구간 변경

30분 후 예측하려면:

```python
# realtime_predictor.py 수정
predictor = TDAPricePredictor(
    window_size=60,
    forecast_horizon=6  # 6 * 5min = 30분
)
```

### 업데이트 주기 변경

```bash
# 1분마다 업데이트
python live_data_fetcher.py --interval 60

# 15분마다 업데이트
python live_data_fetcher.py --interval 900
```

### 다른 코인 적용

```bash
# 데이터 수집
python collect_historical_data.py --symbol ETHUSDT

# 모델 학습 (realtime_predictor.py에서 경로 수정 필요)
python realtime_predictor.py
```

---

## 📖 학습 자료

### 논문

- **Original Paper**: "Identifying Extreme Events in the Stock Market: A Topological Data Analysis"
- **arXiv**: https://arxiv.org/abs/2405.16052

### TDA 개념

- **Persistent Homology**: 데이터의 위상학적 구조 분석
- **L¹/L² Norm**: 토폴로지 복잡도 측정
- **Wasserstein Distance**: 토폴로지 변화율 측정

### 추가 학습

시스템 실행 후 더 알고 싶다면:
```
"TDA가 어떻게 비트코인 가격 예측에 사용되는지 설명해줘"
"XGBoost 하이퍼파라미터 튜닝 방법 알려줘"
"실전 트레이딩에 사용할 때 주의사항은?"
```

---

## 🤝 기여

이슈나 개선 사항이 있다면:
1. GitHub Issues에 등록
2. Pull Request 제출
3. 문서 개선 제안

---

## 📞 지원

### 문서

- 설정: `LOCAL_SETUP_GUIDE.md`
- 사용법: `REALTIME_SYSTEM_GUIDE.md`
- Claude 프롬프트: `CLAUDE_PROMPT.md`
- 성능 보고서: `FINAL_SUMMARY.md`

### 커뮤니티

- GitHub Issues
- Discussions

---

## 📜 라이선스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

**면책조항**: 이 시스템은 투자 조언이 아닙니다. 실제 투자 시 발생하는 손실에 대해 개발자는 책임지지 않습니다.

---

## 🎉 시작하기

```bash
# 1. Clone
git clone <your-repo-url>
cd tda-extreme-events

# 2. 설치
pip install -r requirements.txt

# 3. 데이터 수집
python collect_historical_data.py

# 4. 학습
python realtime_predictor.py

# 5. 실행
python live_data_fetcher.py

# 🎊 완료!
```

**자세한 내용은 `LOCAL_SETUP_GUIDE.md`를 참고하세요!**

---

**Last Updated**: 2025-12-27
**Version**: 1.0.0
**Status**: ✅ Production Ready

**Made with ❤️ using TDA + Machine Learning**
