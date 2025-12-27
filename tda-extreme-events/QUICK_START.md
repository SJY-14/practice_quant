# ⚡ 빠른 시작 가이드 (로컬 Claude용)

## 📥 Git에서 Pull하고 바로 실행하기

### 1️⃣ Repository Clone/Pull

```bash
# 처음이라면 clone
git clone https://github.com/SJY-14/practice_quant.git
cd practice_quant/tda-extreme-events

# 이미 있다면 pull
cd practice_quant
git pull origin main
cd tda-extreme-events
```

---

## 🤖 로컬 Claude에게 복사해서 붙여넣을 프롬프트

### ✨ 초간단 프롬프트 (권장)

Claude Code나 Claude Desktop에 이 프롬프트를 **그대로 복사해서 붙여넣기**하세요:

```
비트코인 TDA 가격 예측 시스템을 실행해줘.

현재 위치: tda-extreme-events 폴더

작업 순서:
1. Python 환경 확인 (Python 3.8+ 필요)
2. 필수 라이브러리 설치: pip install -r requirements.txt
   (GUDHI 설치 실패하면 conda install -c conda-forge gudhi 사용)
3. 데이터 수집: python collect_historical_data.py (10-20분 소요, 바이낸스 API 사용)
4. 모델 학습: python realtime_predictor.py (10-15분 소요)
5. 테스트: python test_prediction_offline.py
6. 결과 확인 및 해석

각 단계마다:
- 진행 상황 보고
- 오류 발생 시 해결책 제시
- 성공 여부 확인

최종 목표: 비트코인 60분 후 가격 예측 시스템 실행
```

---

## 📋 실행 플로우 요약

```
git pull
   ↓
pip install -r requirements.txt
   ↓
python collect_historical_data.py  (365일 데이터 수집)
   ↓
python realtime_predictor.py       (TDA + XGBoost 학습)
   ↓
python live_data_fetcher.py        (실시간 예측 실행)
   ↓
jupyter notebook live_dashboard.ipynb  (대시보드 확인)
```

---

## 🎯 핵심 파일 가이드

### 실행 파일
- **`collect_historical_data.py`** - 바이낸스 데이터 수집 (최초 1회)
- **`realtime_predictor.py`** - 모델 학습 (최초 1회, 이후 주 1회)
- **`live_data_fetcher.py`** - 실시간 예측 실행
- **`test_prediction_offline.py`** - 오프라인 테스트 (바이낸스 API 막힐 때)

### 문서
- **`README.md`** - 시스템 개요
- **`LOCAL_SETUP_GUIDE.md`** - 상세 설치 가이드
- **`CLAUDE_PROMPT.md`** - Claude AI 프롬프트 모음
- **`REALTIME_SYSTEM_GUIDE.md`** - 시스템 사용 가이드

---

## ⚡ 최소 명령어만으로 실행

바로 시작하고 싶다면:

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 데이터 수집 (20분)
python collect_historical_data.py

# 3. 모델 학습 (15분)
python realtime_predictor.py

# 4. 테스트 실행
python test_prediction_offline.py

# 완료! 예측 결과 확인
cat live_prediction_status.json
```

---

## 🔧 문제 해결

### GUDHI 설치 실패
```bash
conda install -c conda-forge gudhi
```

### 바이낸스 API 막힘 (HTTP 451)
```bash
# 오프라인 모드 사용
python test_prediction_offline.py
```

### 더 자세한 문서 필요
- `LOCAL_SETUP_GUIDE.md` 참고
- `CLAUDE_PROMPT.md`의 상세 프롬프트 사용

---

## 💡 로컬 Claude에게 물어볼 질문들

시스템 실행 후 Claude에게 물어보세요:

```
"예측 결과를 해석해줘"
"TDA L1 Norm이 0.8이면 무슨 의미야?"
"모델 성능을 개선하려면?"
"실시간 모니터링을 백그라운드로 실행하는 방법"
"주간 자동 재학습 스크립트 만들어줘"
```

---

## ✅ 성공 확인

다음 파일들이 생성되면 성공:

- [x] `tda_prediction_model.pkl` - 학습된 모델
- [x] `training_metrics.json` - 성능 지표
- [x] `live_prediction_status.json` - 예측 결과
- [x] `../binance-data-collector/BTCUSDT_perp_5m.csv` - 선물 데이터
- [x] `../binance-data-collector/BTCUSDT_spot_5m.csv` - 현물 데이터

---

**Last Updated**: 2025-12-27
**Status**: ✅ Ready to use

**Git에서 pull하고 위 프롬프트를 Claude에게 복사/붙여넣기만 하면 끝!**
