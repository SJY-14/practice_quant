# ML Pipeline: N-BEATS with TDA Features

논문 기반 암호화폐 가격 예측 모델

**Reference Paper**: [Enhancing financial time series forecasting through topological data analysis](https://link.springer.com/article/10.1007/s00521-024-10787-x)

## 방법론

### TDA (Topological Data Analysis) 특징

시계열 데이터에서 위상적 특징을 추출하여 기존 모델에 추가 정보로 활용:

| 특징 | 설명 | 의미 |
|-----|------|------|
| **Entropy** | Persistence diagram의 엔트로피 | 시장 복잡도/변동성 |
| **Amplitude** | Persistence diagram의 spread | 패턴 안정성 |
| **Num Points** | 유의미한 위상적 특징 수 | 반복 사이클/추세 |

### 모델: N-BEATS

[N-BEATS](https://arxiv.org/abs/1905.10437) (Neural Basis Expansion Analysis)
- Residual 연결 기반 딥러닝 모델
- 시계열 예측에 특화
- TDA 특징을 추가 입력으로 통합

## 설치

```bash
cd ml
pip install -r requirements.txt
```

**주의**: `giotto-tda`와 `ripser`는 C++ 컴파일러가 필요할 수 있음

## 사용법

### 1. 기본 학습

```bash
python ml/train.py --data data/BTCUSDT_perp_5m.csv
```

### 2. 옵션

```bash
python ml/train.py \
    --data data/BTCUSDT_perp_5m.csv \
    --lookback 96 \      # 입력 길이 (96 = 8시간)
    --horizon 12 \       # 예측 길이 (12 = 1시간)
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001

# TDA 없이 학습 (비교용)
python ml/train.py --data data/BTCUSDT_perp_5m.csv --no-tda

# 외생 변수 없이 학습
python ml/train.py --data data/BTCUSDT_perp_5m.csv --no-exog
```

## 입력 데이터

### 필수 컬럼
- `close`: 종가

### 선택 컬럼 (외생 변수)
- `volume`: 거래량
- `buy_volume`: 매수 거래량
- `sell_volume`: 매도 거래량
- `volume_delta`: 거래량 델타
- `cvd`: Cumulative Volume Delta

## 학습 파라미터 권장값

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `lookback` | 96 | 8시간 (5분봉 × 96) |
| `horizon` | 12 | 1시간 예측 |
| `tda_window` | 50 | TDA 윈도우 크기 |
| `batch_size` | 64 | 배치 크기 |
| `epochs` | 100 | 학습 에포크 |
| `lr` | 0.001 | 학습률 |

## 출력

```
checkpoints/
├── best_model.pt     # 최적 모델 가중치
└── results.png       # 학습 결과 시각화
```

## 평가 지표

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Direction Accuracy**: 방향 예측 정확도

## 프로젝트 구조

```
ml/
├── features/
│   └── tda_features.py    # TDA 특징 추출
├── models/
│   └── nbeats.py          # N-BEATS 모델
├── train.py               # 학습 파이프라인
├── requirements.txt       # 의존성
└── README.md
```

## GPU 사용

CUDA가 설치되어 있으면 자동으로 GPU 사용:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 실험 아이디어

1. **TDA 효과 비교**: `--no-tda` 옵션으로 TDA 없이 학습 후 비교
2. **Lookback 튜닝**: 4시간(48) vs 8시간(96) vs 24시간(288)
3. **Horizon 튜닝**: 30분(6) vs 1시간(12) vs 4시간(48)
4. **다중 심볼**: BTCUSDT, ETHUSDT 비교
5. **앙상블**: 여러 모델 결합
