# Binance Data Collector

Binance에서 5분봉 데이터를 수집하는 Python 스크립트입니다.

## 수집 데이터

| 데이터 | 설명 |
|--------|------|
| OHLCV | 시가, 고가, 저가, 종가, 거래량 |
| Buy/Sell Volume | 매수/매도 거래량 (Taker 기준) |
| CVD | Cumulative Volume Delta |
| Open Interest | 미결제약정 (선물만, 최근 30일) |

## 설치

```bash
git clone https://github.com/YOUR_USERNAME/binance-data-collector.git
cd binance-data-collector
pip install -r requirements.txt
```

## 사용법

### 기본 실행 (BTCUSDT, 1년치)

```bash
python main.py
```

### 옵션

```bash
# 특정 심볼
python main.py --symbol ETHUSDT

# 기간 설정 (일)
python main.py --days 180

# 선물만 또는 현물만
python main.py --market futures
python main.py --market spot

# 모든 옵션 조합
python main.py --symbol BTCUSDT --days 365 --market both
```

## 출력 파일

```
data/
├── BTCUSDT_perp_5m.csv   # 선물 데이터
└── BTCUSDT_spot_5m.csv   # 현물 데이터
```

### CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| open_time | 캔들 시작 시간 (UTC) |
| open, high, low, close | OHLC 가격 |
| volume | 총 거래량 |
| buy_volume | 매수 거래량 (Taker buy) |
| sell_volume | 매도 거래량 |
| volume_delta | 거래량 델타 (buy - sell) |
| cvd | Cumulative Volume Delta |
| open_interest | 미결제약정 (선물만, 최근 30일) |

## CVD 계산 방식

```
buy_volume = taker_buy_base_volume (Binance Klines index[9])
sell_volume = total_volume - buy_volume
volume_delta = buy_volume - sell_volume
cvd = cumsum(volume_delta)
```

## API 제한사항

- **Open Interest**: Binance API에서 최근 30일만 제공
- **Rate Limit**: 분당 2400 weight (자동 조절됨)

## 요구사항

- Python 3.8+
- requests
- pandas

## License

MIT
