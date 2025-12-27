# TDA Extreme Event Detection in Bitcoin Market

This project reproduces the methodology from the paper:

**"Identifying Extreme Events in the Stock Market: A Topological Data Analysis"**
by Anish Rai, Buddha Nath Sharma, Salam Rabindrajit Luwang, Md. Nurujjaman, and Sushovan Majhi
arXiv:2405.16052 (2024)

## Paper Summary

The paper uses **Topological Data Analysis (TDA)** to detect extreme events in stock markets. Key contributions:

- Uses **L¹ and L² norms** of persistence landscapes
- Uses **Wasserstein distance** between consecutive persistence diagrams
- Defines extreme events as those exceeding **μ + 4σ threshold**
- Successfully detected 2008 financial crisis and COVID-19 pandemic crashes

## Implementation

This implementation applies the same methodology to **Bitcoin** (BTCUSDT) 5-minute data.

### Two Approaches

1. **Multivariate Approach**:
   - Uses multiple features (close price, volume, volume_delta, cvd)
   - Simulates the paper's method of analyzing multiple stocks simultaneously

2. **Takens Embedding Approach**:
   - Uses only close price with time-delay embedding
   - Converts 1D time series to multi-dimensional point cloud

### Key Components

- **Vietoris-Rips Complex**: Constructs simplicial complexes from point clouds
- **Persistent Homology**: Tracks topological features across scales
- **Persistence Landscapes**: Statistical representation of persistence diagrams
- **Lp Norms**: Quantifies topological complexity
- **Wasserstein Distance**: Measures similarity between topologies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Output

The program generates:

1. **results_multivariate.png**: Analysis using multiple features
2. **results_takens.png**: Analysis using Takens embedding
3. **comparison.png**: Comparison of both methods
4. **results_multivariate.csv**: Numerical results (multivariate)
5. **results_takens.csv**: Numerical results (Takens)

## Data Source

Bitcoin data from `../- binance-data-collector/BTCUSDT_5m.csv`
- 30,000 rows of 5-minute candle data
- Columns: timestamp, open, high, low, close, volume, buy_volume, sell_volume, volume_delta, cvd

## Methodology

### Step 1: Data Preparation
- Compute log returns or normalize features
- Create multi-dimensional point cloud

### Step 2: Sliding Window
- Window size: 60 (5 hours of 5-min data)
- Slide through entire dataset

### Step 3: TDA Computation
For each window:
1. Construct Vietoris-Rips complex
2. Compute persistence diagram (H₀ and H₁)
3. Convert to persistence landscape
4. Calculate L¹ and L² norms
5. Compute Wasserstein distance with previous window

### Step 4: Extreme Event Detection
- Calculate mean (μ) and standard deviation (σ) for each metric
- Define threshold: μ + 4σ
- Flag events exceeding threshold as "extreme events"

## Parameters

- `window_size`: 60 (adjustable)
- `maxdim`: 1 (compute H₀ and H₁)
- `threshold_sigma`: 4 (for μ + 4σ threshold)
- `embedding_dim`: 3 (for Takens embedding)

## Interpretation

**Extreme events** are market conditions where:
- Topological complexity (L¹/L² norms) significantly increases
- Market structure (Wasserstein distance) changes dramatically
- These often correspond to crashes, flash crashes, or high volatility periods

## Comparison with Paper

| Aspect | Paper | This Implementation |
|--------|-------|---------------------|
| Data | Multiple stock indices | Bitcoin (single asset) |
| Timeframe | Daily data | 5-minute data |
| Window Size | 60 days | 60 x 5-min = 5 hours |
| Approach | Multiple stocks as dimensions | Multiple features or Takens embedding |
| Threshold | μ + 4σ | μ + 4σ (same) |
| Metrics | L¹, L², WD | L¹, L², WD (same) |

## References

1. Rai, A., et al. (2024). "Identifying Extreme Events in the Stock Market: A Topological Data Analysis." arXiv:2405.16052
2. Gidea, M., & Katz, Y. (2018). "Topological data analysis of financial time series: Landscapes of crashes." Physica A, 491, 820-834.
3. Bubenik, P. (2015). "Statistical topological data analysis using persistence landscapes." JMLR, 16, 77-102.

## Author

Reproduced from the original paper for educational purposes.

## License

This implementation is for educational and research purposes.
