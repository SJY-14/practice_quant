import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Test basic prediction
print("Loading data...")
df = pd.read_csv('BTCUSDT_5m.csv')
prices = df['close'].values[-500:]  # Last 500 points

# Simple plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(prices, linewidth=2)
ax.set_title('BTC Price - Last 500 Points (Recent ~42 Hours)')
ax.set_xlabel('Time (5-min intervals)')
ax.set_ylabel('Price (USDT)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('checkpoints/simple_price.png', dpi=150)
print(f"Current price: ${prices[-1]:,.2f}")
print("Saved to checkpoints/simple_price.png")
