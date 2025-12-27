"""
Quick visualization of current TDA status
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Load current status
with open('monitoring_status.json', 'r') as f:
    status = json.load(f)

# Extract data
timestamp = status['timestamp']
price = status['price']
l1_val = status['l1_norm']['value']
l1_pct = status['l1_norm']['percent']
l1_thr = status['l1_norm']['threshold']
l2_val = status['l2_norm']['value']
l2_pct = status['l2_norm']['percent']
l2_thr = status['l2_norm']['threshold']
alert = status['alert']

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('🔍 Bitcoin TDA Monitor - Current Status', fontsize=20, fontweight='bold', y=0.98)

# Alert status box
ax = axes[0, 0]
ax.axis('off')

alert_colors = {
    'NORMAL': '#4caf50',
    'WARNING': '#ffeb3b',
    'SEVERE': '#ff9800',
    'CRITICAL': '#f44336'
}

color = alert_colors[alert['level']]
rect = mpatches.FancyBboxPatch((0.1, 0.3), 0.8, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor=color,
                               edgecolor='black',
                               linewidth=3)
ax.add_patch(rect)

ax.text(0.5, 0.6, f"{alert['symbol']} {alert['level']}",
        ha='center', va='center', fontsize=24, fontweight='bold', color='white')
ax.text(0.5, 0.4, alert['message'],
        ha='center', va='center', fontsize=14, color='white', style='italic')
ax.text(0.5, 0.15, f"💰 Price: ${price:,.2f}",
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(0.5, 0.05, f"⏰ {timestamp}",
        ha='center', va='center', fontsize=12, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Current Alert Status', fontsize=16, fontweight='bold', pad=20)

# L1 Norm Progress Bar
ax = axes[0, 1]
ax.axis('off')

# Draw progress bar
bar_width = 0.7
bar_height = 0.15

# Background
rect_bg = mpatches.Rectangle((0.15, 0.6), bar_width, bar_height,
                             facecolor='#f0f0f0', edgecolor='black', linewidth=2)
ax.add_patch(rect_bg)

# Progress
progress = min(l1_pct / 100, 1.5)
if l1_pct < 70:
    color = '#4caf50'
elif l1_pct < 90:
    color = '#ffeb3b'
elif l1_pct < 100:
    color = '#ff9800'
else:
    color = '#f44336'

rect_progress = mpatches.Rectangle((0.15, 0.6), bar_width * progress, bar_height,
                                   facecolor=color, edgecolor='black', linewidth=2)
ax.add_patch(rect_progress)

# Threshold line
threshold_x = 0.15 + bar_width
ax.plot([threshold_x, threshold_x], [0.6, 0.75], 'r-', linewidth=3)
ax.text(threshold_x, 0.78, '100%', ha='center', fontsize=10, fontweight='bold', color='red')

# Labels
ax.text(0.5, 0.85, 'L¹ Norm', ha='center', fontsize=18, fontweight='bold')
ax.text(0.5, 0.50, f'{l1_pct:.1f}%', ha='center', fontsize=24, fontweight='bold', color=color)
ax.text(0.5, 0.35, f'Value: {l1_val:.4f}', ha='center', fontsize=12)
ax.text(0.5, 0.25, f'Threshold: {l1_thr:.4f}', ha='center', fontsize=12, color='red')

# Zones
ax.text(0.15, 0.10, '✅ Safe\n(< 70%)', ha='left', fontsize=9, color='#4caf50')
ax.text(0.40, 0.10, '⚡ Caution\n(70-90%)', ha='center', fontsize=9, color='#ffeb3b')
ax.text(0.65, 0.10, '⚠️ Warning\n(90-100%)', ha='center', fontsize=9, color='#ff9800')
ax.text(0.85, 0.10, '🚨 Critical\n(> 100%)', ha='right', fontsize=9, color='#f44336')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('L¹ Norm Status', fontsize=16, fontweight='bold', pad=20)

# L2 Norm Progress Bar
ax = axes[1, 0]
ax.axis('off')

# Draw progress bar
rect_bg = mpatches.Rectangle((0.15, 0.6), bar_width, bar_height,
                             facecolor='#f0f0f0', edgecolor='black', linewidth=2)
ax.add_patch(rect_bg)

# Progress
progress = min(l2_pct / 100, 1.5)
if l2_pct < 70:
    color = '#4caf50'
elif l2_pct < 90:
    color = '#ffeb3b'
elif l2_pct < 100:
    color = '#ff9800'
else:
    color = '#f44336'

rect_progress = mpatches.Rectangle((0.15, 0.6), bar_width * progress, bar_height,
                                   facecolor=color, edgecolor='black', linewidth=2)
ax.add_patch(rect_progress)

# Threshold line
threshold_x = 0.15 + bar_width
ax.plot([threshold_x, threshold_x], [0.6, 0.75], 'r-', linewidth=3)
ax.text(threshold_x, 0.78, '100%', ha='center', fontsize=10, fontweight='bold', color='red')

# Labels
ax.text(0.5, 0.85, 'L² Norm', ha='center', fontsize=18, fontweight='bold')
ax.text(0.5, 0.50, f'{l2_pct:.1f}%', ha='center', fontsize=24, fontweight='bold', color=color)
ax.text(0.5, 0.35, f'Value: {l2_val:.4f}', ha='center', fontsize=12)
ax.text(0.5, 0.25, f'Threshold: {l2_thr:.4f}', ha='center', fontsize=12, color='red')

# Zones
ax.text(0.15, 0.10, '✅ Safe\n(< 70%)', ha='left', fontsize=9, color='#4caf50')
ax.text(0.40, 0.10, '⚡ Caution\n(70-90%)', ha='center', fontsize=9, color='#ffeb3b')
ax.text(0.65, 0.10, '⚠️ Warning\n(90-100%)', ha='center', fontsize=9, color='#ff9800')
ax.text(0.85, 0.10, '🚨 Critical\n(> 100%)', ha='right', fontsize=9, color='#f44336')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('L² Norm Status', fontsize=16, fontweight='bold', pad=20)

# Info box
ax = axes[1, 1]
ax.axis('off')

info_text = f"""
📊 TDA METRICS EXPLAINED

L1 Norm = 시장의 위상학적 복잡도
  • 낮음 (< 70%): 안정적 시장 ✅
  • 중간 (70-90%): 변동성 증가 ⚡
  • 높음 (> 100%): 극단 이벤트! 🚨

현재 상태:
  • L¹ Norm이 {l1_pct:.1f}%로
    {"안정적" if l1_pct < 70 else "주의 필요" if l1_pct < 90 else "경고" if l1_pct < 100 else "위험"}한 수준입니다

  • L² Norm이 {l2_pct:.1f}%로
    {"안정적" if l2_pct < 70 else "주의 필요" if l2_pct < 90 else "경고" if l2_pct < 100 else "위험"}한 수준입니다

📖 자세한 분석은 Jupyter Notebook에서
   dashboard_notebook.ipynb를 실행하세요!
"""

ax.text(0.05, 0.95, info_text, ha='left', va='top', fontsize=11,
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Quick Reference', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('current_status.png', dpi=150, bbox_inches='tight')
print("✅ 현재 상태 시각화 저장: current_status.png")
plt.show()
