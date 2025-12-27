"""
Real-time Web Dashboard for Bitcoin TDA Monitoring
Uses Streamlit for interactive visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from monitor import TDAMonitor
import time


# Page config
st.set_page_config(
    page_title="Bitcoin TDA Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-severe {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #fffde7;
        border-left: 4px solid #ffeb3b;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-normal {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_monitor():
    """Load TDA monitor (cached)."""
    monitor = TDAMonitor(
        historical_data_path='/notebooks/binance-data-collector/BTCUSDT_5m.csv',
        window_size=60,
        lookback_days=30
    )
    return monitor


def create_gauge_chart(value, threshold, title, color='blue'):
    """Create a gauge chart for metrics."""
    percent = (value / threshold) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 100, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 150], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': '#e8f5e9'},
                {'range': [70, 90], 'color': '#fffde7'},
                {'range': [90, 100], 'color': '#fff3e0'},
                {'range': [100, 150], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_time_series_plot(df, window_size):
    """Create time series plot with TDA metrics."""
    # Calculate rolling metrics for visualization
    features = {
        'close': df['close'].values[-1000:],  # Last 1000 points
        'volume': df['volume'].values[-1000:],
        'volume_delta': df['volume_delta'].values[-1000:],
        'cvd': df['cvd'].values[-1000:]
    }

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for key in features:
        features[key] = scaler.fit_transform(features[key].reshape(-1, 1)).flatten()

    monitor = load_monitor()
    point_cloud = monitor.detector.create_multivariate_point_cloud(features)
    results = monitor.detector.sliding_window_analysis(point_cloud, homology_dim=1)

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Bitcoin Price', 'L¹ Norm', 'L² Norm'),
        row_heights=[0.4, 0.3, 0.3]
    )

    # Price
    timestamps = df['timestamp'].values[-1000:]
    plot_timestamps = timestamps[window_size-1:][:len(results['l1_norms'])]

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'].values[-1000:],
            y=df['close'].values[-1000:],
            mode='lines',
            name='Price',
            line=dict(color='#667eea', width=2)
        ),
        row=1, col=1
    )

    # L1 Norm
    l1_threshold = monitor.baseline_stats['l1_norm']['threshold']
    fig.add_trace(
        go.Scatter(
            x=plot_timestamps,
            y=results['l1_norms'],
            mode='lines',
            name='L¹ Norm',
            line=dict(color='#f59e0b', width=2)
        ),
        row=2, col=1
    )
    fig.add_hline(
        y=l1_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        row=2, col=1
    )

    # L2 Norm
    l2_threshold = monitor.baseline_stats['l2_norm']['threshold']
    fig.add_trace(
        go.Scatter(
            x=plot_timestamps,
            y=results['l2_norms'],
            mode='lines',
            name='L² Norm',
            line=dict(color='#10b981', width=2)
        ),
        row=3, col=1
    )
    fig.add_hline(
        y=l2_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="L¹ Norm", row=2, col=1)
    fig.update_yaxes(title_text="L² Norm", row=3, col=1)

    return fig


def create_persistence_diagram_plot(dgm):
    """Create persistence diagram visualization."""
    dgm = dgm[dgm[:, 1] < np.inf]  # Remove infinite points

    if len(dgm) == 0:
        return None

    fig = go.Figure()

    # Add points
    fig.add_trace(go.Scatter(
        x=dgm[:, 0],
        y=dgm[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=dgm[:, 1] - dgm[:, 0],  # Color by persistence
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Persistence")
        ),
        name='H₁ features',
        text=[f'Birth: {b:.4f}<br>Death: {d:.4f}<br>Persistence: {d-b:.4f}'
              for b, d in dgm],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add diagonal
    max_val = max(dgm[:, 1].max(), dgm[:, 0].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Diagonal',
        showlegend=True
    ))

    fig.update_layout(
        title='Persistence Diagram (H₁)',
        xaxis_title='Birth',
        yaxis_title='Death',
        height=400,
        hovermode='closest'
    )

    return fig


def main():
    """Main dashboard function."""

    # Header
    st.markdown('<div class="main-header">📊 Bitcoin TDA Monitor</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 10, 300, 60)

        st.markdown("---")

        st.header("📖 About")
        st.markdown("""
        **Topological Data Analysis Monitor**

        This dashboard uses TDA to detect extreme events in Bitcoin market.

        **Metrics:**
        - **L¹ Norm**: Topological complexity
        - **L² Norm**: Structural stability
        - **Alert Levels**:
          - 🟢 Normal: < 70% of threshold
          - 🟡 Warning: 70-90%
          - 🟠 Severe: 90-100%
          - 🔴 Critical: > 100%
        """)

        st.markdown("---")
        st.markdown("**Paper**: [arXiv:2405.16052](https://arxiv.org/abs/2405.16052)")

        if st.button("🔄 Force Refresh"):
            st.cache_resource.clear()
            st.rerun()

    # Load monitor and run analysis
    with st.spinner('Initializing TDA Monitor...'):
        monitor = load_monitor()

    with st.spinner('Analyzing current market condition...'):
        analysis = monitor.run_once()

    if analysis is None:
        st.error("❌ Unable to perform analysis. Check data availability.")
        return

    # Alert Banner
    alert = analysis['alert_level']
    alert_class = f"alert-{alert['level'].lower()}"

    st.markdown(f"""
    <div class="{alert_class}">
        <h2>{alert['symbol']} {alert['level']}</h2>
        <p>{alert['message']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Current Price",
            f"${analysis['price']:,.2f}",
            delta=None
        )

    with col2:
        l1_val = analysis['metrics']['l1_norm']['value']
        l1_pct = analysis['metrics']['l1_norm']['percent_of_threshold']
        st.metric(
            "📊 L¹ Norm",
            f"{l1_val:.4f}",
            delta=f"{l1_pct:.1f}% of threshold"
        )

    with col3:
        l2_val = analysis['metrics']['l2_norm']['value']
        l2_pct = analysis['metrics']['l2_norm']['percent_of_threshold']
        st.metric(
            "📈 L² Norm",
            f"{l2_val:.4f}",
            delta=f"{l2_pct:.1f}% of threshold"
        )

    with col4:
        st.metric(
            "⏰ Last Update",
            analysis['timestamp'].strftime("%H:%M:%S"),
            delta=None
        )

    st.markdown("---")

    # Gauge Charts
    st.subheader("📊 Metric Gauges")
    col1, col2 = st.columns(2)

    with col1:
        l1_gauge = create_gauge_chart(
            analysis['metrics']['l1_norm']['value'],
            analysis['metrics']['l1_norm']['threshold'],
            "L¹ Norm Status",
            color='#f59e0b'
        )
        st.plotly_chart(l1_gauge, use_container_width=True)

    with col2:
        l2_gauge = create_gauge_chart(
            analysis['metrics']['l2_norm']['value'],
            analysis['metrics']['l2_norm']['threshold'],
            "L² Norm Status",
            color='#10b981'
        )
        st.plotly_chart(l2_gauge, use_container_width=True)

    st.markdown("---")

    # Time Series Plot
    st.subheader("📈 Historical Analysis")
    df = monitor.df_history
    time_series_fig = create_time_series_plot(df, monitor.window_size)
    st.plotly_chart(time_series_fig, use_container_width=True)

    st.markdown("---")

    # Persistence Diagram
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔬 Persistence Diagram")
        pd_fig = create_persistence_diagram_plot(analysis['persistence_diagram'])
        if pd_fig:
            st.plotly_chart(pd_fig, use_container_width=True)
        else:
            st.info("No persistent features detected in current window")

    with col2:
        st.subheader("📊 Baseline Statistics")

        stats_df = pd.DataFrame({
            'Metric': ['L¹ Norm', 'L² Norm'],
            'Mean (μ)': [
                monitor.baseline_stats['l1_norm']['mean'],
                monitor.baseline_stats['l2_norm']['mean']
            ],
            'Std (σ)': [
                monitor.baseline_stats['l1_norm']['std'],
                monitor.baseline_stats['l2_norm']['std']
            ],
            'Threshold (μ+4σ)': [
                monitor.baseline_stats['l1_norm']['threshold'],
                monitor.baseline_stats['l2_norm']['threshold']
            ]
        })

        st.dataframe(
            stats_df.style.format({
                'Mean (μ)': '{:.6f}',
                'Std (σ)': '{:.6f}',
                'Threshold (μ+4σ)': '{:.6f}'
            }),
            use_container_width=True
        )

        st.markdown("### 📝 What is L1 Norm?")
        st.markdown("""
        **L1 Norm** measures the **topological complexity** of the market:

        - **Low value**: Simple, stable market structure
        - **Medium value**: Moderate volatility
        - **High value**: Complex, unstable structure (potential extreme event)

        It's calculated from the persistence landscape, which captures
        the "shape" of the price data in a topological sense.
        """)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()
