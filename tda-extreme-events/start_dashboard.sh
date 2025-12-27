#!/bin/bash

# Start the TDA Dashboard
cd /notebooks/tda-extreme-events
echo "🚀 Starting Bitcoin TDA Dashboard..."
echo "📊 Access the dashboard at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop"
echo ""

streamlit run dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress=localhost \
    --browser.gatherUsageStats=false
