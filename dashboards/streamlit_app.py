# Streamlit Dashboard for Multi-UE Traffic Classification
# Run with: streamlit run dashboards/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-UE Traffic Classifier",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.success-metric {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
}
.warning-metric {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
}
.error-metric {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚀 Multi-UE Traffic Classification Dashboard")
st.markdown("Real-time network traffic classification for 5G QoS optimization")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)

# QoS mapping for reference
QOS_MAPPING = {
    'gaming': {'priority': 1, 'qci': 3, 'delay_budget': 50, 'color': '#FF6B6B'},
    'audio_calls': {'priority': 2, 'qci': 1, 'delay_budget': 100, 'color': '#4ECDC4'},
    'video_calls': {'priority': 3, 'qci': 2, 'delay_budget': 150, 'color': '#45B7D1'},
    'video_streaming': {'priority': 4, 'qci': 6, 'delay_budget': 300, 'color': '#96CEB4'},
    'browsing': {'priority': 5, 'qci': 8, 'delay_budget': 300, 'color': '#FFEAA7'},
    'video_uploads': {'priority': 6, 'qci': 9, 'delay_budget': 300, 'color': '#DDA0DD'},
    'texting': {'priority': 7, 'qci': 9, 'delay_budget': 300, 'color': '#98D8C8'}
}

# API health check
def check_api_health():
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# Check API status
api_healthy, health_info = check_api_health()

if api_healthy:
    st.sidebar.success("🟢 API Online")
    if health_info:
        st.sidebar.metric("Uptime", f"{health_info.get('uptime_seconds', 0):.0f}s")
        st.sidebar.metric("Model Status", "Loaded" if health_info.get('model_loaded') else "Not Loaded")
else:
    st.sidebar.error("🔴 API Offline")
    st.error("Cannot connect to API. Please ensure the server is running at the specified URL.")
    st.info("Start the API server with: `python scripts/serve_api.py`")

# Main content
if api_healthy:

    # File upload section
    st.header("📤 Upload Flow Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with network flow data", 
        type=['csv'],
        help="Upload a CSV file containing network flow features"
    )

    # Sample data generator
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎲 Generate Sample Data"):
            # Generate synthetic sample data
            np.random.seed(42)
            sample_data = []

            classes = ['gaming', 'video_streaming', 'browsing', 'audio_calls', 'video_calls']

            for i in range(50):
                app_class = np.random.choice(classes)

                if app_class == 'gaming':
                    flow_duration = max(0.1, np.random.exponential(2))
                    total_fwd = max(1, int(np.random.poisson(50)))
                    total_bwd = max(1, int(np.random.poisson(30)))
                elif app_class == 'video_streaming':
                    flow_duration = max(1, np.random.exponential(180))
                    total_fwd = max(1, int(np.random.poisson(1000)))
                    total_bwd = max(1, int(np.random.poisson(200)))
                else:  # browsing, calls, etc.
                    flow_duration = max(0.5, np.random.exponential(30))
                    total_fwd = max(1, int(np.random.poisson(100)))
                    total_bwd = max(1, int(np.random.poisson(150)))

                sample_data.append({
                    'Flow Duration': flow_duration,
                    'Total Fwd Packets': total_fwd,
                    'Total Backward Packets': total_bwd,
                    'Flow Bytes/s': (total_fwd + total_bwd) * 800 / flow_duration,
                    'Flow Packets/s': (total_fwd + total_bwd) / flow_duration
                })

            st.session_state.sample_data = pd.DataFrame(sample_data)
            st.success("Generated 50 sample flows!")

    with col2:
        ue_id = st.text_input("UE Identifier", value="UE_001")

    # Process data
    df = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} flows from uploaded file")
    elif 'sample_data' in st.session_state:
        df = st.session_state.sample_data
        st.info("Using generated sample data")

    if df is not None:

        # Display data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Flows", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")

        # Prediction section
        st.header("🔮 Traffic Classification")

        col1, col2 = st.columns([3, 1])

        with col2:
            include_probabilities = st.checkbox("Include Probabilities", value=True)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)

        with col1:
            if st.button("🚀 Classify Traffic", type="primary"):

                with st.spinner("Classifying traffic..."):
                    try:
                        # Prepare request
                        flows_data = df.to_dict('records')

                        request_payload = {
                            "flows": flows_data,
                            "ue_id": ue_id,
                            "include_probabilities": include_probabilities,
                            "confidence_threshold": confidence_threshold
                        }

                        # Make API request
                        response = requests.post(
                            f"{api_url}/predict",
                            json=request_payload,
                            timeout=30
                        )

                        if response.status_code == 200:
                            results = response.json()
                            predictions = results["predictions"]
                            processing_time = results["processing_time_ms"]

                            # Store results in session state
                            st.session_state.predictions = predictions
                            st.session_state.processing_time = processing_time
                            st.session_state.ue_id = results["ue_id"]

                            st.success(f"Classification complete! Processed {len(predictions)} flows in {processing_time:.1f}ms")

                        else:
                            st.error(f"API Error: {response.status_code} - {response.text}")

                    except Exception as e:
                        st.error(f"Request failed: {str(e)}")

        # Display results
        if 'predictions' in st.session_state:

            predictions = st.session_state.predictions
            processing_time = st.session_state.processing_time

            st.header("📈 Classification Results")

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Flows", len(predictions))

            with col2:
                st.metric("Processing Time", f"{processing_time:.1f}ms")

            with col3:
                avg_confidence = np.mean([p["confidence"] for p in predictions])
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")

            with col4:
                throughput = len(predictions) / (processing_time / 1000) if processing_time > 0 else 0
                st.metric("Throughput", f"{throughput:.0f} flows/sec")

            # Extract prediction data
            pred_classes = [p["predicted_class"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            priorities = [p["qos_priority"] for p in predictions]

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Class distribution pie chart
                class_counts = pd.Series(pred_classes).value_counts()
                colors = [QOS_MAPPING.get(cls, {}).get('color', '#CCCCCC') for cls in class_counts.index]

                fig_pie = px.pie(
                    values=class_counts.values, 
                    names=class_counts.index,
                    title="Traffic Class Distribution",
                    color_discrete_sequence=colors
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Confidence distribution histogram
                fig_hist = px.histogram(
                    x=confidences, 
                    nbins=20,
                    title="Prediction Confidence Distribution",
                    labels={'x': 'Confidence Score', 'y': 'Count'}
                )
                fig_hist.add_vline(x=confidence_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)

            # QoS Priority Analysis
            col1, col2 = st.columns(2)

            with col1:
                # QoS Priority distribution
                priority_counts = pd.Series(priorities).value_counts().sort_index()

                fig_bar = px.bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    title="QoS Priority Distribution",
                    labels={'x': 'QoS Priority Level', 'y': 'Number of Flows'},
                    color=priority_counts.values,
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Confidence vs Priority scatter
                fig_scatter = px.scatter(
                    x=priorities,
                    y=confidences,
                    color=pred_classes,
                    title="Confidence vs QoS Priority",
                    labels={'x': 'QoS Priority', 'y': 'Confidence Score'},
                    color_discrete_map={cls: info['color'] for cls, info in QOS_MAPPING.items()}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Detailed results table
            with st.expander("📋 Detailed Results", expanded=False):

                # Create results DataFrame
                results_df = pd.DataFrame([
                    {
                        'Flow_ID': i,
                        'Predicted_Class': p["predicted_class"],
                        'Confidence': f"{p['confidence']:.3f}",
                        'QoS_Priority': p["qos_priority"],
                        'QCI': p["qci"],
                        'Delay_Budget_ms': p["delay_budget_ms"]
                    }
                    for i, p in enumerate(predictions)
                ])

                st.dataframe(results_df, use_container_width=True)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results CSV",
                    data=csv,
                    file_name=f"traffic_classification_results_{ue_id}.csv",
                    mime="text/csv"
                )

            # QoS Mapping Reference
            with st.expander("🚦 QoS Mapping Reference"):
                qos_df = pd.DataFrame.from_dict(QOS_MAPPING, orient='index')
                qos_df.index.name = 'Application Class'
                qos_df = qos_df.reset_index()
                st.dataframe(qos_df, use_container_width=True)

# Auto-refresh logic
if auto_refresh and api_healthy:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("🚀 **Multi-UE Traffic Classification Dashboard**")
