import streamlit as st
import pandas as pd
import psycopg2
import os
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="ML Model Monitor", layout="wide")

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def load_data():
    conn = get_db_connection()
    
    # 1. Fetch Metrics History
    metrics_query = """
    SELECT window_end, metric_name, metric_value 
    FROM metrics 
    ORDER BY window_end ASC
    """
    metrics_df = pd.read_sql(metrics_query, conn)
    
    # 2. Fetch Recent Predictions (Raw Logs)
    preds_query = """
    SELECT timestamp, prediction_prob, prediction_class 
    FROM predictions 
    ORDER BY timestamp DESC LIMIT 100
    """
    preds_df = pd.read_sql(preds_query, conn)
    
    conn.close()
    return metrics_df, preds_df

# --- UI LAYOUT ---
st.title("ML Model Monitoring Dashboard")

# Refresh Button
if st.button('Refresh Data'):
    st.rerun()

try:
    metrics_df, preds_df = load_data()

    # --- ROW 1: Key Metrics (Accuracy & F1) ---
    st.header("Model Performance (Last 7 Days)")
    
    if not metrics_df.empty:
        # Create a Line Chart
        fig = px.line(
            metrics_df, 
            x='window_end', 
            y='metric_value', 
            color='metric_name', 
            markers=True,
            title="Accuracy & F1 Score Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics found. The scheduler is likely still warming up.")

    # --- ROW 2: Prediction Drift ---
    st.subheader("Recent Traffic Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if not preds_df.empty:
            # Histogram of Probabilities (Drift Detection)
            fig2 = px.histogram(
                preds_df, 
                x='prediction_prob', 
                nbins=20, 
                title="Prediction Probability Distribution (Last 100 Requests)",
                range_x=[0, 1]
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.caption("Latest Raw Logs")
        st.dataframe(preds_df, height=300)

except Exception as e:
    st.error(f"Database Connection Error: {e}")