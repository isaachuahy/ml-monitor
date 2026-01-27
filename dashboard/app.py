import streamlit as st
import pandas as pd
import psycopg2
import os
import plotly.express as px
import time

# Set page configuration
st.set_page_config(page_title="ML Model Monitor", layout="wide")

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def load_data():
    conn = get_db_connection()
    
    # 1. Fetch Metrics History
    performance_query = """
    SELECT window_end, metric_name, metric_value 
    FROM metrics 
    WHERE metric_name != 'drift_income_p_value'
    ORDER BY window_end ASC
    """
    performance_df = pd.read_sql(performance_query, conn)
    
    # 2. Fetch Recent Predictions (Raw Logs)
    preds_query = """
    SELECT timestamp, prediction_prob, prediction_class 
    FROM predictions 
    ORDER BY timestamp DESC LIMIT 100
    """
    preds_df = pd.read_sql(preds_query, conn)

    # 3. Fetch Drift History
    drift_query = """
    SELECT window_end, metric_name, metric_value 
    FROM metrics 
    WHERE metric_name = 'drift_income_p_value'
    ORDER BY window_end ASC
    """
    drift_df = pd.read_sql(drift_query, conn)
    
    conn.close()
    return performance_df, drift_df, preds_df

# --- UI LAYOUT ---
st.title("ML Model Monitoring Dashboard")

# Create toggle for auto refresh
# Default to True, but allow user to toggle off to save DB resources
auto_refresh = st.toggle("Auto Refresh", value=True)

# Placeholders for charts
metrics_placeholder = st.empty()
drift_placeholder = st.empty()

while True:
    try:
        performance_df, drift_df, preds_df = load_data()

        # Update performance metrics & drift p-values chart 
    
        with metrics_placeholder.container():
            st.header("Model Performance (Last 7 Days)")        
            if not performance_df.empty:
                # Create a Line Chart
                fig = px.line(
                    performance_df, 
                    x='window_end', 
                    y='metric_value', 
                    color='metric_name', 
                    markers=True,
                    title="Accuracy & F1 Score Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No metrics found. The scheduler is likely still warming up.")

            st.subheader("Data Drift Monitor (KS-Test P-Values) (Last 7 Days)")
            st.text("The KS-Test P-Values are the p-values of the KS-Test between the training data and the new data. A p-value < 0.05 indicates a significant drift.")
            
            if not drift_df.empty:
                fig_drift = px.line(
                    drift_df, 
                    x='window_end', 
                    y='metric_value', 
                    color='metric_name',
                    markers=True,
                    title="Data Drift P-Values (Target > 0.05)"
                )
                # Add a horizontal line at y=0.05
                fig_drift.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Threshold (0.05)")
                st.plotly_chart(fig_drift, use_container_width=True)
            else:
                st.warning("No drift data found. The drift detection is likely still warming up.")

        # Update drift & logs chart
        with drift_placeholder.container():
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

        if not auto_refresh:
            break

        # Sleep for 3 seconds before refreshing because we don't want to overwhelm the DB
        time.sleep(3)

    except Exception as e:
        st.error(f"Database Connection Error: {e}")