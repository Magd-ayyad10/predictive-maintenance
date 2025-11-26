"""
================================================================================
AI4I 2020 Professional Dashboard (v4 - Custom Refined)
================================================================================

Description:
This Streamlit application provides a comprehensive, multi-page professional
dashboard for analyzing the *ENRICHED* AI4I 2020 dataset.

Customizations:
- Uses 'ai4i2020_enriched.csv'.
- Executive Summary:
    - Removed "Predictive Model KPIs".
    - Removed "Model Performance Snapshot".
    - Removed "Machine Health Pie Chart".
- Risk Heatmap:
    - Filters out "Low" risk.
    - Hides hierarchy labels.
    - Order: Red -> Orange -> Yellow.
- Includes Maintenance Priority List, ROI Calculator, and Live Prediction.

To run this app:
1.  Ensure you have all required libraries installed:
    pip install streamlit pandas numpy plotly scikit-learn xgboost lightgbm imbalanced-learn joblib xlsxwriter
2.  Make sure the following files are in the same directory:
    - ai4i2020_enriched.csv
    - voting_model_ai4i_enriched.joblib
    - scaler_ai4i_enriched.joblib
3.  Run the following command in your terminal:
    streamlit run ai4i_dashboard_v4_custom.py
"""

# ------------------------------------------------------------------------------
# 1. IMPORTS & INITIAL SETUP
# ------------------------------------------------------------------------------

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import warnings
from io import BytesIO
import os

# Plotting libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests  # Added for API calls
from datetime import datetime

# Scikit-learn & Model libraries
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_recall_curve, 
    auc,
    f1_score
)

# Suppress warnings for a cleaner dashboard output
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 2. APP CONFIGURATION & STYLING
# ------------------------------------------------------------------------------

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_custom_styling():
    """
    Injects custom CSS for a professional, "magnificent" look.
    """
    custom_css = """
    <style>
        /* --- Main App Styling --- */
        .stApp {
            background-color: #f0f2f6; /* Light gray background */
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: #0c1e3a; /* Dark blue sidebar */
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #ffffff;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            color: #ffffff;
            font-size: 28px;
            font-weight: bold;
        }

        /* --- Main Content Area --- */
        .block-container {
            padding-top: 2rem;
        }
        
        /* --- Title Styling --- */
        h1 {
            color: #0c1e3a; /* Dark blue title */
            font-weight: 700;
        }
        
        h2 {
            color: #1a3a69; /* Medium blue sub-title */
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
            font-weight: 600;
        }
        
        h3 {
            color: #2c508c;
            font-weight: 600;
        }

        /* --- Custom Card Styling for KPIs --- */
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            text-align: left;
            height: 170px; /* Set fixed height for alignment */
        }
        
        .kpi-card h3 {
            color: #555555;
            font-size: 1.1rem;
            margin-bottom: 12px;
            text-transform: uppercase;
            font-weight: 500;
        }
        
        .kpi-card .value {
            color: #0c1e3a;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .kpi-card .subtext {
            color: #666666;
            font-size: 0.9rem;
        }
        
        /* --- Button Styling --- */
        .stButton > button {
            background-color: #0068c9; /* Bright blue button */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #00509e;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* --- Tabs Styling --- */
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            color: #555;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            color: #0068c9;
            border-bottom: 3px solid #0068c9;
        }
        
        /* --- Metric Styling --- */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        /* --- Toggle Styling --- */
        [data-testid="stToggle"] {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Apply the custom styling
apply_custom_styling()

# ------------------------------------------------------------------------------
# 3. GLOBAL CONSTANTS & DATA LOADING
# ------------------------------------------------------------------------------

# --- V4 CUSTOM FILE PATHS ---
DATA_FILE = 'ai4i2020_enriched.csv' 
MODEL_FILE = 'voting_model_ai4i_enriched.joblib'
SCALER_FILE = 'scaler_ai4i_enriched.joblib'
LOG_FILE = 'prediction_logs.xlsx'


# API Configuration
API_URL = "http://127.0.0.1:8000/predict"

# Define column groups
IDENTIFIER_COLS = ['UDI', 'Product ID']
TARGET_COL = 'Machine failure'
FAILURE_FLAGS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
NUMERICAL_COLS = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]
CATEGORICAL_COLS = ['Type']
# NEW columns to use for KPIs
BUSINESS_COLS = [
    'event_timestamp', 
    'downtime_hours', 
    'repair_cost', 
    'repair_notes'
]


@st.cache_data(ttl=3600)
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the AI4I 2020 ENRICHED CSV data.
    """
    try:
        df = pd.read_csv(filepath)
        # Convert timestamp column to datetime objects
        if 'event_timestamp' in df.columns:
            df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        # Use Time Index from UDI for plotting if Time Index is missing
        if 'Time Index' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'Time Index'})
            
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file '{filepath}' not found.")
        st.info(f"Please make sure '{DATA_FILE}' is in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None
    
def save_to_excel(data_dict):
    """
    Saves a dictionary of data to an Excel file. 
    Appends if file exists, creates new if not.
    """
    df_new = pd.DataFrame([data_dict])
    
    if os.path.exists(LOG_FILE):
        try:
            # Read existing data
            df_existing = pd.read_excel(LOG_FILE)
            # Concat new data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Save back
            df_combined.to_excel(LOG_FILE, index=False)
        except Exception as e:
            st.error(f"Error appending to Excel: {e}")
    else:
        try:
            df_new.to_excel(LOG_FILE, index=False)
        except Exception as e:
            st.error(f"Error creating Excel file: {e}")
@st.cache_resource(ttl=3600)
def load_production_model(model_path: str, scaler_path: str) -> dict:
    """
    Loads the pre-trained production model and preprocessor.
    """
    try:
        loaded_model = joblib.load(model_path)
        loaded_preprocessor = joblib.load(scaler_path)
        
        return {
            "model": loaded_model,
            "preprocessor": loaded_preprocessor
        }
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e.filename} not found.")
        st.info(f"""
            Please make sure '{model_path}' and '{scaler_path}' 
            are in the same directory as the app.
        """)
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None

# ------------------------------------------------------------------------------
# 4. PLOTTING & HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def create_kpi_card(title, value, subtext):
    """
    Helper function to render a styled KPI card using HTML.
    """
    st.markdown(f"""
    <div class="kpi-card">
        <h3>{title}</h3>
        <p class="value">{value}</p>
        <p class="subtext">{subtext}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Business KPI Calculation Functions ---
@st.cache_data(ttl=3600)
def calculate_business_kpis(df: pd.DataFrame) -> dict:
    """
    Calculates MTBF, MTTR, Cost metrics.
    """
    kpis = {
        "mtbf": "N/A", "mttr": "N/A", "total_downtime": "N/A",
        "total_cost": "N/A", "avg_failure_cost": 0, "total_failures": 0
    }
    
    if df is None:
        kpis['error'] = "Data not loaded."
        return kpis

    try:
        # 1. MTTR, Cost, Downtime
        df_failures_only = df[df['Machine failure'] == 1].copy()
        kpis['total_failures'] = len(df_failures_only)
        
        if 'downtime_hours' in df_failures_only.columns:
            kpis['mttr'] = f"{df_failures_only['downtime_hours'].mean():.1f} hrs"
            kpis['total_downtime'] = f"{df['downtime_hours'].sum():,.0f} hrs"
            
        if 'repair_cost' in df_failures_only.columns:
            kpis['total_cost'] = f"${df['repair_cost'].sum():,.0f}"
            kpis['avg_failure_cost'] = df_failures_only['repair_cost'].mean()
            
        # 2. MTBF
        if 'event_timestamp' in df_failures_only.columns and len(df_failures_only) >= 2:
            df_failures_sorted = df_failures_only.sort_values(by='event_timestamp')
            time_diffs_hours = df_failures_sorted['event_timestamp'].diff().dt.total_seconds() / 3600
            kpis['mtbf'] = f"{time_diffs_hours.mean():.1f} hrs"
            
        return kpis
        
    except Exception as e:
        kpis['error'] = str(e)
        return kpis

# --- Removed plot_machine_health_pie as requested ---

def plot_failure_type_bars(df: pd.DataFrame):
    """
    Plots a bar chart showing the counts of different failure types.
    """
    failure_counts = df[FAILURE_FLAGS].sum().sort_values(ascending=False)
    failure_df = failure_counts.reset_index()
    failure_df.columns = ['Failure Type', 'Count']
    
    fig = px.bar(
        failure_df, x='Failure Type', y='Count',
        title="Breakdown of Specific Failure Modes",
        color='Failure Type', text_auto=True
    )
    fig.update_layout(xaxis_title="Failure Type", yaxis_title="Total Occurrences",
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_sensor_trends(df: pd.DataFrame, machine_id: str):
    """
    Plots sensor trends for a specific machine.
    """
    machine_data = df[df['Product ID'] == machine_id].copy()
    
    if machine_data.empty:
        st.warning(f"No data found for Machine ID: {machine_id}")
        return
        
    # Mark failure events
    machine_data['Failure Event'] = machine_data[TARGET_COL].astype(str)
    
    x_axis_col = 'event_timestamp' if 'event_timestamp' in machine_data.columns else 'Time Index'
    
    fig = make_subplots(rows=len(NUMERICAL_COLS), cols=1, shared_xaxes=True, subplot_titles=NUMERICAL_COLS)
    colors = px.colors.qualitative.Plotly
    
    for i, col in enumerate(NUMERICAL_COLS, 1):
        # Create a temporary df for this plot, dropping NaNs for this specific column
        plot_data = machine_data[[x_axis_col, col]].dropna()

        # Plot the main sensor line
        fig.add_trace(go.Scatter(
            x=plot_data[x_axis_col], y=plot_data[col], name=col,
            mode='lines', line=dict(color=colors[i % len(colors)])
        ), row=i, col=1)
        
        # Add markers for failures
        failure_points = machine_data[machine_data[TARGET_COL] == 1]
        failure_plot_data = failure_points[[x_axis_col, col]].dropna()
        fig.add_trace(go.Scatter(
            x=failure_plot_data[x_axis_col], y=failure_plot_data[col],
            name='Failure Event', mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            showlegend=(i==1)
        ), row=i, col=1)

    fig.update_layout(
        height=200 * len(NUMERICAL_COLS), title=f"Sensor Trends for Machine: {machine_id}",
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', showlegend=True
    )
    x_axis_title = "Timestamp" if x_axis_col == 'event_timestamp' else "Time Index"
    fig.update_xaxes(title_text=x_axis_title, row=len(NUMERICAL_COLS), col=1)
    return fig

@st.cache_data(ttl=3600)
def get_model_performance(_prod_models: dict, df: pd.DataFrame):
    """
    Calculates performance metrics for the loaded model on the full dataset.
    """
    if not _prod_models or df is None:
        return None
        
    model = _prod_models['model']
    preprocessor = _prod_models['preprocessor']
    
    try:
        # Define X and y from the full dataset
        # Drop all non-training columns
        cols_to_drop = ([TARGET_COL] + FAILURE_FLAGS + BUSINESS_COLS + 
                        IDENTIFIER_COLS + ['Time Index'])
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = df[TARGET_COL]
        
        # Get column names from the preprocessor
        cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
        num_features = preprocessor.named_transformers_['num'].feature_names_in_
        X_ordered = X[list(num_features) + list(cat_features)]
        
        # Process the data
        X_processed = preprocessor.transform(X_ordered)
        
        # Make predictions
        y_pred = model.predict(X_processed)
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        
        # Generate metrics
        report_str = classification_report(y, y_pred)
        report_dict = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        auc_score = auc(recall, precision)
        f1 = f1_score(y, y_pred, average='weighted')
        
        # Get feature importances
        feature_names = list(num_features) + \
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
        
        importances = []
        if hasattr(model, 'estimators_'): # VotingClassifier
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
            else:
                avg_importance = np.zeros(len(feature_names))
        else: # Single model
            avg_importance = getattr(model, 'feature_importances_', np.zeros(len(feature_names)))

        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': avg_importance})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)
        
        return {
            "classification_report_str": report_str,
            "classification_report_dict": report_dict,
            "confusion_matrix": cm,
            "pr_curve": {"precision": precision, "recall": recall, "auc": auc_score},
            "feature_importances": imp_df,
            "f1_score": f1,
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }
    except Exception as e:
        st.error(f"Error in get_model_performance: {e}")
        return None

def plot_confusion_matrix_plotly(cm: np.ndarray, title: str):
    """Plots an interactive confusion matrix heatmap using Plotly."""
    labels = ['Normal (0)', 'Failure (1)']
    fig = px.imshow(
        cm, labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=labels, y=labels, text_auto=True, color_continuous_scale='Blues', title=title
    )
    fig.update_xaxes(side="bottom")
    fig.update_layout(title_x=0.5, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_pr_curve_plotly(precision: np.ndarray, recall: np.ndarray, auc_score: float, title: str):
    """Plots an interactive Precision-Recall curve using Plotly."""
    df_pr = pd.DataFrame({'Recall': recall, 'Precision': precision})
    fig = px.area(
        df_pr, x='Recall', y='Precision', title=f"{title} (AUC = {auc_score:.4f})",
        labels={'Recall': 'Recall', 'Precision': 'Precision'}, template="plotly_white"
    )
    fig.update_traces(fillcolor='rgba(0,104,201,0.2)', line=dict(color='#0068c9', width=2))
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(range=[0, 1.05])
    fig.update_layout(title_x=0.5, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_feature_importance_plotly(imp_df: pd.DataFrame, title: str):
    """Plots feature importances using Plotly."""
    fig = px.bar(
        imp_df.head(15), x="Importance", y="Feature",
        orientation='h', title=title
    )
    fig.update_layout(
        yaxis_title="Feature", xaxis_title="Importance",
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@st.cache_data(ttl=3600)
def plot_risk_treemap(df: pd.DataFrame, _prod_models: dict):
    """
    Generates a treemap of all machines, categorized by risk level.
    CUSTOMIZED: 
    - Filters out "Low" (Green) risk.
    - Shows only Red (Critical) -> Orange (High) -> Yellow (Medium).
    - Removes hierarchy labels ("High", "Medium") from boxes, flattening the view.
    """
    if not _prod_models:
        return None

    try:
        # 1. Get latest reading for each machine
        sort_col = 'event_timestamp' if 'event_timestamp' in df.columns else 'Time Index'
        latest_readings = df.loc[df.groupby('Product ID')[sort_col].idxmax()].copy()
        
        # 2. Get preprocessor and model
        preprocessor = _prod_models['preprocessor']
        model = _prod_models['model']
        
        # 3. Order columns for preprocessor
        cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
        num_features = preprocessor.named_transformers_['num'].feature_names_in_
        X_ordered = latest_readings[list(num_features) + list(cat_features)]
        
        # 4. Get failure probabilities
        probs = model.predict_proba(preprocessor.transform(X_ordered))[:, 1]
        latest_readings['Failure Probability'] = probs
        
        # 5. Define Risk Level
        def assign_risk(prob):
            if prob > 0.9: return "Critical"
            elif prob > 0.5: return "High"
            elif prob > 0.2: return "Medium"
            else: return "Low"
        
        latest_readings['Risk Level'] = latest_readings['Failure Probability'].apply(assign_risk)
        
        # --- CUSTOMIZATION 1: Filter out "Low" risk machines ---
        latest_readings = latest_readings[latest_readings['Risk Level'] != "Low"]
        
        if latest_readings.empty:
            st.success("Great news! No machines are currently at Medium, High, or Critical risk.")
            return None
        
        # --- CUSTOMIZATION 2: Explicit Color Map (Red -> Orange -> Yellow) ---
        color_map = {
            'Critical': '#dc3545',  # Red
            'High': '#fd7e14',      # Orange
            'Medium': '#ffc107'     # Yellow
        }

        # --- CUSTOMIZATION 3: Flattened path (Remove H,M,L boxes) ---
        # By removing 'Risk Level' from the path, we remove the hierarchy boxes.
        # We can still color by 'Risk Level'.
        fig = px.treemap(
            latest_readings,
            path=[px.Constant("Machines at Risk"), 'Risk Level', 'Product ID'], # Flattened path
            values='Failure Probability',
            color='Risk Level',
            color_discrete_map=color_map,
            title="Machine Risk Level Heatmap (Medium to Critical Only)",
            hover_data={'Failure Probability': ':.2%'}
        )
        
        fig.update_traces(
            textinfo="label", 
            marker=dict(cornerradius=5),
            root_color="lightgrey"
        )
        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25)
        )
        return fig

    except Exception as e:
        st.warning(f"Could not generate risk treemap: {e}")
        return None

# ------------------------------------------------------------------------------
# 5. PAGE RENDERING FUNCTIONS
# ------------------------------------------------------------------------------

def render_executive_summary(df: pd.DataFrame, prod_models: dict, perf: dict, kpis: dict):
    """
    Renders Page 1: Executive Summary.
    """
    st.title("Executive Summary")
    st.markdown("A high-level health snapshot of the machine fleet.")
    
    if df is None:
        st.error("Data not loaded. Cannot display summary.")
        return

    # --- KPI Logic ---
    total_machines = df['Product ID'].nunique()
    total_failures = kpis.get("total_failures", 0)
    failure_rate = (total_failures / len(df)) * 100
    
    st.header("Live Fleet KPIs")
    # --- KPI Row 1: Fleet Overview ---
    kpi_cols_1 = st.columns(4)
    with kpi_cols_1[0]:
        create_kpi_card("Total Machines", f"{total_machines}", "Unique machines monitored")
    with kpi_cols_1[1]:
        create_kpi_card("Total Data Points", f"{len(df):,}", "Total sensor readings")
    with kpi_cols_1[2]:
        create_kpi_card("Total Failures", f"{total_failures}", "Historical failures recorded")
    with kpi_cols_1[3]:
        create_kpi_card("Overall Failure Rate", f"{failure_rate:.2f}%", "Of all log entries")
    
    st.write("") # Add some spacing

    # --- REMOVED: Predictive Model KPIs Row (as requested) ---

    # --- KPI Row 3: Business & Reliability KPIs ---
    st.header("Business & Reliability KPIs")
    st.markdown("Calculated from enriched data (timestamps, downtime, cost).")
    
    kpi_cols_3 = st.columns(4)
    with kpi_cols_3[0]:
        create_kpi_card("Mean Time (MTBF)", 
                        kpis.get("mtbf", "N/A"), 
                        "Avg. time between failures")
    with kpi_cols_3[1]:
        create_kpi_card("Mean Time (MTTR)", 
                        kpis.get("mttr", "N/A"), 
                        "Avg. time to repair")
    with kpi_cols_3[2]:
        create_kpi_card("Total Downtime", 
                        kpis.get("total_downtime", "N/A"), 
                        "Total recorded downtime")
    with kpi_cols_3[3]:
        create_kpi_card("Total Repair Costs", 
                        kpis.get("total_cost", "N/A"), 
                        "Total recorded repair costs")
    
    if "error" in kpis:
        st.warning(f"Could not calculate all business KPIs: {kpis['error']}")

    st.markdown("---")

    # --- PREVENTIVE MAINTENANCE ROI CALCULATOR ---
    st.header("Preventive Maintenance ROI Calculator")
    st.markdown("Estimate the potential savings from a proactive maintenance program.")
    
    roi_cols = st.columns([1, 2])
    with roi_cols[0]:
        avg_failure_cost = kpis.get("avg_failure_cost", 0)
        st.metric("Average Cost of a Reactive Repair", f"${avg_failure_cost:,.2f}")
        
        pm_cost = st.slider(
            "Estimated Cost of one Preventive Action", 
            min_value=0, 
            max_value=int(avg_failure_cost) if avg_failure_cost > 0 else 1000, 
            value=int(avg_failure_cost / 4) if avg_failure_cost > 0 else 0, 
            step=25
        )
        st.caption(f"This is the cost to proactively fix a problem *before* it fails (e.g., replace a tool early).")

    with roi_cols[1]:
        total_reactive_cost = total_failures * avg_failure_cost
        total_proactive_cost = total_failures * pm_cost
        net_savings = total_reactive_cost - total_proactive_cost
        
        st.markdown("#### Estimated Savings")
        st.markdown(f"""
        Based on **{total_failures}** historical failures:
        - **Total Reactive Cost:** {total_failures} failures x ${avg_failure_cost:,.2f} = **${total_reactive_cost:,.2f}**
        - **Est. Proactive Cost:** {total_failures} interventions x ${pm_cost:,.2f} = **${total_proactive_cost:,.2f}**
        """)
        st.markdown("---")
        st.metric("Estimated Net Savings", f"${net_savings:,.2f}")
    
    st.markdown("---")

   

    # --- Visuals Row ---
    st.header("Fleet Health Overview")
    
    # New Treemap Visual
    fig_treemap = plot_risk_treemap(df, prod_models)
    if fig_treemap:
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    st.write("") # Add some spacing
    
    
    
    # Only Bar Chart - Centered
    st.markdown("### Failure Modes Breakdown")
    fig_bar = plot_failure_type_bars(df)
    st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")

    # --- Recent Failures Table ---
    st.header("Recent Failure Events")
    st.markdown("A log of the 5 most recent failure events detected in the data.")
    cols_to_show = ['event_timestamp', 'Product ID', 'Type'] + NUMERICAL_COLS + ['downtime_hours', 'repair_cost']
    
    sort_col = 'event_timestamp' if 'event_timestamp' in df.columns else 'Time Index'
    recent_failures = df[df[TARGET_COL] == 1].sort_values(by=sort_col, ascending=False).head(5)
    
    if recent_failures.empty:
        st.info("No failure events found in the dataset.")
    else:
        st.dataframe(recent_failures[cols_to_show], use_container_width=True)

# --- "WOW" PAGE: PRIORITY LIST ---
def render_priority_list(df: pd.DataFrame, prod_models: dict, perf: dict):
    """
    Renders Page 2: The Maintenance Priority List.
    """
    st.title("🚨 Maintenance Priority List")
    st.markdown("A real-time, AI-driven action list of machines at high risk of failure.")
    
    if df is None or prod_models is None or perf is None:
        st.error("Data, model, or performance metrics not loaded. Cannot generate list.")
        return

    try:
        # 1. Get latest reading for each machine
        sort_col = 'event_timestamp' if 'event_timestamp' in df.columns else 'Time Index'
        latest_readings = df.loc[df.groupby('Product ID')[sort_col].idxmax()].copy()
        
        # 2. Get preprocessor and model
        preprocessor = prod_models['preprocessor']
        model = prod_models['model']
        
        # 3. Order columns for preprocessor
        cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
        num_features = preprocessor.named_transformers_['num'].feature_names_in_
        X_ordered = latest_readings[list(num_features) + list(cat_features)]
        
        # 4. Get failure probabilities
        probs = model.predict_proba(preprocessor.transform(X_ordered))[:, 1]
        latest_readings['Failure Probability'] = probs
        
        # 5. Define Risk Level
        def assign_risk(prob):
            if prob > 0.9: return "Critical"
            elif prob > 0.5: return "High"
            elif prob > 0.2: return "Medium"
            else: return "Low"
        latest_readings['Risk Level'] = latest_readings['Failure Probability'].apply(assign_risk)
        
        # 6. Get Top 2 Features (Key Drivers)
        top_feature_1 = perf['feature_importances'].iloc[0]['Feature']
        top_feature_2 = perf['feature_importances'].iloc[1]['Feature']
        
        # 7. Prescriptive Logic
        def get_recommendation(row):
            if row['TWF'] == 1: return "Schedule Tool Change"
            if row['HDF'] == 1: return "Inspect Cooling System"
            if row['PWF'] == 1: return "Inspect Power Supply"
            if row['OSF'] == 1: return "Check for Overstrain"
            return f"Inspect (Check {top_feature_1})"

        latest_readings['Recommended Action'] = latest_readings.apply(get_recommendation, axis=1)
        
        # 8. Filter for the priority list
        priority_list = latest_readings[
            latest_readings['Risk Level'].isin(['Critical', 'High'])
        ].sort_values(by='Failure Probability', ascending=False)
        
        st.header(f"Machines Requiring Immediate Attention: {len(priority_list)}")
        
        if priority_list.empty:
            st.success("✅ All machines are operating at low risk.")
            return

        # 9. Display the list
        cols_to_show = [
            'Product ID', 
            'Risk Level', 
            'Failure Probability',
            'Recommended Action',
            'Type',
            top_feature_1, # Show the value of the key driver
            top_feature_2  # Show the value of the 2nd key driver
        ]
        
        final_cols = [col for col in cols_to_show if col in priority_list.columns]
        
        priority_display = priority_list[final_cols].copy()
        priority_display['Failure Probability'] = priority_display['Failure Probability'].map('{:.1%}'.format)
        
        st.dataframe(priority_display, use_container_width=True, height=300)

        with st.expander("Risk Level Definitions"):
            st.markdown("""
            - **Critical:** > 90% Failure Probability
            - **High:** 50% - 90% Failure Probability
            - **Medium:** 20% - 50% Failure Probability
            """)
            
    except Exception as e:
        st.error(f"An error occurred while generating the priority list: {e}")


def render_machine_deep_dive(df: pd.DataFrame):
    """
    Renders Page 3: Machine Health Monitoring.
    """
    st.title("Machine Deep Dive")
    st.markdown("Select an individual machine (that has experienced a failure) to analyze its sensor data over time.")
    
    if df is None:
        st.error("Data not loaded. Cannot display page.")
        return

    # --- UPDATED LOGIC START ---
    # Filter to only unique Product IDs where 'Machine failure' (TARGET_COL) == 1
    failed_machines = df[df[TARGET_COL] == 1]['Product ID'].unique()

    if len(failed_machines) == 0:
        st.warning("No machines in the dataset have recorded failures.")
        return

    # Populate selectbox with only failed machines
    selected_machine = st.selectbox(
        "Select a Failed Machine ID:",
        options=failed_machines,
        index=0
    )
    # --- UPDATED LOGIC END ---
    
    if selected_machine:
        st.header(f"Sensor Trends: {selected_machine}")
        fig_trends = plot_sensor_trends(df, selected_machine)
        if fig_trends:
            st.plotly_chart(fig_trends, use_container_width=True)
            
        st.header(f"Historical Data: {selected_machine}")
        st.dataframe(df[df['Product ID'] == selected_machine], use_container_width=True)

    

    st.title("Data Explorer & Reports")
    st.markdown("Filter, explore, and download the historical machine data.")
    
    if df is None:
        st.error("Data not loaded. Cannot display page.")
        return

    st.header("Filter Data")
    df_filtered = df.copy()
    
    filter_cols = st.columns(4)
    with filter_cols[0]:
        types = ['All'] + list(df_filtered['Type'].unique())
        selected_type = st.multiselect("Machine Type", options=types, default='All')
        if 'All' not in selected_type:
            df_filtered = df_filtered[df_filtered['Type'].isin(selected_type)]
            
    with filter_cols[1]:
        failure_status = st.selectbox("Failure Status", options=['All', 'Failed', 'Normal'])
        if failure_status == 'Failed':
            df_filtered = df_filtered[df_filtered[TARGET_COL] == 1]
        elif failure_status == 'Normal':
            df_filtered = df_filtered[df_filtered[TARGET_COL] == 0]
            
    with filter_cols[2]:
        min_t, max_t = float(df['Torque [Nm]'].min()), float(df['Torque [Nm]'].max())
        torque_range = st.slider("Torque [Nm] Range", min_t, max_t, (min_t, max_t))
        df_filtered = df_filtered[
            (df_filtered['Torque [Nm]'] >= torque_range[0]) &
            (df_filtered['Torque [Nm]'] <= torque_range[1])
        ]
        
    with filter_cols[3]:
        min_w, max_w = float(df['Tool wear [min]'].min()), float(df['Tool wear [min]'].max())
        wear_range = st.slider("Tool wear [min] Range", min_w, max_w, (min_w, max_w))
        df_filtered = df_filtered[
            (df_filtered['Tool wear [min]'] >= wear_range[0]) &
            (df_filtered['Tool wear [min]'] <= wear_range[1])
        ]

    st.header(f"Filtered Data ({len(df_filtered)} rows)")
    st.dataframe(df_filtered, use_container_width=True, height=400)
    
    @st.cache_data
    def to_excel(df: pd.DataFrame):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Report')
        return output.getvalue()

    excel_data = to_excel(df_filtered)
    st.download_button(
        label="📥 Download Data as Excel", data=excel_data,
        file_name=f"ai4i_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    

def render_model_performance(df: pd.DataFrame, prod_models: dict, perf: dict):
    """
    Renders Page 4: Prediction Results.
    """
    st.title("Model Performance Analysis")
    st.markdown("A deep dive into the classification model's performance on the *entire* dataset.")
    
    if perf is None:
        st.error("Model performance metrics could not be calculated.")
        return
    
    st.header("Model Evaluation Metrics")
    
    # --- KPI Row ---
    report_dict = perf['classification_report_dict']
    f1 = report_dict['weighted avg']['f1-score']
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    auc_score = perf['pr_curve']['auc']
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Weighted F1-Score", f"{f1:.4f}")
    kpi_cols[1].metric("Weighted Precision", f"{precision:.4f}")
    kpi_cols[2].metric("Weighted Recall", f"{recall:.4f}")
    kpi_cols[3].metric("Precision-Recall AUC", f"{auc_score:.4f}")
        
    st.markdown("---")
    
    # --- Visuals Row ---
    st.header("Performance Visuals")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig_cm = plot_confusion_matrix_plotly(
            perf['confusion_matrix'], title="Confusion Matrix (Full Dataset)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with viz_col2:
        fig_pr = plot_pr_curve_plotly(
            perf['pr_curve']['precision'], perf['pr_curve']['recall'],
            perf['pr_curve']['auc'], title="Precision-Recall Curve (Full Dataset)"
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        
    st.markdown("---")
    
    # --- Feature Importance ---
    st.header("Maintenance Insights: Key Failure Drivers")
    fig_imp = plot_feature_importance_plotly(
        perf['feature_importances'], title="Top 15 Feature Importances"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    with st.expander("Show Full Classification Report"):
        st.text(perf['classification_report_str'])

def render_live_prediction(prod_models: dict, df: pd.DataFrame):
    """
    Renders Page 5: Live Prediction Simulator.
    """
    st.title("Live Prediction Simulator")
    st.markdown("""
    Interact with the final, pre-trained `VotingClassifier` model.
    Adjust the sliders and inputs below to match a machine's current
    state and click 'Predict' to get a real-time failure prediction.
    """)
    
    if prod_models is None or df is None:
        st.error("Production model or data could not be loaded. This page is unavailable.")
        return

    model = prod_models['model']
    preprocessor = prod_models['preprocessor']
    
    # Get range for sliders from the original training data
    X_train_raw = df # Use full df for ranges

    st.subheader("Machine Sensor Inputs")
    
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        inputs = {}
        
        with col1:
            st.markdown("#### Machine & Environment")
            inputs['Type'] = st.selectbox("Machine Type", options=X_train_raw['Type'].unique(), index=0)
            inputs['Air temperature [K]'] = st.slider(
                "Air temperature [K]",
                min_value=float(X_train_raw['Air temperature [K]'].min()),
                max_value=float(X_train_raw['Air temperature [K]'].max()),
                value=float(X_train_raw['Air temperature [K]'].mean())
            )
            inputs['Process temperature [K]'] = st.slider(
                "Process temperature [K]",
                min_value=float(X_train_raw['Process temperature [K]'].min()),
                max_value=float(X_train_raw['Process temperature [K]'].max()),
                value=float(X_train_raw['Process temperature [K]'].mean())
            )
            
        with col2:
            st.markdown("#### Operational Parameters")
            inputs['Rotational speed [rpm]'] = st.slider(
                "Rotational speed [rpm]",
                min_value=float(X_train_raw['Rotational speed [rpm]'].min()),
                max_value=float(X_train_raw['Rotational speed [rpm]'].max()),
                value=float(X_train_raw['Rotational speed [rpm]'].mean())
            )
            inputs['Torque [Nm]'] = st.slider(
                "Torque [Nm]",
                min_value=float(X_train_raw['Torque [Nm]'].min()),
                max_value=float(X_train_raw['Torque [Nm]'].max()),
                value=float(X_train_raw['Torque [Nm]'].mean())
            )
            inputs['Tool wear [min]'] = st.slider(
                "Tool wear [min]",
                min_value=float(X_train_raw['Tool wear [min]'].min()),
                max_value=float(X_train_raw['Tool wear [min]'].max()),
                value=float(X_train_raw['Tool wear [min]'].mean())
            )

        submit_button = st.form_submit_button(label="Analyze and Predict Failure")

    # --- Process the prediction ---
    if submit_button:
        with st.spinner("Analyzing parameters and running prediction..."):
            input_df_raw = pd.DataFrame([inputs])
            cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
            num_features = preprocessor.named_transformers_['num'].feature_names_in_
            input_df_ordered = input_df_raw[list(num_features) + list(cat_features)]
            
            try:
                input_processed = preprocessor.transform(input_df_ordered)
                prediction = model.predict(input_processed)[0]
                prediction_proba = model.predict_proba(input_processed)[0]
                
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("🔴 **Prediction: FAILURE LIKELY**", icon="🚨")
                    prob_text = f"{prediction_proba[1]*100:.2f}%"
                    st.markdown("The model predicts a **high probability** of machine failure.")
                else:
                    st.success("🟢 **Prediction: NO FAILURE DETECTED**", icon="✅")
                    prob_text = f"{prediction_proba[1]*100:.2f}%"
                    st.markdown("The model predicts a **low probability** of machine failure.")

                prob_fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prediction_proba[1],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Failure Probability: {prob_text}"},
                    gauge = {
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#dc3545" if prediction == 1 else "#28a745"},
                        'steps': [
                            {'range': [0, 0.5], 'color': '#d4edda'},
                            {'range': [0.5, 1], 'color': '#f8d7da'}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.5}
                    }
                ))
                prob_fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))
                st.plotly_chart(prob_fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

def render_data_explorer(df: pd.DataFrame):
    """
    Renders Page 6: Data & Reports.
    """
    st.title("Data Explorer & Reports")
    st.markdown("Filter, explore, and download the historical machine data.")
    
    if df is None:
        st.error("Data not loaded. Cannot display page.")
        return

    st.header("Filter Data")
    df_filtered = df.copy()
    
    filter_cols = st.columns(4)
    with filter_cols[0]:
        types = ['All'] + list(df_filtered['Type'].unique())
        selected_type = st.multiselect("Machine Type", options=types, default='All')
        if 'All' not in selected_type:
            df_filtered = df_filtered[df_filtered['Type'].isin(selected_type)]
            
    with filter_cols[1]:
        failure_status = st.selectbox("Failure Status", options=['All', 'Failed', 'Normal'])
        if failure_status == 'Failed':
            df_filtered = df_filtered[df_filtered[TARGET_COL] == 1]
        elif failure_status == 'Normal':
            df_filtered = df_filtered[df_filtered[TARGET_COL] == 0]
            
    with filter_cols[2]:
        min_t, max_t = float(df['Torque [Nm]'].min()), float(df['Torque [Nm]'].max())
        torque_range = st.slider("Torque [Nm] Range", min_t, max_t, (min_t, max_t))
        df_filtered = df_filtered[
            (df_filtered['Torque [Nm]'] >= torque_range[0]) &
            (df_filtered['Torque [Nm]'] <= torque_range[1])
        ]
        
    with filter_cols[3]:
        min_w, max_w = float(df['Tool wear [min]'].min()), float(df['Tool wear [min]'].max())
        wear_range = st.slider("Tool wear [min] Range", min_w, max_w, (min_w, max_w))
        df_filtered = df_filtered[
            (df_filtered['Tool wear [min]'] >= wear_range[0]) &
            (df_filtered['Tool wear [min]'] <= wear_range[1])
        ]

    st.header(f"Filtered Data ({len(df_filtered)} rows)")
    st.dataframe(df_filtered, use_container_width=True, height=400)
    
    @st.cache_data
    def to_excel(df: pd.DataFrame):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Report')
        return output.getvalue()

    excel_data = to_excel(df_filtered)
    st.download_button(
        label="📥 Download Data as Excel", data=excel_data,
        file_name=f"ai4i_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
def render_live_prediction_via_api():
    """
    Collects user input and calls the FastAPI endpoint for prediction.
    """
    st.header("🛠️ Live Prediction Interface (API Connected)")
    st.markdown("""
    Adjust the sensor parameters below to simulate a machine state. 
    The app will send this data to the **FastAPI Model Server** to get a prediction.
    """)

    # --- Input Form ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sensor Readings")
        air_temp = st.slider("Air Temperature [K]", 290.0, 310.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature [K]", 300.0, 320.0, 310.0, 0.1)
        torque = st.slider("Torque [Nm]", 0.0, 80.0, 40.0, 0.1)

    with col2:
        st.subheader("Machine Settings")
        rot_speed = st.slider("Rotational Speed [rpm]", 1100, 2900, 1500, 10)
        tool_wear = st.slider("Tool Wear [min]", 0, 300, 0, 1)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

    # --- Prediction Action ---
    if st.button("🚀 Predict Failure Risk", type="primary"):
        
        # 1. Construct Payload
        payload = {
            "Type": machine_type,
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rot_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }

        # 2. Call API
        with st.spinner("Contacting Inference Server..."):
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 3. Process Result
                    prob = result['failure_probability']
                    is_failure = result['prediction'] == 1
                    
                    # 4. Display Result
                    st.divider()
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        if is_failure:
                            st.error(f"## {result['prediction_label']}")
                            st.metric("Failure Probability", f"{prob*100:.2f}%")
                        else:
                            st.success(f"## {result['prediction_label']}")
                            st.metric("Safety Confidence", f"{(1-prob)*100:.2f}%")
                            
                    with c2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Failure Probability"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "red" if is_failure else "green"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50}}))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # 5. Log to Excel
                    log_entry = payload.copy()
                    log_entry['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry['Prediction'] = result['prediction_label']
                    log_entry['Probability'] = prob
                    
                    save_to_excel(log_entry)
                    st.toast("✅ Prediction logged to Excel file!")

                else:
                    st.error(f"Server Error: {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Refused: Is the FastAPI server running?")
                st.info("Run `uvicorn api_server:app --reload` in a separate terminal.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # --- View Logs Section ---
    st.divider()
    with st.expander("📂 View Prediction History"):
        if os.path.exists(LOG_FILE):
            try:
                history_df = pd.read_excel(LOG_FILE)
                st.dataframe(history_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
                
                # Download Button
                with open(LOG_FILE, "rb") as file:
                    st.download_button(
                        label="📥 Download Excel Logs",
                        data=file,
                        file_name="prediction_logs.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Could not read log file: {e}")
        else:
            st.info("No predictions logged yet.")
# ------------------------------------------------------------------------------
# 6. MAIN APPLICATION LOGIC
# ------------------------------------------------------------------------------

def main():
    """
    Main function to run the Streamlit app.
    It controls the sidebar navigation and renders the selected page.
    """
    
    # --- Sidebar Navigation ---
    st.sidebar.markdown("## 🏭 AI4I Maintenance")
    
    # --- Page Options ---
    page_options = {
        "Executive Summary": "SUMMARY",
        "Maintenance Priority List": "PRIORITY", # <-- "WOW" FEATURE
        "Machine Deep Dive": "MACHINE",
        #"Model Performance": "PERFORMANCE",
        "Live Prediction": "PREDICT",
        #"Data Explorer & Reports": "DATA"
    }
    
    selected_page_title = st.sidebar.radio(
        "Navigation",
        options=page_options.keys(),
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data File:** `{DATA_FILE}`")
    st.sidebar.markdown(f"**Model File:** `{MODEL_FILE}`")
    st.sidebar.markdown("---")
    
    is_dark = st.sidebar.toggle("Dark Mode 🌙", value=False, key="dark_mode_toggle")
    
    if is_dark:
        st.markdown("""
            <style>
            /* Main Background */
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #262730;
            }
            /* Headers and Text */
            h1, h2, h3, h4, h5, h6, p, li, span {
                color: #FAFAFA !important;
            }
            /* Metrics */
            [data-testid="stMetricLabel"] {
                color: #A0A0A0 !important;
            }
            [data-testid="stMetricValue"] {
                color: #FAFAFA !important;
            }
            </style>
            """, unsafe_allow_html=True)
    
    # --- Load Data & Models (runs once and is cached) ---
    with st.spinner("Loading dashboard assets..."):
        raw_df = load_data(DATA_FILE)
        prod_models = load_production_model(MODEL_FILE, SCALER_FILE)
        
        # Calculate performance and KPIs once
        perf = get_model_performance(prod_models, raw_df) if prod_models and raw_df is not None else None
        kpis = calculate_business_kpis(raw_df) if raw_df is not None else {}
    
    # --- Page Routing ---
    if selected_page_title == "Executive Summary":
        render_executive_summary(raw_df, prod_models, perf, kpis)
        
    elif selected_page_title == "Maintenance Priority List":
        render_priority_list(raw_df, prod_models, perf)
        
    elif selected_page_title == "Machine Deep Dive":
        render_machine_deep_dive(raw_df)
        
    #elif selected_page_title == "Model Performance":
      # render_model_performance(raw_df, prod_models, perf)
        
    #elif selected_page_title == "Live Prediction":
        #render_live_prediction(prod_models, raw_df)

    elif selected_page_title == "Live Prediction":
        render_live_prediction_via_api()
        
    #elif selected_page_title == "Data Explorer & Reports":
        #render_data_explorer(raw_df)

if __name__ == "__main__":
    main()

# End of file. Total lines: ~1270+