import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    color: white;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    text-align: center;
    border: 1px solid #334155;
}
.metric-title {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: white;
}
.alert-box {
    padding: 12px 16px;
    border-radius: 12px;
    font-weight: 600;
    color: white;
    margin-bottom: 10px;
}
.critical-box {
    background-color: #b91c1c;
}
.warning-box {
    background-color: #d97706;
}
.healthy-box {
    background-color: #15803d;
}
.small-text {
    color: #cbd5e1;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def normalize_alert_columns(df):
    df = df.copy()

    if 'Final_Alert' in df.columns:
        df['Display_Alert'] = df['Final_Alert']
    elif 'Adaptive_Alert' in df.columns:
        df['Display_Alert'] = df['Adaptive_Alert']
    elif 'Alert' in df.columns:
        df['Display_Alert'] = df['Alert']
    else:
        df['Display_Alert'] = 'UNKNOWN'

    if 'Final_Message' in df.columns:
        df['Display_Message'] = df['Final_Message']
    elif 'Adaptive_Message' in df.columns:
        df['Display_Message'] = df['Adaptive_Message']
    elif 'Recommendation' in df.columns:
        df['Display_Message'] = df['Recommendation']
    else:
        df['Display_Message'] = 'No recommendation available'

    return df

def alert_icon(alert):
    alert = str(alert).upper()
    if alert == "CRITICAL":
        return "🔴"
    elif alert == "WARNING":
        return "🟠"
    elif alert == "HEALTHY":
        return "🟢"
    return "⚪"

def alert_html(alert):
    alert = str(alert).upper()
    if alert == "CRITICAL":
        return '<div class="alert-box critical-box">🔴 CRITICAL - Immediate maintenance required</div>'
    elif alert == "WARNING":
        return '<div class="alert-box warning-box">🟠 WARNING - Maintenance should be scheduled soon</div>'
    elif alert == "HEALTHY":
        return '<div class="alert-box healthy-box">🟢 HEALTHY - Machine is operating normally</div>'
    return '<div class="alert-box">⚪ UNKNOWN</div>'

# -----------------------------
# Load Data
# -----------------------------
results_df = load_csv_safe("final_test_alert_results.csv")
model_comparison_df = load_csv_safe("model_comparison.csv")
ensemble_metrics_df = load_csv_safe("ensemble_metrics.csv")
gru_loss_df = load_csv_safe("gru_loss_history.csv")
lstm_loss_df = load_csv_safe("lstm_loss_history.csv")

if results_df is None:
    st.error("final_test_alert_results.csv not found. Please keep it in the same folder as dashboard.py.")
    st.stop()

results_df = normalize_alert_columns(results_df)

for col in ['GRU_RUL', 'LSTM_RUL', 'Ensemble_RUL']:
    if col in results_df.columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

# -----------------------------
# Header
# -----------------------------
st.title("⚙️ Predictive Maintenance Dashboard")
st.markdown('<p class="small-text">RUL prediction, model comparison, alert monitoring, and engine-level analysis</p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Dashboard Filters")

alert_options = sorted(results_df['Display_Alert'].dropna().unique().tolist())
selected_alerts = st.sidebar.multiselect(
    "Filter by Alert Level",
    options=alert_options,
    default=alert_options
)

search_engine = st.sidebar.text_input("Search Engine UID")

min_rul = float(results_df['Ensemble_RUL'].min())
max_rul = float(results_df['Ensemble_RUL'].max())

rul_range = st.sidebar.slider(
    "Filter by Ensemble RUL",
    min_value=min_rul,
    max_value=max_rul,
    value=(min_rul, max_rul)
)

filtered_df = results_df.copy()
filtered_df = filtered_df[filtered_df['Display_Alert'].isin(selected_alerts)]
filtered_df = filtered_df[
    (filtered_df['Ensemble_RUL'] >= rul_range[0]) &
    (filtered_df['Ensemble_RUL'] <= rul_range[1])
]

if search_engine:
    filtered_df = filtered_df[
        filtered_df['engine_uid'].astype(str).str.contains(search_engine, case=False, na=False)
    ]

# -----------------------------
# KPI Cards
# -----------------------------
total_engines = len(results_df)
critical_count = int((results_df['Display_Alert'] == 'CRITICAL').sum())
warning_count = int((results_df['Display_Alert'] == 'WARNING').sum())
healthy_count = int((results_df['Display_Alert'] == 'HEALTHY').sum())
avg_rul = float(results_df['Ensemble_RUL'].mean())

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Engines</div>
        <div class="metric-value">{total_engines}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Critical</div>
        <div class="metric-value">{critical_count}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Warning</div>
        <div class="metric-value">{warning_count}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Healthy</div>
        <div class="metric-value">{healthy_count}</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Ensemble RUL</div>
        <div class="metric-value">{avg_rul:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Engine Explorer",
    "📈 Model Performance",
    "🚨 Priority Engines"
])

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Alert Distribution")
        alert_counts = results_df['Display_Alert'].value_counts().reindex(
            ['CRITICAL', 'WARNING', 'HEALTHY']
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(alert_counts.index, alert_counts.values)
        ax.set_title("Maintenance Alert Distribution")
        ax.set_xlabel("Alert Level")
        ax.set_ylabel("Number of Engines")
        ax.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Predicted RUL Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(results_df['Ensemble_RUL'].dropna(), bins=40)
        ax.set_title("Distribution of Ensemble Predicted RUL")
        ax.set_xlabel("Predicted RUL")
        ax.set_ylabel("Number of Engines")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.subheader("Filtered Engine Results")
    display_df = filtered_df.copy()
    display_df.insert(0, "Status", display_df['Display_Alert'].map(alert_icon))

    st.dataframe(
        display_df[[
            'Status', 'engine_uid', 'GRU_RUL', 'LSTM_RUL',
            'Ensemble_RUL', 'Display_Alert', 'Display_Message'
        ]],
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# Tab 2: Engine Explorer
# -----------------------------
with tab2:
    st.subheader("Engine-wise Prediction Explorer")

    engine_list = filtered_df['engine_uid'].dropna().unique().tolist()

    if not engine_list:
        st.info("No engines match current filters.")
    else:
        selected_engine = st.selectbox("Select Engine UID", engine_list)
        engine_row = filtered_df[filtered_df['engine_uid'] == selected_engine].iloc[0]

        st.markdown(alert_html(engine_row['Display_Alert']), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("GRU Predicted RUL", f"{engine_row['GRU_RUL']:.2f}")
        c2.metric("LSTM Predicted RUL", f"{engine_row['LSTM_RUL']:.2f}")
        c3.metric("Ensemble Predicted RUL", f"{engine_row['Ensemble_RUL']:.2f}")

        st.markdown(f"**Recommendation:** {engine_row['Display_Message']}")

        compare_df = pd.DataFrame({
            'Model': ['GRU', 'LSTM', 'Ensemble'],
            'Predicted RUL': [
                engine_row['GRU_RUL'],
                engine_row['LSTM_RUL'],
                engine_row['Ensemble_RUL']
            ]
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(compare_df['Model'], compare_df['Predicted RUL'])
        ax.set_title(f"Model-wise Prediction for {selected_engine}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Predicted RUL")
        ax.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig, use_container_width=True)

# -----------------------------
# Tab 3: Model Performance
# -----------------------------
with tab3:
    st.subheader("Model Comparison")

    if model_comparison_df is not None:
        st.dataframe(model_comparison_df, use_container_width=True, hide_index=True)

        metric_choice = st.selectbox("Select Metric", ['MAE', 'RMSE', 'R2 Score'])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(model_comparison_df['Model'], model_comparison_df[metric_choice])
        ax.set_title(f"{metric_choice} Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_choice)
        ax.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("model_comparison.csv not found.")

    if ensemble_metrics_df is not None:
        st.subheader("Ensemble Metrics")
        st.dataframe(ensemble_metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Training Curves")
    if gru_loss_df is not None or lstm_loss_df is not None:
        fig, ax = plt.subplots(figsize=(10, 5))

        if gru_loss_df is not None:
            if 'GRU_Train_Loss' in gru_loss_df.columns:
                ax.plot(gru_loss_df['Epoch'], gru_loss_df['GRU_Train_Loss'], label='GRU Train Loss')
            if 'GRU_Val_Loss' in gru_loss_df.columns:
                ax.plot(gru_loss_df['Epoch'], gru_loss_df['GRU_Val_Loss'], label='GRU Val Loss')

        if lstm_loss_df is not None:
            if 'LSTM_Train_Loss' in lstm_loss_df.columns:
                ax.plot(lstm_loss_df['Epoch'], lstm_loss_df['LSTM_Train_Loss'], label='LSTM Train Loss')
            if 'LSTM_Val_Loss' in lstm_loss_df.columns:
                ax.plot(lstm_loss_df['Epoch'], lstm_loss_df['LSTM_Val_Loss'], label='LSTM Val Loss')

        ax.set_title("Training and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("Loss history CSV files not found.")

# -----------------------------
# Tab 4: Priority Engines
# -----------------------------
with tab4:
    st.subheader("Most Critical Engines")

    top_n = st.slider("Select number of top risky engines", 5, 50, 10, 5)
    top_df = results_df.sort_values("Ensemble_RUL", ascending=True).head(top_n).copy()
    top_df.insert(0, "Status", top_df['Display_Alert'].map(alert_icon))

    st.dataframe(
        top_df[[
            'Status', 'engine_uid', 'GRU_RUL', 'LSTM_RUL',
            'Ensemble_RUL', 'Display_Alert', 'Display_Message'
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Action Summary")
    st.markdown("""
- **CRITICAL** → Immediate maintenance required  
- **WARNING** → Maintenance should be scheduled soon  
- **HEALTHY** → Machine is operating normally  
""")

st.markdown("---")
st.caption("Run with: streamlit run dashboard.py")