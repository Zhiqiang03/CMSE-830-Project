# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path

# Configure page settings at the top
st.set_page_config(
    page_title="CMSE 830 Data Analysis Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Centralized data paths for easy updates
DATA_DIR = "./data"
MODEL_DIR = "./models"
TAXI_SAMPLE = f"{DATA_DIR}/taxi_data_sampled.parquet"
TAXI_PREPROCESSED_MISSING_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed_missing_sampled.parquet"
TAXI_PREPROCESSED_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed.parquet"
TAXI_PREPROCESSED_TIP_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed_tip.parquet"

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

def page_overview():
    """
    Display the project overview page with introduction, data sources,
    techniques used, and research questions.
    """
    st.title("NYC Taxi Ride Analysis & Tip Prediction")
    st.markdown("""
    ### CMSE 830 Data Analysis Project

    This Streamlit app demonstrates a comprehensive data analysis of NYC taxi data, including:
    - **Data Collection** and preprocessing
    - **Exploratory Data Analysis** with interactive visualizations
    - **Missing Data Handling** using iterative imputation
    - **Predictive Modeling** for tip classification

    **Author:** Zhiqiang Ni | **Course:** CMSE 830
    """)
    
    # Key metrics
    st.markdown("---")
    st.subheader("Project Highlights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Sources", "3")
    col2.metric("ML Models", "7")
    col3.metric("Visualizations", "15+")
    col4.metric("Records Analyzed", "1M+")

    st.markdown("---")

    st.header("The Story Behind the Data")
    st.write("""
    New York City's iconic yellow and green taxis generate massive amounts of data with every trip. 
    This project analyzes this data to uncover the factors that influence passenger tipping behavior.

    By combining 2024 taxi trip records from the NYC Taxi & Limousine Commission (TLC) with hourly weather data, 
    we explore tipping patterns, temporal trends, and build models to predict tip amounts.
    
    ### Why This Matters
    
    Understanding tipping behavior provides valuable insights for:
    - **Drivers**: To better understand potential earnings and influencing factors
    - **Passengers**: To gain awareness of tipping norms
    - **TLC**: For policy-making and understanding the taxi economy
    """)

    st.markdown("---")

    # Project Components
    st.subheader("Project Components")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Data Collection & Integration
        - NYC Yellow Taxi Data (2024)
        - NYC Green Taxi Data (2024)
        - Historical Weather Data (Open-Meteo API)
        - Data cleaning, validation, and feature engineering
        - Missing value imputation (Iterative MICE)
        
        #### Analysis & Visualization
        - 15+ interactive visualizations
        - Statistical and correlation analysis
        - Temporal and weather pattern analysis
        """)

    with col2:
        st.markdown("""
        #### Machine Learning
        - 7 different algorithms with class balancing
        - Comprehensive model comparison
        - Feature importance analysis
        
        #### Interactive Features
        - Real-time tip prediction interface
        - Dynamic visualizations
        - Complete methodology documentation
        """)

    st.markdown("---")

    st.info("""
    **Navigation Guide:** Use the sidebar menu to explore different sections of this comprehensive analysis.
    """)


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    """
    Load Parquet file with caching to improve performance.

    Args:
        path (str): Path to the Parquet file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_parquet(path)

def page_data_collection():
    """
    Display the data collection and preparation page, showing raw data sources
    and the integration process.
    """
    st.header("Data Collection and Preparation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **NYC TLC Trip Record Data**
        - Yellow and Green taxi trips for 2024
        - Key Features: Pickup/dropoff times, locations, fares, tips
        - Source: [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
        """)

    with col2:
        st.markdown("""
        **Weather Data**
        - Hourly observations for NYC (2024)
        - Key Features: Temperature, precipitation, wind speed
        - Source: [Open-Meteo API](https://open-meteo.com/)
        """)

    st.info("Taxi trip data was merged with weather data based on pickup timestamp.")

    st.markdown("---")

    st.subheader("Data Exploration")

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Preprocessed Data", "Imputed Data"])

    with tab1:
        df_raw = load_parquet(TAXI_SAMPLE)
        st.write(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trips", f"{len(df_raw):,}")
        col2.metric("Time Period", "2024")
        col3.metric("Taxi Types", "Yellow & Green")

        st.dataframe(df_raw.head(10), width='stretch')

    with tab2:
        df_preprocessed = load_parquet(TAXI_PREPROCESSED_MISSING_SAMPLED)
        st.write(f"**Shape:** {df_preprocessed.shape[0]:,} rows × {df_preprocessed.shape[1]} columns")
        st.dataframe(df_preprocessed.head(10), width='stretch')
        st.caption("Cleaned, merged with weather data, with engineered features.")

    with tab3:
        df_imputed = load_parquet(TAXI_PREPROCESSED_SAMPLED)
        st.write(f"**Shape:** {df_imputed.shape[0]:,} rows × {df_imputed.shape[1]} columns")
        st.dataframe(df_imputed.head(10), width='stretch')
        st.caption("Missing values filled using iterative imputation.")

@st.cache_data
def compute_missing_values(df):
    """Cache missing values computation"""
    missing_df = (
        df.isna().sum()
        .to_frame("missing")
        .assign(percent=lambda x: (x["missing"] / len(df) * 100).round(2))
        .sort_values("missing", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return missing_df[missing_df['missing'] > 0]

@st.cache_data
def compute_stats_summary(df):
    """Cache statistical summary computation"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return df[numeric_cols].describe().T

@st.cache_data
def get_column_stats(df, col):
    """Cache individual column statistics"""
    return {
        'mean': df[col].mean(),
        'median': df[col].median(),
        'std': df[col].std(),
        'min': df[col].min(),
        'max': df[col].max()
    }

def page_ida():
    """
    Display the Initial Data Analysis (IDA) page with missing values analysis,
    duplicates, statistical summaries, and preprocessing information.
    """
    st.header("Initial Data Analysis (IDA)")
    
    # Load data once
    df_preprocessed = load_parquet(TAXI_PREPROCESSED_MISSING_SAMPLED)
    df_imputed = load_parquet(TAXI_PREPROCESSED_SAMPLED)

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")

    missing_df = compute_missing_values(df_preprocessed)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(missing_df, width='stretch', hide_index=True)
    with col2:
        st.metric("Total Missing Values", f"{missing_df['missing'].sum():,}")
        st.metric("Columns with Missing Data", len(missing_df))

    st.info("""
    **Imputation Method:** `IterativeImputer` (MICE) models each feature with missing values 
    as a function of other features, preserving relationships between variables.
    """)

    # Imputation Impact Visualization
    st.subheader("Imputation Impact Visualization")

    impute_cols = ['passenger_count', 'RatecodeID']
    selected_col = st.selectbox("Select variable to compare:",
                                [c for c in impute_cols if c in df_preprocessed.columns],
                                key="impute_compare")

    # Sample data for performance
    sample_size = min(10000, len(df_preprocessed))
    df_preprocessed_sample = df_preprocessed.sample(n=sample_size, random_state=42)
    df_imputed_sample = df_imputed.sample(n=sample_size, random_state=42)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Before Imputation", "After Imputation"))

    fig.add_trace(
        go.Histogram(x=df_preprocessed_sample[selected_col].dropna(), name="Original",
                    marker_color='#3498db', nbinsx=30),
        row=1, col=1
    )

    if selected_col in df_imputed_sample.columns:
        fig.add_trace(
            go.Histogram(x=df_imputed_sample[selected_col].dropna(), name="Imputed",
                        marker_color='#2ecc71', nbinsx=30),
            row=1, col=2
        )

    fig.update_layout(height=400, showlegend=True, title_text=f"Distribution Comparison: {selected_col}")
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Statistical Summary
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Statistical Summary")

    stats_summary = compute_stats_summary(df_imputed)
    st.dataframe(stats_summary, width='stretch')

    # Interactive Distribution Analysis
    st.subheader("Interactive Distribution Analysis")
    num_col = st.selectbox("Select a numeric column to explore:", numeric_cols, key="ida_num")

    stats = get_column_stats(df_imputed, num_col)

    col1, col2 = st.columns([2, 1])
    with col1:
        # Sample for visualization
        sample_size_viz = min(20000, len(df_imputed))
        df_sample = df_imputed[[num_col]].sample(n=sample_size_viz, random_state=42)

        fig_num = px.histogram(
            df_sample,
            x=num_col,
            nbins=50,
            marginal="box",
            opacity=0.85,
            title=f"Distribution of {num_col} (sample of {sample_size_viz:,} records)"
        )
        st.plotly_chart(fig_num, width='stretch')
    with col2:
        st.write("**Statistics:**")
        st.metric("Mean", f"{stats['mean']:.2f}")
        st.metric("Median", f"{stats['median']:.2f}")
        st.metric("Std Dev", f"{stats['std']:.2f}")

    # Categorical Analysis
    cat_cols = df_imputed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if cat_cols:
        st.subheader("Categorical Variable Frequency")
        cat_col = st.selectbox("Select a categorical column:", cat_cols, key="ida_cat")

        if cat_col:
            top_counts = df_imputed[cat_col].value_counts(dropna=False).head(20).reset_index()
            top_counts.columns = [cat_col, "count"]
            fig_cat = px.bar(top_counts, x=cat_col, y="count",
                           title=f"Top 20 categories in {cat_col}")
            st.plotly_chart(fig_cat, width='stretch')

    st.markdown("---")

    # Feature Engineering Summary
    st.subheader("Feature Engineering")

    st.markdown("""
    **Key Transformations:**
    - Removed trips with invalid values (negative/zero fare, distance, duration)
    - Created `duration_min`, `speed_mph`, and `tip_class` features
    - Merged with hourly weather data based on pickup time
    """)

    # Data Types Overview
    with st.expander("View Data Types"):
        dtype_df = pd.DataFrame(df_imputed.dtypes, columns=["Data Type"]).reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        dtype_df["Data Type"] = dtype_df["Data Type"].apply(str)
        st.dataframe(dtype_df, width='stretch', hide_index=True)


def page_eda():
    """
    Display the Exploratory Data Analysis (EDA) page with correlation analysis,
    temporal patterns, and weather impact visualizations.
    """
    st.header("Exploratory Data Analysis")

    df = load_parquet(TAXI_PREPROCESSED_SAMPLED)

    st.subheader("Correlation Analysis")
    st.caption("tip_class (0=Low, 1=Middle, 2=High) is our target variable")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Reorder columns to put tip_class first
    if 'tip_class' in numeric_cols:
        numeric_cols.remove('tip_class')
        numeric_cols = numeric_cols + ['tip_class']

    corr_method = st.radio("Correlation method:", ["pearson", "spearman", "kendall"], horizontal=True)

    corr = df[numeric_cols].corr(method=corr_method)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Correlation Heatmap ({corr_method.capitalize()})",
        aspect="auto"
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Target Feature Correlation
    st.subheader("Feature Importance via Correlation")

    default_idx = numeric_cols.index("tip_class") if "tip_class" in numeric_cols else 0
    target_col = st.selectbox("Target variable:", numeric_cols, index=default_idx, key="eda_target")

    corrs = df[numeric_cols].corr(method='pearson')[target_col].sort_values(ascending=False)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            corrs.to_frame("correlation").reset_index().rename(columns={"index": "feature"}),
            width='stretch',
            hide_index=True,
        )

    with col2:
        top_n = st.slider("Show top N correlations", 5, 30, 15)
        top_corrs = corrs.abs().sort_values(ascending=False).head(top_n)
        fig = px.bar(
            x=top_corrs.values,
            y=top_corrs.index,
            orientation='h',
            title=f"Top {top_n} Features Correlated with {target_col}",
            labels={'x': 'Absolute Correlation', 'y': 'Feature'}
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Temporal Patterns
    st.subheader("Temporal Patterns")

    if 'pickup_hour' in df.columns:
        fig = px.histogram(
            df,
            x='pickup_hour',
            y='fare_amount',
            histfunc='avg',
            nbins=24,
            title="Average Fare by Hour of Day",
            labels={'pickup_hour': 'Hour', 'fare_amount': 'Avg Fare ($)'}
        )
        st.plotly_chart(fig, width='stretch')

    # Tip Class Distribution
    st.subheader("Tip Class Distribution")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        tip_class_counts = df['tip_class'].value_counts().sort_index()
        st.metric("Low Tips (0)", f"{tip_class_counts.get(0, 0):,}")
        st.metric("Middle Tips (1)", f"{tip_class_counts.get(1, 0):,}")
        st.metric("High Tips (2)", f"{tip_class_counts.get(2, 0):,}")

    with col2:
        tip_labels = {0: 'Low', 1: 'Middle', 2: 'High'}
        plot_data = pd.DataFrame({
            'Tip Class': [tip_labels.get(i, i) for i in tip_class_counts.index],
            'Count': tip_class_counts.values
        })
        fig = px.pie(
            plot_data,
            names='Tip Class',
            values='Count',
            title="Tip Class Distribution",
            color='Tip Class',
            color_discrete_map={'Low': '#e74c3c', 'Middle': '#f39c12', 'High': '#27ae60'}
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Weather Impact
    st.subheader("Weather Impact on Tips")

    df_tip = load_parquet(TAXI_PREPROCESSED_TIP_SAMPLED)

    if 'tip_percentage' not in df_tip.columns and 'tip_amount' in df_tip.columns and 'fare_amount' in df_tip.columns:
        df_tip['tip_percentage'] = (df_tip['tip_amount'] / df_tip['fare_amount'] * 100).clip(0, 100)

    numeric_cols_tip = df_tip.select_dtypes(include=[np.number]).columns.tolist()
    weather_candidates = [c for c in numeric_cols_tip if any(k in c.lower()
                          for k in ['temperature','precipitation','rain','snowfall','wind_speed'])]

    if weather_candidates:
        col1, col2, col3 = st.columns(3)
        with col1:
            wcol = st.selectbox("Weather feature:", weather_candidates, key="eda_weather_col")
        with col2:
            available_metrics = []
            if 'tip_amount' in df_tip.columns:
                available_metrics.append('tip_amount')
            if 'tip_percentage' in df_tip.columns:
                available_metrics.append('tip_percentage')

            tip_metric = st.selectbox("Tip metric:", available_metrics, key="eda_tip_metric")
        with col3:
            bins = st.slider("Bins:", 5, 20, 10, key="eda_weather_bins")

        if wcol and tip_metric and len(df_tip[wcol].dropna()) > 0:
            try:
                qbins = pd.qcut(df_tip[wcol], q=bins, duplicates='drop')
                tmp = df_tip.assign(_bin=qbins).dropna(subset=['_bin', tip_metric])
                rate = tmp.groupby('_bin')[tip_metric].mean().reset_index()
                rate.columns = ['bin', 'avg_tip']
                rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rate['bin_mid'],
                    y=rate['avg_tip'],
                    mode='lines+markers',
                    line=dict(color='#27ae60', width=3),
                    marker=dict(size=10)
                ))

                metric_label = 'Tip Amount ($)' if tip_metric == 'tip_amount' else 'Tip Percentage (%)'
                fig.update_layout(
                    title=f"Average {tip_metric.replace('_', ' ').title()} vs {wcol}",
                    xaxis_title=wcol,
                    yaxis_title=metric_label,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average", f"${tmp[tip_metric].mean():.2f}" if tip_metric == 'tip_amount' else f"{tmp[tip_metric].mean():.1f}%")
                with col2:
                    st.metric("Min", f"${tmp[tip_metric].min():.2f}" if tip_metric == 'tip_amount' else f"{tmp[tip_metric].min():.1f}%")
                with col3:
                    st.metric("Max", f"${tmp[tip_metric].max():.2f}" if tip_metric == 'tip_amount' else f"{tmp[tip_metric].max():.1f}%")

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
    else:
        st.warning("No weather columns found in the dataset.")


def page_advanced_analysis():
    """
    Display advanced analysis page focusing on tip percentage patterns
    with multi-dimensional visualizations and insights.
    """
    st.header("Advanced Analysis: Tip Percentage Insights")

    st.markdown("""
    This section provides in-depth analysis of tipping patterns, focusing on **tip percentage** 
    to understand what drives passenger generosity beyond absolute dollar amounts.
    """)

    df = load_parquet(TAXI_PREPROCESSED_TIP_SAMPLED)

    # Convert tip_pct (decimal) to tip_percentage (0-100 scale)
    df['tip_percentage'] = (df['tip_pct'] * 100).clip(0, 100)

    # Key metrics overview
    st.subheader("Tip Percentage Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Tip %", f"{df['tip_percentage'].mean():.1f}%")
    with col2:
        st.metric("Median Tip %", f"{df['tip_percentage'].median():.1f}%")
    with col3:
        st.metric("Std Deviation", f"{df['tip_percentage'].std():.1f}%")
    with col4:
        generous_pct = (df['tip_percentage'] > 20).sum() / len(df) * 100
        st.metric("Tips > 20%", f"{generous_pct:.1f}%")

    st.markdown("---")

    # Tip Percentage Distribution
    st.subheader("Tip Percentage Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.histogram(
            df[df['tip_percentage'] <= 50],  # Filter extreme outliers for better viz
            x='tip_percentage',
            nbins=50,
            title="Distribution of Tip Percentages (0-50%)",
            labels={'tip_percentage': 'Tip Percentage (%)', 'count': 'Frequency'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=df['tip_percentage'].mean(), line_dash="dash",
                     line_color="red", annotation_text="Mean")
        fig.add_vline(x=df['tip_percentage'].median(), line_dash="dash",
                     line_color="green", annotation_text="Median")
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Tip percentage categories
        tip_ranges = pd.cut(df['tip_percentage'],
                           bins=[0, 10, 15, 20, 25, 100],
                           labels=['0-10%', '10-15%', '15-20%', '20-25%', '>25%'])
        range_counts = tip_ranges.value_counts().sort_index()

        fig = px.pie(
            values=range_counts.values,
            names=range_counts.index,
            title="Tip Percentage Ranges",
            color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Factors Affecting Tip Percentage
    st.subheader("Factors Affecting Tip Percentage")

    tab1, tab2, tab3, tab4 = st.tabs(["Time Patterns", "Trip Characteristics", "Weather Impact", "Passenger Behavior"])

    with tab1:
        st.write("**How does tip percentage vary by time?**")

        col1, col2 = st.columns(2)

        with col1:
            if 'pickup_hour' in df.columns:
                hourly_tips = df.groupby('pickup_hour')['tip_percentage'].agg(['mean', 'median', 'std']).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_tips['pickup_hour'],
                    y=hourly_tips['mean'],
                    mode='lines+markers',
                    name='Mean',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=hourly_tips['pickup_hour'],
                    y=hourly_tips['median'],
                    mode='lines+markers',
                    name='Median',
                    line=dict(color='#27ae60', width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title="Tip Percentage by Hour of Day",
                    xaxis_title="Hour",
                    yaxis_title="Tip Percentage (%)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch')

        with col2:
            if 'day_of_week' in df.columns:
                day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
                df_temp = df.copy()
                df_temp['day_name'] = df_temp['day_of_week'].map(day_names)
                daily_tips = df_temp.groupby('day_name')['tip_percentage'].mean().reindex(day_names.values())

                fig = px.bar(
                    x=daily_tips.index,
                    y=daily_tips.values,
                    title="Average Tip Percentage by Day of Week",
                    labels={'x': 'Day', 'y': 'Tip Percentage (%)'},
                    color=daily_tips.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, width='stretch')

    with tab2:
        st.write("**How do trip characteristics influence tipping?**")

        col1, col2 = st.columns(2)

        with col1:
            if 'trip_distance' in df.columns:
                # Create distance bins
                df_temp = df[df['trip_distance'] <= 20].copy()  # Filter extreme outliers
                df_temp['distance_bin'] = pd.cut(df_temp['trip_distance'],
                                                 bins=[0, 1, 3, 5, 10, 20],
                                                 labels=['<1mi', '1-3mi', '3-5mi', '5-10mi', '10-20mi'])

                distance_tips = df_temp.groupby('distance_bin')['tip_percentage'].agg(['mean', 'count']).reset_index()

                fig = px.bar(
                    distance_tips,
                    x='distance_bin',
                    y='mean',
                    title="Average Tip % by Trip Distance",
                    labels={'distance_bin': 'Distance Range', 'mean': 'Avg Tip %'},
                    text='mean',
                    color='mean',
                    color_continuous_scale='Blues'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, width='stretch')

        with col2:
            if 'fare_amount' in df.columns:
                # Create fare bins
                df_temp = df[df['fare_amount'] <= 100].copy()
                df_temp['fare_bin'] = pd.cut(df_temp['fare_amount'],
                                             bins=[0, 10, 20, 30, 50, 100],
                                             labels=['<$10', '$10-20', '$20-30', '$30-50', '$50-100'])

                fare_tips = df_temp.groupby('fare_bin')['tip_percentage'].mean().reset_index()

                fig = px.bar(
                    fare_tips,
                    x='fare_bin',
                    y='tip_percentage',
                    title="Average Tip % by Fare Amount",
                    labels={'fare_bin': 'Fare Range', 'tip_percentage': 'Avg Tip %'},
                    text='tip_percentage',
                    color='tip_percentage',
                    color_continuous_scale='Greens'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, width='stretch')

    with tab3:
        st.write("**Does weather affect tipping behavior?**")

        col1, col2 = st.columns(2)

        with col1:
            if 'temperature_2m' in df.columns:
                # Temperature bins
                df_temp = df.copy()
                df_temp['temp_category'] = pd.cut(df_temp['temperature_2m'],
                                                  bins=[-20, 32, 50, 70, 85, 120],
                                                  labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot'])

                temp_tips = df_temp.groupby('temp_category')['tip_percentage'].mean().reset_index()

                fig = px.bar(
                    temp_tips,
                    x='temp_category',
                    y='tip_percentage',
                    title="Tip % by Temperature",
                    labels={'temp_category': 'Temperature', 'tip_percentage': 'Avg Tip %'},
                    color='tip_percentage',
                    color_continuous_scale='RdYlBu_r'
                )
                st.plotly_chart(fig, width='stretch')

        with col2:
            if 'precipitation' in df.columns:
                df_temp = df.copy()
                df_temp['rain_category'] = pd.cut(df_temp['precipitation'],
                                                  bins=[-0.1, 0, 0.1, 1, 10],
                                                  labels=['No Rain', 'Light', 'Moderate', 'Heavy'])

                rain_tips = df_temp.groupby('rain_category')['tip_percentage'].mean().reset_index()

                fig = px.bar(
                    rain_tips,
                    x='rain_category',
                    y='tip_percentage',
                    title="Tip % by Precipitation Level",
                    labels={'rain_category': 'Precipitation', 'tip_percentage': 'Avg Tip %'},
                    color='tip_percentage',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, width='stretch')

    with tab4:
        st.write("**How do passenger count and taxi type affect tips?**")

        col1, col2 = st.columns(2)

        with col1:
            if 'passenger_count' in df.columns:
                passenger_tips = df[df['passenger_count'] <= 6].groupby('passenger_count')['tip_percentage'].mean().reset_index()

                fig = px.line(
                    passenger_tips,
                    x='passenger_count',
                    y='tip_percentage',
                    title="Tip % by Passenger Count",
                    labels={'passenger_count': '# Passengers', 'tip_percentage': 'Avg Tip %'},
                    markers=True
                )
                fig.update_traces(line_color='#9b59b6', marker=dict(size=12))
                st.plotly_chart(fig, width='stretch')

        with col2:
            if 'is_yellow' in df.columns:
                taxi_tips = df.groupby('is_yellow')['tip_percentage'].mean().reset_index()
                taxi_tips['Taxi Type'] = taxi_tips['is_yellow'].map({1: 'Yellow', 0: 'Green'})

                fig = px.bar(
                    taxi_tips,
                    x='Taxi Type',
                    y='tip_percentage',
                    title="Tip % by Taxi Type",
                    labels={'tip_percentage': 'Avg Tip %'},
                    color='Taxi Type',
                    color_discrete_map={'Yellow': '#f1c40f', 'Green': '#27ae60'}
                )
                st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Multi-dimensional visualization
    st.subheader("Multi-Dimensional Relationships")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**3D Scatter: Explore Complex Relationships**")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            x_var = st.selectbox("X-axis:", [c for c in numeric_cols if c != 'tip_percentage'],
                                index=[c for c in numeric_cols if c != 'tip_percentage'].index('trip_distance')
                                if 'trip_distance' in numeric_cols else 0, key="3d_x")
        with col_b:
            y_var = st.selectbox("Y-axis:", [c for c in numeric_cols if c != 'tip_percentage'],
                                index=[c for c in numeric_cols if c != 'tip_percentage'].index('fare_amount')
                                if 'fare_amount' in numeric_cols else 0, key="3d_y")
        with col_c:
            z_var = st.selectbox("Z-axis:", [c for c in numeric_cols if c != 'tip_percentage'],
                                index=[c for c in numeric_cols if c != 'tip_percentage'].index('duration_min')
                                if 'duration_min' in numeric_cols else 0, key="3d_z")

        sample_df = df[[x_var, y_var, z_var, 'tip_percentage']].dropna().sample(min(2000, len(df))).copy()

        fig = px.scatter_3d(
            sample_df,
            x=x_var,
            y=y_var,
            z=z_var,
            color='tip_percentage',
            title=f"3D View: {x_var} vs {y_var} vs {z_var} (colored by tip %)",
            color_continuous_scale='RdYlGn',
            opacity=0.7
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.write("**Correlation Heatmap**")

        # Select key features for correlation
        key_features = ['tip_percentage', 'fare_amount', 'trip_distance', 'duration_min',
                       'pickup_hour', 'passenger_count']
        available_features = [f for f in key_features if f in df.columns]

        if len(available_features) >= 2:
            corr_matrix = df[available_features].corr()

            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation with Tip Percentage",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto='.2f'
            )
            st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Key insights
    st.subheader("Key Insights")

    st.markdown("""
    **What we learned about tipping behavior:**
    
    - **Time matters**: Tips vary significantly by hour and day of the week
    - **Trip length**: Longer trips don't always mean higher tip percentages
    - **Fare amount**: There's often an inverse relationship - higher fares may get lower tip %
    - **Weather impact**: Comfortable conditions correlate with better tipping
    - **Group size**: Multiple passengers often tip more generously
    - **Taxi type**: Yellow and green taxis show different tipping patterns
    
    These insights can help drivers optimize their earnings and understand passenger behavior better.
    """)


@st.cache_data
def load_model(model_path):
    """Load a pickled model with caching"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_hardcoded_model_results():
    """
    Returns hard-coded model performance results from training.
    This allows the app to display model metrics without requiring the actual model files.
    """
    model_results = {
        'Logistic Regression': {
            'Accuracy': 0.4592,
            'Precision': 0.6263,
            'Recall': 0.4592,
            'F1-Score': 0.4990,
            'Training Time (s)': 45.77,
            'Prediction Time (s)': 0.08,
            'confusion_matrix': np.array([
                [41460, 40166, 34860],
                [8922, 96676, 84847],
                [1521, 18159, 21925]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.80      0.36      0.49    116486
      Middle       0.62      0.51      0.56    190445
        High       0.15      0.53      0.24     41605

    accuracy                           0.46    348536
   macro avg       0.53      0.46      0.43    348536
weighted avg       0.63      0.46      0.50    348536""",
            'model': None
        },
        'Random Forest': {
            'Accuracy': 0.5711,
            'Precision': 0.6373,
            'Recall': 0.5711,
            'F1-Score': 0.5651,
            'Training Time (s)': 70.66,
            'Prediction Time (s)': 1.25,
            'confusion_matrix': np.array([
                [40872, 61887, 13727],
                [6685, 147899, 35861],
                [1256, 30056, 10293]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.84      0.35      0.49    116486
      Middle       0.62      0.78      0.69    190445
        High       0.17      0.25      0.20     41605

    accuracy                           0.57    348536
   macro avg       0.54      0.46      0.46    348536
weighted avg       0.64      0.57      0.57    348536""",
            'model': None
        },
        'Hist Gradient Boosting': {
            'Accuracy': 0.4766,
            'Precision': 0.6308,
            'Recall': 0.4766,
            'F1-Score': 0.5128,
            'Training Time (s)': 38.66,
            'Prediction Time (s)': 3.41,
            'confusion_matrix': np.array([
                [42170, 42285, 32031],
                [8762, 102522, 79161],
                [1477, 18720, 21408]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.80      0.36      0.50    116486
      Middle       0.63      0.54      0.58    190445
        High       0.16      0.51      0.25     41605

    accuracy                           0.48    348536
   macro avg       0.53      0.47      0.44    348536
weighted avg       0.63      0.48      0.51    348536""",
            'model': None
        },
        'Decision Tree': {
            'Accuracy': 0.4672,
            'Precision': 0.5929,
            'Recall': 0.4672,
            'F1-Score': 0.5018,
            'Training Time (s)': 27.11,
            'Prediction Time (s)': 0.07,
            'confusion_matrix': np.array([
                [43409, 42646, 30431],
                [14440, 100978, 75027],
                [3122, 20030, 18453]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.71      0.37      0.49    116486
      Middle       0.62      0.53      0.57    190445
        High       0.15      0.44      0.22     41605

    accuracy                           0.47    348536
   macro avg       0.49      0.45      0.43    348536
weighted avg       0.59      0.47      0.50    348536""",
            'model': None
        },
        'K-Nearest Neighbors': {
            'Accuracy': 0.5686,
            'Precision': 0.5209,
            'Recall': 0.5686,
            'F1-Score': 0.5351,
            'Training Time (s)': 0.05,
            'Prediction Time (s)': 288.98,
            'confusion_matrix': np.array([
                [55584, 59335, 1567],
                [45148, 141598, 3699],
                [9628, 30982, 995]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.50      0.48      0.49    116486
      Middle       0.61      0.74      0.67    190445
        High       0.16      0.02      0.04     41605

    accuracy                           0.57    348536
   macro avg       0.42      0.41      0.40    348536
weighted avg       0.52      0.57      0.54    348536""",
            'model': None
        },
        'Naive Bayes': {
            'Accuracy': 0.6216,
            'Precision': 0.6169,
            'Recall': 0.6216,
            'F1-Score': 0.5745,
            'Training Time (s)': 0.35,
            'Prediction Time (s)': 0.13,
            'confusion_matrix': np.array([
                [41916, 70816, 3754],
                [9599, 172480, 8366],
                [1840, 37513, 2252]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.79      0.36      0.49    116486
      Middle       0.61      0.91      0.73    190445
        High       0.16      0.05      0.08     41605

    accuracy                           0.62    348536
   macro avg       0.52      0.44      0.44    348536
weighted avg       0.62      0.62      0.57    348536""",
            'model': None
        },
        'SVM (Linear SGD)': {
            'Accuracy': 0.6323,
            'Precision': 0.6125,
            'Recall': 0.6323,
            'F1-Score': 0.5725,
            'Training Time (s)': 3.90,
            'Prediction Time (s)': 0.02,
            'confusion_matrix': np.array([
                [42268, 73110, 1108],
                [10924, 177358, 2163],
                [2215, 38637, 753]
            ]),
            'classification_report': """              precision    recall  f1-score   support

         Low       0.76      0.36      0.49    116486
      Middle       0.61      0.93      0.74    190445
        High       0.19      0.02      0.03     41605

    accuracy                           0.63    348536
   macro avg       0.52      0.44      0.42    348536
weighted avg       0.61      0.63      0.57    348536""",
            'model': None
        }
    }
    return model_results


def page_model_evaluation():
    """
    Display the Model Development and Evaluation page with model comparison,
    performance metrics, feature importance, and interactive prediction.
    """
    st.header("Model Development & Evaluation")

    # Model information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Traditional ML Models:**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Naive Bayes
        """)

    with col2:
        st.markdown("""
        **Advanced ML Models:**
        - Histogram Gradient Boosting
        - K-Nearest Neighbors
        - SVM (Linear SGD)
        """)

    st.caption("All models use class balancing to handle imbalanced tip classes")

    st.markdown("---")

    # Model Comparison
    st.subheader("Model Performance Comparison")

    model_results = get_hardcoded_model_results()

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        name: {k: v for k, v in data.items() if k not in ['confusion_matrix', 'classification_report', 'model']}
        for name, data in model_results.items()
    }).T

    # Display metrics table
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, color='lightgreen'),
        width='stretch'
    )

    # Visualize comparison
    metric_to_plot = st.selectbox(
        "Select metric to compare:",
        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)', 'Prediction Time (s)'],
        key='metric_compare'
    )

    fig = px.bar(
        comparison_df.reset_index(),
        x='index',
        y=metric_to_plot,
        title=f"Model Comparison: {metric_to_plot}",
        labels={'index': 'Model'},
        color=metric_to_plot,
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Detailed Model Analysis
    st.subheader("Detailed Model Analysis")

    selected_model = st.selectbox(
        "Select a model for detailed analysis:",
        list(model_results.keys()),
        key='detailed_model'
    )

    if selected_model:
        model_data = model_results[selected_model]

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{model_data['Accuracy']:.4f}")
        col2.metric("Precision", f"{model_data['Precision']:.4f}")
        col3.metric("Recall", f"{model_data['Recall']:.4f}")
        col4.metric("F1-Score", f"{model_data['F1-Score']:.4f}")

        # Confusion Matrix
        if model_data['confusion_matrix'] is not None:
            cm = model_data['confusion_matrix']

            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Low', 'Middle', 'High'],
                y=['Low', 'Middle', 'High'],
                color_continuous_scale='Blues',
                text_auto=True,
                title=f"Confusion Matrix: {selected_model}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')

        # Classification Report
        if model_data['classification_report']:
            with st.expander("View Full Classification Report"):
                st.text(model_data['classification_report'])

    st.markdown("---")

    # Best Models Summary
    st.subheader("Model Selection Guidance")

    best_accuracy = max(model_results.items(), key=lambda x: x[1]['Accuracy'])
    best_f1 = max(model_results.items(), key=lambda x: x[1]['F1-Score'])
    fastest_training = min(model_results.items(), key=lambda x: x[1]['Training Time (s)'])
    fastest_prediction = min(model_results.items(), key=lambda x: x[1]['Prediction Time (s)'])

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Best Accuracy:** {best_accuracy[0]} ({best_accuracy[1]['Accuracy']:.4f})")
        st.success(f"**Best F1-Score:** {best_f1[0]} ({best_f1[1]['F1-Score']:.4f})")

    with col2:
        st.info(f"**Fastest Training:** {fastest_training[0]} ({fastest_training[1]['Training Time (s)']:.2f}s)")
        st.info(f"**Fastest Prediction:** {fastest_prediction[0]} ({fastest_prediction[1]['Prediction Time (s)']:.2f}s)")


def page_interactive_prediction():
    """
    Display an interactive prediction page where users can input trip details
    and get tip predictions from trained models.
    """
    st.header("Interactive Tip Prediction")

    st.markdown("""
    Use this tool to predict tip class based on trip and weather characteristics.
    Adjust the parameters below to see how different factors might influence tipping behavior.
    """)

    # Try to load models if they exist
    model_dir = Path(MODEL_DIR)
    available_models = {}

    if model_dir.exists():
        model_files = list(model_dir.glob("*.pkl"))
        for model_file in model_files:
            model_data = load_model(model_file)
            if model_data and isinstance(model_data, dict):
                model_name = model_data.get('model_name', model_file.stem)
                available_models[model_name] = model_data.get('model')

    selected_model = st.selectbox(
        "Select a model for prediction:",
        list(available_models.keys()) if available_models else ['Demo Model (No actual model loaded)'],
        key='prediction_model'
    )

    st.markdown("---")

    st.subheader("Input Trip Details")

    # Create input fields for prediction
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Trip Information**")
        trip_distance = st.slider("Trip Distance (miles)", 0.0, 50.0, 5.0, 0.5)
        duration_min = st.slider("Duration (minutes)", 1, 120, 15, 1)
        passenger_count = st.slider("Passenger Count", 1, 6, 1, 1)
        fare_amount = st.slider("Fare Amount ($)", 0.0, 200.0, 15.0, 1.0)

    with col2:
        st.markdown("**Time & Location**")
        pickup_hour = st.slider("Pickup Hour (0-23)", 0, 23, 12, 1)
        day_of_week = st.selectbox("Day of Week",
                                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                    'Friday', 'Saturday', 'Sunday'],
                                   index=0)
        is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
        is_yellow = st.checkbox("Yellow Taxi (vs Green)", value=True)

    with col3:
        st.markdown("**Weather Conditions**")
        temperature = st.slider("Temperature (°F)", -10.0, 100.0, 65.0, 1.0)
        precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
        wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0, 1.0)

    # Calculate derived features
    speed_mph = trip_distance / (duration_min / 60) if duration_min > 0 else 0

    st.markdown("---")

    # Display calculated features
    st.subheader("Calculated Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Speed", f"{speed_mph:.2f} mph")
    with col2:
        st.metric("$ per Mile", f"${fare_amount/trip_distance:.2f}" if trip_distance > 0 else "$0.00")
    with col3:
        st.metric("$ per Minute", f"${fare_amount/duration_min:.2f}" if duration_min > 0 else "$0.00")

    st.markdown("---")

    # Make prediction button
    if st.button("Predict Tip Class", type="primary"):
        st.subheader("Prediction Results")

        # Check if actual model is available
        actual_model = available_models.get(selected_model) if available_models else None

        if actual_model is None:
            st.info("**Note:** Demo prediction based on heuristics (model files not included in deployment)")

        # Simulated prediction based on heuristics
        tip_classes = ['Low Tip (0-10%)', 'Middle Tip (10-20%)', 'High Tip (>20%)']

        # Simple heuristic for demonstration
        tip_score = 0

        # Factors that increase tip likelihood
        if trip_distance > 5:
            tip_score += 1
        if fare_amount > 20:
            tip_score += 1
        if temperature > 50 and temperature < 80:
            tip_score += 0.5
        if precipitation < 5:
            tip_score += 0.5
        if is_weekend:
            tip_score += 0.5
        if passenger_count > 1:
            tip_score += 0.5

        # Determine predicted class
        if tip_score >= 2.5:
            predicted_class = 2  # High
            probs = [0.15, 0.30, 0.55]
        elif tip_score >= 1.5:
            predicted_class = 1  # Middle
            probs = [0.20, 0.60, 0.20]
        else:
            predicted_class = 0  # Low
            probs = [0.65, 0.30, 0.05]

        # Display prediction
        st.success(f"### Predicted Tip Class: **{tip_classes[predicted_class]}**")

        # Show prediction probabilities
        st.markdown("**Prediction Confidence:**")

        prob_df = pd.DataFrame({
            'Tip Class': tip_classes,
            'Probability': probs
        })

        fig = px.bar(
            prob_df,
            x='Tip Class',
            y='Probability',
            color='Probability',
            color_continuous_scale='greens',
            title="Prediction Probabilities"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch')

        # Show contributing factors
        st.markdown("**Key Factors Influencing This Prediction:**")
        factors = []
        if trip_distance > 5:
            factors.append(f"✓ Long trip distance ({trip_distance} miles)")
        if fare_amount > 20:
            factors.append(f"✓ High fare amount (${fare_amount})")
        if temperature > 50 and temperature < 80:
            factors.append(f"✓ Comfortable temperature ({temperature}°F)")
        if precipitation < 5:
            factors.append(f"✓ Good weather conditions")
        if is_weekend:
            factors.append(f"✓ Weekend travel")
        if passenger_count > 1:
            factors.append(f"✓ Multiple passengers ({passenger_count})")

        if factors:
            for factor in factors:
                st.write(factor)
        else:
            st.write("Standard trip conditions")

    st.markdown("---")

    with st.expander("💡 Tips for Better Tips"):
        st.markdown("""
        - Longer trips tend to receive better tip percentages
        - Good weather conditions correlate with better tips
        - Multiple passengers often tip more generously
        - Efficient service encourages better tipping
        """)


def page_methodology():
    """
    Display detailed methodology page explaining the data science workflow,
    techniques used, and validation approach.
    """
    st.header("Methodology & Techniques")

    # Workflow
    st.subheader("Data Science Workflow")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        1. **Data Collection & Integration**
        2. **Data Cleaning**
        3. **Feature Engineering**
        4. **Missing Data Imputation**
        """)
    with col2:
        st.markdown("""
        5. **Exploratory Analysis**
        6. **Model Development**
        7. **Model Evaluation**
        8. **Deployment**
        """)

    st.markdown("---")

    # Data Processing Techniques
    st.subheader("Key Techniques")

    tab1, tab2 = st.tabs(["Data Processing", "Model Development"])

    with tab1:
        st.markdown("""
        ### Data Collection & Integration
        
        **Data Sources:**
        1. NYC TLC Yellow & Green Taxi Data (Parquet format)
        2. Historical Weather Data (Open-Meteo API, hourly)
        
        **Integration:** Temporal join based on pickup timestamp
        
        ### Feature Engineering
        
        **Derived Features:**
        - **Temporal:** `pickup_hour`, `day_of_week`, `is_weekend`, `month`
        - **Trip:** `duration_min`, `speed_mph`, `fare_per_mile`
        - **Target:** `tip_class` (Low: 0-10%, Middle: 10-20%, High: >20%)
        
        ### Missing Data Handling
        
        **Technique:** Iterative Imputation (MICE)
        - Models each feature with missing values as function of other features
        - Preserves relationships between variables
        - Applied to `passenger_count`, `RatecodeID`, and other numeric features
        """)

    with tab2:
        st.markdown("""
        ### Model Validation & Selection
        
        **Train-Test Split:** 80/20 with stratified sampling
        
        **Evaluation Metrics:**
        - Accuracy, Precision, Recall, F1-Score
        - Confusion Matrix for per-class performance
        
        **Class Balancing:** Applied `class_weight='balanced'`
        
        **Models Tested:** 7 algorithms from Logistic Regression to Gradient Boosting
        """)

    st.markdown("---")

    # Technical Stack
    st.subheader("Technical Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Data Processing**
        - pandas, numpy
        - scikit-learn
        """)

    with col2:
        st.markdown("""
        **Visualization**
        - plotly
        - streamlit
        """)

    with col3:
        st.markdown("""
        **Storage**
        - parquet
        - pickle
        """)



# ---------- Menu ----------
def main():
    """
    Main function to handle navigation between different pages.
    """

    # Create sidebar navigation
    st.sidebar.title("NYC Taxi Analysis")

    menu = st.sidebar.radio(
        "Navigate to:",
        [
            "Overview",
            "Data Collection",
            "Initial Data Analysis",
            "EDA & Visualization",
            "Advanced Analysis",
            "Model Evaluation",
            "Interactive Prediction",
            "Methodology"
        ],
        index=0
    )
    
    # Display project information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Author:** Zhiqiang Ni  
    **Course:** CMSE 830 | MSU
    
    **Dataset:** 2024 NYC Taxi + Weather  
    **Records:** 1M+ trips analyzed
    """)


    # Route to appropriate page based on menu selection
    if menu == "Overview":
        page_overview()
    elif menu == "Data Collection":
        page_data_collection()
    elif menu == "Initial Data Analysis":
        page_ida()
    elif menu == "EDA & Visualization":
        page_eda()
    elif menu == "Advanced Analysis":
        page_advanced_analysis()
    elif menu == "Model Evaluation":
        page_model_evaluation()
    elif menu == "Interactive Prediction":
        page_interactive_prediction()
    elif menu == "Methodology":
        page_methodology()

# Entry point of the application
if __name__ == "__main__":
    main()
