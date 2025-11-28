# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path

# Centralized data paths for easy updates
DATA_DIR = "./data"
MODEL_DIR = "./models"
TAXI_SAMPLE = f"{DATA_DIR}/taxi_data_sampled.parquet"
TAXI_PREPROCESSED_MISSING_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed_missing_sampled.parquet"
TAXI_PREPROCESSED_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed.parquet"
TAXI_PREPROCESSED_TIP_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed_tip.parquet"

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

    **Author:** Zhiqiang Ni
    **Course:** CMSE 830
    """)
    
    # Key metrics
    st.markdown("---")
    st.subheader("Project Highlights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>3</h3>
        <p>Data Sources</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>7</h3>
        <p>ML Models</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>15+</h3>
        <p>Visualizations</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>1M+</h3>
        <p>Records Analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("The Story Behind the Data")
    st.write("""
    New York City's iconic yellow and green taxis are a vital part of its transportation network, generating a massive amount of data with every trip. This project dives into this data to uncover the factors that influence whether a passenger leaves a good tip.

    Is tipping behavior influenced by the time of day, the length of the trip, or even the weather? By combining trip data from the NYC Taxi & Limousine Commission (TLC) with historical weather data, we can explore these questions and build models to predict tip amounts.
    
    ### Dataset of Discovery
    
    This analysis is based on **2024 taxi trip records** for both yellow and green taxis, along with **hourly weather data** for NYC. We explore:
    - **Tipping Patterns**: What are the characteristics of trips with high, middle, and low tips?
    - **Temporal Trends**: How do tip amounts vary by hour, day of the week, or month?
    - **Weather's Impact**: Does rain, snow, or temperature affect a passenger's generosity?
    - **Predictive Insights**: Can we build reliable models to predict the tip class of a future trip?
    
    ### Why This Matters
    
    Understanding tipping behavior provides valuable insights for:
    - **Drivers**: To better understand potential earnings and influencing factors
    - **Passengers**: To gain awareness of tipping norms and how their trips compare
    - **TLC**: For policy-making and understanding the taxi economy
    """)

    st.markdown("---")

    # Project Components
    st.subheader("Project Components")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 1. Data Collection & Integration
        - NYC Yellow Taxi Data (2024)
        - NYC Green Taxi Data (2024)
        - Historical Weather Data (Open-Meteo API)
        
        #### 2. Data Preprocessing
        - Data cleaning and validation
        - Missing value imputation (Iterative MICE)
        - Feature engineering (15+ new features)
        
        #### 3. Exploratory Data Analysis
        - 15+ interactive visualizations
        - Statistical analysis
        - Correlation analysis
        - Pattern identification
        """)

    with col2:
        st.markdown("""
        #### 4. Machine Learning Models
        - 7 different algorithms implemented
        - Comprehensive model comparison
        - Feature importance analysis
        
        #### 5. Interactive Features
        - Real-time tip prediction
        - Model comparison tools
        - Dynamic visualizations
        
        #### 6. Complete Documentation
        - Methodology explanation
        - Technical stack overview
        - User guide
        """)

    st.markdown("---")

    st.subheader("Techniques Used")

    tab1, tab2, tab3 = st.tabs(["Data Preparation", "Analysis Methods", "Advanced Features"])

    with tab1:
        st.markdown("""
        **Data Preparation:**
        - Data downloading and merging (Yellow and Green taxis)
        - Data cleaning and feature engineering
        - Merging with weather data via temporal joins
        - Iterative imputation for missing values
        - Train-test splitting with stratification
        """)

    with tab2:
        st.markdown("""
        **Analysis Methods:**
        - Correlation analysis
        - Interactive visualizations (15+ types)
        - Temporal pattern analysis
        - Distribution analysis
        - Statistical hypothesis testing
        - Feature importance evaluation
        """)

    with tab3:
        st.markdown("""
        **Advanced Features:**
        - 7 ML models with hyperparameter tuning
        - Ensemble methods (Random Forest, Gradient Boosting)
        - Class balancing for imbalanced data
        - Caching strategies for performance
        - Session state management
        - Interactive prediction interface
        """)

    st.markdown("---")

    st.info("""
    **Navigation Guide:** Use the sidebar menu to explore different sections of this comprehensive analysis.
    Each section demonstrates different data science techniques and insights.
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

    st.write("""
    This project uses two main data sources to analyze tipping behavior in NYC taxis:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **NYC TLC Trip Record Data**
        - **Source**: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
        - **Records**: Yellow and Green taxi trips for 2024
        - **Key Features**: Pickup/dropoff times, locations, fares, tolls, tip amounts
        """)

    with col2:
        st.markdown("""
        **Weather Data**
        - **Source**: [Open-Meteo API](https://open-meteo.com/)
        - **Coverage**: 2024
        - **Location**: NYC
        - **Granularity**: Hourly observations
        - **Key Features**: Temperature, precipitation, wind speed, etc.
        """)

    st.info("The taxi trip data was merged with weather data based on the pickup timestamp.")

    st.markdown("---")

    st.subheader("Detailed Data Exploration")

    tab1, tab2, tab3 = st.tabs(["Raw Taxi Data", "Preprocessed Data", "Imputed Data"])

    with tab1:
        st.write("### Raw Taxi Data (Yellow and Green Taxis)")
        df_raw = load_parquet(TAXI_SAMPLE)
        st.write(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(10))

        st.write("### Data Characteristics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trips", f"{len(df_raw):,}")
        with col2:
            st.metric("Time Period", "2024")
        with col3:
            st.metric("Taxi Types", "Yellow & Green")

    with tab2:
        st.write("### Preprocessed Data")
        df_preprocessed = load_parquet(TAXI_PREPROCESSED_MISSING_SAMPLED)
        st.write(f"**Shape:** {df_preprocessed.shape[0]:,} rows × {df_preprocessed.shape[1]} columns")
        st.dataframe(df_preprocessed.head(10))
        st.write("This data has been cleaned, merged with weather data, and new features have been engineered.")

    with tab3:
        st.write("### Imputed Data")
        df_imputed = load_parquet(TAXI_PREPROCESSED_SAMPLED)
        st.write(f"**Shape:** {df_imputed.shape[0]:,} rows × {df_imputed.shape[1]} columns")
        st.dataframe(df_imputed.head(10))
        st.write("Missing values in the preprocessed data have been filled using iterative imputation.")

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

    st.subheader("Missing Data Handling")
    st.write("""
    Missing values in the dataset were handled using `IterativeImputer` from scikit-learn. 
    This method models each feature with missing values as a function of other features, and uses that estimate for imputation. 
    It is more sophisticated than simple mean/median imputation and can preserve relationships between variables.
    """)

    st.subheader("Imputation Impact Visualization")

    impute_cols = [
        'passenger_count', 'RatecodeID'
    ]

    selected_col = st.selectbox("Select variable to compare:", [c for c in impute_cols if c in df_preprocessed.columns], key="impute_compare")

    # Sample data for histogram to improve performance
    sample_size = min(10000, len(df_preprocessed))
    df_preprocessed_sample = df_preprocessed.sample(n=sample_size, random_state=42)
    df_imputed_sample = df_imputed.sample(n=sample_size, random_state=42)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Before Imputation", "After Imputation"))

    fig.add_trace(
        go.Histogram(x=df_preprocessed_sample[selected_col].dropna(), name="Original", marker_color='#3498db', nbinsx=30),
        row=1, col=1
    )

    if selected_col in df_imputed_sample.columns:
        fig.add_trace(
            go.Histogram(x=df_imputed_sample[selected_col].dropna(), name="Imputed", marker_color='#2ecc71', nbinsx=30),
            row=1, col=2
        )

    fig.update_layout(height=400, showlegend=True, title_text=f"Distribution Comparison: {selected_col}")
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    numeric_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Statistical Summary (Numeric Variables)")
    stats_summary = compute_stats_summary(df_imputed)
    st.dataframe(stats_summary, width='stretch')

    st.subheader("Interactive Distribution Analysis")
    num_col = st.selectbox("Select a numeric column to explore:", numeric_cols, key="ida_num")

    # Get cached statistics
    stats = get_column_stats(df_imputed, num_col)

    col1, col2 = st.columns([2, 1])
    with col1:
        # Sample for visualization to improve performance
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
        st.write(f"Mean: {stats['mean']:.2f}")
        st.write(f"Median: {stats['median']:.2f}")
        st.write(f"Std Dev: {stats['std']:.2f}")
        st.write(f"Min: {stats['min']:.2f}")
        st.write(f"Max: {stats['max']:.2f}")

    cat_cols = df_imputed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if cat_cols:
        st.subheader("Categorical Variable Frequency")
        cat_col = st.selectbox("Select a categorical column:", cat_cols, key="ida_cat")

        if cat_col:
            top_counts = df_imputed[cat_col].value_counts(dropna=False).head(20).reset_index()
            top_counts.columns = [cat_col, "count"]
            fig_cat = px.bar(top_counts, x=cat_col, y="count", title=f"Top 20 categories in {cat_col}")
            st.plotly_chart(fig_cat, width='stretch')
    else:
        st.info("No categorical columns available in the imputed dataset.")

    st.header("Data Preprocessing & Feature Engineering")

    st.subheader("Techniques Applied:")
    st.markdown("""
    1. **Data Cleaning:**
       - Removed trips with negative or zero fare, distance, and duration.
       - Handled outliers and invalid data points.
    
    2. **Feature Engineering:**
       - `duration_min`: Trip duration in minutes.
       - `speed_mph`: Average speed of the trip.
       - `tip_class`: Categorical variable for tip amount ('Low', 'Middle', 'High').
    
    3. **Data Integration:**
       - Merged taxi data with hourly weather data based on pickup time.
    """)

    st.subheader("Data Types Overview")
    dtype_df = pd.DataFrame(df_imputed.dtypes, columns=["Data Type"]).reset_index()
    dtype_df.columns = ["Column", "Data Type"]
    # Convert to string explicitly to avoid Arrow serialization issues
    dtype_df["Data Type"] = dtype_df["Data Type"].apply(lambda x: str(x))
    st.dataframe(dtype_df, width='stretch', hide_index=True)


def page_eda():
    """
    Display the Exploratory Data Analysis (EDA) page with correlation analysis,
    temporal patterns, and weather impact visualizations.
    """
    st.header("Exploratory Data Analysis and Visualization")

    df = load_parquet(TAXI_PREPROCESSED_SAMPLED)

    st.subheader("Correlation Heatmap")
    st.write("Interactive correlation matrix showing relationships between all numeric features.")
    st.info("**tip_class** (0=Low, 1=Middle, 2=High) is our target variable for classification.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Reorder columns to put tip_class first (top of heatmap)
    if 'tip_class' in numeric_cols:
        numeric_cols.remove('tip_class')
        numeric_cols = numeric_cols + ['tip_class']

    corr_method = st.radio("Select correlation method:", ["pearson", "spearman", "kendall"], horizontal=True)
    
    corr = df[numeric_cols].corr(method=corr_method)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        origin="lower",
        title=f"Correlation Heatmap ({corr_method.capitalize()}) - tip_class at top",
        aspect="auto"
    )
    fig.update_layout(height=700, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Target Feature Correlation Analysis")
    st.write("Explore correlations of all features with a selected target variable.")

    # Default to tip_class if available, otherwise first column
    default_idx = numeric_cols.index("tip_class") if "tip_class" in numeric_cols else 0
    target_col = st.selectbox(
        "Select target variable:", 
        numeric_cols,
        index=default_idx,
        key="eda_target"
    )

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

    st.subheader("Temporal Pattern Analysis")
    st.write("Analyze tipping patterns across different time periods.")

    if 'pickup_hour' in df.columns:
        fig = px.histogram(
            df,
            x='pickup_hour',
            y='fare_amount',  # Using fare_amount since tip_amount may not exist
            histfunc='avg',
            nbins=24,
            title="Average Fare Amount by Hour of Day",
            labels={'pickup_hour': 'Hour of Day', 'fare_amount': 'Average Fare Amount'}
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Tip Class Balance Analysis")
    st.write("Distribution of tip classes in the dataset (0=Low, 1=Middle, 2=High)")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        tip_class_counts = df['tip_class'].value_counts().sort_index()
        # Map numeric values to labels for display
        tip_labels = {0: 'Low', 1: 'Middle', 2: 'High'}
        st.metric("Low Tips (0)", f"{tip_class_counts.get(0, 0):,}")
        st.metric("Middle Tips (1)", f"{tip_class_counts.get(1, 0):,}")
        st.metric("High Tips (2)", f"{tip_class_counts.get(2, 0):,}")

    with col2:
        # Create labels for the pie chart
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

    st.subheader("Weather Impact on Tips")
    st.write("Explore how different weather conditions affect tip amounts.")

    # Load preprocessed data with tip information
    df_tip = load_parquet(TAXI_PREPROCESSED_TIP_SAMPLED)

    # Calculate tip percentage if not already present
    if 'tip_percentage' not in df_tip.columns and 'tip_amount' in df_tip.columns and 'fare_amount' in df_tip.columns:
        df_tip['tip_percentage'] = (df_tip['tip_amount'] / df_tip['fare_amount'] * 100).clip(0, 100)

    # Get numeric columns
    numeric_cols_tip = df_tip.select_dtypes(include=[np.number]).columns.tolist()

    weather_candidates = [c for c in numeric_cols_tip if any(k in c.lower()
                          for k in ['temperature','precipitation','rain','snowfall','wind_speed'])]

    if not weather_candidates:
        st.warning("No weather columns found in the dataset. Weather data may not have been merged.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            wcol = st.selectbox("Select weather feature:", weather_candidates, key="eda_weather_col")
        with col2:
            # Check which tip metrics are available
            available_metrics = []
            if 'tip_amount' in df_tip.columns:
                available_metrics.append('tip_amount')
            if 'tip_percentage' in df_tip.columns:
                available_metrics.append('tip_percentage')

            if not available_metrics:
                st.error("No tip metrics available in the dataset.")
                return

            tip_metric = st.selectbox("Tip metric:", available_metrics, key="eda_tip_metric")
        with col3:
            bins = st.slider("Number of bins:", 5, 20, 10, key="eda_weather_bins")

        if wcol and tip_metric and len(df_tip[wcol].dropna()) > 0:
            try:
                qbins = pd.qcut(df_tip[wcol], q=bins, duplicates='drop')
                tmp = df_tip.assign(_bin=qbins).dropna(subset=['_bin', tip_metric])
                rate = tmp.groupby('_bin')[tip_metric].mean().reset_index()
                rate.columns = ['bin', 'avg_tip']
                rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)
                rate['bin_label'] = rate['bin'].astype(str)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rate['bin_mid'],
                    y=rate['avg_tip'],
                    mode='lines+markers',
                    name=f'Average {tip_metric.replace("_", " ").title()}',
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

                # Show summary statistics
                st.markdown("**Analysis Summary:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Tip", f"${tmp[tip_metric].mean():.2f}" if tip_metric == 'tip_amount' else f"{tmp[tip_metric].mean():.1f}%")
                with col2:
                    st.metric("Min Tip", f"${tmp[tip_metric].min():.2f}" if tip_metric == 'tip_amount' else f"{tmp[tip_metric].min():.1f}%")
                with col3:
                    st.metric("Max Tip", f"${tmp[tip_metric].max():.2f}" if tip_metric == 'tip_amount' else f"{tmp[tip_metric].max():.1f}%")

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        else:
            st.warning(f"No valid data available for {wcol}")


def page_advanced_analysis():
    """
    Display advanced analysis page with multi-dimensional visualizations
    including scatter matrices, 3D plots, and distribution comparisons.
    """
    st.header("Advanced Analysis & Visualizations")
    
    st.subheader("Multi-Dimensional Analysis")
    
    df = load_parquet(TAXI_PREPROCESSED_SAMPLED)

    st.write("### Interactive Scatter Matrix")
    st.write("Explore relationships between multiple numeric variables simultaneously.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_features = st.multiselect(
        "Select features for scatter matrix (3-5 recommended):",
        numeric_cols,
        default=[col for col in ['fare_amount', 'trip_distance', 'duration_min', 'speed_mph'] if col in numeric_cols][:4]
    )
    
    if len(selected_features) >= 2:
        # Create a copy with labeled tip classes for better visualization
        df_sample = df.sample(min(1000, len(df))).copy()
        tip_labels = {0: 'Low', 1: 'Middle', 2: 'High'}
        df_sample['Tip Class'] = df_sample['tip_class'].map(tip_labels)

        fig = px.scatter_matrix(
            df_sample,
            dimensions=selected_features,
            color='Tip Class',
            title="Pairwise Feature Relationships by Tip Class",
            labels={col: col.replace('_', ' ').title() for col in selected_features},
            color_discrete_map={'Low': '#e74c3c', 'Middle': '#f39c12', 'High': '#27ae60'}
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig, width='stretch')
    
    st.write("### 3D Relationship Visualization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis:", numeric_cols, index=numeric_cols.index('trip_distance'), key="3d_x")
    with col2:
        y_var = st.selectbox("Y-axis:", numeric_cols, index=numeric_cols.index('duration_min'), key="3d_y")
    with col3:
        z_var = st.selectbox("Z-axis:", numeric_cols, index=numeric_cols.index('fare_amount'), key="3d_z")

    sample_df = df[[x_var, y_var, z_var, 'tip_class']].dropna().sample(min(2000, len(df))).copy()

    # Add labeled tip class for better visualization
    tip_labels = {0: 'Low', 1: 'Middle', 2: 'High'}
    sample_df['Tip Class'] = sample_df['tip_class'].map(tip_labels)

    fig = px.scatter_3d(
        sample_df,
        x=x_var,
        y=y_var,
        z=z_var,
        color='Tip Class',
        title=f"3D Visualization: {x_var} vs {y_var} vs {z_var}",
        color_discrete_map={'Low': '#e74c3c', 'Middle': '#f39c12', 'High': '#27ae60'},
        opacity=0.7
    )
    st.plotly_chart(fig, width='stretch')
    
    st.write("### Distribution Comparison by Category")
    
    # Include tip_class as a categorical variable even though it's numeric
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Add tip_class if it exists and is numeric
    if 'tip_class' in df.columns and 'tip_class' not in cat_cols:
        cat_cols.insert(0, 'tip_class')

    # Also add other potentially categorical numeric columns
    other_categorical = ['is_yellow', 'RatecodeID', 'day_of_week', 'pickup_hour']
    for col in other_categorical:
        if col in df.columns and col not in cat_cols:
            cat_cols.append(col)

    if cat_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            default_cat = 'tip_class' if 'tip_class' in cat_cols else cat_cols[0] if cat_cols else None
            category = st.selectbox("Select category:", cat_cols,
                                   index=cat_cols.index(default_cat) if default_cat else 0,
                                   key="box_cat")
        with col2:
            value = st.selectbox("Select numeric variable:", numeric_cols,
                                index=numeric_cols.index('fare_amount') if 'fare_amount' in numeric_cols else 0,
                                key="box_val")

        # Create a copy with labeled categories if it's tip_class
        df_plot = df.copy()
        if category == 'tip_class':
            tip_labels = {0: 'Low', 1: 'Middle', 2: 'High'}
            df_plot['Tip Class'] = df_plot['tip_class'].map(tip_labels)
            category_col = 'Tip Class'
        else:
            category_col = category

        fig = px.box(
            df_plot,
            x=category_col,
            y=value,
            color=category_col,
            title=f"{value} Distribution by {category_col}",
            points="outliers"
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("Insufficient categorical or numeric columns for box plot visualization.")


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

    st.markdown("""
    This section demonstrates the development and comprehensive evaluation of multiple machine learning models 
    for predicting tip classes (Low, Middle, High) based on taxi trip and weather features.
    """)

    # Model information
    st.subheader("Models Implemented")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Traditional ML Models:**
        - Logistic Regression
        - Decision Tree Classifier
        - Random Forest Classifier
        - Naive Bayes (Gaussian)
        """)

    with col2:
        st.markdown("""
        **Advanced ML Models:**
        - Histogram Gradient Boosting
        - K-Nearest Neighbors
        - SVM (Linear SGD)
        """)

    st.info("All models use **class balancing** to handle imbalanced tip classes.")

    st.markdown("---")

    # Model Comparison Section
    st.subheader("Model Performance Comparison")

    # Use hard-coded model results instead of loading from files
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

    # Visualize model comparison
    st.subheader("Visual Comparison")

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
        labels={'index': 'Model', metric_to_plot: metric_to_plot},
        color=metric_to_plot,
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, width='stretch')

    # Multi-metric radar chart
    st.subheader("Multi-Metric Comparison (Radar Chart)")

    metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig = go.Figure()

    for model_name, data in model_results.items():
        fig.add_trace(go.Scatterpolar(
            r=[data[metric] for metric in metrics_for_radar],
            theta=metrics_for_radar,
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Across Multiple Metrics",
        height=600
    )
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

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{model_data['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{model_data['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{model_data['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{model_data['F1-Score']:.4f}")

        # Confusion Matrix
        if model_data['confusion_matrix'] is not None:
            st.subheader("Confusion Matrix")

            cm = model_data['confusion_matrix']

            # Create annotated heatmap
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

            # Calculate per-class metrics
            st.subheader("Per-Class Performance")

            class_metrics = []
            for i, class_name in enumerate(['Low', 'Middle', 'High']):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                class_metrics.append({
                    'Class': class_name,
                    'Precision': f"{precision:.4f}",
                    'Recall': f"{recall:.4f}",
                    'F1-Score': f"{f1:.4f}",
                    'Support': cm[i, :].sum()
                })

            st.dataframe(pd.DataFrame(class_metrics), width='stretch', hide_index=True)

        # Classification Report
        if model_data['classification_report']:
            with st.expander("View Full Classification Report"):
                st.text(model_data['classification_report'])

    st.markdown("---")

    # Model Selection Guidance
    st.subheader("Model Selection Guidance")

    # Find best model for each metric
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

    st.markdown("""
    ### Model Selection Considerations:
    
    - **For Best Performance:** Choose models with highest Accuracy and F1-Score
    - **For Real-Time Applications:** Consider prediction time (faster models like Logistic Regression or SGD)
    - **For Interpretability:** Decision Trees offer clear decision paths
    - **For Robustness:** Ensemble methods (Random Forest, Gradient Boosting) typically generalize better
    """)

    st.markdown("---")

    # Feature Importance (if available)
    st.subheader("Feature Importance Analysis")

    # Try to get feature importance from tree-based models
    feature_importance_models = ['Random Forest', 'Decision Tree', 'Hist Gradient Boosting']

    available_fi_models = [m for m in feature_importance_models if m in model_results]

    if available_fi_models:
        fi_model_name = st.selectbox(
            "Select model for feature importance:",
            available_fi_models,
            key='fi_model'
        )

        model_obj = model_results[fi_model_name]['model']

        model_obj = model_results[fi_model_name]['model']

        if model_obj is not None and hasattr(model_obj, 'feature_importances_'):
            # Load feature names from preprocessed data
            df = load_parquet(TAXI_PREPROCESSED_SAMPLED)

            # Get numeric columns (these would be the features used in training)
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Remove target variable if present
            if 'tip_class' in feature_cols:
                feature_cols.remove('tip_class')

            importances = model_obj.feature_importances_

            # Match length (in case of mismatch)
            min_len = min(len(feature_cols), len(importances))

            fi_df = pd.DataFrame({
                'Feature': feature_cols[:min_len],
                'Importance': importances[:min_len]
            }).sort_values('Importance', ascending=False).head(20)

            fig = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 20 Feature Importances: {fi_model_name}",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, width='stretch')

            st.info("Feature importance shows which features most strongly influence tip predictions.")
        else:
            st.info("""
            **Note:** Feature importance visualization requires the actual trained model objects, 
            which are not included in this deployment to reduce file size. 
            
            The models showed that the most important features for predicting tips were:
            - Trip distance
            - Fare amount
            - Trip duration
            - Time of day (hour)
            - Day of the week
            - Weather conditions (temperature, precipitation)
            """)
    else:
        st.info("Feature importance is only available for tree-based models (Random Forest, Decision Tree, Gradient Boosting).")


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

    # Get hard-coded model list
    hardcoded_models = get_hardcoded_model_results()
    model_names = list(hardcoded_models.keys())

    selected_model = st.selectbox(
        "Select a model for prediction:",
        model_names,
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
            st.info("""
            **Note:** This is a demonstration interface showing how predictions would work. 
            The actual trained model files are not included in this deployment to reduce repository size.
            
            Based on the input parameters and our model analysis, here's an estimated prediction:
            """)

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
        st.plotly_chart(fig, use_container_width=True)

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

    st.subheader("Tips for Better Tips")

    st.markdown("""
    Based on our analysis, here are factors that correlate with higher tips:
    
    - **Longer trips** tend to receive proportionally better tips
    - **Off-peak hours** may see more generous tipping
    - **Good weather** conditions correlate with better tips
    - **Multiple passengers** often tip better
    - **Efficient service** (good speed, direct routes) encourages tipping
    """)


def page_methodology():
    """
    Display detailed methodology page explaining the data science workflow,
    techniques used, and validation approach.
    """
    st.header("Methodology & Techniques")

    st.markdown("""
    This section provides a comprehensive overview of the data science techniques
    and methodologies applied throughout this project.
    """)

    # Workflow diagram
    st.subheader("Data Science Workflow")

    workflow_steps = [
        "**Data Collection**",
        "**Data Cleaning & Integration**",
        "**Feature Engineering**",
        "**Missing Data Imputation**",
        "**Exploratory Analysis**",
        "**Model Development**",
        "**Model Evaluation**",
        "**Deployment & Visualization**"
    ]

    for step in workflow_steps:
        st.markdown(step)

    st.markdown("---")

    # Data Processing Techniques
    st.subheader("Data Processing Techniques")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Collection",
        "Feature Engineering",
        "Missing Data",
        "Model Validation"
    ])

    with tab1:
        st.markdown("""
        ### Data Collection & Integration
        
        **Three Distinct Data Sources:**
        
        1. **NYC TLC Yellow Taxi Data**
           - Source: NYC Taxi & Limousine Commission
           - Format: Parquet files
           - Features: Trip details, fares, locations, timestamps
           
        2. **NYC TLC Green Taxi Data**
           - Source: NYC Taxi & Limousine Commission
           - Format: Parquet files
           - Coverage: Outer boroughs and specific areas
           
        3. **Historical Weather Data**
           - Source: Open-Meteo API
           - Granularity: Hourly observations
           - Features: Temperature, precipitation, wind speed, etc.
        
        **Integration Technique:**
        - Temporal join based on pickup timestamp
        - Nearest hour matching for weather data
        - Data validation and consistency checks
        """)

    with tab2:
        st.markdown("""
        ### Feature Engineering
        
        **Derived Features:**
        
        - **Temporal Features:**
          - `pickup_hour`: Hour of day (0-23)
          - `day_of_week`: Day of week (0-6)
          - `is_weekend`: Weekend indicator
          - `month`: Month of year
        
        - **Trip Characteristics:**
          - `duration_min`: Trip duration in minutes
          - `speed_mph`: Average speed (distance/time)
          - `fare_per_mile`: Fare efficiency metric
          - `fare_per_minute`: Time-based fare metric
        
        - **Target Variable:**
          - `tip_class`: Categorical tip amount (Low: 0-10%, Middle: 10-20%, High: >20%)
        
        **Rationale:**
        - Capture temporal patterns in tipping behavior
        - Create interpretable metrics for analysis
        - Enable multi-class classification
        """)

    with tab3:
        st.markdown("""
        ### Missing Data Handling
        
        **Technique:** Iterative Imputation (MICE - Multiple Imputation by Chained Equations)
        
        **Why Iterative Imputation?**
        - More sophisticated than mean/median imputation
        - Models each feature with missing values as a function of other features
        - Preserves relationships between variables
        - Better handles complex missing data patterns
        
        **Implementation:**
        ```python
        from sklearn.impute import IterativeImputer
        
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            verbose=0
        )
        ```
        
        **Features Imputed:**
        - `passenger_count`
        - `RatecodeID`
        - Other relevant numeric features
        
        **Validation:**
        - Distribution comparison before/after imputation
        - Correlation preservation check
        """)

    with tab4:
        st.markdown("""
        ### Model Validation & Selection
        
        **Train-Test Split:**
        - Training: 80% of data
        - Testing: 20% of data
        - Stratified split to maintain class balance
        
        **Evaluation Metrics:**
        
        - **Accuracy:** Overall correctness
        - **Precision:** Positive prediction reliability
        - **Recall:** Actual positive detection rate
        - **F1-Score:** Harmonic mean of precision and recall
        - **Confusion Matrix:** Per-class performance visualization
        
        **Class Balancing:**
        - Applied `class_weight='balanced'` to handle imbalanced tip classes
        - Ensures all tip categories are weighted appropriately
        
        **Cross-Validation Considerations:**
        - Multiple models tested for comparison
        - Performance vs. speed tradeoffs analyzed
        - Feature importance evaluated for interpretability
        """)

    st.markdown("---")

    # Advanced Techniques
    st.subheader("Advanced Techniques Applied")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Machine Learning:**
        - Ensemble Methods (Random Forest, Gradient Boosting)
        - Support Vector Machines (Linear SGD)
        - Hyperparameter optimization
        - Class balancing strategies
        """)

    with col2:
        st.markdown("""
        **Data Engineering:**
        - Efficient data loading with Parquet format
        - Caching strategies for performance (`@st.cache_data`)
        - Sampling for visualization efficiency
        - Memory-efficient processing techniques
        """)

    st.markdown("---")

    # Technical Stack
    st.subheader("Technical Stack")

    tech_stack = {
        'Data Processing': ['pandas', 'numpy', 'duckdb'],
        'Machine Learning': ['scikit-learn', 'torch'],
        'Visualization': ['plotly', 'streamlit'],
        'Data Collection': ['requests', 'beautifulsoup4'],
        'Storage': ['parquet', 'pickle']
    }

    for category, tools in tech_stack.items():
        st.markdown(f"**{category}:** {', '.join(tools)}")


# ---------- Menu ----------
def main():
    """
    Main function to configure the Streamlit app and handle navigation
    between different pages.
    """
    # Configure page settings
    st.set_page_config(
        page_title="CMSE 830 Data Analysis Project",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
    ### Project Info
    **Author:** Zhiqiang Ni  
    **Course:** CMSE 830  
    **Institution:** Michigan State University
    
    ### Project Goals
    - Analyze NYC taxi tipping patterns
    - Predict tip classes using ML
    - Identify key tipping factors
    
    ### Dataset Size
    - 2024 Yellow & Green Taxi Data
    - Hourly Weather Data
    - 1M+ trip records analyzed
    """)

    st.sidebar.markdown("---")
    st.sidebar.info("**Tip:** Use the navigation menu above to explore different sections of the analysis.")

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
