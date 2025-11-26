# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb

# Convert Google Drive share links to direct download URLs
taxi_data_url = "https://drive.google.com/uc?id=1yP2LKP4k9SNu-o2B4JclqVkPZv2aUgqr"
taxi_data_preprocessed_url = "https://drive.google.com/uc?id=16sJ85mmvrWe73f0Y0edbZXJ75j2AUTcv"
taxi_data_preprocessed_missing_url = "https://drive.google.com/uc?id=16sJ85mmvrWe73f0Y0edbZXJ75j2AUTcv"

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
    
    st.header("The Story Behind the Data")
    st.write("""
    New York City's iconic yellow and green taxis are a vital part of its transportation network, generating a massive amount of data with every trip. This project dives into this data to uncover the factors that influence whether a passenger leaves a good tip.

    Is tipping behavior influenced by the time of day, the length of the trip, or even the weather? By combining trip data from the NYC Taxi & Limousine Commission (TLC) with historical weather data, we can explore these questions and build a model to predict tip amounts.
    
    ### Dataset of Discovery
    
    This analysis is based on **2024 taxi trip records** for both yellow and green taxis, along with **hourly weather data** for NYC. We will explore:
    - **Tipping Patterns**: What are the characteristics of trips with high, middle, and low tips?
    - **Temporal Trends**: How do tip amounts vary by hour, day of the week, or month?
    - **Weather's Impact**: Does rain, snow, or temperature affect a passenger's generosity?
    - **Predictive Insights**: Can we build a reliable model to predict the tip class of a future trip?
    
    ### Why This Matters
    
    Understanding tipping behavior can provide valuable insights for:
    - **Drivers**: To better understand their potential earnings and the factors that influence them.
    - **Passengers**: To gain awareness of tipping norms and how their trips compare.
    - **TLC**: For policy-making and understanding the taxi economy.
    """)

    st.subheader("Techniques Used")
    st.markdown("""
    **Data Preparation:**
    - Data downloading and merging (Yellow and Green taxis)
    - Data cleaning and feature engineering
    - Merging with weather data
    - Iterative imputation for missing values
    
    **Analysis Methods:**
    - Correlation analysis
    - Interactive visualizations of distributions and relationships
    - Temporal pattern analysis
    - Building and evaluating a classification model for tip prediction
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
        df_raw = load_parquet(taxi_data_url)
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
        df_preprocessed = load_parquet(taxi_data_preprocessed_missing_url)
        st.write(f"**Shape:** {df_preprocessed.shape[0]:,} rows × {df_preprocessed.shape[1]} columns")
        st.dataframe(df_preprocessed.head(10))
        st.write("This data has been cleaned, merged with weather data, and new features have been engineered.")

    with tab3:
        st.write("### Imputed Data")
        df_imputed = load_parquet(taxi_data_preprocessed_url)
        st.write(f"**Shape:** {df_imputed.shape[0]:,} rows × {df_imputed.shape[1]} columns")
        st.dataframe(df_imputed.head(10))
        st.write("Missing values in the preprocessed data have been filled using iterative imputation.")

def page_ida():
    """
    Display the Initial Data Analysis (IDA) page with missing values analysis,
    duplicates, statistical summaries, and preprocessing information.
    """
    st.header("Initial Data Analysis (IDA)")
    
    df_preprocessed = load_parquet(taxi_data_preprocessed_missing_url)

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")

    missing_df = (
        df_preprocessed.isna().sum()
        .to_frame("missing")
        .assign(percent=lambda x: (x["missing"] / len(df_preprocessed) * 100).round(2))
        .sort_values("missing", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_df = missing_df[missing_df['missing'] > 0]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
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
    df_imputed = load_parquet(taxi_data_preprocessed_url)

    impute_cols = [
        'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax',
        'tip_amount', 'tolls_amount', 'total_amount', 'duration_min', 'speed_mph',
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall',
        'wind_speed_10m'
    ]

    selected_col = st.selectbox("Select variable to compare:", [c for c in impute_cols if c in df_preprocessed.columns], key="impute_compare")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Before Imputation", "After Imputation"))

    fig.add_trace(
        go.Histogram(x=df_preprocessed[selected_col].dropna(), name="Original", marker_color='#3498db'),
        row=1, col=1
    )

    if selected_col in df_imputed.columns:
        fig.add_trace(
            go.Histogram(x=df_imputed[selected_col].dropna(), name="Imputed", marker_color='#2ecc71'),
            row=1, col=2
        )

    fig.update_layout(height=400, showlegend=True, title_text=f"Distribution Comparison: {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    numeric_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Statistical Summary (Numeric Variables)")
    st.dataframe(df_imputed[numeric_cols].describe().T, use_container_width=True)

    st.subheader("Interactive Distribution Analysis")
    num_col = st.selectbox("Select a numeric column to explore:", numeric_cols, key="ida_num")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_num = px.histogram(
            df_imputed,
            x=num_col,
            nbins=50,
            marginal="box",
            opacity=0.85,
            title=f"Distribution of {num_col}"
        )
        st.plotly_chart(fig_num, use_container_width=True)
    with col2:
        st.write("**Statistics:**")
        st.write(f"Mean: {df_imputed[num_col].mean():.2f}")
        st.write(f"Median: {df_imputed[num_col].median():.2f}")
        st.write(f"Std Dev: {df_imputed[num_col].std():.2f}")
        st.write(f"Min: {df_imputed[num_col].min():.2f}")
        st.write(f"Max: {df_imputed[num_col].max():.2f}")

    cat_cols = df_imputed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    st.subheader("Categorical Variable Frequency")
    cat_col = st.selectbox("Select a categorical column:", cat_cols, key="ida_cat")

    top_counts = df_imputed[cat_col].value_counts(dropna=False).head(20).reset_index()
    top_counts.columns = [cat_col, "count"]
    fig_cat = px.bar(top_counts, x=cat_col, y="count", title=f"Top 20 categories in {cat_col}")
    st.plotly_chart(fig_cat, use_container_width=True)

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
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def page_eda():
    """
    Display the Exploratory Data Analysis (EDA) page with correlation analysis,
    temporal patterns, and weather impact visualizations.
    """
    st.header("Exploratory Data Analysis and Visualization")

    df = load_parquet(taxi_data_preprocessed_url)

    st.subheader("Correlation Heatmap")
    st.write("Interactive correlation matrix showing relationships between all numeric features.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_method = st.radio("Select correlation method:", ["pearson", "spearman", "kendall"], horizontal=True)
    
    corr = df[numeric_cols].corr(method=corr_method)

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        origin="lower",
        title=f"Correlation Heatmap ({corr_method.capitalize()})",
        aspect="auto"
    )
    fig.update_layout(height=700, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Target Feature Correlation Analysis")
    st.write("Explore correlations of all features with a selected target variable.")

    target_col = st.selectbox(
        "Select target variable:", 
        numeric_cols,
        index=numeric_cols.index("tip_amount") if "tip_amount" in numeric_cols else 0,
        key="eda_target"
    )

    corrs = df[numeric_cols].corr(method='pearson')[target_col].sort_values(ascending=False)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            corrs.to_frame("correlation").reset_index().rename(columns={"index": "feature"}),
            use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Temporal Pattern Analysis")
    st.write("Analyze tipping patterns across different time periods.")

    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()

    fig = px.histogram(
        df,
        x='hour',
        y='tip_amount',
        histfunc='avg',
        nbins=24,
        title="Average Tip Amount by Hour of Day",
        labels={'hour': 'Hour of Day', 'tip_amount': 'Average Tip Amount'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tip Class Balance Analysis")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        tip_class_counts = df['tip_class'].value_counts()
        st.metric("Low Tips", f"{tip_class_counts.get('Low', 0):,}")
        st.metric("Middle Tips", f"{tip_class_counts.get('Middle', 0):,}")
        st.metric("High Tips", f"{tip_class_counts.get('High', 0):,}")

    with col2:
        fig = px.pie(
            names=tip_class_counts.index,
            values=tip_class_counts.values,
            title="Tip Class Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weather Impact on Tips")
    st.write("Explore how different weather conditions affect tip amounts.")

    weather_candidates = [c for c in numeric_cols if any(k in c.lower()
                          for k in ['temperature','precipitation','rain','snowfall','wind_speed'])]

    col1, col2 = st.columns(2)
    with col1:
        wcol = st.selectbox("Select weather feature:", weather_candidates, key="eda_weather_col")
    with col2:
        bins = st.slider("Number of bins:", 5, 20, 10, key="eda_weather_bins")

    if len(df[wcol].dropna()) > 0:
        qbins = pd.qcut(df[wcol], q=bins, duplicates='drop')
        tmp = df.assign(_bin=qbins).dropna(subset=['_bin'])
        rate = tmp.groupby('_bin')['tip_amount'].mean().reset_index()
        rate.columns = ['bin', 'avg_tip']
        rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)
        rate['bin_label'] = rate['bin'].astype(str)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rate['bin_mid'],
            y=rate['avg_tip'],
            mode='lines+markers',
            name='Average Tip',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title=f"Average Tip Amount vs {wcol}",
            xaxis_title=wcol,
            yaxis_title="Average Tip Amount",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)


def page_advanced_analysis():
    """
    Display advanced analysis page with multi-dimensional visualizations
    including scatter matrices, 3D plots, and distribution comparisons.
    """
    st.header("Advanced Analysis & Visualizations")
    
    st.subheader("Multi-Dimensional Analysis")
    
    df = load_parquet(taxi_data_preprocessed_url)

    st.write("### Interactive Scatter Matrix")
    st.write("Explore relationships between multiple numeric variables simultaneously.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_features = st.multiselect(
        "Select features for scatter matrix (3-5 recommended):",
        numeric_cols,
        default=[col for col in ['fare_amount', 'trip_distance', 'duration_min', 'speed_mph'] if col in numeric_cols][:4]
    )
    
    if len(selected_features) >= 2:
        fig = px.scatter_matrix(
            df.sample(min(1000, len(df))),
            dimensions=selected_features,
            color='tip_class',
            title="Pairwise Feature Relationships by Tip Class",
            labels={col: col.replace('_', ' ').title() for col in selected_features}
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.write("### 3D Relationship Visualization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis:", numeric_cols, index=numeric_cols.index('trip_distance'), key="3d_x")
    with col2:
        y_var = st.selectbox("Y-axis:", numeric_cols, index=numeric_cols.index('duration_min'), key="3d_y")
    with col3:
        z_var = st.selectbox("Z-axis:", numeric_cols, index=numeric_cols.index('fare_amount'), key="3d_z")

    sample_df = df[[x_var, y_var, z_var, 'tip_class']].dropna().sample(min(2000, len(df)))

    fig = px.scatter_3d(
        sample_df,
        x=x_var,
        y=y_var,
        z=z_var,
        color='tip_class',
        title=f"3D Visualization: {x_var} vs {y_var} vs {z_var}",
        labels={'tip_class': 'Tip Class'},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Distribution Comparison by Category")
    
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Select category:", cat_cols, index=cat_cols.index('tip_class'), key="box_cat")
    with col2:
        value = st.selectbox("Select numeric variable:", numeric_cols, index=numeric_cols.index('fare_amount'), key="box_val")

    fig = px.box(
        df,
        x=category,
        y=value,
        color=category,
        title=f"{value} Distribution by {category}",
        points="outliers"
    )
    st.plotly_chart(fig, use_container_width=True)


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
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Select Section:",
        ["Overview", "Data Collection", "IDA", "EDA & Visualization", "Advanced Analysis"],
        index=0
    )
    
    # Display project information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Project Info
    **Author:** Zhiqiang Ni  
    **Course:** CMSE 830  
    """)

    # Route to appropriate page based on menu selection
    if menu == "Overview":
        page_overview()
    elif menu == "Data Collection":
        page_data_collection()
    elif menu == "IDA":
        page_ida()
    elif menu == "EDA & Visualization":
        page_eda()
    elif menu == "Advanced Analysis":
        page_advanced_analysis()

# Entry point of the application
if __name__ == "__main__":
    main()