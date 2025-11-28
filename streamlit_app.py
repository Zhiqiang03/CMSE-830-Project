# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Centralized data paths for easy updates
DATA_DIR = "./data"
TAXI_SAMPLE = f"{DATA_DIR}/taxi_data_sampled.parquet"
TAXI_PREPROCESSED_MISSING_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed_missing_sampled.parquet"
TAXI_PREPROCESSED_SAMPLED = f"{DATA_DIR}/taxi_data_preprocessed_sampled.parquet"

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
    st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig_num, use_container_width=True)
    with col2:
        st.write("**Statistics:**")
        st.write(f"Mean: {stats['mean']:.2f}")
        st.write(f"Median: {stats['median']:.2f}")
        st.write(f"Std Dev: {stats['std']:.2f}")
        st.write(f"Min: {stats['min']:.2f}")
        st.write(f"Max: {stats['max']:.2f}")

    cat_cols = df_imputed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    st.subheader("Categorical Variable Frequency")
    cat_col = st.selectbox("Select a categorical column:", cat_cols, key="ida_cat")

    top_counts = df_imputed[cat_col].value_counts(dropna=False).head(20).reset_index()
    top_counts.columns = [cat_col, "count"]
    fig_cat = px.bar(top_counts, x=cat_col, y="count", title=f"Top 20 categories in {cat_col}")
    st.plotly_chart(fig_cat, width='stretch')

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
    st.info("💡 **tip_class** (0=Low, 1=Middle, 2=High) is our target variable for classification.")

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
        rate = tmp.groupby('_bin')['fare_amount'].mean().reset_index()
        rate.columns = ['bin', 'avg_fare']
        rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)
        rate['bin_label'] = rate['bin'].astype(str)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rate['bin_mid'],
            y=rate['avg_fare'],
            mode='lines+markers',
            name='Average Fare',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title=f"Average Fare Amount vs {wcol}",
            xaxis_title=wcol,
            yaxis_title="Average Fare Amount",
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')


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