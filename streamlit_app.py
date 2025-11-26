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


@st.cache_resource
def get_duckdb_connection():
    """
    Create and cache a DuckDB connection.
    """
    # Using an in-memory database
    con = duckdb.connect(database=':memory:', read_only=False)
    # Load the parquet file into a table
    con.execute("CREATE OR REPLACE TABLE rides AS SELECT * FROM read_parquet('./data/combined.parquet')")
    return con

@st.cache_data
def query_duckdb(query: str) -> pd.DataFrame:
    """
    Run a query on the DuckDB connection and return the result as a DataFrame.
    """
    con = get_duckdb_connection()
    return con.execute(query).fetchdf()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV file with caching to improve performance.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_csv(path)

def page_overview():
    """
    Display the project overview page with introduction, data sources,
    techniques used, and research questions.
    """
    st.title("NYC Taxi Tip Classification & Weather Impact")
    st.markdown("""
    ### CMSE 830 Data Analysis Project
    
    This Streamlit app explores **tipping behavior in NYC taxi rides** and builds a model to 
    **classify whether a ride will receive a high tip** based on ride characteristics and weather.

    We combine:
    - **NYC Yellow / Green Taxi trip records**
    - **NYC weather data** matched by time

    to study how **time, distance, location, and weather** relate to tipping behavior.

    **Target:** High vs low/no tip (e.g., tip ≥ 20% of fare)
    
    **Author:** Zhiqiang Ni  
    **Course:** CMSE 830  
    """)
    
    st.header("The Story Behind the Data")
    st.write("""
    Imagine you're taking a yellow cab in Manhattan on a rainy Friday night.
    You ride across town, pay the fare, and then decide how much to tip.

    Tipping is influenced by many factors:
    - Trip distance and fare
    - Time of day, day of week
    - Weather conditions
    - Payment type and passenger behavior

    This project asks: **When are riders most likely to give a *high* tip?**
    And how do **weather and trip characteristics** shape tipping behavior in NYC taxis?
    """)

    st.subheader("Techniques Used")
    st.markdown("""
    **Data Preparation:**
    - Join NYC taxi trip data with hourly/daily NYC weather
    - Feature engineering: tip rate, time-of-day, day-of-week, weekend, etc.
    - Filtering unrealistic trips (e.g., zero distance, negative fares)
    - Handling outliers in fare, distance, and tip

    **Analysis Methods:**
    - Exploratory analysis of tip amount and tip percentage
    - Temporal and weather-based pattern analysis
    - Class definition: high vs low tip based on tip percentage
    - Supervised classification models (e.g., Logistic Regression, Random Forest)
    - Model evaluation: confusion matrix, precision, recall, F1-score, ROC-AUC
    """)

def page_data_collection():
    """
    Display the data collection and preparation page, showing raw data sources
    and the integration process.
    """
    st.header("Data Collection and Preparation")

    # Display data source information at the top
    st.write("""
    This project combines **NYC taxi trip records** with **NYC weather data** 
    to analyze and predict taxi tipping behavior.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **NYC Taxi Trip Dataset**
        - **Source**: NYC Taxi & Limousine Commission (TLC)
        - **Rides**: Yellow and Green taxi trips in NYC
        - **Time Period**: 2022 and 2023
        - **Key Features**: pickup/dropoff datetime, locations, distance, fare, tip, payment type, passenger count
        """)

    with col2:
        st.markdown("""
        **NYC Weather Dataset**
        - **Source**: Visual Crossing Weather API (or similar)
        - **Coverage**: January 2022 - December 2023
        - **Location**: New York City
        - **Key Features**: Temperature, precipitation, humidity, wind speed, visibility, atmospheric pressure
        """)

    st.info("These datasets were combined by matching the taxi pickup time to the corresponding hourly weather record.")

    st.markdown("---")

    st.subheader("Detailed Data Exploration")

    # Create tabs for different data sources
    tab1, tab2 = st.tabs(["Combined Taxi & Weather Data", "Data Integration Strategy"])

    with tab1:
        st.write("### Combined NYC Taxi and Weather Data")
        # Load and display combined data
        df_trips = query_duckdb("SELECT * FROM rides LIMIT 1000")
        st.write(f"**Shape (sample):** {df_trips.shape[0]:,} rows × {df_trips.shape[1]} columns")
        st.dataframe(df_trips.head(10))

        st.write("### Data Characteristics")
        # Display key metrics about the dataset
        total_rides_query = "SELECT COUNT(*) FROM rides"
        total_rides = query_duckdb(total_rides_query).iloc[0,0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rides", f"{total_rides:,}")
        with col2:
            st.metric("Time Period", "2022")
        with col3:
            st.metric("Taxi Types", "Yellow & Green")

    with tab2:
        st.write("### Data Integration & Feature Engineering")
        st.markdown("""
        **Merging Strategy:**
        1. Taxi trip records (Yellow & Green) and hourly weather data for NYC were loaded.
        2. The taxi pickup datetime was rounded to the nearest hour.
        3. Trips were joined with the weather data based on this rounded hourly timestamp.
        
        **Feature Engineering & Target Label:**
        1. **`tip_rate`**: Calculated as `tip_amount / fare_amount` to normalize for trip cost.
        2. **`high_tip` (Target Label)**: A binary flag created to classify tips:
           - **High Tip (1)**: `tip_rate` is 20% or more (`>= 0.20`).
           - **Low/No Tip (0)**: `tip_rate` is less than 20% (`< 0.20`).
        3. Trips with invalid data (e.g., zero fare, negative tips) were filtered out.
        """)
        
        # Display merged dataset with new features
        df_merged = query_duckdb("SELECT *, (tip_amount / fare_amount) as tip_rate, (tip_rate >= 0.2) as high_tip FROM rides WHERE fare_amount > 0 LIMIT 10")
        st.write(f"**Example of Integrated Data with Engineered Features:**")
        st.dataframe(df_merged)

        st.info(f"Successfully created a unified dataset with {total_rides:,} records for analysis.")

def page_ida():
    """
    Display the Initial Data Analysis (IDA) page with missing values analysis,
    duplicates, statistical summaries, and preprocessing information.
    """
    st.header("Initial Data Analysis (IDA)")
    
    # Get a sample of the data for analysis to keep it fast
    df_raw = query_duckdb("SELECT * FROM rides USING SAMPLE 100000 ROWS")
    df_raw['tip_rate'] = (df_raw['tip_amount'] / df_raw['fare_amount']).replace([np.inf, -np.inf], np.nan)

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")
    st.write("Analysis of missing data in a 100,000-row sample. The full dataset was cleaned during preprocessing.")

    # Calculate missing values and percentages
    missing_df = (
        df_raw.isna().sum()
        .to_frame("missing")
        .assign(percent=lambda x: (x["missing"] / len(df_raw) * 100).round(2))
        .sort_values("missing", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_df = missing_df[missing_df['missing'] > 0]

    # Display missing values table and summary metrics
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    with col2:
        st.metric("Total Missing Values", f"{missing_df['missing'].sum():,}")
        st.metric("Columns with Missing Data", len(missing_df))

    st.write("**Action Taken:** Missing values were handled during preprocessing, often by removing rows where essential data (like fare or distance) was missing, or by imputation for less critical fields.")

    st.markdown("---")

    # Numeric Summary
    numeric_cols_raw = df_raw.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Statistical Summary (Numeric Variables)")
    st.write("Summary statistics for key numeric columns in the dataset.")
    st.dataframe(df_raw[numeric_cols_raw].describe().T, use_container_width=True)

    # Interactive Distribution
    st.subheader("Interactive Distribution Analysis")
    default_numeric = 'tip_rate' if 'tip_rate' in numeric_cols_raw else numeric_cols_raw[0]
    num_col = st.selectbox("Select a numeric column to explore:", numeric_cols_raw, index=numeric_cols_raw.index(default_numeric), key="ida_num")

    if num_col:
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create histogram with marginal box plot
            fig_num = px.histogram(
                df_raw,
                x=num_col,
                nbins=50,
                marginal="box",
                opacity=0.85,
                title=f"Distribution of {num_col}"
            )
            st.plotly_chart(fig_num, use_container_width=True)
        with col2:
            # Display summary statistics
            st.write("**Statistics:**")
            st.write(f"Mean: {df_raw[num_col].mean():.2f}")
            st.write(f"Median: {df_raw[num_col].median():.2f}")
            st.write(f"Std Dev: {df_raw[num_col].std():.2f}")
            st.write(f"Min: {df_raw[num_col].min():.2f}")
            st.write(f"Max: {df_raw[num_col].max():.2f}")

    # Categorical Analysis
    cat_cols = df_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Add some high-cardinality numeric columns that act as categories
    for col in ['payment_type', 'VendorID', 'RatecodeID']:
        if col in df_raw.columns and col not in cat_cols:
            cat_cols.append(col)

    st.subheader("Categorical Variable Frequency")
    default_cat = 'payment_type' if 'payment_type' in cat_cols else cat_cols[0]
    cat_col = st.selectbox("Select a categorical column:", cat_cols, index=cat_cols.index(default_cat), key="ida_cat")

    if cat_col:
        # Display top 20 categories in bar chart
        top_counts = df_raw[cat_col].value_counts(dropna=False).head(20).reset_index()
        top_counts.columns = [cat_col, "count"]
        fig_cat = px.bar(top_counts, x=cat_col, y="count", title=f"Top 20 categories in {cat_col}")
        st.plotly_chart(fig_cat, use_container_width=True)

    # Initial Visualizations
    st.header("Initial Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment type distribution
        if 'payment_type' in df_raw.columns:
            fig = px.histogram(
                df_raw,
                x='payment_type',
                color='payment_type',
                title="Payment Type Distribution",
                labels={'payment_type': 'Payment Type', 'count': 'Number of Rides'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Vendor distribution
        if 'VendorID' in df_raw.columns:
            vendor_counts = df_raw['VendorID'].value_counts().index.tolist()
            fig = px.histogram(
                df_raw,
                x='VendorID',
                color='VendorID',
                title="Taxi Vendor Distribution",
                category_orders={"VendorID": vendor_counts},
                labels={'VendorID': 'Vendor ID', 'count': 'Count'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)



def page_eda():
    """
    Display the Exploratory Data Analysis (EDA) page with correlation analysis,
    temporal patterns, and weather impact visualizations.
    """
    st.header("Exploratory Data Analysis: Tipping Behavior")

    # Load data and perform initial transformations
    df = query_duckdb("""
        SELECT *, 
               (tip_amount / fare_amount) as tip_rate,
               (tip_rate >= 0.2) as high_tip
        FROM rides 
        WHERE fare_amount > 0 AND tip_amount >= 0
    """)
    df['high_tip'] = df['high_tip'].astype(int)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("Interactive correlation matrix showing relationships between numeric features.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude irrelevant or redundant columns
    exclude_cols = ['VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'passenger_count', 'payment_type']
    corr_cols = [col for col in numeric_cols if col not in exclude_cols and 'id' not in col.lower()]

    # Allow user to select correlation method
    corr_method = st.radio("Select correlation method:", ["pearson", "spearman"], horizontal=True)

    # Calculate correlation matrix
    corr = df[corr_cols].corr(method=corr_method)

    # Create interactive heatmap
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Correlation Heatmap ({corr_method.capitalize()})",
        aspect="auto"
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # Feature Correlation Analysis
    st.subheader("Target Feature Correlation Analysis")
    st.write("Explore correlations of all features with a selected target variable.")

    # Allow user to select target variable
    target_col = st.selectbox(
        "Select target variable:", 
        ['high_tip', 'tip_rate', 'tip_amount'],
        index=0,
        key="eda_target"
    )

    # Calculate correlations with target variable
    corrs = df[corr_cols].corr(method='pearson')[target_col].sort_values(ascending=False)

    col1, col2 = st.columns([1, 2])

    with col1:
        # Display correlation table
        st.dataframe(
            corrs.to_frame("correlation").reset_index().rename(columns={"index": "feature"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        # Display bar chart of top correlations
        top_n = st.slider("Show top N correlations", 5, 30, 15)
        top_corrs = corrs.abs().drop(target_col).sort_values(ascending=False).head(top_n)
        fig = px.bar(
            x=top_corrs.values,
            y=top_corrs.index,
            orientation='h',
            title=f"Top {top_n} Features Correlated with {target_col}",
            labels={'x': 'Absolute Correlation', 'y': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Temporal Analysis
    st.subheader("Temporal Pattern Analysis")
    st.write("Analyze tipping patterns across different time periods.")

    df['hour'] = df['tpep_pickup_datetime'].dt.hour

    # Create histogram of rides by hour, colored by high_tip
    fig = px.histogram(
        df,
        x='hour',
        color='high_tip',
        barmode='group',
        category_orders={'hour': list(range(24))},
        nbins=24,
        title="Taxi Rides by Hour of Day (High Tip vs. Low/No Tip)",
        labels={'hour': 'Hour of Day', 'count': 'Number of Rides', 'high_tip': 'Tip Class'}
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    # Class Balance Analysis
    st.subheader("Tip Classification Balance")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display tipping metrics
        high_tip_rate = df['high_tip'].mean()
        st.metric("High-Tip Rate", f"{high_tip_rate:.2%}")
        st.metric("Total Rides Analyzed", f"{len(df):,}")
        st.metric("High-Tip Rides", f"{df['high_tip'].sum():,}")
        st.metric("Low/No-Tip Rides", f"{(df['high_tip'] == 0).sum():,}")

    with col2:
        # Create pie chart of tip classes
        counts = df['high_tip'].value_counts().reset_index()
        counts.columns = ['Tip Class', 'Count']
        counts['Tip Class'] = counts['Tip Class'].map({0: 'Low/No Tip', 1: 'High Tip'})

        fig = px.pie(
            counts, 
            values='Count', 
            names='Tip Class',
            title="High Tip vs. Low/No Tip Distribution",
            color_discrete_map={'Low/No Tip': '#3498db', 'High Tip': '#2ecc71'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Weather Impact Analysis
    st.subheader("Weather Impact on Tipping")
    st.write("Explore how different weather conditions affect tip rates.")

    weather_candidates = [c for c in df.columns if any(k in c.lower()
                          for k in ['temp','precip','humidity','wind','pressure','visibility','snow'])]

    col1, col2 = st.columns(2)
    with col1:
        wcol = st.selectbox("Select weather feature:", weather_candidates, index=weather_candidates.index('temp'), key="eda_weather_col")
    with col2:
        bins = st.slider("Number of bins:", 5, 20, 10, key="eda_weather_bins")

    # Calculate average tip rate across weather bins
    df[wcol] = pd.to_numeric(df[wcol], errors='coerce')
    valid = df[[wcol, 'tip_rate']].dropna()

    if len(valid) > 0:
        # Create quantile-based bins
        qbins = pd.qcut(valid[wcol], q=bins, duplicates='drop')
        tmp = valid.assign(_bin=qbins)
        rate = tmp.groupby('_bin')['tip_rate'].mean().reset_index()
        rate.columns = ['bin', 'avg_tip_rate']
        rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)

        # Create line plot of tip rate vs weather feature
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rate['bin_mid'],
            y=rate['avg_tip_rate'],
            mode='lines+markers',
            name='Average Tip Rate',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title=f"Average Tip Rate vs {wcol}",
            xaxis_title=wcol,
            yaxis_title="Average Tip Rate (%)",
            yaxis_tickformat=".2%",
            hovermode='x unified'
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
        ["Overview", "Data Collection", "IDA", "EDA & Visualization"],
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

# Entry point of the application
if __name__ == "__main__":
    main()