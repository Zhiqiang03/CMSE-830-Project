# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    st.title("Uber Ride Analytics & Weather Impact Analysis")
    st.markdown("""
    ### CMSE 830 Data Analysis Project
    
    This Streamlit app demonstrates comprehensive data analysis including:
    - **Data Collection** from multiple sources
    - **Data Cleaning** and preprocessing techniques
    - **Exploratory Data Analysis** with advanced visualizations
    - **Missing Data Handling** with multiple imputation techniques
    
    **Author:** Zhiqiang Ni  
    **Course:** CMSE 830  
    """)
    
    st.header("The Story Behind the Data")
    st.write("""
    Imagine you in Delhi, and you need to book an Uber to get home. 
    You open the app, request a ride, and it gets cancelled. Then another. And another. 
    Frustrated, you wonder: *Is this just bad luck, or is there a pattern here?*
    
    This project will explore **Do weather conditions actually impact ride-sharing behavior?** 
    And if so, how much does rain, temperature, or humidity affect whether your ride gets completed or cancelled?
    
    ### Our Dataset of Discovery
    
    Collected with over **15,000 Uber ride bookings** from Delhi's National Capital Region and detailed 
    **hourly weather data** spanning all of 2024, we set out to uncover the hidden patterns. This isn't 
    just about numbers on a screen—it's about understanding the real-world interplay between weather, 
    human behavior, and urban mobility.
    
    Through this analysis, we explore:
    - **Weather's Hidden Influence**: How do rain, temperature extremes, and humidity affect cancellation rates?
    - **Temporal Rhythms**: When are rides most likely to be cancelled? Morning rush? Late night?
    - **Vehicle Dynamics**: Do different vehicle types show different patterns in challenging weather?
    - **Data-Driven Insights**: What do correlations reveal about the factors that truly matter?
    
    ### Why This Matters
    
    Understanding these patterns has real implications:
    - **For Riders**: Better predict when you might face cancellations and plan accordingly
    - **For Drivers**: Optimize when and where to drive based on weather conditions
    - **For Platforms**: Improve algorithms to better match supply and demand during weather events
    - **For Cities**: Plan better transportation infrastructure and backup options
    """)

    st.subheader("Techniques Used")
    st.markdown("""
    **Data Preparation:**
    - Missing value analysis and imputation (KNN, EM algorithms)
    - Duplicate removal and data validation
    - One-hot encoding for categorical variables
    - MinMaxScaler normalization
    - Timestamp merging (30-min windows)
    
    **Analysis Methods:**
    - Correlation analysis (Pearson, Spearman, Kendall)
    - Distribution analysis with interactive visualizations
    - Temporal pattern detection
    - Weather impact assessment
    - Multi-dimensional relationship exploration
    - Interactive filtering and exploration
    """)

def page_data_collection():
    """
    Display the data collection and preparation page, showing raw data sources
    and the integration process.
    """
    st.header("Data Collection and Preparation")

    # Display data source information at the top
    st.write("""
    This project combines two primary data sources to analyze the relationship between 
    weather conditions and Uber ride cancellations in Delhi, India throughout 2024.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Uber Ride Booking Dataset**
        - **Source**: [Kaggle - Uber Ride Analytics](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
        - **Records**: ~15,000 ride bookings
        - **Time Period**: Throughout 2024
        - **Location**: Delhi NCR (National Capital Region), India
        - **Key Features**: Booking details, timestamps, vehicle types, locations, booking status
        """)

    with col2:
        st.markdown("""
        **Weather Dataset**
        - **Source**: [Visual Crossing Weather API](https://www.visualcrossing.com/)
        - **Coverage**: January 2024 - December 2024
        - **Location**: Delhi, India (NCR region)
        - **Granularity**: Hourly observations
        - **Key Features**: Temperature, precipitation, humidity, wind speed, visibility, atmospheric pressure
        """)

    st.info("These datasets were combined based on timestamp matching within 30-minute windows to create a comprehensive dataset for analysis.")

    st.markdown("---")

    st.subheader("Detailed Data Exploration")

    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["Uber Ride Data", "Weather Data", "Data Integration"])
    
    with tab1:
        st.write("### Raw Uber Ride Booking Data")
        # Load and display raw ride booking data
        df_rides = load_csv('./data/ncr_ride_bookings.csv')
        st.write(f"**Shape:** {df_rides.shape[0]:,} rows × {df_rides.shape[1]} columns")
        st.dataframe(df_rides.head(10))
        
        st.write("### Data Characteristics")
        # Display key metrics about the dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bookings", f"{len(df_rides):,}")
        with col2:
            st.metric("Unique Customers", f"{df_rides['Customer ID'].nunique():,}" if 'Customer ID' in df_rides.columns else "N/A")
        with col3:
            st.metric("Time Period", "2024")
    
    with tab2:
        st.write("### Weather Data Collection")
        # Load and display weather data
        df_weather = load_csv('./data/weather_data.csv')
        st.write(f"**Shape:** {df_weather.shape[0]:,} rows × {df_weather.shape[1]} columns")
        st.dataframe(df_weather.head(10))
        
        st.write("### Weather Variables Collected")
        # Extract and display weather-related column names
        weather_vars = [col for col in df_weather.columns if any(x in col.lower()
                       for x in ['temp', 'precip', 'humid', 'wind', 'pressure', 'visibility'])]
        st.write(", ".join(weather_vars[:10]))
    
    with tab3:
        st.write("### Data Integration Process")
        st.markdown("""
        **Merging Strategy:**
        1. Convert timestamps to datetime format
        2. Round ride booking times to nearest 30-minute interval
        3. Match with corresponding weather observations
        4. Handle unmatched records appropriately
        """)
        
        # Display merged dataset
        df_merged = load_csv('./data/rides_with_weather.csv')
        st.write(f"**Merged Dataset Shape:** {df_merged.shape[0]:,} rows × {df_merged.shape[1]} columns")
        st.dataframe(df_merged.head(10))
        
        st.info(f"Successfully merged {len(df_merged):,} records with weather data")

def page_ida():
    """
    Display the Initial Data Analysis (IDA) page with missing values analysis,
    duplicates, statistical summaries, and preprocessing information.
    """
    st.header("Initial Data Analysis (IDA)")
    
    # Duplicate Analysis
    st.subheader("Duplicate Records")
    df_raw = load_csv('./data/ncr_ride_bookings.csv')
    duplicates = df_raw.duplicated().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duplicate Rows", f"{duplicates:,}")
    with col2:
        st.metric("Duplicate Percentage", f"{(duplicates/len(df_raw)*100):.2f}%")
    
    st.write("**Action Taken:** All duplicate records were removed during preprocessing.")

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")

    # Calculate missing values and percentages
    missing_df = (
        df_raw.isna().sum()
        .to_frame("missing")
        .assign(percent=lambda x: (x["missing"] / len(df_raw) * 100).round(2))
        .sort_values("missing", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )

    # Display missing values table and summary metrics
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    with col2:
        st.metric("Total Missing Values", f"{missing_df['missing'].sum():,}")
        st.metric("Columns with Missing Data", len(missing_df[missing_df['missing'] > 0]))

    # Visualize missingness pattern using heatmap
    st.subheader("Missingness Pattern Visualization")
    sample = df_raw.sample(n=min(500, len(df_raw)), random_state=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(sample.isna(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Data Pattern (500 sample rows)")
    st.pyplot(fig)
    plt.close()

    # Imputation Techniques Section
    st.header("Missing Data Handling & Imputation Techniques")

    st.write("""
    Multiple imputation techniques were evaluated to handle missing values in the dataset.
    """)

    st.subheader("Imputation Methods Comparison")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **1. K-Nearest Neighbors (KNN) Imputation**
        - Uses similarity between samples to impute missing values
        - Considers k=5 nearest neighbors
        - **Pros**: Preserves local patterns
        - **Cons**: Computationally expensive for large datasets
        
        **2. Expectation-Maximization (EM) Algorithm**
        - Iterative method based on maximum likelihood estimation
        - Better suited for datasets with complex missing patterns
        - **Pros**: Handles multivariate missingness well
        - **Cons**: Assumes multivariate normality
        """)

    with col2:
        st.markdown("""
        **3. Mean/Median Imputation (Baseline)**
        - Simple replacement with column mean or median
        - Used as baseline for comparison
        - **Pros**: Fast and simple
        - **Cons**: Doesn't preserve variance or relationships
        
        **Performance Comparison:**
        """)

        # Create comparison table
        comparison_data = {
            'Method': ['Mean/Median', 'KNN (k=5)', 'EM Algorithm'],
            'Computation Time': ['< 1 sec', '>5 min', '< 1 min'],
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.success("**Method Selected:** EM Algorithm was chosen as the primary imputation method due to its balance of computational efficiency and data quality preservation for this large dataset.")

    # Imputation Impact Visualization
    st.subheader("Imputation Impact Visualization")

    st.write("Compare distributions before and after imputation:")

    # Load data before and after imputation
    df_imputed = load_csv('./data/rides_with_weather.csv')
    targets = ['Avg VTAT', 'Avg CTAT', 'Driver Ratings', 'Customer Rating', 'Ride Distance', 'Booking Value']
    selected_col = st.selectbox("Select variable to compare:", targets, key="impute_compare")

    # Create side-by-side histograms
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Before Imputation", "After Imputation"))

    # Before imputation histogram
    fig.add_trace(
        go.Histogram(x=df_raw[selected_col].dropna(), name="Original", marker_color='#3498db'),
        row=1, col=1
    )

    # After imputation histogram
    if selected_col in df_imputed.columns:
        fig.add_trace(
            go.Histogram(x=df_imputed[selected_col].dropna(), name="Imputed", marker_color='#2ecc71'),
            row=1, col=2
        )

    fig.update_layout(height=400, showlegend=True, title_text=f"Distribution Comparison: {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Numeric Summary
    numeric_cols_raw = df_raw.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Statistical Summary (Numeric Variables)")
    st.dataframe(df_raw[numeric_cols_raw].describe().T, use_container_width=True)

    # Interactive Distribution
    st.subheader("Interactive Distribution Analysis")
    num_col = st.selectbox("Select a numeric column to explore:", numeric_cols_raw, key="ida_num")
    
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
    st.subheader("Categorical Variable Frequency")
    cat_col = st.selectbox("Select a categorical column:", cat_cols, key="ida_cat")

    # Display top 20 categories in bar chart
    top_counts = df_raw[cat_col].value_counts(dropna=False).head(20).reset_index()
    top_counts.columns = [cat_col, "count"]
    fig_cat = px.bar(top_counts, x=cat_col, y="count", title=f"Top 20 categories in {cat_col}")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Data Processing Section
    st.header("Data Preprocessing & Cleaning")
    
    st.subheader("Techniques Applied:")
    st.markdown("""
    1. **Missing Value Handling:**
       - KNN Imputation: K-Nearest Neighbors algorithm for numeric features
       - EM Algorithm: Expectation-Maximization for complex patterns
       - Note: KNN was computationally intensive for large dataset
    
    2. **Data Encoding:**
       - One-hot encoding for categorical variables (Booking Status, Vehicle Type, etc.)
       - Binary encoding for boolean features
    
    3. **Feature Scaling:**
       - MinMaxScaler applied to normalize numeric features to [0, 1] range
       - Improves model convergence and performance
    
    4. **Data Quality:**
       - Removed duplicate records
       - Standardized date/time formats
       - Validated data types
    """)

    # Display preprocessed data
    df_processed = load_csv('./data/rides_with_weather.csv')
    st.write("**Preprocessed Data Preview:**")
    st.dataframe(df_processed.head(10))

    st.subheader("Normalized Data (MinMaxScaler)")
    st.write("All numeric values scaled to range [0, 1]")
    df_normalized = load_csv('./data/rides_with_weather_processed.csv')
    st.dataframe(df_normalized.head(10))

    # Visualizations
    st.header("Initial Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Booking status distribution
        ride_data = load_csv("./data/ncr_ride_bookings.csv")
        fig = px.histogram(
            ride_data,
            x='Booking Status',
            color='Booking Status',
            title="Booking Status Distribution",
            labels={'Booking Status': 'Status', 'count': 'Number of Rides'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vehicle type distribution
        vehicle_counts = ride_data['Vehicle Type'].value_counts().index.tolist()
        fig = px.histogram(
            ride_data,
            x='Vehicle Type',
            color='Vehicle Type',
            title="Vehicle Type Distribution",
            category_orders={"Vehicle Type": vehicle_counts},
            labels={'Vehicle Type': 'Type', 'count': 'Count'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data Types
    st.subheader("Data Types Overview")
    dtype_df = pd.DataFrame(df_normalized.dtypes, columns=["Data Type"]).reset_index()
    dtype_df.columns = ["Column", "Data Type"]
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def page_eda():
    """
    Display the Exploratory Data Analysis (EDA) page with correlation analysis,
    temporal patterns, and weather impact visualizations.
    """
    st.header("Exploratory Data Analysis and Visualization")

    # Load processed data
    df = load_csv('./data/rides_with_weather_processed.csv')
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("Interactive correlation matrix showing relationships between all numeric features.")
    
    # Convert boolean columns to integers for correlation calculation
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    df_corr = df.copy()
    df_corr[bool_cols] = df_corr[bool_cols].astype(int)

    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()

    # i want to exclude some columns that are not relevant for correlation analysis
    exclude_cols = ['Booking Status_Cancelled by Customer',
       'Booking Status_Cancelled by Driver', 'Booking Status_Completed',
       'Booking Status_Incomplete', 'Booking Status_No Driver Found']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_corr = df_corr[numeric_cols].copy()


    # Allow user to select correlation method
    corr_method = st.radio("Select correlation method:", ["pearson", "spearman", "kendall"], horizontal=True)
    
    # Calculate correlation matrix
    corr = df_corr[numeric_cols].corr(method=corr_method)

    # Create interactive heatmap
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

    # Feature Correlation Analysis
    st.subheader("Target Feature Correlation Analysis")
    st.write("Explore correlations of all features with a selected target variable.")

    # Allow user to select target variable
    target_col = st.selectbox(
        "Select target variable:", 
        df_corr.columns, 
        index=df_corr.columns.get_loc("Booking Value") if "Booking Value" in df_corr.columns else 0, 
        key="eda_target"
    )

    # Calculate correlations with target variable
    df_corr[target_col] = pd.to_numeric(df_corr[target_col], errors='coerce')
    num_cols = df_corr.select_dtypes(include=[np.number]).columns
    
    if target_col in num_cols:
        corr_mat = df_corr[num_cols].corr(method='pearson')
        corrs = corr_mat[target_col].sort_values(ascending=False)
        
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
            top_corrs = corrs.abs().sort_values(ascending=False).head(top_n)
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
    st.write("Analyze booking patterns across different time periods.")
    
    # Reconstruct hour from cyclical encoding (sin/cos)
    if 'hour_sin' in df.columns and 'hour_cos' in df.columns:
        # Convert sin/cos back to hour using arctan2
        angle = np.arctan2(df['hour_sin'], df['hour_cos'])
        hour_float = (np.mod(angle, 2*np.pi) / (2*np.pi)) * 24
        df['hour'] = hour_float.round().astype(int) % 24
        df['hour'] = pd.Categorical(df['hour'], categories=list(range(24)), ordered=True)

        # Create histogram of bookings by hour
        fig = px.histogram(
            df,
            x='hour',
            color='is_cancelled',
            barmode='group',
            category_orders={'hour': list(range(24))},
            nbins=24,
            title="Ride Bookings by Hour of Day (Cancelled vs Non-cancelled)",
            labels={'hour': 'Hour of Day', 'count': 'Number of Bookings'}
        )
        fig.update_traces(xbins=dict(start=-0.5, end=23.5, size=1))
        fig.update_layout(bargap=0.0, bargroupgap=0.0)
        fig.update_xaxes(title='Hour (0–23)', tickmode='linear', tick0=0, dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    # Class Balance Analysis
    st.subheader("Class Balance Analysis")

    # Load raw data and create cancellation indicator
    df_raw = load_csv('./data/rides_with_weather.csv')
    df_raw['is_cancelled'] = df_raw[['Booking Status_Cancelled by Customer', 'Booking Status_Cancelled by Driver']].any(axis=1).astype(int)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display cancellation metrics
        cancel_rate = df_raw['is_cancelled'].mean()
        st.metric("Cancellation Rate", f"{cancel_rate:.2%}")
        st.metric("Total Rides", f"{len(df_raw):,}")
        st.metric("Cancelled Rides", f"{df_raw['is_cancelled'].sum():,}")
        st.metric("Completed Rides", f"{(~df_raw['is_cancelled'].astype(bool)).sum():,}")
    
    with col2:
        # Create pie chart of booking status
        counts = df_raw['is_cancelled'].value_counts().reset_index()
        counts.columns = ['Cancelled', 'Count']
        counts['Cancelled'] = counts['Cancelled'].map({0: 'Completed', 1: 'Cancelled'})
        
        fig = px.pie(
            counts, 
            values='Count', 
            names='Cancelled',
            title="Booking Status Distribution",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Weather Impact Analysis
    st.subheader("Weather Impact on Cancellations")
    st.write("Explore how different weather conditions affect ride cancellation rates.")
    
    # Prepare data for weather analysis
    df_corr = df_raw.copy()
    bool_cols = df_corr.select_dtypes(include='bool').columns
    df_corr[bool_cols] = df_corr[bool_cols].astype(int)
    num_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()

    # Filter for weather-related columns
    weather_candidates = [c for c in num_cols if any(k in c.lower()
                          for k in ['temp','precip','rain','humidity','wind','pressure','visibility','snow'])]

    col1, col2 = st.columns(2)
    with col1:
        wcol = st.selectbox("Select weather feature:", weather_candidates, key="eda_weather_col")
    with col2:
        bins = st.slider("Number of bins:", 5, 20, 10, key="eda_weather_bins")

    # Calculate cancellation rate across weather bins
    df_raw[wcol] = pd.to_numeric(df_raw[wcol], errors='coerce')
    valid = df_raw[wcol].dropna()

    if len(valid) > 0:
        # Create quantile-based bins
        qbins = pd.qcut(df_raw[wcol], q=bins, duplicates='drop')
        tmp = df_raw.assign(_bin=qbins).dropna(subset=['_bin'])
        rate = tmp.groupby('_bin')['is_cancelled'].mean().reset_index()
        rate.columns = ['bin', 'cancel_rate']
        rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)
        rate['bin_label'] = rate['bin'].astype(str)

        # Create line plot of cancellation rate vs weather feature
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rate['bin_mid'],
            y=rate['cancel_rate'],
            mode='lines+markers',
            name='Cancellation Rate',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title=f"Cancellation Rate vs {wcol}",
            xaxis_title=wcol,
            yaxis_title="Cancellation Rate",
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
    
    # Load data and create cancellation indicator
    df = load_csv('./data/rides_with_weather.csv')
    df['is_cancelled'] = df[['Booking Status_Cancelled by Customer', 'Booking Status_Cancelled by Driver']].any(axis=1).astype(int)
    
    # Scatter Matrix
    st.write("### Interactive Scatter Matrix")
    st.write("Explore relationships between multiple numeric variables simultaneously.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Allow user to select features for scatter matrix
    selected_features = st.multiselect(
        "Select features for scatter matrix (3-5 recommended):",
        numeric_cols,
        default=[col for col in ['Booking Value', 'Ride Distance', 'temperature', 'humidity'] if col in numeric_cols][:4]
    )
    
    if len(selected_features) >= 2:
        # Create scatter matrix with sample data (for performance)
        fig = px.scatter_matrix(
            df.sample(min(1000, len(df))),
            dimensions=selected_features,
            color='is_cancelled',
            title="Pairwise Feature Relationships",
            labels={col: col.replace('_', ' ').title() for col in selected_features}
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D Scatter Plot
    st.write("### 3D Relationship Visualization")
    
    # Allow user to select 3 variables for 3D plot
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis:", numeric_cols, index=0, key="3d_x")
    with col2:
        y_var = st.selectbox("Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1), key="3d_y")
    with col3:
        z_var = st.selectbox("Z-axis:", numeric_cols, index=min(2, len(numeric_cols)-1), key="3d_z")
    
    # Sample data for performance
    sample_df = df[[x_var, y_var, z_var, 'is_cancelled']].dropna().sample(min(2000, len(df)))
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        sample_df,
        x=x_var,
        y=y_var,
        z=z_var,
        color='is_cancelled',
        title=f"3D Visualization: {x_var} vs {y_var} vs {z_var}",
        labels={'is_cancelled': 'Cancelled'},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot comparison
    st.write("### Distribution Comparison by Category")
    
    # Get categorical columns including boolean
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols_with_bool = cat_cols + ['is_cancelled']
    
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Select category:", cat_cols_with_bool, key="box_cat")
    with col2:
        value = st.selectbox("Select numeric variable:", numeric_cols, key="box_val")
    
    # Create box plot comparing distributions across categories
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