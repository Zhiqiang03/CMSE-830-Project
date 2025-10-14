# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def page_overview():
    st.title("CMSE 830 Data Analysis Project")
    st.markdown("""
    This Streamlit app demonstrates data collection, cleaning,
    exploratory analysis, and visualization using multiple data sources.
    \*Author:\* Zhiqiang Ni  
    \*Course:\* CMSE 830
    """)

def page_ida():
    st.header("Initial Data Analysis (IDA)")
    st.write("""
    Uber dataset is from https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard, and all the weather data is from https://www.visualcrossing.com/
    these two dataset is combine into one by time column with in 30min.
    """)

    st.header("Initial Data Analysis (IDA)")
    df_raw = load_csv('./data/ncr_ride_bookings.csv')
    st.subheader("Missing Values")
    missing_df = (
        df_raw.isna().sum()
        .to_frame("missing")
        .assign(percent=lambda x: (x["missing"] / len(df_raw) * 100).round(2))
        .sort_values("missing", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.subheader("Missingness Pattern")
    sample = df_raw.sample(n=min(500, len(df_raw)), random_state=0)
    plt.figure(figsize=(12, 4))
    sns.heatmap(sample.isna(), cbar=False)
    st.pyplot(plt.gcf())
    plt.clf()

    numeric_cols_raw = df_raw.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Numeric Summary (raw scale)")
    st.dataframe(df_raw[numeric_cols_raw].describe().T, use_container_width=True)

    st.subheader("Numeric Distribution")
    num_col = st.selectbox("Select a numeric column", numeric_cols_raw, key="ida_num")
    fig_num = px.histogram(
        df_raw,
        x=num_col,
        nbins=50,
        marginal="box",
        opacity=0.85,
        title=f"Distribution of {num_col}"
    )
    st.plotly_chart(fig_num, use_container_width=True)

    cat_cols = df_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    st.subheader("Categorical Frequency")
    cat_col = st.selectbox("Select a categorical column", cat_cols, key="ida_cat")
    top_counts = df_raw[cat_col].value_counts(dropna=False).head(20).reset_index()
    top_counts.columns = [cat_col, "count"]
    fig_cat = px.bar(top_counts, x=cat_col, y="count", title=f"Top levels in {cat_col}")
    st.plotly_chart(fig_cat, use_container_width=True)


    st.write("""
    I have preprocessed the data to handle missing values, remove duplicates, and ensure consistent formatting. also did one-hot encoding for categorical variables.
    
    I used `MinMaxScaler` to normalize the numeric features, scaling them to a range between 0 and 1. This helps improve model performance and convergence during training.
    
    I used KNN and EM algorithms to impute missing values in the dataset, the KNN is very slow for this large dataset, so I mainly used EM algorithm to impute missing values.
    """)

    df = load_csv('./data/rides_with_weather.csv')

    st.write("**Preprocessed data preview:**")
    st.dataframe(df.head())

    st.header("Normalized Data")
    st.write("After applying `MinMaxScaler`, all values are scaled to a range between 0 and 1.")

    df = load_csv('./data/rides_with_weather_processed.csv')

    st.subheader("Normalized DataFrame Preview")
    st.dataframe(df.head())

    # visualization
    st.header("Data Visualization")
    ride_data = load_csv("./data/ncr_ride_bookings.csv")
    fig = px.histogram(
        ride_data,
        x='Booking Status',
        color='Booking Status',  # Colors the bars based on the booking status
        title="Booking Status Distribution",
        labels={'Booking Status': 'Booking Status', 'count': 'Number of Rides'}, # Customize axis labels
        height=450  # Set the height of the plot
    )
    st.plotly_chart(fig)

    vehicle_counts = ride_data['Vehicle Type'].value_counts().index.tolist()

    fig = px.histogram(
        ride_data,
        x='Vehicle Type',
        color='Vehicle Type',  # Colors the bars based on the vehicle type (like 'hue')
        title="Vehicle Type Distribution",
        category_orders={"Vehicle Type": vehicle_counts},
        labels={'Vehicle Type': 'Vehicle Type', 'count': 'Number of Vehicles'}, # Custom axis labels
        height=450  # Set the height of the plot
    )
    st.plotly_chart(fig)

    st.subheader("data types")
    dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index()
    dtype_df.columns = ["Column", "Data Type"]
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def page_eda():
    st.header("Exploratory Data Analysis and Visualization")

    st.subheader("Correlation Heatmap")
    st.write("Add a column 'is_cancelled' to indicate whether a ride was cancelled (1) or not (0).")

    df = load_csv('./data/rides_with_weather_processed.csv')
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    df_corr = df.copy()
    df_corr[bool_cols] = df_corr[bool_cols].astype(int)

    corr = df_corr[df_corr.select_dtypes(include=[np.number]).columns.tolist()].corr(method="pearson")

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        origin="lower",
        title=f"Correlation Heatmap"
    )
    fig.update_layout(height=600, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation")

    df = load_csv('./data/rides_with_weather_processed.csv')

    df_corr = df.copy()
    bool_cols = df_corr.select_dtypes(include='bool').columns
    df_corr[bool_cols] = df_corr[bool_cols].astype(int)

    target_col = st.selectbox("Select target column", df_corr.columns, index=df_corr.columns.get_loc("Booking Value"), key="eda_target")

    df_corr[target_col] = pd.to_numeric(df_corr[target_col], errors='coerce')

    num_cols = df_corr.select_dtypes(include=[np.number]).columns
    if target_col not in num_cols:
        st.error(f"Target column '{target_col}' is not numeric.")
        return

    corr_mat = df_corr[num_cols].corr(method='pearson')
    corrs = corr_mat[target_col].sort_values(ascending=False)

    st.dataframe(
        corrs.to_frame("correlation").reset_index().rename(columns={"index": "feature"}),
        use_container_width=True,
        hide_index=True,
    )

    angle = np.arctan2(df['hour_sin'], df['hour_cos'])            # [-pi, pi]
    hour_float = (np.mod(angle, 2*np.pi) / (2*np.pi)) * 24         # [0, 24)
    df['hour'] = hour_float.round().astype(int) % 24
    df['hour'] = pd.Categorical(df['hour'], categories=list(range(24)), ordered=True)

    fig = px.histogram(
        df,
        x='hour',
        color='is_cancelled',
        barmode='group',
        category_orders={'hour': list(range(24))},
        nbins=24,
        title="Ride Bookings by Hour of Day (Cancelled vs Non-cancelled)"
    )
    fig.update_traces(xbins=dict(start=-0.5, end=23.5, size=1))
    fig.update_layout(bargap=0.0, bargroupgap=0.0)
    fig.update_xaxes(title='Hour (0–23)', tickmode='linear', tick0=0, dtick=1)

    st.plotly_chart(fig, use_container_width=True)

    df = load_csv('./data/rides_with_weather.csv')

    df['is_cancelled'] = df[['Booking Status_Cancelled by Customer', 'Booking Status_Cancelled by Driver']].any(axis=1)
    df['is_cancelled'] = df['is_cancelled'].astype(int)

    counts = df['is_cancelled'].value_counts(dropna=False).rename_axis('class').reset_index(name='count')
    counts['class'] = counts['class'].map({0: 'not_cancelled', 1: 'cancelled'}).fillna(counts['class'])
    fig = px.bar(counts, x='class', y='count', text='count', title="Class Balance")
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Cancel rate: {df['is_cancelled'].mean():.2%}")

    # Make bools numeric for correlation
    df_corr = df.copy()
    bool_cols = df_corr.select_dtypes(include='bool').columns
    df_corr[bool_cols] = df_corr[bool_cols].astype(int)

    num_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()

    # Weather correlation curve on original or normalized df
    weather_candidates = [c for c in num_cols if any(k in c.lower()
                           for k in ['temp','precip','rain','humidity','wind','pressure','visibility','snow'])]

    wcol = st.selectbox("Weather feature", weather_candidates, key="eda_weather_col")
    bins = st.slider("Quantile bins", 5, 20, 10, key="eda_weather_bins")

    # Ensure numeric for selected column
    df[wcol] = pd.to_numeric(df[wcol], errors='coerce')
    valid = df[wcol].dropna()

    if len(valid) > 0:
        qbins = pd.qcut(df[wcol], q=bins, duplicates='drop')
        tmp = df.assign(_bin=qbins).dropna(subset=['_bin'])
        rate = tmp.groupby('_bin')['is_cancelled'].mean().reset_index()
        rate.columns = ['bin', 'cancel_rate']
        rate['bin_mid'] = rate['bin'].apply(lambda iv: iv.mid if hasattr(iv, 'mid') else np.nan)

        fig = px.line(
            rate.sort_values('bin_mid'),
            x='bin_mid',
            y='cancel_rate',
            markers=True,
            labels={'bin_mid': wcol, 'cancel_rate': 'cancel rate'},
            title=f"Cancellation Rate vs {wcol} (quantile-binned)"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- Menu ----------
def main():
    menu = st.sidebar.radio(
        "Navigate",
        ["Overview", "IDA",  "EDA"],
        index=0
    )

    if menu == "Overview":
        page_overview()
    elif menu == "IDA":
        page_ida()
    elif menu == "EDA":
        page_eda()

if __name__ == "__main__":
    main()