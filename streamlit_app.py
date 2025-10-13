# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------
# 1. APP TITLE AND DESCRIPTION
# ------------------------------------------------------
st.set_page_config(page_title="CMSE 830 Data Project", layout="wide")

st.title("CMSE 830 Data Analysis Project")
st.markdown("""
This Streamlit app demonstrates data collection, cleaning, 
exploratory analysis, and visualization using multiple data sources.  
*Author:* Zhiqiang Ni  
*Course:* CMSE 830  
""")

st.header("Initial Data Analysis (IDA)")
df_raw = pd.read_csv('./data/ncr_ride_bookings.csv')
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

dup_count = df_raw.duplicated().sum()
st.write(f"Duplicate rows: {dup_count:,}")

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

st.header("Data Collection and Preparation")
st.write("""
Uber dataset is from https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard, and all the weather data is from https://www.visualcrossing.com/
these two dataset is combine into one by time column with in 30min.
""")

st.write("""
I have preprocessed the data to handle missing values, remove duplicates, and ensure consistent formatting. also did one-hot encoding for categorical variables.
""")

df = pd.read_csv('./data/rides_with_weather.csv')

st.write("**Preprocessed data preview:**")
st.dataframe(df.head())

st.header("Normalized Data")
st.write("After applying `MinMaxScaler`, all values are scaled to a range between 0 and 1.")
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

st.subheader("Normalized DataFrame Preview")
st.dataframe(df.head())

# visualization
st.header("Data Visualization")
ride_data = pd.read_csv("./data/ncr_ride_bookings.csv")
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


#draw a correlation heatmap
st.subheader("Correlation Heatmap")
st.write("Add a column 'is_cancelled' to indicate whether a ride was cancelled (1) or not (0).")
df['is_cancelled'] = ride_data['Booking Status'].apply(lambda x: 1 if 'cancel' in x.lower() else 0)

bool_cols = df.select_dtypes(include='bool').columns.tolist()
df_corr = df.copy()
if bool_cols:
    df_corr[bool_cols] = df_corr[bool_cols].astype(int)

selected_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df_corr[selected_cols].corr(method="pearson")
fig = px.imshow(
    corr,
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    origin="lower",
    title=f"Correlation Heatmap"
)
fig.update_layout(height=600, xaxis_title="", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.header("ℹ️ Documentation")
st.markdown("""

**GitHub Repo:** [your_repo_link_here](https://github.com/yourusername/your-repo)  
**Deployed App:** [your_streamlit_link_here](https://yourapp.streamlit.app)  
""")
