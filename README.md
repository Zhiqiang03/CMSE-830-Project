# Uber Ride Analytics & Weather Impact Analysis

A comprehensive data analysis project examining the relationship between Uber ride bookings and weather conditions in Delhi, India. This project demonstrates data collection, preprocessing, exploratory data analysis, and interactive visualization using Streamlit.

**Author:** Zhiqiang Ni  
**Course:** CMSE 830 - Foundations of Data Science  
**Institution:** Michigan State University

## Project Overview

This project analyzes Uber ride booking data from the NCR (National Capital Region) of Delhi combined with historical weather data to understand:
- Patterns in ride cancellations
- Impact of weather conditions on booking behavior
- Temporal trends in ride demand
- Vehicle type preferences and their relationship to cancellations

### Data Sources

- **Uber Dataset:** [Kaggle - Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
- **Weather Data:** [Visual Crossing Weather API](https://www.visualcrossing.com/) (January 2024 - December 2024)

The datasets are merged based on timestamp matching (within 30-minute windows) to correlate ride bookings with weather conditions.

## Features

### 1. Initial Data Analysis (IDA)
- Missing value analysis and visualization
- Data quality assessment
- Summary statistics for numeric and categorical variables
- Distribution analysis with interactive visualizations

### 2. Data Preprocessing
- Handling missing values using KNN and EM algorithms
- One-hot encoding for categorical variables
- Feature scaling using MinMaxScaler (0-1 normalization)
- Duplicate removal and data cleaning

### 3. Exploratory Data Analysis (EDA)
- Correlation analysis between features
- Booking status distribution analysis
- Vehicle type analysis
- Temporal pattern analysis (hourly booking trends)
- Weather impact on cancellation rates
- Interactive visualizations with Plotly

## Installation & Setup

### Prerequisites

- Python 3.13 or higher

### Step 1: Install Dependencies

```bash
cd CMSE-830-Project
pip install -r requirements.txt
```

### Step 3: Verify Data Files

Ensure all data files are present in the `data/` directory. The main datasets should be:
- `ncr_ride_bookings.csv`
- `rides_with_weather.csv`
- `rides_with_weather_processed.csv`

## Usage

### Running the Streamlit App

To launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Navigation

The app has three main sections accessible via the sidebar:

1. **Overview** - Project introduction and context
2. **IDA** - Initial Data Analysis with missing value analysis, distributions, and data types
3. **EDA** - Exploratory Data Analysis with correlations, temporal patterns, and weather impact

### Working with Notebooks

To explore the analysis notebooks:

```bash
jupyter notebook
```

Then open:
- `main.ipynb` - Complete analysis workflow
- `preprocess.ipynb` - Data preprocessing steps
- `basic_weather.ipynb` - Weather data exploration

## Key Findings

- **Cancellation Rate:** The dataset shows a measurable cancellation rate for ride bookings
- **Temporal Patterns:** Booking patterns vary significantly by hour of day
- **Weather Impact:** Weather conditions (temperature, precipitation, humidity) correlate with cancellation rates
- **Vehicle Type Distribution:** Different vehicle types show varying booking and cancellation patterns