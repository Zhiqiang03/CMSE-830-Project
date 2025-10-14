# Uber Ride Analytics & Weather Impact Analysis

A comprehensive data analysis project examining the relationship between Uber ride bookings and weather conditions in Delhi, India. This project demonstrates data collection, preprocessing, exploratory data analysis, and interactive visualization using Streamlit.

**Author:** Zhiqiang Ni  
**Course:** CMSE 830 - Foundations of Data Science  
**Institution:** Michigan State University

**Live Demo:** https://cmse-830-project-ni.streamlit.app/

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

### 1. Data Collection & Preparation
- **Multiple Data Sources:** Integration of Uber booking data with weather API data
- **Data Cleaning:** Comprehensive missing value analysis and handling
- **Advanced Integration:** Time-based merging with 30-minute window tolerance

### 2. Initial Data Analysis (IDA)
- Missing value analysis and visualization
- Duplicate detection and removal
- Data quality assessment
- Summary statistics for numeric and categorical variables
- Distribution analysis with interactive visualizations
- Data type documentation

### 3. Data Preprocessing
- **Multiple Imputation Techniques:**
  - K-Nearest Neighbors (KNN) imputation
  - Expectation-Maximization (EM) algorithm
  - Mean/Median baseline imputation
- **Feature Engineering:**
  - One-hot encoding for categorical variables
  - Cyclical encoding for temporal features
  - Feature scaling using MinMaxScaler (0-1 normalization)
- **Data Quality Improvements:**
  - Duplicate removal
  - Timestamp standardization
  - Data type validation

### 4. Exploratory Data Analysis (EDA)
- **Correlation Analysis:**
  - Interactive heatmaps with multiple methods (Pearson, Spearman, Kendall)
  - Target feature correlation ranking
- **Visualization Types:**
  - Histograms with marginal distributions
  - Box plots with outlier detection
  - Pie charts for class balance
  - Bar charts for categorical frequencies
- **Temporal Analysis:**
  - Hourly booking patterns
  - Cancellation trends over time
- **Weather Impact:**
  - Quantile-based weather impact curves
  - Interactive binning controls

### 5. Advanced Analysis
- **Multi-dimensional Visualizations:**
  - 3D scatter plots with rotation
  - Interactive scatter matrices
  - Distribution comparisons by category
- **Statistical Analysis:**
  - In-depth correlation studies
  - Class balance analysis
  - Feature importance assessment

### 6. Imputation Comparison
- Side-by-side comparison of imputation methods
- Performance metrics (computation time, variance preservation)
- Visual distribution comparisons before/after imputation
- Recommendations for method selection

## Interactive Features

The Streamlit app includes 10+ interactive elements:
- Navigation menu with 6 distinct pages
- Variable selectors for custom analysis
- Correlation method toggles
- Sliders for binning and feature selection
- Multi-select for scatter matrix dimensions
- Tabs for organized content exploration
- Interactive Plotly charts (zoom, pan, hover)

## Installation & Setup

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone [your-repo-url]
cd CMSE-830-Project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data files:**
Ensure all data files are present in the `data/` directory:
- `ncr_ride_bookings.csv` (raw Uber data)
- `weather_data.csv` (compiled weather data)
- `rides_with_weather.csv` (merged dataset)
- `rides_with_weather_processed.csv` (normalized dataset)

## Usage

### Running Locally

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Navigation

The app has 6 main sections:

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

- **Cancellation Rate:** Measurable cancellation rate varies with weather conditions
- **Weather Impact:** Temperature, humidity, and precipitation show correlation with cancellations
- **Temporal Patterns:** Peak booking hours identified with distinct cancellation patterns
- **Vehicle Types:** Different vehicle types show varying performance across conditions
- **Data Quality:** EM algorithm provided best balance for imputation on this dataset


## Requirements

See `requirements.txt` for full list. Key packages:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.14.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0