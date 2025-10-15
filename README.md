# Uber Ride Analytics & Weather Impact Analysis

A comprehensive data analysis project examining the relationship between Uber ride bookings and weather conditions in Delhi, India. This project demonstrates data collection, preprocessing, exploratory data analysis, and interactive visualization using Streamlit.

**Author:** Zhiqiang Ni  
**Course:** CMSE 830 - Foundations of Data Science  
**Institution:** Michigan State University

**Live Demo:** https://cmse-830-project-ni.streamlit.app/

- **Uber Dataset:** [Kaggle - Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
  - ~15,000 ride bookings from 2024
  - Includes booking status, timestamps, vehicle types, locations, ratings
- **Weather Data:** [Visual Crossing Weather API](https://www.visualcrossing.com/)
  - Hourly weather observations for Delhi (January 2024 - December 2024)
  - Temperature, humidity, precipitation, wind speed, visibility, and more

The datasets are merged based on timestamp matching (within 30-minute windows) to correlate ride bookings with weather conditions.

## Features

- **Weather Data:** [Visual Crossing Weather API](https://www.visualcrossing.com/) (January 2024 - December 2024)
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
  - Cyclical encoding for temporal features (hour, minute, wind direction)
  - Feature scaling using MinMaxScaler (0-1 normalization)
- **Data Quality Improvements:**
  - Duplicate removal
  - Timestamp standardization
  - Data type validation
  - Creation of aggregated `is_cancelled` indicator

### 4. Exploratory Data Analysis (EDA)
- **Correlation Analysis:**
  - Interactive heatmaps with multiple methods (Pearson, Spearman, Kendall)
  - Target feature correlation ranking
  - Weather vs. cancellation correlation analysis
- **Visualization Types:**
  - Histograms with marginal distributions
  - Box plots with outlier detection
  - Pie charts for class balance
  - Bar charts for categorical frequencies
- **Temporal Analysis:**
  - Hourly booking patterns
  - Cancellation trends over time
  - Peak hour identification
- **Weather Impact:**
  - Quantile-based weather impact curves
  - Interactive binning controls
  - Cancellation rate vs. weather conditions

### 5. Advanced Analysis
- **Multi-dimensional Visualizations:**
  - 3D scatter plots with rotation
  - Interactive scatter matrices
  - Distribution comparisons by category
- **Statistical Analysis:**
  - In-depth correlation studies
  - Class balance analysis
  - Feature importance assessment

## Interactive Features

The Streamlit app includes 10+ interactive elements:
- Navigation menu with 5 distinct pages
- Variable selectors for custom analysis
- Correlation method toggles (Pearson, Spearman, Kendall)
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

The app has 5 main sections:

1. **Overview** - Project introduction, context, and story behind the data
2. **Data Collection** - Raw data sources, integration process, and data characteristics
3. **IDA** - Initial Data Analysis with missing value analysis, duplicates, distributions, and data types
4. **EDA & Visualization** - Exploratory Data Analysis with correlations, temporal patterns, weather impact, and class balance
5. **Advanced Analysis** - Multi-dimensional visualizations including 3D plots and scatter matrices

### Working with Notebooks

To explore the analysis notebooks:

```bash
jupyter notebook
```

Then open:
- `preprocess.ipynb` - Data preprocessing and feature engineering steps
- `visualization.ipynb` - Visualization experiments and exploratory analysis
- `basic_weather.ipynb` - Weather data exploration

## Key Findings

- **Cancellation Rate:** The dataset shows measurable cancellation patterns that vary with time and conditions
- **Weather Impact:** Weather variables show correlations with ride cancellations, though relationships may be non-linear
- **Temporal Patterns:** Peak booking hours identified with distinct cancellation patterns (morning/evening rush hours)
- **Vehicle Types:** Different vehicle types show varying performance across conditions
- **Data Quality:** EM algorithm provided best balance of computational efficiency and data quality preservation for imputation
- **Class Balance:** Cancellation vs. completion rates analyzed to understand booking outcome distributions

## Technical Details

### Data Processing Pipeline

1. **Data Collection:** Raw Uber booking data + Weather API data
2. **Data Cleaning:** Remove duplicates, handle missing values, standardize formats
3. **Data Integration:** Timestamp-based merging with 30-minute tolerance
4. **Imputation:** EM algorithm for missing values
5. **Feature Engineering:** One-hot encoding, cyclical encoding, normalization
6. **Analysis:** Correlation analysis, visualization, pattern detection

### Technologies Used

- **Frontend/Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Analysis:** SciPy, scikit-learn
- **Development:** Jupyter Notebook, Python 3.13

## Requirements

See `requirements.txt` for full list. Key packages:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.14.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
