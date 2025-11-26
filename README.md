# NYC Taxi Ride Analysis & Tip Prediction

A comprehensive data analysis project examining the factors that influence taxi trip tips in New York City. This project demonstrates data collection, preprocessing, exploratory data analysis, and machine learning modeling to predict tip amounts.

**Author:** Zhiqiang Ni
**Course:** CMSE 830 - Foundations of Data Science
**Institution:** Michigan State University

**Live Demo:** https://cmse-830-project-ni.streamlit.app/

## Data Sources

- **NYC Taxi & Limousine Commission (TLC) Trip Record Data:** [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  - Yellow and Green taxi trip records for 2024.
  - Includes pickup/dropoff times, locations, fares, tolls, and other trip details.
- **Open-Meteo Weather API:** [Open-Meteo](https://open-meteo.com/)
  - Historical hourly weather data for NYC (2024).
  - Includes temperature, precipitation, wind speed, and other weather conditions.

## Features

### 1. Data Collection
- Downloads 2024 yellow and green taxi data from the NYC TLC website.
- Fetches historical weather data from the Open-Meteo API.

### 2. Data Preprocessing
- Merges yellow and green taxi data.
- Cleans the data by removing trips with invalid values (e.g., negative fares, zero distance).
- Engineers new features such as `duration_min`, `speed_mph`, and `tip_class`.
- Merges the trip data with weather data based on the pickup time.

### 3. Imputation
- Uses `IterativeImputer` from scikit-learn to fill in missing values for key numerical features.

### 4. Exploratory Data Analysis (EDA)
- Interactive visualizations of trip data, including distributions of fares, distances, and durations.
- Analysis of the relationship between weather conditions and trip characteristics.
- Correlation analysis to identify factors that may influence tip amounts.

### 5. Tip Prediction
- The goal of the project is to build a model to predict the tip class ('Low', 'Middle', 'High') based on trip and weather data.

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
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

3. **Run the data pipeline:**
   ```bash
   python download_data.py
   python pre_processing.py
   python impute.py
   ```

## Usage

### Running Locally

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Technologies Used

- **Frontend/Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Data Collection:** Requests, BeautifulSoup
- **Machine Learning:** scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Development:** Jupyter Notebook, Python

## Requirements

See `requirements.txt` for the full list of packages.
