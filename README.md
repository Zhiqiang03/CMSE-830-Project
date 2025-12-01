# NYC Taxi Ride Analysis & Tip Prediction

> **🌟 LIVE DEPLOYMENT NOTICE 🌟**  
> This application is **Streamlit Cloud's free tier** so it will run out of memory.
> If you encounter "😦 Oh no." errors, see the [Troubleshooting](#troubleshooting) section below.

A comprehensive data science project examining the factors that influence taxi trip tips in New York City. This project demonstrates advanced data collection, preprocessing, exploratory data analysis, feature engineering, and machine learning modeling to predict tip amounts using over 1 million trip records.

**Author:** Zhiqiang Ni  
**Course:** CMSE 830 - Foundations of Data Science  
**Institution:** Michigan State University  

**Live Demo:** https://cmse-830-project-ni.streamlit.app/

---

## Project Overview

This project analyzes NYC taxi data to uncover the factors that influence passenger tipping behavior by:
- Combining 2024 taxi trip records with hourly weather data
- Processing and analyzing 3M+ taxi trips (smart sampling for analysis)
- Building and comparing 8 models
- Creating an interactive Streamlit dashboard with memory-optimized operations
- Deploying on Streamlit Cloud with resource-efficient techniques

### Key Metrics
- **Data Sources:** 3 (Yellow Taxi, Green Taxi, Weather API)
- **Records Processed:** 3M+ trips
- **ML Models Trained:** 7
- **Best Model Accuracy:** 63.42% (SVM Linear SGD)
- **Interactive Visualizations:** 15+

---

## Data Sources

### 1. NYC Taxi & Limousine Commission (TLC) Trip Record Data
**Source:** [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- Yellow and Green taxi trip records for 2024
- **Key Features:**
  - Pickup/dropoff timestamps and locations
  - Trip distance and duration
  - Fare amount, tips, tolls, and surcharges
  - Payment type and passenger count
  - Rate code and store/forward flag

### 2. Open-Meteo Weather API
**Source:** [Open-Meteo](https://open-meteo.com/)

- Historical hourly weather data for NYC (2024)
- **Key Features:**
  - Temperature (2m above ground)
  - Precipitation and rain
  - Snowfall and snow depth
  - Wind speed and direction
  - Weather code (condition)

---

## Features & Capabilities

### 1. Data Collection (`download_data.ipynb`)
- Automated download of 2024 yellow and green taxi data from NYC TLC
- Historical weather data fetching from Open-Meteo API
- Data validation and initial quality checks

### 2. Data Preprocessing (`pre_processing.ipynb`)
- **Data Cleaning:**
  - Removes trips with invalid values (negative fares, zero distance, extreme outliers)
  - Filters out unrealistic trip durations and speeds
  - Handles duplicate records
  
- **Feature Engineering:**
  - `duration_min`: Trip duration in minutes
  - `speed_mph`: Average trip speed
  - `tip_percentage`: Tip as percentage of fare
  - `tip_class`: Categorical tip classification (Low: 0-10%, Middle: 10-20%, High: 20%+)
  - Temporal features: `pickup_hour`, `pickup_day`, `day_of_week`
  
- **Data Integration:**
  - Merges yellow and green taxi datasets
  - Joins weather data based on pickup timestamp
  - Creates unified feature set with 104 features

- **Smart Sampling for Analysis:**
  - Implements random sampling for memory-efficient analysis
- Imputes key numerical features while preserving data distributions
- Comparative analysis of pre/post imputation data quality

### 4. Exploratory Data Analysis (Interactive Streamlit App)
- **Distribution Analysis:**
  - Fare amounts, trip distances, and durations
  - Tip percentages and tip class distributions
  - Temporal patterns (hourly, daily, weekly)

- **Correlation Analysis:**
  - Feature correlation heatmaps
  - Top features influencing tip behavior
  - Multi-dimensional relationship exploration

- **Weather Impact Analysis:**
  - Temperature vs. tipping patterns
  - Precipitation effects on tips
  - Weather condition categorization

- **Geospatial Analysis:**
  - Pickup/dropoff location patterns
  - Borough-level tipping behavior

### 5. Machine Learning Models (`models.py`)

Seven classification models trained to predict tip classes:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Prediction Time |
|-------|----------|-----------|--------|----------|---------------|-----------------|
| **SVM (Linear SGD)** ⭐ | **63.42%** | **0.619** | **0.634** | **0.572** | **6.16s** | **0.03s** |
| K-Nearest Neighbors | 56.83% | 0.521 | 0.568 | 0.535 | 0.07s | 422.42s |
| Random Forest | 54.32% | 0.635 | 0.543 | 0.552 | 84.49s | 1.00s |
| Hist Gradient Boosting | 47.53% | 0.631 | 0.475 | 0.512 | 67.00s | 3.06s |
| Decision Tree | 47.19% | 0.582 | 0.472 | 0.504 | 26.07s | 0.06s |
| Logistic Regression | 45.96% | 0.626 | 0.460 | 0.499 | 104.74s | 0.13s |
| Naive Bayes | 22.65% | 0.386 | 0.227 | 0.187 | 0.69s | 0.23s |

**Best Model: SVM (Linear SGD)**
- Highest accuracy at 63.42%
- Fast training (6.16s) and prediction (0.03s)
- Excellent balance between performance and efficiency
- Particularly strong at identifying middle-tip class (94% recall)

### 6. Interactive Prediction Tool
- Real-time tip prediction using the best-trained model
- User-friendly input interface for trip parameters
- Instant classification with probability scores
- Explanatory insights about prediction factors

---

## Installation & Setup

### Prerequisites

- Python 3.13 or higher
- pip package manager
- 2GB+ free disk space (for data storage)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd CMSE-830-Project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the data pipeline (optional - pre-processed data included):**
   ```bash
   # Step 1: Download raw data
   jupyter notebook download_data.ipynb
   
   # Step 2: Preprocess and engineer features
   jupyter notebook pre_processing.ipynb
   
   # Step 3: Handle missing values
   jupyter notebook impute.ipynb
   
   # Step 4: Train models
   python models.py
   ```

   **Note:** Pre-processed data and trained models are already included in the repository, so you can skip directly to running the Streamlit app.

---

## Usage

### Running the Streamlit Dashboard

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Streamlit App Navigation

The application includes 8 interactive pages with memory-optimized rendering:

1. **Overview** - Project introduction and key highlights
2. **Data Collection** - Information about data sources and collection process
3. **Initial Data Analysis** - Basic statistics and data quality assessment (uses sampling)
4. **EDA & Visualization** - Interactive exploratory data analysis (optimized with 50K samples)
5. **Advanced Analysis** - Deep-dive statistical analysis and insights (optimized with 50K samples)
6. **Model Evaluation** - Model comparison and performance metrics
7. **Interactive Prediction** - Real-time tip prediction tool
8. **Methodology** - Technical details and research approach

**Note:** Pages showing "Analysis based on a sample of X records" use representative sampling to ensure fast loading and smooth performance on Streamlit Cloud while maintaining statistical accuracy.

### Training Models

To retrain models with your own data:

```bash
python models.py
```

This will:
- Load preprocessed data from `./data/`
- Train all 7 models
- Display performance metrics
- Save trained models to `./models/`

---

## Technologies Used

### Data Processing & Analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **DuckDB** - In-memory analytical database for efficient data queries and memory-optimized operations

### Machine Learning
- **scikit-learn** - ML models, preprocessing, and evaluation
  - Linear models (Logistic Regression, SVM)
  - Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
  - Instance-based (K-Nearest Neighbors)
  - Probabilistic (Naive Bayes)
  - IterativeImputer for missing data

### Visualization
- **Plotly** - Interactive plots and dashboards
- **Matplotlib** - Static plots and figures
- **Seaborn** - Statistical data visualization

### Web Application
- **Streamlit** - Interactive web dashboard framework with optimized caching
- **Requests** - HTTP library for API calls
- **BeautifulSoup4** - Web scraping and HTML parsing

### Development Tools
- **Jupyter Notebook** - Interactive development and analysis
- **Python 3.13+** - Core programming language

## Key Findings

### Tipping Behavior Insights

1. **Time Matters**
   - Peak tipping occurs during evening hours (6-9 PM)
   - Weekends show slightly higher tip percentages
   - Late-night trips (after midnight) have more variable tipping

2. **Trip Characteristics**
   - Longer trips don't always mean higher tip percentages
   - Inverse relationship: higher fares often get lower tip percentages
   - Short trips (<2 miles) show more generous tipping behavior

3. **Weather Impact**
   - Comfortable temperatures (60-75°F) correlate with better tipping
   - Heavy precipitation slightly reduces tip percentages
   - Extreme weather conditions increase tip variability

4. **Passenger Behavior**
   - Multiple passengers tend to tip more generously
   - Credit card payments show higher tips than cash
   - Yellow and green taxis exhibit different tipping patterns

5. **Model Performance**
   - SVM (Linear SGD) achieves 63.42% accuracy
   - Middle-tip class is easiest to predict (94% recall)
   - High-tip class is most challenging (only 2% recall)
   - Feature engineering significantly improved model performance

---

## Model Performance Details

### SVM (Linear SGD) - Best Model

**Overall Metrics:**
- Accuracy: 63.42%
- Precision: 0.619
- Recall: 0.634
- F1-Score: 0.572

**Per-Class Performance:**
- **Low Tips (0-10%):** 78% precision, 36% recall
- **Middle Tips (10-20%):** 61% precision, 94% recall
- **High Tips (20%+):** 18% precision, 2% recall

**Why This Model Works:**
- Efficiently handles high-dimensional data (104 features)
- Fast training and prediction times
- Robust to outliers with linear decision boundaries
- Excellent at identifying the dominant middle-tip class

---

## Troubleshooting

### ⚠️ IMPORTANT: Streamlit Cloud Memory Issues ⚠️
### Streamlit Cloud Issues
**"😦 Oh no." Error:**
If you encounter this error when viewing the app on Streamlit Cloud:
- This may be due to resource limitations on the free tier.
- Try running the app locally using `streamlit run streamlit_app.py` after installing dependencies
---

## Author

**Zhiqiang Ni**
- Course: CMSE 830 - Foundations of Data Science
- Institution: Michigan State University
- Project: NYC Taxi Ride Analysis & Tip Prediction

**Live Demo:** https://cmse-830-project-ni.streamlit.app/

