#%%
import pandas as pd
import numpy as np
#%%
df = pd.read_parquet('./data/taxi_data.parquet')
df.head()
#%%
df = df[df['total_amount'] > 0] #Remove bad data (negative fares and zero fares)
df = df[df['trip_distance'] > 0] #Remove bad data (negative distances and zero distances)
df = df[df['payment_type'] != 2] #Remove cash payments
df = df[df['fare_amount'] > 0] #Remove bad data (negative fares and zero fares)
df = df[df['mta_tax'] >= 0] # Remove bad data (negative MTA tax)
df = df[df['tip_amount'] >= 0] # Remove bad data (negative tips)
df = df[df['tolls_amount'] >= 0] # Remove bad data (negative tolls)
df = df[df['extra'] >= 0] # Remove bad data (negative extra charges)
#%%
def classify_tip(pct):
    if pct < 0.15:
        return 'Low'
    elif 0.15 <= pct <= 0.22:
        return 'Middle'
    else:
        return 'High'

df['tip_pct'] = df['tip_amount'] / (df['total_amount'] - df['tip_amount']) #Calculate tip percentage
df['tip_class'] = df['tip_pct'].apply(classify_tip)
df.drop(columns=['tip_pct', 'tip_amount', 'total_amount'], inplace=True)
#%%
df.columns
#%%
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# remove wrong datetime entries only 2024 data is kept
df = df[(df['pickup_datetime'].dt.year == 2024)]
df = df[df['dropoff_datetime'] > df['pickup_datetime']]

df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Calculate duration in minutes
df['duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60

df = df[df['duration_min'] > 0]  # Remove trips with non-positive duration
#%%
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 40.7834,
	"longitude": -73.9663,
	"start_date": "2024-01-01",
	"end_date": "2024-12-31",
	"hourly": ["temperature_2m", "apparent_temperature", "rain", "snowfall", "precipitation", "wind_speed_10m", "weather_code"],
	"timezone": "America/New_York",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(6).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["precipitation"] = hourly_precipitation
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["weather_code"] = hourly_weather_code

weather_df = pd.DataFrame(data = hourly_data)

#%%
df
#%%
df['Airport_fee'] = df['Airport_fee'].fillna(0)
df['congestion_surcharge'] = df['congestion_surcharge'].fillna(0)