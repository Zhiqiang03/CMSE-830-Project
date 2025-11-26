#%%
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
#%%
BASE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
year = 2024

res = requests.get(BASE_URL)
soup = BeautifulSoup(res.content, 'html.parser')

links = soup.find_all('a')
links = []

pattern = re.compile(r'.*Taxi Trip.*', re.IGNORECASE)
for link in soup.find_all('a', title=pattern):
    href = link.get('href')
    if href and str(year) in href:
        links.append(href)
#%%
yellow_links = [link for link in links if 'yellow' in link.lower()]
green_links = [link for link in links if 'green' in link.lower()]
#%%
yellow_data = pd.concat(
    [pd.read_parquet(link) for link in yellow_links],
    ignore_index=True
)
green_data = pd.concat(
    [pd.read_parquet(link) for link in green_links],
    ignore_index=True
)
#%%
# add a column to identify the taxi type
yellow_data['is_yellow'] = True
green_data['is_yellow'] = False
#%%
yellow_data.columns
#%%
green_data.columns
#%%
yellow_data = yellow_data.rename(columns={
    'tpep_pickup_datetime': 'pickup_datetime',
    'tpep_dropoff_datetime': 'dropoff_datetime'
})
green_data = green_data.rename(columns={
    'lpep_pickup_datetime': 'pickup_datetime',
    'lpep_dropoff_datetime': 'dropoff_datetime'
})
#%%
yellow_data.columns
#%%
green_data.columns
#%%
combined_data = pd.concat([yellow_data, green_data], ignore_index=True)
#%%
combined_data.to_parquet("./data/taxi_data.parquet", index=False)
#%%
