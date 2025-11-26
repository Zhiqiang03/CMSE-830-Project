#%%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#%%
df = pd.read_parquet('./data/taxi_data_preprocessed_missing.parquet')
#%%


mice_cols = [
    'passenger_count', 'RatecodeID', 'trip_distance',
    'fare_amount', 'PULocationID', 'DOLocationID',
    'extra', 'mta_tax', 'tolls_amount',
    'congestion_surcharge', 'Airport_fee', 'duration_min',
    'speed_mph'
]

df_mice_input = df[mice_cols].copy()
mice_imputer = IterativeImputer(max_iter=10, random_state=0)

df_mice_output = mice_imputer.fit_transform(df_mice_input)
df_imputed = pd.DataFrame(df_mice_output, columns=mice_cols)

df['passenger_count'] = df_imputed['passenger_count'].round().astype(int)
df['RatecodeID'] = df_imputed['RatecodeID'].round().astype(int)

print(df[['passenger_count', 'RatecodeID']].isnull().sum())
#%%
df.to_parquet('./data/taxi_data_preprocessed.parquet', index=False)
#%%
