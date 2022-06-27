
import sys
import pickle
import pandas as pd
import numpy as np

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

year = int(sys.argv[1])
month = int(sys.argv[2])

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


y_pred

print(np.mean(y_pred))

year = 2021
month = 2

df_result = pd.DataFrame()
df_result['prediction'] = y_pred
df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df_result.index.astype('str')
#df_result

#    df_result.to_parquet(
#        'output.parquet',
#        engine='pyarrow',
#        compression=None,
#        index=False
#    )



