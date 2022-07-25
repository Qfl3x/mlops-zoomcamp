import pandas as pd
from datetime import datetime

from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def test_output():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df_prepared = prepare_data(df, ['PUlocationID', 'DOlocationID'])

    assert df_prepared.shape[0] == 2

