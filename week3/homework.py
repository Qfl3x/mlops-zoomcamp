import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from datetime import date, datetime

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

import pickle

@task
def get_paths(date):
    if date == None:
        date = date.today()
    elif type(date) == str:
        time_format = '%Y-%m-%d'
        date = datetime.strptime(date, time_format)

    current_month = date.month

    training_month = current_month - 2
    validation_month = current_month - 1

    train_path = f"./data/fhv_tripdata_2021-{training_month:02d}.parquet"
    val_path = f"./data/fhv_tripdata_2021-{validation_month:02d}.parquet"
    return train_path, val_path, str(date)

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def get_vectorizer(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    return dv

@task
def train_model(df, categorical, dv):
    train_dicts = df[categorical].to_dict(orient='records')
    X_train = dv.transform(train_dicts)
    y_train = df.duration.values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    train_path, val_path, date = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical)
    print("Done Processing")
    #Vectorize
    dv = get_vectorizer(df_train_processed, categorical)
    # train the model
    lr = train_model(df_train_processed, categorical, dv)
    run_model(df_val_processed, categorical, dv, lr)

    saved_model_file = f"models/model-{date}.bin"
    saved_dv_file = f"models/dv-{date}.b"

    with open(saved_model_file, "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(saved_dv_file, "wb") as f_out:
        pickle.dump(dv, f_out)


# DeploymentSpec(
#   flow=main,
#   name="model_training",
#   schedule=CronSchedule(cron="0 9 15 * *"),
#   flow_runner=SubprocessFlowRunner(),
#   tags=["homework"]
# )

main()
