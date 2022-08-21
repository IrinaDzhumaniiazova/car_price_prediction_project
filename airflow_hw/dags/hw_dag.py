import logging
import os, sys
from datetime import datetime

import dill
import pandas as pd
import json
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from airflow.models import DAG
from airflow.operators.python import PythonOperator

path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

path_model = os.environ.get('PROJECT_PATH', '..')


files = os.listdir(f'{path_model}/data/models')
files = [file.split('.')[0].split('_')[-1] for file in files]
model_number = max(files)


with open(f'{path_model}/data/models/cars_pipe_{model_number}.pkl', 'rb') as file:
    model = dill.load(file)


from modules.pipeline import pipeline, filter_data, remove_outliers, create_features
from modules.predict import predict


args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}


with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag = dag
    )

    prediction = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag = dag
    )

    pipeline >> prediction
