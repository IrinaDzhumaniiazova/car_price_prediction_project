import logging
import dill
import json
import pandas as pd
from datetime import datetime

import os

path = os.environ.get('PROJECT_PATH', '..')

files = os.listdir(f'{path}/data/models')
files = [file.split('.')[0].split('_')[-1] for file in files]
model_number = max(files)
print(model_number)


with open(f'{path}/data/models/cars_pipe_{model_number}.pkl', 'rb') as file:
    model = dill.load(file)


def predict():
    results  = pd.DataFrame(columns=['id', 'prediction', 'price'])
    dirs = os.listdir(f'{path}/data/test')
    for f in dirs:
        f = str(f)
        paths = f'{path}/data/test/'+f
        with open(paths) as file:
            form = json.load(file)
            df = pd.DataFrame([form])
            y = model.predict(df)
            to_append = [f.split('.')[0], y[0], df.loc[0, 'price']]
            results.loc[len(results.index)]=to_append

    prediction_filename = f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    results.to_csv(prediction_filename, index=False)
    logging.info(f'Prediction is saved as {prediction_filename}')


if __name__ == '__main__':
    predict()
