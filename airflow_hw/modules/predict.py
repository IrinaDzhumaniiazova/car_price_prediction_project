import logging
import dill
import json
import pandas as pd
from datetime import datetime

import os


with open('../data/models/cars_pipe_202208211029.pkl', 'rb') as file:
    model = dill.load(file)


def predict():
    results  = pd.DataFrame(columns=['id', 'prediction', 'price'])
    dirs = os.listdir('../data/test')
    for f in dirs:
        f = str(f)
        path = '../data/test/'+f
        with open(path) as file:
            form = json.load(file)
            df = pd.DataFrame([form])
            y = model.predict(df)
            to_append = [f.split('.')[0], y[0], df.loc[0, 'price']]
            results.loc[len(results.index)]=to_append

    prediction_filename = f'../data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    results.to_csv(prediction_filename, index=False)
    logging.info(f'Prediction is saved as {prediction_filename}')


if __name__ == '__main__':
    predict()
