import schedule
import time
from datetime import datetime
from os import getenv
import os
import pandas as pd
import tempfile
import numpy as np

from .model import get_stocks, prepare_training_data, process_data, get_model, \
    save_model, train_model, get_latest_stocks, prepare_prediction_data, save_prediction, filter_data, \
    create_model


def get_input_columns(name: str):
    return [
        f"EUR_USD_{name}",
        f"USD_CHF_{name}",
        f"GBP_USD_{name}",
        f"AUD_USD_{name}",
        f"EUR_USD_{name}_EMA12",
        f"EUR_USD_{name}_EMA26",
        f"EUR_USD_{name}_MACD",
        f"EUR_USD_{name}_BU",
        f"EUR_USD_{name}_BD"
    ]


def train():
    print('Retrieving stocks data...')
    recv_start = datetime.now()
    stocks = get_stocks()
    print(f'Retrieved {len(stocks)} entries in {datetime.now() - recv_start}.')

    print('Processing stocks data...')
    stocks = process_data(stocks)

    for name in ['ask', 'bid']:
        print(f'Training model for "{name}"')
        start = datetime.now()

        input_columns = get_input_columns(name)

        filtered_data = filter_data(stocks, input_columns)
        print(
            f'Working with {len(filtered_data)} entries and {len(filtered_data.columns)} columns.')

        train_data = prepare_training_data(filtered_data)
    
        print('Loading model from database...')
        model = get_model(name)
        if model is None:
            print('Model not found, creating a new one.')
            model = create_model(
                (train_data.x_train.shape[-2], train_data.x_train.shape[-1]))

        train_model(model, train_data)

        print(f'Training time: {datetime.now()-start}')
        save_model(name, model)
        print(f'New model for "{name}"" has been saved.')


def predict(name: str, n=200):
    stocks = get_latest_stocks(n)
    stocks = process_data(stocks)

    input_columns = get_input_columns(name)

    stocks = pd.DataFrame([stocks.loc[:, input_columns].mean()])

    if np.isnan(np.sum(stocks.values)):
        raise Exception('NaN in data, try a higher entry count.')

    prediction_data = prepare_prediction_data(stocks)

    model = get_model(name)

    if model is None:
        raise Exception('No model available')

    prediction = model.predict_classes(prediction_data)

    save_prediction(name, prediction[0].item())


def main():
    tasks_str = getenv('TASKS')
    if tasks_str is None:
        raise Exception('TASKS env is required.')

    tasks = tasks_str.lower().split(',')

    for task in tasks:
        if task == 'train':
            train_interval = getenv('TRAIN_INTERVAL_SECONDS')
            if train_interval is None:
                train_interval = 60 * 2
            else:
                train_interval = float(train_interval)
            schedule.every(train_interval).seconds.do(train)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
