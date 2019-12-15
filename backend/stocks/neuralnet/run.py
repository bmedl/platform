import schedule
import time
from datetime import datetime, timedelta
from os import getenv
import os
import pandas as pd
import tempfile
import numpy as np

from .model import get_stocks, prepare_model_data, process_data, get_model, \
    save_model, train_model, get_stocks_by_date_range, save_prediction, \
    create_model, get_latest_stocks, get_stocks_before_index, get_stocks_by_index


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
        print(f'Training model for "{name}".')
        start = datetime.now()

        input_columns = get_input_columns(name)

        filtered_data = stocks.loc[:, input_columns]

        print(
            f'Working with {len(filtered_data)} entries and {len(filtered_data.columns)} columns.')

        x, y = prepare_model_data(filtered_data)

        print('Loading model from database...')
        model = get_model(name)
        if model is None:
            print('Model not found, creating a new one.')
            model = create_model(
                (x.shape[-2], x.shape[-1]))

        train_model(model, x, y)

        print(f'Training time: {datetime.now()-start}')
        save_model(name, model)
        print(f'New model for "{name}"" has been saved.')


def predict(name: str, time_range: timedelta = None):
    latest_stock = get_latest_stocks(1)
    latest_stock.price_date = pd.to_datetime(latest_stock.price_date)

    if time_range is None:
        time_range = timedelta(hours=24)

    latest_date = latest_stock.iloc[0]['price_date']

    stocks = get_stocks_by_date_range(min=latest_date - time_range, max=latest_date)
    if len(stocks) < 8000:
        stocks = get_stocks_before_index(latest_stock.id, 8000)
        time_range = latest_date - get_stocks_by_index(latest_stock.iloc[0]['id'] - 8000 + 1).iloc[0]['price_date']
        if time_range is None:
            raise Exception('Should never happen')

    stocks = process_data(stocks)

    input_columns = get_input_columns(name)

    filtered_data = stocks.loc[:, input_columns]

    x, _ = prepare_model_data(filtered_data)

    model = get_model(name)

    if model is None:
        raise Exception('No model available')

    prediction = model.predict_classes(x)
 
    save_prediction(name, prediction[0].item(), time_range, latest_date)


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
