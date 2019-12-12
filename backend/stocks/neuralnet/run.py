import schedule
import time
from os import getenv

from .model import get_stocks, prepare_data, process_data, get_model, save_model, train_model


def train():
    stocks = get_stocks()
    stocks = process_data(stocks)

    for name in ['ask', 'bid']:
        print(f'training model for {name}')
        input_columns = [
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

        model_data = prepare_data(stocks, input_columns)

        model = get_model(
            name,
            (model_data.x_train.shape[-2], model_data.x_train.shape[-1])
        )

        train_model(model, model_data)
        save_model(name, model)

def predict():
    pass

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
                train_interval = int(train_interval)
            schedule.every(train_interval).seconds.do(train)
        if task == 'predict':
            predict_interval = getenv('PREDICT_INTERVAL_SECONDS')
            if predict_interval is None:
                predict_interval = 60 * 2
            else:
                predict_interval = int(predict_interval)
            schedule.every(predict_interval).seconds.do(predict)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
