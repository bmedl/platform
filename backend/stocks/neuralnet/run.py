import schedule
import time
from datetime import datetime
from os import getenv

from .model import get_stocks, prepare_training_data, process_data, get_model, \
    save_model, train_model, get_latest_stocks, prepare_prediction_data, save_prediction


def train():
    print('Retrieving stocks data...')
    stocks = process_data(get_stocks())
    print(f'Retrieved {len(stocks)} entries.')

    for name in ['ask', 'bid']:
        print(f'Training model for {name}')
        start = datetime.now()
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

        model_data = prepare_training_data(stocks, input_columns)

        model = get_model(
            name,
            (model_data.x_train.shape[-2], model_data.x_train.shape[-1])
        )

        train_model(model, model_data)

        print(f'Training time: {datetime.now()-start}')
        save_model(name, model)
        print(f'New model {name} has been saved.')


def predict(name: str, n=1):
    stocks = process_data(get_latest_stocks(1))

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

    prediction_data = prepare_prediction_data(stocks, input_columns)
    model = get_model(
        name, (prediction_data.shape[-2], prediction_data.shape[-1]))

    prediction = model.predict(prediction_data)[0]

    save_prediction(name, prediction.item())


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
