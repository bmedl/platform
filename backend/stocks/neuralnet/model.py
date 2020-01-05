#!/usr/bin/env python

import keras
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import Model
import tempfile

from io import BytesIO
import pytz
from typing import Any
from datetime import datetime, timedelta
import v20
import os
import pandas as pd
from pandas import DataFrame
from django_pandas.io import read_frame
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, NamedTuple

from .util import get_timeseries, trim_dataset

import django
django.setup()


def get_batch_size():
    """
    Gets the batch size that should be used.
    """
    return 64


def get_stocks() -> pd.DataFrame:
    """
    Gets all the stock data from the database.
    """
    from stocks.stocks.models import Stock

    return read_frame(Stock.objects.all())


def get_latest_stocks(n=1) -> pd.DataFrame:
    """
    Gets the latest n the stock data entries from the database.
    """
    from stocks.stocks.models import Stock

    return read_frame(Stock.objects.all().order_by('-id')[:n])


def get_stocks_by_date_range(
    min: Optional[datetime] = None,
    max: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Gets stock data from the database based on date range.
    """
    from stocks.stocks.models import Stock

    if min is None:
        return get_stocks()

    if max is None:
        max = datetime.now()

    return read_frame(Stock.objects.filter(price_date__range=(min, max)))


def get_stocks_before_index(
    from_idx: int,
    n: int = 1
) -> pd.DataFrame:
    """
    Gets n stock data entries from the database before a given index.
    """
    from stocks.stocks.models import Stock

    return read_frame(Stock.objects.filter(id__range=(from_idx-n+1, from_idx)))


def get_stocks_by_index(
    i: int,
) -> pd.DataFrame:
    """
    Gets a stock data entry by a given index.
    """
    from stocks.stocks.models import Stock

    return read_frame(Stock.objects.filter(id=i))


def calculate_meta(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Calculates various values for a given column in a dataframe,
    then returns them. 

    These values include:
        - EMA (12)
        - EMA (26)
        - MACD
        - Upper bounds
        - Lower bounds
    """
    col_data = data[[col]].copy()

    ema12 = col_data.ewm(span=12, adjust=True).mean()
    ema26 = col_data.ewm(span=26, adjust=True).mean()
    macd = ema12 - ema26

    sma = col_data.rolling(window=20).mean()
    rstd = col_data.rolling(window=20).std()

    bu = sma + 2 * rstd
    bd = sma - 2 * rstd

    final_columns = [
        ema12.add_suffix('_EMA12'),
        ema26.add_suffix('_EMA26'),
        macd.add_suffix('_MACD'),
        bu.add_suffix('_BU'),
        bd.add_suffix('_BD')
    ]

    return pd.concat(final_columns, sort=False, axis=1)


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes all the data and calculates additional necessary values.
    """

    currencies = data.name.unique().astype(str)

    data = aggregate_data(data)

    all_data: List[pd.DataFrame] = [data]
    for name in ['ask', 'bid']:
        for currency in currencies:
            all_data.append(calculate_meta(data, f'{currency}_{name}'))

    return pd.concat(all_data, sort=False, axis=1).dropna()


def aggregate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates all the currencies into a single dataframe,
    then groups them by a time interval.
    """
    data = data.copy()
    data.event_date = pd.to_datetime(data.event_date)
    data = data.set_index('event_date')

    currencies = data.name.unique().astype(str)

    all_currencies: List[pd.DataFrame] = []
    for currency in currencies:
        data_currency = data.loc[data['name'] == currency, [
            'ask', 'bid']].copy().add_prefix(f'{currency}_')

        all_currencies.append(data_currency)

    return pd.concat(all_currencies, sort=False, axis=1).astype(float) \
        .groupby(pd.Grouper(freq='30s')) \
        .mean() \
        .dropna()


def prepare_model_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for learning.

    A dataframe is turned into a numpy array,
    then y is calculated in batches depending on whether
    the values of x ascended or descended in a time interval.
    """

    scaler = StandardScaler()

    output_col_num = 3

    x = data.values

    x = scaler.fit_transform(x)

    x, y = get_timeseries(
        x, 120, output_col_num, 0.05, 10)

    batch_size = get_batch_size()

    x = trim_dataset(x, batch_size)
    y = trim_dataset(y, batch_size)

    y = np_utils.to_categorical(y, 3)

    return x, y


def create_model(lstm_shape) -> Model:
    """
    Initializes a new Keras model.

    This model is already optimized.
    """

    model = Sequential()
    model.add(LSTM(64, input_shape=lstm_shape, return_sequences=True))
    model.add(Dropout(0.0744))
    model.add(Activation('relu'))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.0716))
    model.add(Activation('relu'))
    model.add(LSTM(64))
    model.add(Dropout(0.1262))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(
    model: Model,
    x: np.ndarray,
    y: np.ndarray
) -> Model:
    """
    Trains the model with the given dataset, and returns the final model.
    """

    early_stopping = EarlyStopping(patience=8, verbose=1, monitor='accuracy')

    model.fit(x,
              y,
              batch_size=get_batch_size(),
              epochs=40,
              verbose=1,
              callbacks=[early_stopping])

    return model


def get_model(name: str) -> Model:
    """
    Returns a model from the database.
    """
    from stocks.stocks.models import NetworkModel
    model = NetworkModel.objects.filter(name=name).first()

    if model is None:
        return None

    return load_model(BytesIO(model.model_blob))


def save_model(name: str, model: Model):
    """
    Saves a model in the database.
    """
    from stocks.stocks.models import NetworkModel

    blob = BytesIO()
    model.save(blob)
    blob.seek(0)

    NetworkModel.objects.update_or_create(
        name=name,
        defaults={
            'model_blob': blob.read()
        }
    )


def save_prediction(name: str, value: int, time_range: timedelta, price_date: datetime):
    """
    Saves a prediction in the database.
    """
    from stocks.stocks.models import Prediction

    Prediction(
        name=name,
        time_range=time_range,
        price_date=price_date,
        value=value
    ).save()


def save_backtest_result(name: str, date: datetime, actual: int, expected: int):
    from stocks.stocks.models import BacktestResult
    BacktestResult(
        name=name,
        price_date=date,
        expected=expected,
        actual=actual
    ).save()


def model_prediction(array: np.ndarray) -> int:
    if array.argmax() == 0:
        return 0
    elif array.argmax() == 1:
        return 1
    else:
        return -1
