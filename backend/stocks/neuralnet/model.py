#!/usr/bin/env python

import keras
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import Model
import tempfile

from io import BytesIO
import pytz
from typing import Any
from datetime import datetime
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

batch_size = 64


class TrainingData(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def get_latest_stocks(n=1):
    from stocks.stocks.models import Stock

    return read_frame(Stock.objects.all().order_by('-id')[:n])


def get_stocks() -> pd.DataFrame:
    """
    Gets all the data from the database,
    and read it into a dataframe.
    """

    from stocks.stocks.models import Stock
    return read_frame(Stock.objects.all())


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes all the data and calculates additional necessary values.
    """
    currencies = data.name.unique().astype(str)

    data.event_date = pd.to_datetime(data.event_date)
    data = data.set_index('event_date')

    data_currencies: List[pd.DataFrame] = []
    for currency in currencies:
        data_currency = data.loc[data['name'] == currency, [
            'bid', 'ask']].add_prefix(f'{currency}_')

        data_currency[f'{currency}_ask_EMA12'] = data_currency[f'{currency}_ask'].ewm(
            span=12, adjust=True).mean()
        data_currency[f'{currency}_ask_EMA26'] = data_currency[f'{currency}_ask'].ewm(
            span=26, adjust=True).mean()
        data_currency[f'{currency}_ask_MACD'] = data_currency[f'{currency}_ask_EMA12'] - \
            data_currency[f'{currency}_ask_EMA26']

        data_currency[f'{currency}_ask_BU'] = data_currency[f'{currency}_ask'].rolling(
            window=20).mean() + data_currency[f'{currency}_ask'].rolling(window=10).std() * 2
        data_currency[f'{currency}_ask_BD'] = data_currency[f'{currency}_ask'].rolling(
            window=20).mean() - data_currency[f'{currency}_ask'].rolling(window=10).std() * 2

        data_currency[f'{currency}_bid_EMA12'] = data_currency[f'{currency}_bid'].ewm(
            span=12, adjust=True).mean()
        data_currency[f'{currency}_bid_EMA26'] = data_currency[f'{currency}_bid'].ewm(
            span=26, adjust=True).mean()
        data_currency[f'{currency}_bid_MACD'] = data_currency[f'{currency}_bid_EMA12'] - \
            data_currency[f'{currency}_bid_EMA26']

        data_currency[f'{currency}_bid_BU'] = data_currency[f'{currency}_bid'].rolling(
            window=20).mean() + data_currency[f'{currency}_bid'].rolling(window=10).std() * 2
        data_currency[f'{currency}_bid_BD'] = data_currency[f'{currency}_bid'].rolling(
            window=20).mean() - data_currency[f'{currency}_bid'].rolling(window=10).std() * 2

        data_currencies.append(data_currency)

    return pd.concat(data_currencies, sort=False).groupby(pd.Grouper(freq='30s')).mean().dropna()


# TODO this is not the best
def prepare_training_data(data: pd.DataFrame, input_columns: List[str]) -> TrainingData:
    """
    Prepares data for learning.
    """

    scaler = StandardScaler()

    batch_size = 64

    output_col_num = 3

    train, test = train_test_split(
        data, train_size=0.8, test_size=0.2, shuffle=False)

    x = train.loc[:, input_columns].values

    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(
        test.loc[:, input_columns].values)

    x_train, y_train = get_timeseries(
        x_train, 120, output_col_num, 0.05, 10)
    x_train = trim_dataset(x_train, batch_size)
    y_train = trim_dataset(y_train, batch_size)

    x_val_test, y_val_test = get_timeseries(
        x_test, 120, output_col_num, 0.05, 10)
    x_val, x_test = np.split(
        trim_dataset(x_val_test, batch_size), 2)
    y_val, y_test = np.split(
        trim_dataset(y_val_test, batch_size), 2)

    y_train = np_utils.to_categorical(y_train, 3)
    y_val = np_utils.to_categorical(y_val, 3)
    y_test = np_utils.to_categorical(y_test, 3)

    return TrainingData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test
    )


def prepare_prediction_data(data: pd.DataFrame, input_columns: List[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data.loc[:, input_columns].values)


def create_model(lstm_shape) -> Model:
    """
    Initializes a new Keras model.
    """

    model = Sequential()
    model.add(LSTM(64, input_shape=lstm_shape, return_sequences=True))
    model.add(Dropout(0.0744))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.0716))
    model.add(LSTM(64))
    model.add(Dropout(0.1262))
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(
        model: Model,
        data: TrainingData) -> Model:
    """
    Trains the model with the given dataset, and returns the final model.
    """

    checkpointer = ModelCheckpoint(
        filepath='model.hdf5', save_best_only=True)

    early_stopping = EarlyStopping(patience=8, verbose=1)

    model.fit(data.x_train,
                data.y_train,
                batch_size=batch_size,
                epochs=10,
                verbose=1,
                validation_data=(data.x_val, data.y_val),
                callbacks=[checkpointer, early_stopping])

    model.evaluate(data.x_test, data.y_test)

    return load_model('model.hdf5')



def get_model(
    name: str,
    lstm_shape: Optional[Tuple[int, int]]
) -> Model:
    from stocks.stocks.models import NetworkModel
    model = NetworkModel.objects.filter(name=name).first()

    if model is None:
        return create_model(lstm_shape)
    else:
        return load_model(BytesIO(model.model_blob))


def save_model(name: str, model: Model):
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


def save_prediction(name: str, value: int):
    from stocks.stocks.models import Prediction

    Prediction(
        name=name,
        value=value
    ).save()
