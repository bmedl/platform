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

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv')

    try:
        stocks = read_frame(Stock.objects.all().order_by('-id')[:n])

        # This solves a lot of issues for whatever reason
        stocks.to_csv(tmp_path)
        return pd.read_csv(tmp_path)
    finally:
        os.remove(tmp_path)


def get_stocks() -> pd.DataFrame:
    """
    Gets all the data from the database,
    and read it into a dataframe.
    """
    from stocks.stocks.models import Stock

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv')

    try:
        stocks = read_frame(Stock.objects.all())

        # This solves a lot of issues for whatever reason
        stocks.to_csv(tmp_path)
        return pd.read_csv(tmp_path)
    finally:
        os.remove(tmp_path)


def calculate_meta(data: pd.DataFrame) -> pd.DataFrame:
    final_columns: List[pd.DataFrame] = []

    for col in data.columns:
        col_data = data[[col]].copy()

        final_columns.append(col_data)

        ema12 = col_data.ewm(span=12, adjust=True).mean().dropna()
        ema26 = col_data.ewm(span=26, adjust=True).mean().dropna()
        macd = ema12 - ema26

        sma = col_data.rolling(window=20).mean().dropna()
        rstd = col_data.rolling(window=20).std().dropna()

        bu = sma + 2 * rstd
        bd = sma - 2 * rstd

        final_columns.append(ema12.add_suffix('_EMA12'))
        final_columns.append(ema26.add_suffix('_EMA26'))
        final_columns.append(macd.add_suffix('_MACD'))
        final_columns.append(bu.add_suffix('_BU'))
        final_columns.append(bd.add_suffix('_BD'))

    return pd.concat(final_columns, sort=False, axis=1)


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
            'ask', 'bid']].copy().add_prefix(f'{currency}_')

        data_currencies.append(calculate_meta(data_currency))

    return pd.concat(data_currencies, sort=False, axis=1)


def filter_data(data, columns: List[str]) -> pd.DataFrame:
    return data.loc[:, columns].copy().groupby(pd.Grouper(freq='30s')).mean().dropna()


# TODO this is not the best
def prepare_training_data(data: pd.DataFrame) -> TrainingData:
    """
    Prepares data for learning.
    """

    scaler = StandardScaler()

    batch_size = 64

    output_col_num = 3

    train, test = train_test_split(
        data, train_size=0.8, test_size=0.2, shuffle=False)

    x = train.values

    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(
        test.values)

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


def prepare_prediction_data(data: pd.DataFrame) -> np.ndarray:
    values = pd.concat([data]*120).values
    
    return StandardScaler().fit_transform(values).reshape((1, 120, 9))


def create_model(lstm_shape) -> Model:
    """
    Initializes a new Keras model.
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
        data: TrainingData) -> Model:
    """
    Trains the model with the given dataset, and returns the final model.
    """

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.hdf5')
    try:
        checkpointer = ModelCheckpoint(
            filepath=tmp_path, save_best_only=True)

        early_stopping = EarlyStopping(patience=8, verbose=1)

        model.fit(data.x_train,
                  data.y_train,
                  batch_size=batch_size,
                  epochs=10,
                  verbose=1,
                  validation_data=(data.x_val, data.y_val),
                  callbacks=[checkpointer, early_stopping])

        model.evaluate(data.x_test, data.y_test)

        return load_model(tmp_path)
    finally:
        os.remove(tmp_path)


def get_model(name: str) -> Model:
    from stocks.stocks.models import NetworkModel
    model = NetworkModel.objects.filter(name=name).first()

    if model is None:
        return None

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
