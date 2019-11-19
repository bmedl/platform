#!/usr/bin/env python

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import pytz
from typing import Any
from datetime import datetime
import v20
import os
import pandas as pd
from django_pandas.io import read_frame
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import django
django.setup()

from stocks.stocks.models import Stock

stocks = read_frame(Stock.objects.all().filter(name='EUR_USD'))

def get_timeseries(df, time_steps):
    dim_0 = df.shape[0] - time_steps
    dim_1 = df.shape[1]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = df[i:time_steps+i]
        y[i] = df[time_steps+i, 3]

    return x, y


def trim_dataset(df, batch_size):
    no_of_rows_drop = df.shape[0] % batch_size
    if(no_of_rows_drop > 0):
        return df[:-no_of_rows_drop]
    else:
        return df


scaler = StandardScaler()

data_cols = ['bid', 'ask', 'closeout_bid', 'closeout_ask']

stock_train, stock_test = train_test_split(
    stocks, train_size=0.8, test_size=0.2, shuffle=False)

stock_x = stock_train.loc[:, data_cols].values

stock_x_train = scaler.fit_transform(stock_x)
stock_x_test = scaler.transform(stock_test.loc[:, data_cols].values)

stock_x_train, stock_y_train = get_timeseries(stock_x_train, 360)
stock_x_train = trim_dataset(stock_x_train, 128)
stock_y_train = trim_dataset(stock_y_train, 128)

stock_x_val_test, stock_y_val_test = get_timeseries(stock_x_test, 360)
stock_x_val, stock_x_test = np.split(trim_dataset(stock_x_val_test, 128), 2)
stock_y_val, stock_y_test = np.split(trim_dataset(stock_y_val_test, 128), 2)

checkpointer = ModelCheckpoint(
    filepath='weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model = Sequential()
model.add(LSTM(128, input_shape=(
    stock_x_train.shape[-2], stock_x_train.shape[-1])))

model.add(Dense(120))
model.add(Activation('linear'))
model.add(Dense(60))
model.add(Activation('linear'))
model.add(Dense(30))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

optimizer = Adam(lr=0.01)
model.compile(loss='MSE', optimizer=optimizer)

model.fit(stock_x_train, stock_y_train, batch_size=128, epochs=10, verbose=1,
          validation_data=(stock_x_val, stock_y_val), callbacks=[checkpointer])
