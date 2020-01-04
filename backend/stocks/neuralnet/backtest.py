import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam

def data():
    dfAll = pd.read_csv("stocks_stock_202001022129.csv", sep=",")
    EUR_USD = dfAll[dfAll["name"]=="EUR_USD"]
    EUR_USD.event_date = pd.to_datetime(dfAll.event_date)
    EUR_USD.index = pd.to_datetime(EUR_USD.event_date)
    EUR_USD = EUR_USD.rename(columns={"bid": "EURUSD_bid", "ask": "EURUSD_ask"})

    USD_CHF = dfAll[dfAll["name"]=="USD_CHF"]
    USD_CHF.event_date = pd.to_datetime(dfAll.event_date)
    USD_CHF.index = pd.to_datetime(USD_CHF.event_date)
    USD_CHF = USD_CHF.rename(columns={"bid": "USDCHF_bid", "ask": "USDCHF_ask"})

    GBP_USD = dfAll[dfAll["name"]=="GBP_USD"]
    GBP_USD.event_date = pd.to_datetime(dfAll.event_date)
    GBP_USD.index = pd.to_datetime(GBP_USD.event_date)
    GBP_USD = GBP_USD.rename(columns={"bid": "GBPUSD_bid", "ask": "GBPUSD_ask"})

    AUD_USD = dfAll[dfAll["name"]=="AUD_USD"]
    AUD_USD.event_date = pd.to_datetime(dfAll.event_date)
    AUD_USD.index = pd.to_datetime(AUD_USD.event_date)
    AUD_USD = AUD_USD.rename(columns={"bid": "AUDUSD_bid", "ask": "AUDUSD_ask"})

    df = EUR_USD.groupby(pd.Grouper(freq='30s')).mean().dropna()
    df = df.merge(USD_CHF.groupby(pd.Grouper(freq='30s')).mean().dropna(), left_index=True, right_index=True)
    df = df.merge(GBP_USD.groupby(pd.Grouper(freq='30s')).mean().dropna(), left_index=True, right_index=True)
    df = df.merge(AUD_USD.groupby(pd.Grouper(freq='30s')).mean().dropna(), left_index=True, right_index=True)

    df["EURUSD_ask_EMA12"] = df["EURUSD_ask"].ewm(span=12,adjust=True).mean()
    df["EURUSD_ask_EMA26"] = df["EURUSD_ask"].ewm(span=26,adjust=True).mean()
    df["EURUSD_ask_MACD"] = df["EURUSD_ask_EMA12"]-df["EURUSD_ask_EMA26"]

    df["EURUSD_bid_EMA12"] = df["EURUSD_bid"].ewm(span=12,adjust=True).mean()
    df["EURUSD_bid_EMA26"] = df["EURUSD_bid"].ewm(span=26,adjust=True).mean()
    df["EURUSD_bid_MACD"] = df["EURUSD_bid_EMA12"]-df["EURUSD_bid_EMA26"]

    symbol = "EURUSD_ask"

    sma = df[symbol].rolling(window=20).mean()

    # calculate the standar deviation
    rstd = df[symbol].rolling(window=20).std()

    upper_band = sma + 2 * rstd
    upper_band = upper_band.rename(columns={symbol: 'upper'})
    lower_band = sma - 2 * rstd
    lower_band = lower_band.rename(columns={symbol: 'lower'})
    df["EURUSD_ask_BU"] = upper_band
    df["EURUSD_ask_BD"] = lower_band
    symbol = "EURUSD_bid"

    sma = df[symbol].rolling(window=20).mean()

    # calculate the standar deviation
    rstd = df[symbol].rolling(window=20).std()

    upper_band = sma + 2 * rstd
    upper_band = upper_band.rename(columns={symbol: 'upper'})
    lower_band = sma - 2 * rstd
    lower_band = lower_band.rename(columns={symbol: 'lower'})

    df["EURUSD_bid_BU"] = upper_band
    df["EURUSD_bid_BD"] = lower_band

    df = df.dropna()
    return df


def get_timeseries(df, time_steps, output_col_num, limit, predict_interval):
    dim_0 = df.shape[0] - time_steps
    dim_1 = 9
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = df[i:time_steps+i]
        if (df[time_steps+i, output_col_num] - df[time_steps+i-predict_interval, output_col_num] > df[time_steps+i, output_col_num]*limit):
            y[i] = 1
        elif (abs(df[time_steps+i, output_col_num] - df[time_steps+i-predict_interval, output_col_num]) < df[time_steps+i, output_col_num]*limit):
            y[i] = 0
        else:
            y[i] = -1
    return x, y


def trim_dataset(df, batch_size):
    no_of_rows_drop = df.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return df[:-no_of_rows_drop]
    else:
        return df


def ModelPrediction(array):
    if array.argmax() == 0:
        return 0
    elif array.argmax() == 1:
        return 1
    else:
        return -1


#Data processing
df = data()
scaler = StandardScaler()
batch_size = 64
ask_input_cols = ["EURUSD_ask", "USDCHF_ask", "GBPUSD_ask", "AUDUSD_ask", "EURUSD_ask_EMA12", "EURUSD_ask_EMA26", "EURUSD_ask_MACD", "EURUSD_ask_BU", "EURUSD_ask_BD" ]
ask_output_col_num = 3
startDate = pd.to_datetime("2019-12-18")
dfScaler = df[df.index<startDate].loc[:, ask_input_cols]

dfTrain = scaler.fit_transform(dfScaler)
X, Y = get_timeseries(dfTrain, 120, ask_output_col_num, 0.05, 10)

XTrain = trim_dataset(X, batch_size)
YTrain = np_utils.to_categorical(trim_dataset(Y, batch_size), 3)

#Model definition
model = Sequential()
model.add(LSTM(64, input_shape=(XTrain.shape[-2], XTrain.shape[-1]), return_sequences=True))
model.add(Dropout(0.0744))
model.add(Activation('relu'))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.0716))
model.add(Activation('relu'))
model.add(LSTM(64))
model.add(Dropout(0.1262))
model.add(Activation('relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001), metrics=['accuracy'])

#callbacks definition
ask_checkpointer = ModelCheckpoint(filepath='ask_weights.hdf5', save_best_only = True)
early_stopping=EarlyStopping(patience=8, verbose=1)
ask_logger = CSVLogger('ask_training_log.csv',separator=',', append= True)

#initial train
model.fit(XTrain,
          YTrain,
          batch_size=batch_size,
          epochs=40,
          verbose=1,
          callbacks = [ask_checkpointer, early_stopping, ask_logger])


#backtest
actual = startDate.strftime("%Y-%m-%d")
for index, row in df[(df.index >= startDate)].iterrows():
    actualDF = get_timeseries(scaler.transform(df[df.index <= index].loc[:, ask_input_cols].tail(121)),
                                                120, 
                                                ask_output_col_num, 
                                                0.05, 
                                                10)
    print(index, ModelPrediction(model.predict(actualDF[0])), int(actualDF[1][0]))
    if actual != index.strftime("%Y-%m-%d"):
        #retrain the network in every day
        actual=index.strftime("%Y-%m-%d")
        
        dfScaler = df[df.index<=index].loc[:, ask_input_cols]
        dfTrain = scaler.fit_transform(dfScaler)
        X, Y = get_timeseries(dfTrain, 120, ask_output_col_num, 0.05, 10)
        XTrain = trim_dataset(X, batch_size)
        YTrain = np_utils.to_categorical(trim_dataset(Y, batch_size), 3)
        model.fit(XTrain,
              YTrain,
              batch_size=batch_size,
              epochs=40,
              verbose=1,
              callbacks = [ask_checkpointer, early_stopping, ask_logger])