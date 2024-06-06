from pymongo import MongoClient
from tqdm.asyncio import tqdm
import logging
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, PReLU
from keras.callbacks import EarlyStopping
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras import metrics, optimizers
import numpy as np
import pandas as pd
import time
import datetime
from datetime import datetime, timedelta
import json
import requests
import os
import yfinance as yf
import asyncio
import aiohttp
# from aiohttp_requests import requests


def fetch_data_from_mongodb(ticker, db_name='SMP_uni', collection_field='data'):
    """
    Fetches stock data for a given ticker from MongoDB.

    Args:
        ticker (str): Stock ticker.
        db_name (str): The name of the MongoDB database.
        collection_field (str): The field name in MongoDB that contains the data.

    Returns:
        pd.DataFrame: DataFrame containing the stock data for the ticker.
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[ticker]

    # Fetch data from MongoDB
    cursor = collection.find()
    data = list(cursor)

    if not data:
        print(f"No data found for ticker {ticker}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Extract nested data
    data_df = pd.json_normalize(df[collection_field])

    # Combine with main DataFrame
    df = pd.concat([df.drop(columns=[collection_field]), data_df], axis=1)

    # Drop Time.$date if it exists
    if 'Time.$date' in df.columns:
        df = df.drop(columns=['Time.$date'])

    if 'company_name' in df.columns:
        df = df.drop(columns=['company_name'])

    if 'ticker' in df.columns:
        df = df.drop(columns=['ticker'])

    # Rename _id to Date and set it as the index
    df.rename(columns={'_id': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)

    return df


def preprocess_data(ticker):
    # if u call preprocess_date() ---> it will set daytogoback=26
    # if u call preprocess_date(daytogoback=n) ---> it will set daytogoback=n
    # Function to calculate RSI
    def calc_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Function to calculate MACD
    def calc_macd(df, short_window=12, long_window=26, signal_window=9):
        df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(
            span=signal_window, adjust=False).mean()
        return df
    df = fetch_data_from_mongodb(ticker)
    # Converting relavent columns to float
    df[
        ['Open', 'High', 'Low', 'Close', 'Volume']
    ] = df[[
        'Open', 'High', 'Low', 'Close', 'Volume']
    ].astype(float)
    df.sort_index(ascending=True, inplace=True)

    # Create 3-Day EMA
    df['EMA_3'] = df['Close'].ewm(span=3, adjust=False).mean()

    # Create 5-Day EMA
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

    # Calculate EMA percentage change
    df['EMA_pct_change'] = ((df['EMA_3'] - df['EMA_5']) / 100) * 100

    # Create 14-Day RSI
    df['RSI'] = calc_rsi(df['Close'])

    # Create 9-Day MACD Signal
    df = calc_macd(df)

    df.dropna(inplace=True)

    if df.isnull().values.any():
        print('Missing Values found. Filling missing valued...')
        df = df.fillna(method='ffill').fillna(method='bfill')

    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)

    return scaled_df, scaler


# Function to create windowed DataFrame
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=7):
    def str_to_datatime(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d')

    # Convert string date to datetime object
    first_date = str_to_datatime(first_date_str)
    last_date = str_to_datatime(last_date_str)

    # Ensure the dataframe index is datetime for easier slicing
    dataframe = dataframe.copy()
    dataframe.index = pd.to_datetime(dataframe.index)

    # filter the dataframe for the given date range
    dataframe = dataframe[(dataframe.index >= first_date)
                          & (dataframe.index <= last_date)]

    # Check the dataframe has enough rows for the window size
    if len(dataframe) < n + 1:
        raise ValueError(
            f"Error: window of size {n} is too larg for the given data range!")

    features = ['Close', 'EMA_pct_change', 'RSI', 'MACD', 'MACD_Signal']
    windowed_data = {}

    for feature in features:
        windowed_data[feature] = np.lib.stride_tricks.sliding_window_view(
            dataframe[feature].values, window_shape=n + 1
        )

    dates = dataframe.index[n:]

    # Extract features and construct the target array
    X_features = np.array([windowed_data[feature][:, :-1]
                          for feature in features]).transpose(1, 2, 0)
    Y_target = windowed_data['Close'][:, -1]

    feature_columns = [f'Target-{n-i}' for i in range(n)]
    ret_df = pd.DataFrame(index=dates)
    for idx, feature in enumerate(feature_columns):
        ret_df[feature] = list(X_features[:, idx])

    ret_df['Target'] = Y_target

    return ret_df.reset_index().rename(columns={'index': 'Date'})


# Function to convert windowed DataFrame into date, X, y
def windowed_df_to_date_X_y(windowed_dataframe, num_features=2):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]

    middle_matrix = np.array([np.array(x) for x in df_as_np[:, 1:-1].tolist()])
    num_samples = len(dates)

    # Reshape X to have correct shape (samples, time steps, features)
    if middle_matrix.size % (num_samples * num_features) != 0:
        raise ValueError(
            'Total Elements in middle metrix not divisible by expected number of features per sample times samples')

    X = middle_matrix.reshape((num_samples, -1, num_features))
    Y = df_as_np[:, -1].astype(np.float32)

    return dates, X.astype(np.float32), Y

# Function to splite data


def split_date(dates, X, y, train_frac=0.8, val_frac=0.1):
    # Ensure fraction are whitin a valid range
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(
            'Fractions must be between 0 and 1, and their sum must be less than 1.')

    total_length = len(dates)
    train_size = int(total_length * train_frac)
    val_size = int(total_length * val_frac)

    if total_length < train_size + val_size:
        raise ValueError(
            'The dataset is too small to split with the given fractions.')

    # Calculate the splite indices
    train_end = train_size
    val_end = train_size + val_size

    # Split the Data
    date_train, X_train, y_train = dates[:
                                         train_end], X[:train_end], y[:train_end]
    date_val, X_val, y_val = dates[train_end:
                                   val_end], X[train_end:val_end], y[train_end:val_end]
    date_test, X_test, y_test = dates[val_end:], X[val_end:], y[val_end:]

    return (date_train, X_train, y_train), (date_val, X_val, y_val), (date_test, X_test, y_test)

# Function to build and compile the LSTM model


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='leaky_relu'))
    model.add(Dense(25, activation='selu'))
    model.add(Dense(25, activation='elu'))
    PReLU()
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mean_squared_error'])
    return model


# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, model_name):
    # Ensure the Models directory exists
    models_dir = 'Models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_checkpoint = ModelCheckpoint(
        f'{models_dir}/{model_name}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=100, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(
        X_val, y_val), callbacks=[model_checkpoint, early_stopping], verbose=0)

# Function to evaluate the model


def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)
    return rmse, r2


# Function to calculate mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def store_prediction_in_mongodb(rmse, r2, mape, accuracy, model_name):

    rmse = rmse.item() if isinstance(rmse, np.number) else rmse
    r2 = r2.item() if isinstance(r2, np.number) else r2
    mape = mape.item() if isinstance(mape, np.number) else mape
    accuracy = accuracy.item() if isinstance(accuracy, np.number) else accuracy

    data = {
        'Time': int(datetime.now().timestamp()),
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': accuracy
    }

    client = MongoClient('mongodb://localhost:27017/')
    db = client['SMP_train_data_uni']
    collection = db[model_name]

    today_date = datetime.today().now().strftime('%Y-%m-%d')

    existing_document = collection.find_one({'_id': today_date})

    if existing_document:
        collection.update_one({'_id': today_date}, {'$set': {'data': data}})
    else:
        collection.insert_one({'_id': today_date, 'data': data})


def SMP_madel(model_name):
    df, scaler = preprocess_data(stock)

    start_date = df.index[7]
    end_date = df.index[-2]
    windowed_df = df_to_windowed_df(df, start_date, end_date, n=5)
    if windowed_df is None:
        return

    dates, X, y = windowed_df_to_date_X_y(windowed_df, 5)

    train, val, test = split_date(dates, X, y, 0.8, 0.1)
    date_train, X_train, y_train = train
    date_val, X_val, y_val = val
    date_test, X_test, y_test = test

    model = build_model((X_train.shape[1], 5))
    train_model(model, X_train, y_train, X_val, y_val, model_name)

    rmse, r2 = evaluate_model(model, X_test, y_test)

    print(f'Test RMSE: {rmse}')
    print(f'Test R-squared: {r2}')

    predictions = model.predict(X_test, verbose=0)
    mape = mean_absolute_percentage_error(y_test, predictions)
    accuracy = 100 - mape
    print(f'Test MAPE: {mape}')
    print(f'Test Accuracy: {accuracy}')

    store_prediction_in_mongodb(rmse, r2, mape, accuracy, model_name)


def get_collection_names(db_name='SMP_uni'):
    """
    Fetches the list of collection names in a MongoDB database.

    Args:
        db_name (str): The name of the MongoDB database.

    Returns:
        list: A list of collection names.
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    return db.list_collection_names()


stocks = get_collection_names()
for stock in tqdm(stocks):
    model_name = stock
    stk = stock
    try:
        SMP_madel(model_name)
    except Exception as e:
        print(f'Error processing {stock}: {e}')
        continue
